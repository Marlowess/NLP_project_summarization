import sys
sys.path.append('../')

import pandas as pd
import os
import subprocess
from utils.analysis import create_summary_analysis
from glimpse.evaluate.evaluate_seahorse_metrics_samples_custom import evaluate_with_seahorse_custom, QUESTION_MAP
import pickle
from utils.constants import CANDIDATES_CREATION_PREFIX, INIT_STEP_PREFIX, PREPROCESSING_PREFIX, RSA_PREFIX, EVALUATION_PREFIX, PROCESSED_DATA_PATH, INPUT_SETTINGS_KEYS_TYPES_AND_DEFAULT_PIPELINE
from utils.constants import CANDIDATES_OUTPUT_PATH, INPUT_DATA_PATH, PROCESSED_DATA_PATH, RSA_OUTPUT_DIR, OUTPUT_LOGS_DIR, WARNING_PREFIX
from handler_abstract import AbstractHandler
from contextlib import redirect_stdout
import time

import nltk
nltk.download('punkt_tab')
class PipelineHandler(AbstractHandler):
    """
    This class handles all the operations related to the multi-language extension.
    It takes a configuration dictionary as a JSON and performs all the steps needed to complete the pipeline: 
    - input file processing
    - generation of extractive and abstractive candidates
    - RSA computation
    - Evalutation using different approaches
    """
    
    def __init__(self, settings, run_name: str = None):
        super(PipelineHandler, self).__init__(settings, INPUT_SETTINGS_KEYS_TYPES_AND_DEFAULT_PIPELINE, run_name) # Call the superclass init
        
        # Paths and prefixes
        self.input_data_path = INPUT_DATA_PATH.format(root_path=self.base_path)
        self.processed_data_path = PROCESSED_DATA_PATH.format(output_path=self.output_path, run_name=self.run_name)
        self.candidates_output_path = CANDIDATES_OUTPUT_PATH.format(output_path=self.output_path, run_name=self.run_name)
        self.rsa_output_dir = RSA_OUTPUT_DIR.format(output_path=self.output_path, run_name=self.run_name)
        self.output_logs_dir = OUTPUT_LOGS_DIR.format(output_path=self.output_path, run_name=self.run_name)
        self.safe_mode = True if self.old_run_loaded else False # Safe mode is enabled by default if the old run is loaded
        self._log_message(INIT_STEP_PREFIX, f"Final settings: {settings}")
        self.statistics = {} # Dictionary of statistics
        self._process_input_data()
        self.rsa_paths = {
            "abstractive": None if not self.old_run_loaded else self._find_file_with_wildcard(f"{self.rsa_output_dir}/abstractive", "*.pk"), 
            "extractive": None if not self.old_run_loaded else self._find_file_with_wildcard(f"{self.rsa_output_dir}/extractive", "*.pk")
        }

    def _create_folder_if_not_exists(self, path):
        """
        This utility method checks if a folder already exists.
        If it does not exist, it creates it.
        """
        if not os.path.exists(path):
            os.makedirs(path)

    def set_safe_mode(self, safe_mode: bool):
        """
        This method sets the safe mode for the pipeline
        """
        self.safe_mode = safe_mode

    def _check_safe_mode(self):
        """
        This method checks if the safe mode is enabled
        """
        if self.safe_mode:
            self._log_message(WARNING_PREFIX, "Safe mode enabled. Skipping the operation.")
            self._log_message(WARNING_PREFIX, "If you want to disable it, please call the set_safe_mode method with False as argument.")
            return True
        return False

    def _process_input_data(self):
        """
        This method processes input data and prepare them to perform the pipeline
        """
        start_time = time.time()
        self._log_message(PREPROCESSING_PREFIX, "Preparing input files")
        if self._check_safe_mode(): return
        self._create_folder_if_not_exists(self.processed_data_path)

        for file in self.settings.get('input_files_to_process'):
            try:
                dataset = pd.read_csv(f"{self.input_data_path}/{file}")
                sub_dataset = dataset[['id','review', 'metareview']]
                sub_dataset.rename(columns={"review": "text", "metareview": "gold"}, inplace=True)
                self._log_message(PREPROCESSING_PREFIX, f"File {file} found. Managing it.")

                output_file_path = f"{self.processed_data_path}/{file}"
                sub_dataset.to_csv(output_file_path, index=False)
                self._log_message(PREPROCESSING_PREFIX, f"Saved file at {output_file_path}")
            except:
                self._log_message(PREPROCESSING_PREFIX, f"Input file {file} not found or already managed. Skipping it.")
        self.statistics['preprocessing_time'] = time.time() - start_time

    def perform_extractive_step(self):
        """
        This methods runs the generation of candidates using the extractive mode
        """
        start_time = time.time()
        self._log_message(CANDIDATES_CREATION_PREFIX, "Running the extractive step.")
        if self._check_safe_mode(): return
        result_extractive = subprocess.run([
            "python",
            f"{self.base_path}/glimpse/data_loading/generate_extractive_candidates.py",
            "--dataset_path", f"{self.processed_data_path}/{self.settings.get('dataset_name')}",
            "--limit", str(self.settings.get('limit')),
            "--output_dir", self.candidates_output_path,
            "--scripted-run" if self.settings.get('print_output_path') else ''
            ], capture_output=True, text=True)

        return_code_extractive = result_extractive.returncode  # return value of the process
        task_complete_success = return_code_extractive == 0 # Boolean indicating if the job completed successfully
        self._log_message(CANDIDATES_CREATION_PREFIX, f"Task completed: {'OK' if task_complete_success else 'KO'}")

        if task_complete_success:
            self.extractive_candidates_path = result_extractive.stdout.split('\n')[-2]
            self._log_message(CANDIDATES_CREATION_PREFIX, f"Extractive dataset path: {self.extractive_candidates_path}")
            self._perform_rsa(self.extractive_candidates_path, 'extractive')
        else:
            self._log_message(CANDIDATES_CREATION_PREFIX, f"Occurred error: {result_extractive.stderr}")
        self.statistics['extractive_time'] = time.time() - start_time

    def perform_abstractive_step(self):
        """
        This methods runs the generation of candidates using the abstractive mode
        """
        start_time = time.time()
        self._log_message(CANDIDATES_CREATION_PREFIX, "Running the abstractive step.")
        if self._check_safe_mode(): return
        result_abstractive = subprocess.run([
            "python",
            f"{self.base_path}/glimpse/data_loading/generate_abstractive_candidates.py",
            "--dataset_path", f"{self.processed_data_path}/{self.settings.get('dataset_name')}",
            "--batch_size", str(self.settings.get('batch_size')),
            "--device", str(self.settings.get('device')),
            "--limit", str(self.settings.get('limit')),
            "--output_dir", self.candidates_output_path,
            "--model_name", str(self.settings.get('abstractive_model')),
            "--scripted-run" if self.settings.get('print_output_path') else ''
            ], capture_output=True, text=True)
        
        return_code_abstractive = result_abstractive.returncode  # return value of the process
        task_complete_success = return_code_abstractive == 0 # Boolean indicating if the job completed successfully
        self._log_message(CANDIDATES_CREATION_PREFIX, f"Task completed: {'OK' if task_complete_success else 'KO'}")

        if task_complete_success:
            self.abstractive_candidates_path = result_abstractive.stdout.split('\n')[-2]
            self._log_message(CANDIDATES_CREATION_PREFIX, f"Abstractive dataset path: {self.extractive_candidates_path}")
            self._perform_rsa(self.abstractive_candidates_path, 'abstractive')
        else:
            self._log_message(CANDIDATES_CREATION_PREFIX, f"Occurred error: {result_abstractive.stderr}")
        self.statistics['abstractive_time'] = time.time() - start_time

    def _perform_rsa(self, candidates_path, step):
        """
        This method receives path of candidates record and performs the RSA method on them
        """
        start_time = time.time()
        self._log_message(RSA_PREFIX, "Computing the RSA score...")
        if self._check_safe_mode(): return
        result_rsa = subprocess.run([
            "python",
            f"{self.base_path}/glimpse/src/compute_rsa.py",
            "--summaries", candidates_path,
            "--output_dir", f"{self.rsa_output_dir}/{step}",
            "--scripted-run" if self.settings.get('print_output_path') else ''
            ], capture_output=True, text=True)

        return_code_rsa = result_rsa.returncode  # return value of the process
        rsa_task_success = return_code_rsa == 0
        self._log_message(RSA_PREFIX, f"Task completed: {'OK' if rsa_task_success else 'KO'}")
        
        if rsa_task_success:
            rsa_path = result_rsa.stdout.split('\n')[-2]
            self._log_message(RSA_PREFIX, f"RSA pickle path for the {step}: {rsa_path}")
            self.rsa_paths[step] = rsa_path
        else:
            self._log_message(RSA_PREFIX, f"Occurred error: {result_rsa.stderr}")
        self.statistics[f'{step}_rsa_time'] = time.time() - start_time

    def perform_evaluation(self, evaluation_name):
        """
        This method runs the evaluation step.
        :param: evaluation_name: seahorse, batbert, common
        """

        start_time = time.time()
        # Check if the output folder for logs already exists
        # output_folder_log_files = self.base_path + '/' + self.settings.get('output_log_files_path')
        self._create_folder_if_not_exists(self.output_logs_dir)

        # Get the abstractive data
        with open(self.rsa_paths.get('abstractive'), 'rb') as f:
            abstractive_data = pickle.load(f)

        # Get the extractive data
        with open(self.rsa_paths.get('extractive'), 'rb') as f:
            extractive_data = pickle.load(f)

        # Create the analysis dataframes
        abstractive_analysis = create_summary_analysis(abstractive_data, 'abstractive')
        extractive_analysis = create_summary_analysis(extractive_data, 'extractive')

        if evaluation_name == 'seahorse':
            self._perform_seahorse_evaluation(abstractive_analysis, extractive_analysis, self.output_logs_dir)
        else:
            self._log_message(EVALUATION_PREFIX, f"Evaluation named {evaluation_name} not implemented yet") 
        self.statistics[f'{evaluation_name}_evaluation_time'] = time.time() - start_time

        # Combine the analyses
        # combined_analysis = pd.concat([abstractive_analysis, extractive_analysis])

    def _perform_seahorse_evaluation(self, abstractive_summaries, extractive_summaries, output_folder_log_files):
        """
        Seahorse evaluation
        """
        self._log_message(EVALUATION_PREFIX, "Evaluating Abstractive Summaries.")
        key_questions = self.settings.get('seahorse_evaluation_key_questions')
        abstractive_metrics = []
        self._log_message(EVALUATION_PREFIX, f"Logs will be written in the {output_folder_log_files} folder")

        for q in key_questions:
            metrics = evaluate_with_seahorse_custom(abstractive_summaries, q, 4, self.settings.get('device'), output_folder_log_files + '/abstractive.log')
            abstractive_metrics.append(metrics)

        self._log_message(EVALUATION_PREFIX, "Evaluating Extractive Summaries:")
        extractive_metrics = []
        for q in key_questions:
            metrics = evaluate_with_seahorse_custom(extractive_summaries, q, 4, self.settings.get('device'), output_folder_log_files + '/extractive.log')
            extractive_metrics.append(metrics)

        # Print summary comparisons
        output_summary_log_file = self.output_logs_dir + '/summary.log'
        context = open(output_summary_log_file, "w")
        with context as file, (redirect_stdout(file)):
            self._log_message(EVALUATION_PREFIX, "--- SEAHORSE ---")
            self._log_message(EVALUATION_PREFIX, "Summary of Results:")
            for i, q in enumerate(key_questions):
                self._log_message(EVALUATION_PREFIX, f"{QUESTION_MAP[q]} Metric:")
                self._log_message(EVALUATION_PREFIX, f"Abstractive average: {abstractive_metrics[i]['SHMetric/' + QUESTION_MAP[q] + '/proba_1'].mean():.3f}")
                self._log_message(EVALUATION_PREFIX, f"Extractive average: {extractive_metrics[i]['SHMetric/' + QUESTION_MAP[q] + '/proba_1'].mean():.3f}")
            self._log_message(EVALUATION_PREFIX, "--- End of Summary ---\n\n")

    def _perform_bartbert_evaluation(self):
        pass

    def perform_common_metrics_evaluation(self):
        pass

    def get_statistics(self):
        """
        This method returns the statistics of the pipeline
        """
        return self.statistics