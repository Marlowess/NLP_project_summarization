import sys
sys.path.append('../')

import pandas as pd
import os
import subprocess
from utils.path_utils import get_git_root
from utils.analysis import create_summary_analysis
from glimpse.evaluate.evaluate_seahorse_metrics_samples_custom import evaluate_with_seahorse, QUESTION_MAP
import pickle

import nltk
nltk.download('punkt_tab')

VALIDATION_PREFIX="[VALIDATION]"
CANDIDATES_CREATION_PREFIX="[CANDIDATES-CREATION]"
INIT_STEP_PREFIX="[INIT]"
PREPROCESSING_PREFIX="[PREPROCESSING]"
RSA_PREFIX="[RSA]"

PROCESSED_DATA_PATH="{root_path}/data/processed"
INPUT_SETTINGS_KEYS_TYPES_AND_DEFAULT = {
    "model": ("gsarti/it5-base", str),
    "batch_size": (8, int),
    "device": ("cuda", str),
    "limit": (1, int),
    "dataset_name": ("all_reviews_2017_translated.csv", str),
    "print_output_path": (True, bool),
    "output_dir": ("data/candidates", str),
    "rsa_output_dir": ("output", str),
    "seahorse_evaluation_key_questions": ([1, 2], list)
}

class PipelineHandler:
    """
    This class handles all the operations related to the multi-language extension.
    It takes a configuration dictionary as a JSON and performs all the steps needed to complete the pipeline: 
    - input file processing
    - generation of extractive and abstractive candidates
    - RSA computation
    - Evalutation using different approaches
    - TODO: model fine tuning
    """
    
    def __init__(self, settings):
        self.base_path = get_git_root()
        self.settings = settings
        self._validate_input_settings()
        self._log_message(INIT_STEP_PREFIX, f"Final settings: {settings}")
        self._process_input_data()
        self.rsa_paths = {"abstractive": None, "extractive": None}

    def _log_message(self, prefix, message):
        print(f"{prefix} {message}")

    def _validate_input_settings(self):
        """
        This methods validates the input settings
        """
        self._log_message(INIT_STEP_PREFIX, f"Validating the provided settings")
        for k, t in INPUT_SETTINGS_KEYS_TYPES_AND_DEFAULT.items():
            default_value, default_type = t[0], t[1] # default_value and item type

            # Check if the input settings dict contains this key
            if k in self.settings:
                setting_item = self.settings.get(k) # Get the settings item in order to validate it
                if type(setting_item) != default_type:
                    # Trying to cast it to the right type
                    self._log_message(VALIDATION_PREFIX, f"Key {k} should be of type {default_type}, but received {type(setting_item)}")
                    self._log_message(VALIDATION_PREFIX, f"Trying to cast it")
                    try:
                        final_item = default_type(setting_item)
                        self._log_message(VALIDATION_PREFIX, f"Cast completed")
                        self._log_message(VALIDATION_PREFIX, f"Setting {k} to the casted value {final_item}")
                        self.settings[k] = final_item
                    except ValueError:
                        self._log_message(VALIDATION_PREFIX, f"Cannot setting {k}(setting_item) to {default_type}")
                        self._log_message(VALIDATION_PREFIX, f"Setting its default value: {default_value}")
                        self.settings[k] = default_value
            else:
                # Matching datatype
                self._log_message(VALIDATION_PREFIX, f"Item {k} not found in the received settings")
                self._log_message(VALIDATION_PREFIX, f"Adding {k} setting with the default value {default_value}")
                self.settings[k] = default_value
        self._log_message(INIT_STEP_PREFIX, f"Validation completed")

    def _process_input_data(self):
        """
        This method processes input data and prepare them to perform the pipeline
        """
        self._log_message(PREPROCESSING_PREFIX, "Preparing input files")
        if not os.path.exists(PROCESSED_DATA_PATH.format(root_path=self.base_path)):
            os.makedirs(PROCESSED_DATA_PATH.format(root_path=self.base_path))

        for year in range (2017, 2021):
            try:
                dataset = pd.read_csv(f"{self.base_path}/data/all_reviews_{year}_translated.csv")
                sub_dataset = dataset[['id','review', 'metareview']]
                sub_dataset.rename(columns={"review": "text", "metareview": "gold"}, inplace=True)
                self._log_message(PREPROCESSING_PREFIX, f"Found a file for year {year}. Managing it...")

                output_file_path = f"{PROCESSED_DATA_PATH.format(root_path=self.base_path)}/all_reviews_{year}_translated.csv"
                sub_dataset.to_csv(output_file_path, index=False)
                self._log_message(PREPROCESSING_PREFIX, f"Saved file at {output_file_path}")
            except:
                self._log_message(PREPROCESSING_PREFIX, f"Input file for year {year} not found or already generated. Skipping it...")

    def perform_extractive_step(self):
        """
        This methods runs the generation of candidates using the extractive mode
        """
        self._log_message(CANDIDATES_CREATION_PREFIX, "Running the extractive step.")
        result_extractive = subprocess.run([
            "python",
            f"{self.base_path}/glimpse/data_loading/generate_extractive_candidates.py",
            "--dataset_path", f"{PROCESSED_DATA_PATH.format(root_path=self.base_path)}/{self.settings.get('dataset_name')}",
            "--limit", str(self.settings.get('limit')),
            "--output_dir", f"{self.base_path}/{self.settings.get('output_dir')}",
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

    def perform_abstractive_step(self):
        """
        This methods runs the generation of candidates using the abstractive mode
        """
        self._log_message(CANDIDATES_CREATION_PREFIX, "Running the abstractive step.")
        result_abstractive = subprocess.run([
            "python",
            f"{self.base_path}/glimpse/data_loading/generate_abstractive_candidates.py",
            "--dataset_path", f"{PROCESSED_DATA_PATH.format(root_path=self.base_path)}/{self.settings.get('dataset_name')}",
            "--batch_size", str(self.settings.get('batch_size')),
            "--device", str(self.settings.get('device')),
            "--limit", str(self.settings.get('limit')),
            "--output_dir", f"{self.base_path}/{self.settings.get('output_dir')}",
            "--model_name", str(self.settings.get('model')),
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

    def _perform_rsa(self, candidates_path, step):
        """
        This method receives path of candidates record and performs the RSA method on them
        """
        self._log_message(RSA_PREFIX, "Computing the RSA score...")
        result_rsa = subprocess.run([
            "python",
            f"{self.base_path}/glimpse/src/compute_rsa.py",
            "--summaries", candidates_path,
            "--output_dir", f"{self.base_path}/{self.settings.get('rsa_output_dir')}",
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

    def perform_evaluation(self, evaluation_name):
        """
        This method runs the evaluation step.
        :param: evaluation_name: seahorse, batbert, common
        """
        # Abstractive data
        with open(self.rsa_paths.get('abstractive'), 'rb') as f:
            abstractive_data = pickle.load(f)

        # Extractive data
        with open(self.rsa_paths.get('extractive'), 'rb') as f:
            extractive_data = pickle.load(f)

        # Create analysis DataFrames
        abstractive_analysis = create_summary_analysis(abstractive_data, 'abstractive')
        extractive_analysis = create_summary_analysis(extractive_data, 'extractive')

        if evaluation_name == 'seahorse':
            self._perform_seahorse_evaluation(abstractive_analysis, extractive_analysis)
        else:
            raise Exception(f"Evaluation named {evaluation_name} not implemented yet")

        # Combine the analyses
        # combined_analysis = pd.concat([abstractive_analysis, extractive_analysis])

    def _perform_seahorse_evaluation(self, abstractive_summaries, extractive_summaries):
        """
        Seahorse evaluation
        """
        print("Evaluating Abstractive Summaries:")
        key_questions = self.settings.get('seahorse_evaluation_key_questions')
        abstractive_metrics = []
        for q in key_questions:
            metrics = evaluate_with_seahorse(abstractive_summaries, q)
            abstractive_metrics.append(metrics)

        print("Evaluating Extractive Summaries:")
        extractive_metrics = []
        for q in key_questions:
            metrics = evaluate_with_seahorse(extractive_summaries, q)
            extractive_metrics.append(metrics)

        # Print summary comparison
        print("Summary of Results:")
        for i, q in enumerate(key_questions):
            print(f"\n{QUESTION_MAP[q]} Metric:")
            print(f"Abstractive average: {abstractive_metrics[i]['SHMetric/' + QUESTION_MAP[q] + '/proba_1'].mean():.3f}")
            print(f"Extractive average: {extractive_metrics[i]['SHMetric/' + QUESTION_MAP[q] + '/proba_1'].mean():.3f}")

    def _perform_bartbert_evaluation(self):
        pass

    def perform_common_metrics_evaluation(self):
        pass