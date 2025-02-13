# PIPELINE HANDLER CONSTANTS
VALIDATION_PREFIX="[VALIDATION]"
CANDIDATES_CREATION_PREFIX="[CANDIDATES-CREATION]"
INIT_STEP_PREFIX="[INIT]"
PREPROCESSING_PREFIX="[PREPROCESSING]"
RSA_PREFIX="[RSA]"
EVALUATION_PREFIX="[EVALUATION]"
WARNING_PREFIX="[WARNING]"

INPUT_DATA_PATH="{root_path}/data"
OUTPUT_PATH="{root_path}/output"
PROCESSED_DATA_PATH="{output_path}/{run_name}/data/processed"
OUTPUT_LOGS_DIR="{output_path}/{run_name}/logs"
CANDIDATES_OUTPUT_PATH="{output_path}/{run_name}/data/candidates"
RSA_OUTPUT_DIR="{output_path}/{run_name}/data/rsa"

INPUT_SETTINGS_KEYS_TYPES_AND_DEFAULT_PIPELINE = {
    "abstractive_model": ("facebook/bart-large-cnn", str),
    "batch_size": (8, int),
    "device": ("cuda", str),
    "limit": (1, int),
    "dataset_name": ("all_reviews_2017_italian.csv", str),
    "print_output_path": (True, bool),
    "seahorse_evaluation_key_questions": ([1, 2], list),
    "input_files_to_process": (["all_reviews_2017_italian.csv"], list)
}

OPENAI_KEY='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'