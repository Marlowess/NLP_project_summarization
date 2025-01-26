# PIPELINE HANDLER CONSTANTS
VALIDATION_PREFIX="[VALIDATION]"
CANDIDATES_CREATION_PREFIX="[CANDIDATES-CREATION]"
INIT_STEP_PREFIX="[INIT]"
PREPROCESSING_PREFIX="[PREPROCESSING]"
RSA_PREFIX="[RSA]"
EVALUATION_PREFIX="[EVALUATION]"

PROCESSED_DATA_PATH="{root_path}/data/processed"
INPUT_SETTINGS_KEYS_TYPES_AND_DEFAULT_PIPELINE = {
    "abstractive_model": ("facebook/bart-large-cnn", str),
    "batch_size": (8, int),
    "device": ("cuda", str),
    "limit": (1, int),
    "dataset_name": ("all_reviews_2017_translated.csv", str),
    "print_output_path": (True, bool),
    "output_dir": ("data/candidates", str),
    "rsa_output_dir": ("output", str),
    "seahorse_evaluation_key_questions": ([1, 2], list),
    "input_files_to_process": (["all_reviews_2017_translated.csv"], list),
    "output_log_files_path": ("output/logs", str)
}