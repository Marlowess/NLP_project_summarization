from abc import ABC
from utils.constants import VALIDATION_PREFIX, INIT_STEP_PREFIX, OUTPUT_PATH
from utils.path_utils import get_git_root
import datetime
import os, glob

class AbstractHandler(ABC):
    """
    This is an abstract class for handlers implementation
    """
    def __init__(self, settings, default_dict, run_name):
        self.base_path = get_git_root()
        self.output_path = OUTPUT_PATH.format(root_path=self.base_path)
        self.settings = settings
        self._validate_input_settings(default_dict)
        self._check_old_run_and_set_run_name(run_name)
    
    def _log_message(self, prefix, message):
        print(f"{prefix} {message}")

    def _check_old_run_and_set_run_name(self, run_name):
        """
        This method checks if the old run is present in the output path
        """
        self._log_message(INIT_STEP_PREFIX, f"Checking if the defined run name already exists")
        if os.path.exists(f"{self.output_path}/{run_name}"):
            self._log_message(INIT_STEP_PREFIX, f"Run {run_name} found")
            self._log_message(INIT_STEP_PREFIX, f"Setting the run name to {run_name}")
            self._log_message(INIT_STEP_PREFIX, f"Since this run already exists, you could overwrite some files if you run it again")
            self.old_run_loaded = True
        else:
            self._log_message(INIT_STEP_PREFIX, f"Old run {run_name} not found")
            self._log_message(INIT_STEP_PREFIX, f"Setting the new run name to {run_name}")
            self.old_run_loaded = False
        self.run_name = run_name

    def _validate_input_settings(self, default_dict):
        """
        This methods validates the input settings
        """
        self._log_message(INIT_STEP_PREFIX, f"Validating the provided settings")
        for k, t in default_dict.items():
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

    def _find_file_with_wildcard(self, directory, pattern):
        self._log_message(INIT_STEP_PREFIX, f"Searching for the file with pattern {pattern} in directory {directory}")
        search_pattern = os.path.join(directory, pattern)
        files = glob.glob(search_pattern)
        if files:
            self._log_message(INIT_STEP_PREFIX, f"Found file {files[0]}")
            return files[0]  # Get the first file
        else:
            self._log_message(INIT_STEP_PREFIX, f"File with pattern {pattern} not found in directory {directory}")
            return None