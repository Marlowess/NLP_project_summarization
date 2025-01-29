from pipeline_handler import PipelineHandler

def main():
    settings = {
        "model": "gsarti/it5-small",
        "batch_size": 8,
        "device": "cpu",
        "limit": 8,
        "dataset_name": "all_reviews_2017_italian.csv",
        "print_output_path": True
    }
    
    handler = PipelineHandler(settings, run_name='20250129_175203')
    handler.perform_extractive_step()
    handler.perform_abstractive_step()
    handler.perform_evaluation('seahorse')

main()