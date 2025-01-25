from pipeline_handler import PipelineHandler

def main():
    settings = {
        "model": "gsarti/it5-small",
        "batch_size": 8,
        "device": "cpu",
        "limit": 8,
        "dataset_name": "all_reviews_2017_translated.csv",
        "print_output_path": True
    }
    
    handler = PipelineHandler(settings)
    handler.perform_extractive_step()

main()