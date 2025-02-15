import pandas as pd
import numpy as np

def create_summary_analysis(data, model_type):
    all_results = []

    for result in data['results']:
        summary_data = {
            'id': result['id'][0],  # Get the first element of the tuple
            'gold': result['gold'],
            'model_type': model_type,
            'best_rsa_summary': ' '.join(result['best_rsa']),  # Join the array elements
            'best_base_summary': ' '.join(result['best_base']),  # Join the array elements
            'num_candidates': len(result['consensuality_scores']),
            'mean_consensuality': np.mean(result['consensuality_scores']),
            'mean_speaker_score': result['speaker_df'].mean().mean(),
            'mean_listener_score': result['listener_df'].mean().mean(),
        }
        all_results.append(summary_data)

    return pd.DataFrame(all_results)