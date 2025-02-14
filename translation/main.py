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

    """ sample_path = handler.perform_data_sampling({
        "sample_path": "" # path of file after input processing
        "sample_fraction": 0.2 # or chosen by user
    }) """

    handler.perform_extractive_step()
    handler.perform_abstractive_step()
    handler.perform_evaluation('seahorse')

    handler._perform_rsa(candidates_path="extractive_step_output_path")

    """ (
        glimpse_unique_summaries_path
        glimpse_speaker_summaries_path
    ) = handler._perform_extract_glimpse_summaries({
        "rsa-res": "" # pickle file path in output from _perform_rsa()
        "reviews": sample_path
    }) """

    #lsa_path = handler._perform_sumy_baseline("LSA")
    #lex_rank_path = handler._perform_sumy_baseline("lex-rank")
    #lsa_path = handler._perform_extract_baseline_summaries({"summaries": lsa_path, "reviews": sample_path})
    #lex_rank_path = handler._perform_extract_baseline_summaries({"summaries": lex_rank_path, "reviews": sample_path})

    glimspe_u_vs_glimpse_s_path = handler._perform_llm_as_expert_pairwise_evaluation({
        "summaries_a": "", # glimpse_unique_summaries_path
        "summaries_b": "", # glimpse_speaker_summaries_path
        "model_a": "extractive_glimpse_unique",
        "model_b": "extractive_glimpse_speaker",
    })

    glimspe_u_vs_lsa_path = handler._perform_llm_as_expert_pairwise_evaluation({
        "summaries_a": "", # glimpse_unique_summaries_path
        "summaries_b": "", # lsa_path
        "model_a": "extractive_glimpse_unique",
        "model_b": "lsa"
    })

    glimspe_u_vs_lexrank_path = handler._perform_llm_as_expert_pairwise_evaluation({
        "summaries_a": "", # glimpse_unique_summaries_path
        "summaries_b": "", # lexrank_path
        "model_a": "extractive_glimpse_unique",
        "model_b": "lexrank"
    })

    """ handler._perform_visualize_results({
        "evaluation_dataset": glimspe_u_vs_glimpse_s_path,
        "type": "pairwise",
        "model": "glimpse_u",
        "model_b": "glimpse_s"
    })
    # idem for lsa and lexrank """

    glimpse_u_discrim_score_path = handler._perform_llm_as_expert_evaluation({
        "summaries_by_documents": "", # glimpse_unique_summaries_path
        "eval_type": "discriminativeness",
        "model": "glimpse_u"
    })

    glimpse_u_SH_like_score_path = handler._perform_llm_as_expert_evaluation({
        "summaries_by_documents": "", # glimpse_unique_summaries_path
        "eval_type": "SH_like",
        "model": "glimpse_u"
    })

    """ handler._perform_visualize_results({
        "evaluation_dataset": glimpse_u_discrim_score_path,
        "type": "discrim_score",
        "model": "glimpse_u",
    })
    """

    """ handler._perform_visualize_results({
        "evaluation_dataset": glimpse_u_SH_like_score_path,
        "type": "seahorse_score",
        "model": "glimpse_u",
        "seahorse_evaluation_dataset": "" #path in output from perform_evaluation(seahorse)
    })
    """




main()