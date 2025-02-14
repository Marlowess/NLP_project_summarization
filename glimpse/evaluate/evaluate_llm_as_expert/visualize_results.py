from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
from pathlib import Path
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation_dataset", type=Path, default="", required=True)
    parser.add_argument("--seahorse_evaluation_dataset", type=Path, default="")
    parser.add_argument("--type", type=str, default="", required=True)
    parser.add_argument("--model", type=str, default="", required=True)
    parser.add_argument("--model_b", type=str, default="")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    folder_path = "plots"
    os.makedirs(folder_path, exist_ok=True)

    evaluation_type = args.type
    evaluation_df = pd.read_csv(args.evaluation_dataset)
    if evaluation_type == 'pairwise':
        cols = ['common_ideas', 'unique_ideas', 'best_overall']
        for col in cols:
            evaluation_df.groupby(col).size().plot(kind='barh', color=sns.palettes.mpl_palette('Dark2'))
            plt.gca().spines[['top', 'right',]].set_visible(False)
            filepath = os.path.join(folder_path, f"pairwise_plot_{col}_{args.model}_vs_{args.model_b}.pdf")
            plt.savefig(filepath, format="pdf", dpi=300, bbox_inches="tight")
            plt.close()
    elif evaluation_type == 'discrim_score':
        for col in evaluation_df.columns:
            evaluation_df[col].plot(kind='hist', bins=20, title=col)
            plt.gca().spines[['top', 'right',]].set_visible(False)
            filepath = os.path.join(folder_path, f"{evaluation_type}_plot_{col}_{args.model}.pdf")
            plt.savefig(filepath, format="pdf", dpi=300, bbox_inches="tight")
            plt.close()
    elif evaluation_type == 'seahorse_score':
        seahorse_evaluation_df = pd.read_csv(args.seahorse_evaluation_dataset)
        metrics_SH = ['comprehensible', 'repetition', 'grammar', 'attribution', 'main_ideas', 'conciseness']
        metrics_SH_like = [
            'SHMetric/Comprehensible/proba_1', 
            'SHMetric/Repetition/proba_1', 
            'SHMetric/Grammar/proba_1', 
            'SHMetric/Attribution/proba_1', 
            'SHMetric/Main ideas/proba_1', 
            'SHMetric/Conciseness/proba_1'
            ]
        plt.figure(figsize=(12, 8))
        for i, (metric_SH, metric_SH_like) in enumerate(zip(metrics_SH, metrics_SH_like), 1):
            plt.subplot(2, 3, i)
            sns.kdeplot(evaluation_df[metric_SH_like], label=f"{metric_SH_like}_SH_like", shade=True, alpha=0.5)
            sns.kdeplot(seahorse_evaluation_df[metric_SH], label=f"{metric_SH_like}_SH", shade=True, alpha=0.5)
            plt.title(f"{metric_SH_like} distribution")
            plt.legend()
        
        plt.tight_layout()
        filepath = os.path.join(folder_path, f"{evaluation_type}_plot_{args.model}.pdf")
        plt.savefig(filepath, format="pdf", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()