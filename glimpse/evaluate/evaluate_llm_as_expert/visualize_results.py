from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
from pathlib import Path
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation_dataset", type=Path, default="", required=True)
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
    else:
        for col in evaluation_df.columns:
            evaluation_df[col].plot(kind='hist', bins=20, title=col)
            plt.gca().spines[['top', 'right',]].set_visible(False)
            filepath = os.path.join(folder_path, f"{evaluation_type}_plot_{col}_{args.model}.pdf")
            plt.savefig(filepath, format="pdf", dpi=300, bbox_inches="tight")
            plt.close()



if __name__ == "__main__":
    main()