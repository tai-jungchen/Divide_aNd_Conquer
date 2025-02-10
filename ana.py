"""
Author: Alex (Tai-Jung) Chen

Offering some plots for analysis purpose.
"""
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


def main():
    # Read the CSV file
    file_name = "results_0129_.csv"
    df = pd.read_csv(file_name)

    df['model'] = [md.split('(')[0] for md in df['model']]
    methods = df['method'].unique()

    # Extract unique metrics (excluding 'Method' and 'Model')
    metrics = [col for col in df.columns if col in ['acc', 'kappa', 'bacc',
                                                    'precision', 'recall',
                                                    'specificity', 'f1']]

    for metric in metrics:
        plt.figure(figsize=(15, 6))
        sns.barplot(data=df, x="model", y=metric, hue="method", ci=None, dodge=True)

        # Labels and title
        plt.xlabel("Model")
        plt.ylabel(metric)
        plt.title(f"{metric} Comparison by Model and Method")
        plt.legend(title="Method")

        # Show plot
        plt.show()
        # plt.savefig(f"{metric}_comparison.png")


if __name__ == '__main__':
    main()
