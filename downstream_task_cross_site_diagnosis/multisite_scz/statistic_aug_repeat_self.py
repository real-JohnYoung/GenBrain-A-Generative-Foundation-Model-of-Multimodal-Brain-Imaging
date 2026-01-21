import os
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

base_dir ="" # baseline results dir  
result_dir = "" # data_aug_results

metric = "val_accuracy"
metric_two =   "val_f1"
test_metric = "test_f1"
percent_list = list(range(0, 301, 10))

all_results = []

for percent in percent_list:
    percent_str = str(percent)
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)

        if not (os.path.isdir(folder_path) and folder.startswith(f"data_aug_{percent_str}_percent_")):
            continue

        match = re.match(rf"data_aug_{percent_str}_percent_train_on_(.+?)_test_on_(.+)", folder)
        if not match:
            continue

        train_site, test_site = match.groups()
        csv_path = os.path.join(folder_path, "param_search", "param_search_results.csv")
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)
        if df.empty or metric not in df.columns:
            continue

        best_row = df.sort_values(by=[metric], ascending=[False]).iloc[0]

        all_results.append({
            "train_site": train_site,
            "test_site": test_site,
            "aug_percent": percent,
            metric: best_row.get(metric, None),
            metric_two: best_row.get(metric_two, None),
            "test_accuracy": best_row.get("test_accuracy", None),
            "test_precision": best_row.get("test_precision", None),
            "test_recall": best_row.get("test_recall", None),
            "test_f1": best_row.get("test_f1", None),
            "test_auroc": best_row.get("test_auroc", None)
        })


df_all = pd.DataFrame(all_results)
summary_csv_path = os.path.join(result_dir, f"all_cross_site_results_with_{metric}_selection.csv")
df_all.to_csv(summary_csv_path, index=False)
print(f"Results saved in: {summary_csv_path}")


df_best = df_all.sort_values(by=[metric, metric_two], ascending=[False, True]) \
                .groupby(["train_site", "test_site"]) \
                .first() \
                .reset_index()


pivot_data = df_best.pivot(index="train_site", columns="test_site", values=test_metric)
annot_data = df_best.pivot(index="train_site", columns="test_site", values="aug_percent")

annotations = df_best.copy()
annotations["annot"] = annotations.apply(lambda row: f"{row[test_metric]:.2f}\n({int(row['aug_percent'])}%)", axis=1)
annot_matrix = annotations.pivot(index="train_site", columns="test_site", values="annot")

plt.figure(figsize=(10, 8))
sns.heatmap(pivot_data, annot=annot_matrix, fmt="", cmap="viridis", vmin=0.5, vmax=1, cbar_kws={"label": test_metric})
plt.title(f"Cross-site {test_metric} Heatmap\n(selected by best {metric})")
plt.xlabel("Test Site")
plt.ylabel("Train Site")
plt.tight_layout()

heatmap_path = os.path.join(result_dir, f"best_cross_site_heatmap_by_{metric}.png")
plt.savefig(heatmap_path)
plt.close()
print(f"Heat map saved in {heatmap_path}")
