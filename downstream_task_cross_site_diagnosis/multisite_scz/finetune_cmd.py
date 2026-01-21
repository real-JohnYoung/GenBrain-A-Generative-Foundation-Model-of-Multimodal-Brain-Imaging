import os

configs = [
    {"site": "SH_JZ1", "epochs": 2000},
    {"site": "SH_JZ2", "epochs": 2000},
    {"site": "SH_drug1", "epochs": 2500},
    {"site": "NUSDAST", "epochs": 2500},
    {"site": "COBRE", "epochs": 3400},
    {"site": "chengdu", "epochs": 2500},
    {"site": "zhengzhou", "epochs": 2500},
    {"site": "MCIC", "epochs": 3400},
    {"site": "ds000030", "epochs": 3400}
    {"site": "fBIRN", "epochs": 5000}
]

base_script = "../finetune_multisite_scz_T1_one_site_step.py"
label_base = "Path to /multisite_scz_one_site/labels"
results_base = "Path to Results/multisite_scz"

# finetuned for 10 sites
for cfg in configs:
    site = cfg["site"]
    epochs = cfg["epochs"]
    label_file = f"{label_base}/{site}/{site}_train_data.csv"
    results_dir = f"{results_base}/one_site_finetune_{site}"

    cmd = (
        f"torchrun --nnodes=1 --nproc_per_node=1 {base_script} "
        f"--data_label_file {label_file} "
        f"--epochs {epochs} "
        f"--results_dir {results_dir}"
    )

    print(f"Running: {site} (epochs={epochs})")
    os.system(cmd)
