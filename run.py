import subprocess

weeks = [str(202000 + x) for x in range(23, 49)]
epochs = [2500]
patience = [100, 400]
batch_size = [32, 64, 128]
start = [-200]

for w in weeks:
    for e in epochs:
        for p in patience:
            for b in batch_size:
                for s in start:
                    model_name = f"myexpt_{w}_{e}_{p}_{b}_{s}"
                    subprocess.run(["bash", "./scripts/mort_preprocess.sh", w])
                    subprocess.run(["python", "./train_covid2.py", "--epiweek", w, "--epochs", str(e), "--patience", str(p), "--batch", str(b), "--start", str(s), "--save", model_name])