import os
import pandas as pd

def process_prediction_logs(log_dir="logs", output_csv="overconfidence_dataset.csv"):

    data = []

    for filename in os.listdir(log_dir):
        file_path = os.path.join(log_dir, filename)

        if "overconfidence" in filename:
            overconfident_label = 1
        elif "predictions" in filename or "confidence" in filename:
            overconfident_label = 0
        else:
            continue

        parts = filename.replace(".txt", "").split("_")
        if len(parts) < 5:
            continue

        model_name  = parts[2]
        dataset_name = parts[3]
        method      = "_".join(parts[4:])  

        with open(file_path, "r") as f:
            lines = f.readlines()[3:]  

            for line in lines:
                line = line.strip()
                parts_line = line.split()
                if len(parts_line) == 4:
                    sample_id_str, pred_str, conf_str, unc_str = parts_line

                    try:
                        sample_id   = int(sample_id_str)
                        prediction  = int(pred_str)
                        confidence  = float(conf_str)
                        uncertainty = float(unc_str)

                        data.append([
                            dataset_name,
                            model_name,
                            method,
                            sample_id,
                            prediction,
                            confidence,
                            uncertainty,
                            overconfident_label
                        ])
                    except ValueError:
                        pass

    df = pd.DataFrame(data, columns=[
        "Dataset", "Model", "Method", "SampleID", 
        "Prediction", "Confidence", "Uncertainty", 
        "Overconfident"
    ])

    df.to_csv(output_csv, index=False)
    print(f"Overconfidence dataset saved as '{output_csv}' with {len(df)} entries.")

if __name__ == "__main__":
    process_prediction_logs(log_dir="logs", output_csv="overconfidence_dataset.csv")
