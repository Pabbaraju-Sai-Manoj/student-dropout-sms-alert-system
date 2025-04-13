import shap
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os

def explain_model(model_path="model.pkl", data_path="data.csv", output_dir="shap_outputs"):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the trained model
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Load the dataset
    df = pd.read_csv(data_path)
    X = df[["attendance", "grade", "parent_income"]]

    # Calculate dropout risk probabilities
    probabilities = model.predict_proba(X)[:, 1]  # Probability of class '1' (dropout risk)
    df["Dropout Risk (%)"] = (probabilities * 100).round(2)

    # Predictions as Yes/No
    predictions = model.predict(X)
    df["Dropout Risk"] = ["Yes" if p == 1 else "No" for p in predictions]

    # Reorder columns
    column_order = ["attendance", "grade", "parent_income", "Dropout Risk (%)", "Dropout Risk"]
    df = df[column_order]

    # Initialize SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Prepare SHAP values DataFrame
    shap_values_df = pd.DataFrame(shap_values, columns=[f"SHAP_{col}" for col in X.columns])

    # Combine with original data
    result_df = pd.concat([df, shap_values_df], axis=1)

    # Save to CSV
    result_csv_path = os.path.join(output_dir, "shap_values_with_risk.csv")
    result_df.to_csv(result_csv_path, index=False)
    print(f"✅ SHAP values and dropout risk saved to {result_csv_path}")

    # Initialize JS for interactive plots
    shap.initjs()

    # Generate and save summary plot
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig(os.path.join(output_dir, "shap_summary_plot.png"), bbox_inches='tight')
    plt.close()
    print(f"✅ SHAP summary plot saved to {os.path.join(output_dir, 'shap_summary_plot.png')}")

    print("✅ SHAP explanation completed.")

if __name__ == "__main__":
    explain_model()
