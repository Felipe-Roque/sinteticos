from sinteticos import (
    load_dataframe, save_dataframe,
    preprocess_dataframe, train_model, generate_synthetic,
    evaluate_hellinger_by_column, plot_hellinger_bar, compare_histograms,
)

# 1) Load your dataset
# Use the bundled example dataset
df = load_dataframe("data/IAFREE_Chile.xlsx")
# or your own file under data/
# df = load_dataframe("data/your_dataset.xlsx")

# 2) Optional: subset columns by prefix (e.g., only EEE*)
_eee_cols = [c for c in df.columns if c.startswith("EEE")]
if _eee_cols:
    df = df[_eee_cols]
else:
    print("No columns starting with 'EEE' found; using all columns instead.")

# 3) Optional preprocessing similar to the notebook
numeric_cols = ["EEE5", "EEE6", "EEE7", "EEE8", "EEE9", "EEE10", "EEE11", "Raça"]
df_prep = preprocess_dataframe(df, dropna=False, numeric_columns=numeric_cols, fill_value=-1)

# 4) Train a model and generate synthetic data
# Try CTGAN first; if unavailable, fall back to GaussianCopula
try:
    model = train_model(df_prep, model="CTGAN", epochs=600)
except ImportError as e:
    print("CTGAN unavailable; falling back to GaussianCopula. Reason:", e)
    model = train_model(df_prep, model="GaussianCopula")

synth = generate_synthetic(model, num_rows=1000)

# 5) Evaluate similarity (Hellinger)
hell = evaluate_hellinger_by_column(df_prep, synth)
print("Hellinger mean:", float(hell["Hellinger"].mean()))
print(hell.head())

# 6) (Optional) Plot
plot_hellinger_bar(hell)
compare_histograms(df_prep, synth, columns=df_prep.columns[:6])

# 7) Save synthetic data
save_dataframe(synth, "data/synthetic_output.csv")