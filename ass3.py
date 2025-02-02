import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import textstat

def perform_topsis(matrix, weights, impacts):
    """
    Apply TOPSIS to rank alternatives based on multiple criteria.

    Parameters:
    matrix: DataFrame with rows as alternatives and columns as criteria.
    weights: List of weights corresponding to each criterion.
    impacts: List indicating the impact direction (+1 for positive, -1 for negative).

    Returns:
    DataFrame containing original metrics, TOPSIS scores, and rankings.
    """
    # Normalize the matrix
    normalized = matrix / np.sqrt((matrix**2).sum())

    # Apply weights
    weighted = normalized * weights

    # Identify ideal best and worst values based on impacts
    best_ideal = weighted.max() * (impacts == 1) + weighted.min() * (impacts == -1)
    worst_ideal = weighted.min() * (impacts == 1) + weighted.max() * (impacts == -1)

    # Compute Euclidean distances
    dist_best = np.sqrt(((weighted - best_ideal) ** 2).sum(axis=1))
    dist_worst = np.sqrt(((weighted - worst_ideal) ** 2).sum(axis=1))

    # Compute TOPSIS scores
    scores = dist_worst / (dist_best + dist_worst)

    # Append results to the DataFrame
    results = matrix.copy()
    results["TOPSIS Score"] = scores
    results["Rank"] = scores.rank(ascending=False).astype(int)

    return results.sort_values("Rank")

# Sample text and reference summary
text = """
The Industrial Revolution transformed agrarian societies into industrial economies, 
spurring urbanization and technological progress. Factories increased efficiency but led to poor labor conditions. 
The era introduced the telegraph, mechanization, and early environmental concerns. Similar technological shifts 
continue today, reshaping modern industries.
"""

reference_summary = """
The Industrial Revolution marked a shift to industrial economies, bringing efficiency, urbanization, and 
technological advances, but also poor labor conditions and environmental concerns.
"""

# Define summarization models
models = {
    "BART": "facebook/bart-large-cnn",
    "T5": "t5-small",
    "Pegasus": "google/pegasus-xsum",
    "LED": "allenai/led-large-16384"
}

# Load semantic similarity model
semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

# Collect results
metrics = []

for model_name, model_path in models.items():
    summarizer = pipeline("summarization", model=model_path)

    try:
        start_time = time.time()
        generated_summary = summarizer(text, max_length=100, min_length=50, do_sample=False)[0]["summary_text"]
        execution_time = time.time() - start_time

        compression_ratio = len(generated_summary) / len(text)
        readability_score = textstat.flesch_reading_ease(generated_summary)
        similarity_score = cosine_similarity(
            semantic_model.encode([reference_summary]), semantic_model.encode([generated_summary])
        )[0][0]

        metrics.append([model_name, compression_ratio, readability_score, similarity_score, execution_time])

    except Exception as error:
        print(f"Error processing {model_name}: {error}")
        metrics.append([model_name, None, None, None, None])

# Create DataFrame
df_metrics = pd.DataFrame(metrics, columns=["Model", "Compression", "Readability", "Similarity", "Time"])

# Define weights and impacts
weights = np.array([0.3, 0.3, 0.3, 0.1])
impacts = np.array([1, 1, 1, -1])  # Positive for Compression, Readability, and Similarity; Negative for Time

# Perform TOPSIS ranking
ranked_models = perform_topsis(df_metrics.set_index("Model"), weights, impacts)

# Display results
print("\nTOPSIS Model Ranking:")
print(ranked_models)

# Visualization: Metric Comparison
plt.figure(figsize=(10, 6))
df_metrics.set_index("Model")[["Compression", "Readability", "Similarity", "Time"]].plot(kind="bar", colormap="Set2")
plt.title("Performance Metrics by Model", fontsize=16)
plt.xlabel("Model", fontsize=14)
plt.ylabel("Metric Values", fontsize=14)
plt.xticks(rotation=45)
plt.legend(title="Metrics")
plt.tight_layout()
plt.show()

# Visualization: TOPSIS Score Comparison
plt.figure(figsize=(8, 6))
sns.barplot(x=ranked_models.index, y=ranked_models["TOPSIS Score"], palette="viridis")
plt.title("TOPSIS Scores by Model", fontsize=16)
plt.xlabel("Model", fontsize=14)
plt.ylabel("TOPSIS Score", fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Visualization: Rank Distribution
plt.figure(figsize=(8, 6))
sns.boxplot(x=ranked_models["Rank"], palette="viridis")
plt.title("Rank Distribution of Models", fontsize=16)
plt.xlabel("Rank", fontsize=14)
plt.tight_layout()
plt.show()

# Visualization: Heatmap of Model Scores
plt.figure(figsize=(10, 6))
sns.heatmap(df_metrics.set_index("Model")[["Compression", "Readability", "Similarity", "Time"]], annot=True, cmap="Blues", linewidths=0.5)
plt.title("Metric Scores by Model", fontsize=16)
plt.xlabel("Metrics", fontsize=14)
plt.ylabel("Models", fontsize=14)
plt.tight_layout()
plt.show()
