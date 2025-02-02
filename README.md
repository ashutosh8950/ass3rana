

**Ranking Summarization Models with TOPSIS**  

### Introduction  

This project evaluates different text summarization models using the TOPSIS (Technique for Order Preference by Similarity to Ideal Solution) approach. The objective is to assess the models based on key performance indicators—Compression, Readability, Similarity, and Inference Time—providing a structured ranking to determine the most effective summarization model. The results are presented with clear visualizations for easier comparison.  

### Project Overview  

The study analyzes four state-of-the-art summarization models to assess their ability to generate concise yet informative summaries. The models included are:  

1. **BART (Bidirectional and Auto-Regressive Transformers)**  

   BART is a transformer-based model optimized for abstractive summarization. It excels in producing high-quality, natural-sounding summaries from complex texts.  

2. **T5 (Text-to-Text Transfer Transformer)**  

   T5 frames all NLP tasks, including summarization, as a text-to-text problem. Its flexible architecture enables it to generate diverse and context-aware summaries.  

3. **Pegasus (Pretrained Text-to-Text Transformer)**  

   Pegasus is pre-trained specifically for abstractive summarization. Its training strategy allows it to capture essential information from long-form text, making its summaries precise and well-structured.  

4. **LED (Long-Document Encoder-Decoder)**  

   LED is designed to handle lengthy documents efficiently. Its modified transformer mechanism enables it to summarize long reports and articles without truncating essential details.  

### Evaluation Metrics  

To ensure a thorough evaluation, models are assessed based on the following key metrics:  

- **Compression:** Measures the text reduction ratio, calculated as the summary length compared to the original text. A lower ratio signifies better compression.  
- **Readability:** Evaluated using the Flesch Reading Ease score, which assesses text simplicity based on word and sentence structure. Higher scores indicate greater readability.  
- **Similarity:** Measured using cosine similarity, which compares the generated summary with a reference summary to determine content retention accuracy. Higher values indicate better similarity.  
- **Inference Time:** Captures the time taken by each model to generate summaries. Faster inference times are preferred, especially for real-time applications.  

### System Requirements  

Ensure the following Python packages are installed before running the analysis:  

```bash
pip install pandas numpy matplotlib seaborn transformers sentence-transformers textstat scikit-learn
```  

### Script Overview  

1. **Data Preparation:**  
   - A reference summary and an extended text (e.g., an article on the Industrial Revolution) are used for evaluation.  

2. **Model Setup:**  
   - The summarization models (BART, T5, Pegasus, LED) are loaded using the Hugging Face Transformers library.  

3. **Metrics Calculation:**  
   - **Compression:** Calculated as the ratio of summary length to original text length.  
   - **Readability:** Determined using the textstat library.  
   - **Similarity:** Computed with sentence-transformers.  
   - **Inference Time:** Measured for each model during execution.  

4. **TOPSIS Evaluation:**  
   - The models are ranked based on their performance across all metrics using the TOPSIS method. The impact of each metric is defined as follows:  

     - **Compression:** Positive impact (higher compression is better).  
     - **Readability:** Positive impact (easier readability is better).  
     - **Similarity:** Positive impact (higher similarity is better).  
     - **Inference Time:** Negative impact (lower inference time is better).  

5. **Visualization:**  
   - The evaluation results are illustrated through various plots for better interpretation:  
     - **Bar Plot:** Comparing model performance across all metrics.  
     - **TOPSIS Score Plot:** Showing the final ranking of the models.  
     - **Box Plot:** Depicting the distribution of model rankings.  
     - **Heatmap:** Providing an overview of model performance across metrics.  

### Usage  

To evaluate the models and generate visualizations, execute:  

```bash
python analysis.py
```  

### Output  

The script provides:  

- **TOPSIS Results:** A ranked list of models based on performance scores.  
- **Visualizations:**  
  - Bar plots comparing the models in terms of Compression, Readability, Similarity, and Inference Time.  
  - A TOPSIS score plot highlighting overall rankings.  
  - A heatmap displaying model performance across all evaluated metrics.  

### Conclusion  

This project offers a structured methodology to compare and rank text summarization models using the TOPSIS approach. By leveraging comprehensive evaluation metrics and data visualizations, users can make informed decisions about selecting the best summarization model based on their specific needs—whether prioritizing speed, readability, or summary accuracy.  

Following this approach ensures a systematic and objective comparison, helping researchers and practitioners choose the most suitable model for diverse NLP applications.  

