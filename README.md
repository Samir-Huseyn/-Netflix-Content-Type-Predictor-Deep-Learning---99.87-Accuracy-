This project implements a Deep Learning binary classification model built on the real-world **Netflix Dataset**. The model analyzes the parsed genres of a specific title to predict whether the content is a **Movie** or a **TV Show** with an outstanding **99.87%** test accuracy.

## 🚀 Advanced Data Engineering & Feature Engineering

The core strength of this project lies in rigorous data cleaning and proactive mitigation of **Data Leakage**, ensuring the model learns actual underlying patterns rather than trivial shortcuts.

1. **Preventing Data Leakage:**
   - The original `duration` column contained explicit markers like "min" for movies and "Season(s)" for TV shows. Keeping this feature would cause severe data leakage, so it was dropped entirely.
   - The words "Movies", "TV Shows", and "Series" were embedded within the `genres` strings. A chained `.str.replace()` pipeline was applied to strip these keywords out, forcing the neural network to learn solely from the pure genre concepts.

2. **Multi-Label Categorical Encoding:**
   - Since titles can belong to multiple genres simultaneously (e.g., `Horror, International, Thrillers`), a standard `pd.get_dummies()` approach would treat the entire combination as a single unique string. 
   - Instead, Pandas' `.str.get_dummies(sep=",")` feature was utilized to split, trim, and binarize each sub-genre independently into its own sparse column (0 or 1).

3. **Dimensionality Reduction:**
   - Uninformative columns such as `show_id`, `title`, `director`, `cast`, `country`, `date_added`, `release_year`, `rating`, and `description` were dropped to optimize computational efficiency and avoid overfitting.

## 🧠 Model Architecture

The deep learning network is built using the **TensorFlow/Keras** Sequential API:

- **Input Layer:** Dynamically shapes itself to the number of processed sub-genres (`input_shape=(67,)`).
- **Hidden Layer 1:** 16 Neurons, `ReLU` activation function.
- **Hidden Layer 2:** 8 Neurons, `ReLU` activation function.
- **Hidden Layer 3:** 6 Neurons, `ReLU` activation function.
- **Output Layer:** 1 Neuron, `Sigmoid` activation function (ideal for Binary Classification).

### Compilation Settings:
- **Optimizer:** `Adam`
- **Loss Function:** `Binary Crossentropy`
- **Metrics:** `Accuracy`

## 📊 Performance Metrics

The model was trained for **25 epochs** with a **batch size of 10**, yielding the following results on the unseen test set:

- **Final Evaluation Loss:** `0.0065`
- **Test Accuracy:** `99.87%`
