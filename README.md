# TFT Model Training Procedures & Artifacts

This document outlines the step-by-step procedures, scripts, and notebooks used to train the Temporal Fusion Transformer (TFT) model for refinery process forecasting.

## 📜 Training Workflow Overview

The training process involves several key stages, designed to prepare the data, reduce complexity, optionally augment the dataset, and finally train the TFT model. The general flow is:

1.  **Initial Data Loading & Cleaning:** Raw monthly CSV reports (`Lab.csv`, `TS Monitoring Tags.csv`, `blend.csv`) are combined and cleaned.
2.  **Exploratory Data Analysis (EDA):** (Assumed, typically done in a notebook) Understanding data distributions, correlations, and identifying potential issues.
3.  **Feature Engineering (Implicit/Explicit):**
    *   Dynamic identification of target, past covariate, and future covariate columns based on naming conventions.
    *   Creation of a `time_idx` for Darts.
4.  **Data Splitting:** The cleaned dataset is split into training and validation sets, preserving temporal order.
5.  **Dimensionality Reduction / Feature Selection (Choose ONE path):**
    *   **Path A (PCA):** Principal Component Analysis is applied to the high-dimensional original past covariates to reduce dimensionality while retaining variance.
    *   **Path B (RFFS):** Random Forest Feature Selection is used to identify the most important original past covariates with respect to the target variables.
6.  **Data Augmentation (Optional - VAR):** Vector Autoregression (VAR) is used to generate synthetic time series data based on the output from either PCA (Path A) or RFFS (Path B), creating an augmented training set.
7.  **TFT Model Training:** The Darts TFT model is trained using one of the following datasets:
    *   PCA-reduced data.
    *   RFFS-selected feature data.
    *   VAR-augmented data (which itself was built upon PCA or RFFS data).
8.  **Model Evaluation & Iteration:** The model is evaluated on the validation set (MAE, MAPE, visual inspection). Iterations on feature engineering, hyperparameter tuning, or model choice are performed based on results.

---

## 📁 Key Files & Scripts/Notebooks Involved

*(Note: Replace placeholders like `[script_name.py]` or `[notebook_name.ipynb]` with your actual filenames. It's good practice to number them if they follow a sequence.)*

1.  **`00_data_extraction_and_cleaning.ipynb` (or `initial_data_prep.py`)**
    *   **Purpose:** Loads the raw monthly CSV files (`tft_ready_data_darts.csv`, `..._JULY.csv`, etc.), combines them into a single DataFrame (`df_original`). Performs initial cleaning (handling NaNs/Infs, sorting by date, removing duplicates). Identifies `target_cols`, `future_covariate_cols`, and original `past_covariate_cols`.
    *   **Input:** Raw CSV files mentioned in the script.
    *   **Output:** A cleaned DataFrame (e.g., saved as `df_original_cleaned.csv` or passed in memory) and lists of column types. This `df_original` is the starting point for the subsequent reduction/selection scripts.

2.  **Dimensionality Reduction / Feature Selection (Choose ONE path's scripts):**

    *   **PATH A: PCA**
        *   **`01a_pca_dimensionality_reduction.py` (or `.ipynb`)**
            *   **Purpose:** Takes the cleaned `df_original`. Splits it into training and validation sets. Scales the original past covariates. Applies PCA to the scaled training past covariates. Reconstructs `df_train_reduced` and `df_val_reduced` with PCA components as the new past covariates.
            *   **Input:** `df_original` (cleaned).
            *   **Output (saved to `PCA_Output_Data/` directory):**
                *   `train_data_pca_reduced.csv`: Training data with targets, futures, and PCA components.
                *   `val_data_pca_reduced.csv`: Validation data with targets, futures, and PCA components.
                *   `target_cols.txt`: List of target column names.
                *   `future_covariate_cols.txt`: List of future covariate names.
                *   `past_covariate_cols_original.txt`: List of the *original* past covariate names (important if preprocessing for prediction needs to redo PCA).
                *   `past_covariate_cols_reduced.txt`: List of the new PCA component names (e.g., `PC_1`, `PC_2`).
                *   `past_cov_scaler_pca.pkl`: The `StandardScaler` object fitted on the original past covariates *before* PCA.
                *   `pca_object.pkl`: The fitted `PCA` object.

    *   **PATH B: Random Forest Feature Selection (RFFS)**
        *   **`01b_rf_feature_selection.py` (or `.ipynb`)**
            *   **Purpose:** Takes the cleaned `df_original`. Splits it. Scales original past covariates. Iteratively trains RandomForestRegressors for each target against the original past covariates. Aggregates feature importances and selects the top `k` original past covariates. Reconstructs `df_train_rf_selected` and `df_val_rf_selected`.
            *   **Input:** `df_original` (cleaned).
            *   **Output (saved to `Feature_Selection_Output/` directory):**
                *   `train_data_rf_selected.csv`: Training data with targets, futures, and *selected original* past covariates.
                *   `val_data_rf_selected.csv`: Validation data with targets, futures, and *selected original* past covariates.
                *   `target_cols.txt`: List of target column names.
                *   `future_covariate_cols.txt`: List of future covariate names.
                *   `past_covariates_selected_rf.txt`: List of the *selected original* past covariate names.
                *   `past_cov_scaler_rf.pkl`: The `StandardScaler` object fitted on the original past covariates (used during RF importance calculation).

3.  **Data Augmentation (Optional - `02_var_augmentation.py` or `.ipynb`)**
    *   **Purpose:** Loads the output from either the PCA script (e.g., `df_train_reduced`, `past_covariate_cols_reduced`) or the RFFS script. Trains a VAR model on the targets and reduced/selected past covariates. Generates synthetic data and combines it with the original training portion to create `augmented_df`.
    *   **Input:** Files from `PCA_Output_Data/` or `Feature_Selection_Output/`.
    *   **Output (saved to `VAR_Augmented_Data/` directory):**
        *   `augmented_data_var_pca.csv` (or `_var_rffs.csv`): The augmented dataset containing original training data + synthetic data, ready for TFT training.

4.  **TFT Model Training (`03_tft_training.py` or `.ipynb`)**
    *   **Purpose:** This is the main Darts TFT training script. It's configured to load data from one of the previous steps:
        *   Loads `train_data_pca_reduced.csv`, `val_data_pca_reduced.csv`, and associated lists if training on PCA-reduced data.
        *   Loads `train_data_rf_selected.csv`, `val_data_rf_selected.csv`, and associated lists if training on RFFS data.
        *   Loads `augmented_data_var_pca.csv` (or `_var_rffs.csv`) and associated lists if training on augmented data.
    *   Performs final scaling specific to this training run (fitting scalers on the training portion of the loaded data).
    *   Initializes and trains the `TFTModel`.
    *   **Input:** CSV data and `.txt` column lists from the chosen preceding step (PCA, RFFS, or VAR augmentation).
    *   **Output (saved to a model-specific directory like `DORC_tft_REF_training_PCA/DORC_TFT_FORCASTOR_v5_PCA_Reduced/`):**
        *   Model checkpoints (`.ckpt` files, including `best-model.ckpt`).
        *   `target_scaler_[pca|rffs|augmented].pkl`: The target scaler fitted during this training run.
        *   `past_cov_[pca|rffs|augmented]_scaler.pkl`: The past covariate scaler fitted during this training run.
        *   TensorBoard logs (if enabled).

5.  **Prediction & Evaluation (`04_tft_prediction_evaluation.py` or `.ipynb`)**
    *   **Purpose:** Loads a trained TFT model checkpoint and its associated scalers. Loads the appropriate validation data (e.g., `val_data_pca_reduced.csv`). Prepares data, generates predictions, inverse transforms, and calculates evaluation metrics (MAE, Relative MAE).
    *   **Input:**
        *   Model checkpoint and scalers from the TFT training output directory.
        *   Validation data CSV (e.g., `val_data_pca_reduced.csv`) and column lists from `PCA_Output_Data/` (or `Feature_Selection_Output/`).
    *   **Output:** MAE scores, Relative MAE scores, plots of predictions vs. actuals.

---

## ⚙️ Key Configuration & Parameters

*   **Dimensionality Reduction/Feature Selection:**
    *   PCA: `pca_variance_threshold` (e.g., 0.95) in `01a_pca_dimensionality_reduction.py`.
    *   RFFS: `n_features_to_keep`, `rf_n_estimators`, `rf_max_depth` in `01b_rf_feature_selection.py`.
*   **Data Augmentation (VAR):**
    *   `var_maxlags`, `var_selected_lag` in `02_var_augmentation.py`.
*   **TFT Model Training:**
    *   `input_chunk_length`, `output_chunk_length`, `n_epochs`, `batch_size`, `learning_rate`, `hidden_size`, `attention_head_size`, `dropout`, `hidden_continuous_size` in `03_tft_training.py`.
    *   The specific data loading paths in `03_tft_training.py` determine which preprocessed dataset is used.
*   **Data Splitting:**
    *   `validation_set_size` (or fixed validation length logic) applied consistently in the PCA or RFFS scripts. If using augmented data, this percentage is applied to the *augmented* dataset in the TFT training script.

---

## 👟 Running the Training Pipeline

1.  **Prepare Raw Data:** Ensure the initial 5 monthly CSV files are available.
2.  **Run `00_data_extraction_and_cleaning`:** Generate `df_original_cleaned.csv` (or ensure `df_original` is correctly in memory for the next script) and the initial column lists.
3.  **Choose Path and Run Reduction/Selection:**
    *   Run `01a_pca_dimensionality_reduction.py` **OR** `01b_rf_feature_selection.py`. This will create a dedicated output directory (e.g., `PCA_Output_Data/`) with reduced/selected data and relevant lists/objects.
4.  **(Optional) Run `02_var_augmentation.py`:** If augmenting, point this script to the output directory from step 3. It will create an `VAR_Augmented_Data/` directory.
5.  **Run `03_tft_training.py`:**
    *   **Crucially, modify the data loading section in this script** to point to the correct directory and files from either step 3 (if no augmentation) or step 4 (if using augmented data).
    *   Ensure `past_covariate_cols` in this script is loaded with the correct list (`past_covariate_cols_reduced.txt` for PCA, or `past_covariates_selected_rf.txt` for RFFS).
    *   This will train the model and save its artifacts.
6.  **Run `04_tft_prediction_evaluation.py`:** Point this script to the model artifacts from step 5 and the corresponding validation data from step 3 to evaluate performance.

---

## 💡 Notes & Considerations

*   **Environment Consistency:** It's recommended to use the same Python environment (and library versions) for all training steps to ensure compatibility of saved objects (especially pickled scalers and models).
*   **Data Paths:** Carefully manage and verify all input and output file paths in each script.
*   **Iterative Process:** Training is often iterative. Results from evaluation (step 6) might lead you to revisit feature selection, hyperparameters, or data augmentation strategies.
*   **Artifact Management:** Keep track of which model checkpoint and scalers correspond to which data preparation pipeline (PCA vs. RFFS, augmented vs. not). Naming conventions for output directories and model files are important.

This README should provide a good overview of how to reproduce your training process. Remember to update filenames and details to match your exact project.
