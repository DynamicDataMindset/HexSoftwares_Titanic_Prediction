# HexSoftwares_Titanic_Survival_Prediction

## Project Overview

This project is a comprehensive application of the Data Science workflow to predict passenger survival on the Titanic, using the public Kaggle dataset. This solution was developed as a portfolio project for the Hex Softwares Data Science Internship Program.

### Goal
To build a robust Machine Learning model capable of predicting the binary outcome (`Survived`: 0=No, 1=Yes) of passengers based on features like socio-economic class, gender, age, and family size.

### Final Model Performance
The final model used was a **Random Forest Classifier**.

| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| **Training Accuracy** | **0.8709** (87.09%) | The model learned the patterns in the training data effectively. |
| **Final Deliverable** | `titanic_submission_rf_v1.csv` | The required prediction file for the internship submission. |

---

## 🛠️ Technical Stack and Libraries

* **Language:** Python (via Google Colab Notebook)
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Modeling:** Scikit-learn (RandomForestClassifier, accuracy_score)

---

## 📋 Data Science Workflow Walkthrough

### 1. Feature Engineering (The Creative Step)
Raw data columns were transformed into highly predictive features:
* **Title:** Extracted from the `Name` column (`Mr.`, `Mrs.`, `Master.`, etc.) to improve Age imputation and capture social status.
* **FamilySize:** Created by summing `SibSp` + `Parch` + 1 (the passenger themselves).
* **Deck:** Extracted from the first letter of the `Cabin` (e.g., 'C' for C Deck), treating missing values as 'U' (Unknown).

### 2. Imputation (Handling Missing Data)
A context-aware imputation strategy was used to fill all missing values:
* **Age:** Imputed using the **median Age specific to each passenger's `Title` group** (e.g., filling a missing age for 'Master' with the median age of all 'Masters').
* **Fare & Embarked:** Imputed using the **median** and **mode**, respectively, as only 1-2 values were missing.

### 3. Preprocessing and Encoding
* All data was converted to a numerical format.
* Categorical features (`Sex`, `Embarked`, `Deck`, `Title`, `Pclass`) were converted using **One-Hot Encoding** (`pd.get_dummies`) to prevent the model from assuming an incorrect ordinal relationship between categories.

### 4. Modeling
A **Random Forest Classifier** (`n_estimators=100`, `random_state=42`) was chosen for its robustness, ability to handle non-linear relationships, and strong baseline performance.

---

## ▶️ How to Reproduce the Project

1.  **Data Source:** Download `train.csv` and `test.csv` from the Kaggle Titanic competition page.
2.  **Setup:** Upload the data to a Google Drive folder.
3.  **Run Notebook:** Open the Colab notebook (`[Your File Name].ipynb`).
4.  **Update Path:** In **Cell 1**, update the file paths to point to your data on Google Drive.
5.  **Execution:** Run all cells sequentially (Cell 1 through Cell 8). The final output will be the `titanic_submission_rf_v1.csv` deliverable file.

---

## 📧 Submission

This repository contains the full source code and the final prediction deliverable (`titanic_submission_rf_v1.csv`) for the Hex Softwares Data Science Internship.
