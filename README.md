# Loan Eligibility Prediction - Machine Learning

## Project Overview
This project focuses on developing a machine learning model to predict loan eligibility, suggest loan amounts, calculate individualized interest rates, and detect fraud. The model is trained using various machine learning algorithms, including:
- **Decision Tree Classifier** (Loan Approval Prediction)
- **Random Forest Regressor** (Loan Amount Recommendation)
- **K-Means Clustering** (Personalized Interest Rate & Fraud Detection)

## 4. Model Development and Training
This chapter describes the process of building and training models using different machine learning algorithms based on dataset features and problem requirements.

### 4.1 Overview of Selected Machine Learning Algorithms
Each algorithm was chosen based on its ability to handle classification, regression, or clustering problems efficiently.

#### a. Decision Tree Classifier (for Loan Approval)
- A supervised learning algorithm used for classification.
- Builds tree-based models where each node represents a decision.
- Suitable for categorical and numerical data.
- Used to predict whether a loan application is approved.

#### b. Random Forest Regressor (for Loan Amount Recommendation)
- An ensemble learning method that aggregates multiple decision trees.
- Reduces overfitting and improves predictive accuracy.
- Ideal for predicting continuous variables such as loan amounts.

#### c. K-Means Clustering (for Personalized Interest Rate and Fraud Detection)
- An unsupervised learning algorithm for clustering data based on similarity.
- Used to group funded loans based on financial characteristics.
- Helps in fraud detection by identifying applicants with abnormal financial behavior.

### 4.2 Implementation of Different Algorithms
Each algorithm is implemented to solve a specific problem within the project.

#### 4.2.1 Decision Tree Algorithm (Loan Approval Prediction)
**Implementation Steps:**
1. **Preprocessing:**
   - Handle missing values.
   - Encode categorical variables using label encoding.
2. **Feature Selection:**
   - Select relevant features using feature selection techniques.
3. **Model Training:**
   - Split dataset: 80% training, 20% testing.
   - Train model using Decision Tree Classifier.
   - Visualize decision trees for better interpretability.
## Installation and Setup
To run this project, follow these steps:

### Prerequisites
Ensure you have the following installed:
- Python 3.x
- Jupyter Notebook (optional)
- Required Python libraries:
  ```sh
  pip install numpy pandas scikit-learn matplotlib seaborn
  ```

### Running the Project
1. Clone this repository:
   ```sh
   git clone https://github.com/SanchalaLiyanage/Loan-Eligibility-Prediction-ML.git
   ```
2. Navigate to the project directory:
   ```sh
   cd Loan-Eligibility-Prediction-ML
   ```
3. Run the Jupyter Notebook or Python scripts:
   ```sh
   jupyter notebook
   ```

## Results and Evaluation
- Model performance is evaluated using accuracy, precision, recall, and F1-score.
- Random Forest and Decision Tree models are tuned for optimal performance.

## Contributing
Feel free to fork this repository and submit pull requests for improvements.

