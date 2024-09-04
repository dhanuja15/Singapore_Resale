# Industrial Copper Modeling

## Project Overview
This project aims to address challenges in the copper industry by developing machine learning models for sales pricing predictions and lead classification. The primary objectives are to improve pricing decisions using regression models and to enhance lead classification using classification models. A Streamlit web application will provide an interactive interface for these predictions.

## Skills and Tools
- Python scripting
- Data Preprocessing
- Exploratory Data Analysis (EDA)
- Machine Learning (Regression & Classification)
- Streamlit

## Domain
Manufacturing

## Problem Statement
The copper industry faces challenges with skewed and noisy sales and pricing data, affecting pricing strategies. The project involves:
- Developing a regression model to predict continuous variable 'Selling_Price'.
- Creating a classification model to predict lead status ('WON' or 'LOST').

## Data Description
The dataset contains various attributes regarding copper sales transactions, including identifiers, dates, quantities, customer details, and transaction status. The key target variables are `selling_price` for regression and `status` for classification.

## Methodology
1. **Data Understanding and Preprocessing:**
   - Identify variable types and distributions.
   - Handle missing values and outliers.
   - Address skewness with transformations.
   - Encode categorical variables.

2. **Exploratory Data Analysis (EDA):**
   - Visualize data distribution and outliers using Seaborn plots.

3. **Feature Engineering:**
   - Create informative features.
   - Use heatmaps to identify and drop highly correlated features.

4. **Model Building and Evaluation:**
   - Regression: Train models like ExtraTreesRegressor, XGBRegressor.
   - Classification: Train models like ExtraTreesClassifier, XGBClassifier.
   - Use appropriate evaluation metrics and optimize hyperparameters.

5. **Streamlit Application:**
   - Develop a Streamlit app with task input and data entry fields.
   - Perform real-time predictions for regression and classification tasks.

## Learning Outcomes
- Proficiency in Python and data analysis libraries (Pandas, NumPy, etc.).
- Experience with data preprocessing and EDA.
- Skills in regression and classification modeling.
- Web app development using Streamlit.

## Usage
1. **Set up the environment:**
   - Clone the repository.
   - Install dependencies: `pip install -r requirements.txt`.

2. **Run the Streamlit app:**
   - Execute: `streamlit run app.py`.

3. **Interact using the web interface:**
   - Provide inputs to obtain predictions for sales price or lead status.
