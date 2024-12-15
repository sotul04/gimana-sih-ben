# Tugas Besar 2 IF3170 - Machine Learning Algorithm Implementation

## Description 📝

This project is an implementation of the common Machine Learning algorithms from scratch. The algorithms implemented are K-Nearest Neighbor, Naive Bayes, and Decision Tree Learning ID3. This project uses Dataset UNSW-NB15 which is a collection of data network traffic that covers all cyber attacks and also normal behaviors. In this project, the data will first be cleaned and preprocessed to ensure the model trained well and produce quality predictions. Then we will compare the statistic metric of performance between model implementation from library and the model we implemented from scratch

**Machine Learning Algorithms**:

- K-Nearest Neighbor: Euclidean, Manhattan, Minkowski
- Gaussian Naive-Bayes
- Decision Tree Learning ID3

## Requirements 🤔

- **Python**: [Install Python](https://python.org/dl/)
- **Python Libraries**  
Run this command to install all the necessary libraries that  
```cmd
pip install pandas numpy seaborn matplotlib scipy scikit-learn imblearn
```  

## Setting Up 💻

### Clone the Repository

```cmd
git clone https://github.com/sotul04/gimana-sih-ben
cd gimana-sih-ben
```

## Running the Application 🏃‍♂️‍➡️

1. Go to the main.ipynb. Run the cells in order that you like. Click 'Run All' to run all the cells from the beginning  
2. Adjust the model that you like to use for creating the submission.csv  
3. If you would like to load an existing model that has been trained, run this command in a cell  
   ```cmd
   with open ("model-nb.pkl", "rb") as file:
    loaded_model = pickle.load(file)
   ```  
   Or if you would like to save an existing model that you have trained, run this command in a cell  
   ```cmd
   with open('model-nb.pkl', 'wb') as file:
       pickle.dump(nb, file)
   ```

## Assignments

**13522029 Ignatius Jhon Hezkiel Chan - Data Cleaning and Preprocessing**<br>
**13522093 Matthew Vladimir Hutabarat - Naive Bayes**<br>
**13522098 Suthasoma Mahardhika Munthe - ID3**<br>
**13522110 Marvin Scifo Yehezkiel Hutahaean - KNN**<br>