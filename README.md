# Microbe DNA Analysis for Tumor Type Prediction

This repository contains the code and data for a study that explores the potential of DNA samples of microbes in human blood as indicators of various tumor types in the human body. The goal of this study is to develop effective artificial intelligence models using the DNA samples to predict tumor types.

## Objective

The objective of this study is to investigate whether the DNA samples of microbes in human blood can serve as reliable indicators for the presence of different tumor types in the human body. By developing and evaluating artificial intelligence models, we aim to determine if the microbial DNA samples can accurately predict tumor types.

## Dataset

The dataset used in this study consists of DNA samples obtained from human blood. Each DNA sample corresponds to a specific tumor type. The dataset includes both dependent variables for predicting tumor types and independent variables representing DNA sequences of the microbes.

## Methodology

1. The dataset is preprocessed to prepare it for model training and evaluation.
2. The dataset is divided into a training set (70% of the data) and a test set (30% of the data) using the train-test split.
3. Two different machine learning algorithms, LightGBM and Random Forest, are implemented for tumor type prediction.
4. The models are trained on the training set and evaluated on the test set.
5. Model performance is assessed using metrics such as accuracy, area under the curve (AUC), and classification report.

## Results

The performance of the models was evaluated using the following metrics:

### LightGBM Algorithm

- Accuracy: 0.85
- AUC: 0.8125
- Classification Report:
  - Precision, recall, f1-score values
- Confusion Matrix:
  - True Positive (TP), True Negative (TN), False Positive (FP), False Negative (FN), specificity, sensitivity values
  - ![alt text](https://github.com/ugurde98/Introduction-to-Machine-Learning-Midterm-Project/blob/main/Ekran%20Resmi%202023-06-07%2022.14.35.png?raw=true)

### Random Forest Algorithm

- Accuracy: 0.875
- AUC: 0.84375
- Classification Report:
  - Precision, recall, f1-score values
- Confusion Matrix:
  - True Positive (TP), True Negative (TN), False Positive (FP), False Negative (FN), specificity, sensitivity values
  - ![alt text](https://github.com/ugurde98/Introduction-to-Machine-Learning-Midterm-Project/blob/main/Ekran%20Resmi%202023-06-07%2022.14.48.png?raw=true)

## Conclusion

The results of this study indicate that the DNA samples of microbes in human blood show promise as potential indicators of different tumor types. Both the LightGBM and Random Forest algorithms achieved high accuracy and AUC scores in predicting tumor types using the microbial DNA samples. These findings suggest that further research and exploration in this area could lead to novel approaches for tumor diagnosis and prediction.

## Dependencies

The following dependencies are required to run the code in this repository:

- pandas
- scikit-learn
- lightgbm
- Other necessary dependencies

Please make sure to install these dependencies before running the code.

## Usage

To replicate the results, follow these steps:

1. Clone this repository to your local machine.
2. Preprocess and split the dataset into training and test sets.
3. Run the `main.py` script to train and evaluate the models.
4. View the results and performance metrics.

Feel free to modify the code or experiment with different parameters to further improve the model performance.

## License

This project is licensed under the [MIT License](LICENSE).
