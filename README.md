# Apziva Project 1: Customer Happiness Prediction

## Overview

This project aims to predict customer happiness based on survey responses. By analyzing customer feedback, we strive to identify key factors influencing satisfaction and develop a predictive model to assess customer happiness.

## Dataset

The primary dataset used in this project is the `ACME-HappinessSurvey2020.csv` file, which contains survey responses from customers. Each row represents an individual customer's feedback, with the following columns:

- **X1**: My order was delivered on time (1-5 scale)
- **X2**: Contents of my order were as expected (1-5 scale)
- **X3**: I ordered everything I wanted (1-5 scale)
- **X4**: I paid a good price for my order (1-5 scale)
- **X5**: I am satisfied with my courier (1-5 scale)
- **X6**: The app makes ordering easy for me (1-5 scale)
- **Y**: Customer happiness (0 = Unhappy, 1 = Happy)

## Project Structure

- **`Project1.ipynb`**: Jupyter Notebook containing the main analysis and modeling workflow.
- **`analysis.py`**: Python script for data preprocessing and exploratory data analysis.
- **`analysis2.py`**: Additional Python script for supplementary analysis.
- **`ACME-HappinessSurvey2020.csv`**: Dataset file containing customer survey responses.
- **`README.md`**: This file, providing an overview of the project.

## Installation

To run the analysis and models locally, ensure you have the following Python packages installed:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

You can install these packages using pip:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Acknowledgments

Apziva for providing the dataset and project framework.​ The open-source community for their invaluable tools and libraries.​
