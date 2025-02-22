# Apziva Project 1: Customer Happiness Prediction

## Overview

This project aims to predict customer happiness based on survey responses. By analyzing customer feedback, we strive to identify key factors influencing satisfaction and develop a predictive model to assess customer happiness.

## Dataset

The primary dataset used in this project is the `ACME-HappinessSurvey2020.csv` file, which contains survey responses from customers. Each row represents an individual customer's feedback, with the following columns:

- **X1**: My order was delivered on time (1-5 scale)&#8203;:contentReference[oaicite:0]{index=0}
- **X2**: Contents of my order were as expected (1-5 scale)&#8203;:contentReference[oaicite:1]{index=1}
- **X3**: I ordered everything I wanted (1-5 scale)&#8203;:contentReference[oaicite:2]{index=2}
- **X4**: I paid a good price for my order (1-5 scale)&#8203;:contentReference[oaicite:3]{index=3}
- **X5**: I am satisfied with my courier (1-5 scale)&#8203;:contentReference[oaicite:4]{index=4}
- **X6**: The app makes ordering easy for me (1-5 scale)&#8203;:contentReference[oaicite:5]{index=5}
- **Y**: Customer happiness (0 = Unhappy, 1 = Happy)&#8203;:contentReference[oaicite:6]{index=6}

## Project Structure

- **`Project1.ipynb`**: Jupyter Notebook containing the main analysis and modeling workflow.&#8203;:contentReference[oaicite:7]{index=7}
- **`analysis.py`**: Python script for data preprocessing and exploratory data analysis.&#8203;:contentReference[oaicite:8]{index=8}
- **`analysis2.py`**: Additional Python script for supplementary analysis.&#8203;:contentReference[oaicite:9]{index=9}
- **`ACME-HappinessSurvey2020.csv`**: Dataset file containing customer survey responses.&#8203;:contentReference[oaicite:10]{index=10}
- **`README.md`**: This file, providing an overview of the project.&#8203;:contentReference[oaicite:11]{index=11}

## Installation

To run the analysis and models locally, ensure you have the following Python packages installed:

- pandas&#8203;:contentReference[oaicite:12]{index=12}
- numpy&#8203;:contentReference[oaicite:13]{index=13}
- scikit-learn&#8203;:contentReference[oaicite:14]{index=14}
- matplotlib&#8203;:contentReference[oaicite:15]{index=15}
- seaborn&#8203;:contentReference[oaicite:16]{index=16}

You can install these packages using pip:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
