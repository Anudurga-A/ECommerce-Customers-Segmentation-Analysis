# E-Commerce Customer Segmentation and Recommendation Analysis

## Project Overview
This project analyzes e-commerce customer transactions and segments customers based on their purchasing behavior using RFM Analysis and K-Means Clustering.

## Objectives
- Analyze online retail transaction data.
- Clean and preprocess customer data.
- Perform customer-level feature engineering.
- Calculate Recency, Frequency, and Monetary (RFM) values.
- Apply K-Means clustering to identify customer segments.
- Visualize customer purchasing patterns.

## Technologies Used
- Python 3.13
- Pandas
- NumPy
- Matplotlib
- Seaborn
- OpenPyXL
- K-Means Clustering

## Dataset
Online Retail Dataset containing 541,909 transaction records and 8 columns.

## Project Results
- Total Customers: 4,338
- Total Revenue: 8,887,208.89
- Number of Clusters: 4
- Silhouette Score: 0.5986

## Output Files
- Customer Feature Table
- RFM Analysis
- Customer Segments
- Cluster Summary
- Customer Spending Graph
- RFM Correlation Heatmap
- Elbow Method Graph
- Customer Clusters Graph

## How to Run
1. Install the required libraries:
   `py -3.13 -m pip install pandas numpy matplotlib seaborn openpyxl`
2. Run the project:
   `py -3.13 customer_segmentation.py`

## Conclusion
This project helps businesses understand customer purchasing behavior, identify valuable customer groups, and make data-driven marketing decisions.