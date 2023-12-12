import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Any
import sqlite3 as sl
from flask import Flask, redirect, render_template, request, session, url_for, send_file
import datetime
import io
import os
import sqlite3 as sl
from matplotlib.figure import Figure
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
def db_create_dataframe(item):
    look_up_item = item
    # establish connection to db
    conn = sl.connect('average_prices.db')
    # create a cursor to execute commends
    curs = conn.cursor()
    # reading in the dataset
    df = pd.read_sql_query(
        "SELECT year, period, value, item_name, area_name FROM prices WHERE item_name = '" + look_up_item + "' ORDER BY value",
        conn)
    # removing rows of data where the average price is null

    df = df[df["value"].notnull()]
    # make only the odd year show up

    #only look at the appropriate one
    df = df[df["item_name"] == look_up_item]
    df["month"] = df["period"].str[-2:]
    # sort your df by MONTH_DAY (use df.sort_values())
    df.sort_values(inplace=True, by='value')
    return df

def main():
    df = db_create_dataframe("American processed cheese, per lb. (453.6 gm)")
    # Define features (X) and target (y)
    X = df[['year']]  # Adjust columns as needed
    y = df['value']
    print(X)
    if len(df) > 0:  # Ensure enough data for splitting
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Continue with modeling, fitting, and predictions...
        # ...
    else:
        print("Insufficient data for splitting into training and testing sets.")

    # Modeling
    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("X_test")
    print(X_test)

    #new data
    years = []
    year_passed = 2028
    month_passed = 1
    print(df["year"].max())
    max_year = df["year"].max()
    years = list(range(max_year+1, year_passed + 1))
    print(years)
    new_data = pd.DataFrame({'year': years}) # Assuming you have this input
    print(new_data)
    # Predict on test data or new inputs
    predictions = model.predict(new_data)  # Or use model.predict(new_data)

    # Evaluate the model (if needed)
    # ... (Metrics, visualization)
    # Visualize the prediction graph
    plt.figure(figsize=(10, 6))
    plt.scatter(df['year'], df['value'], color='blue', label='Original Data')  # Original data
    plt.plot(new_data['year'], predictions, color='red', label='Predicted Data')  # Predicted data
    plt.xlabel('Year')
    plt.ylabel('Price')
    plt.title(f'Price Prediction up to {year_passed}')
    plt.legend()
    plt.show()

    # Example output/printing predictions
    print("predictions")
    print(predictions)

if __name__ == "__main__":
    main()
