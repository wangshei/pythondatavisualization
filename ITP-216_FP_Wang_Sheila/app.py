# Sheila Wang, wangshei@usc.edu
# ITP 216, Fall 2023
# Section: Wednesday
# Final Project
from statistics import mean
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



app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
db = "average_prices.db"


@app.route("/")
def home():
    #no year to predict should be passed at this point
    session.clear()
    options = {
        "area": "Compare Yearly Change By Area",
        "average": "Compare Fluctuation Each Year By U.S Average"
    }
    return render_template("home.html", items=db_get_items(), message="Please enter an item to search for.",
                           options=options)

@app.route("/submit_item", methods=["POST"])
def submit_item():
    # session["item"] = request.form["ite "].capitalize()
    print(request.form['item'])
    session["item"] = request.form["item"]
    if 'item' not in session or session["item"] == "":
        return redirect(url_for("home"))
    if "data_request" not in request.form:
        return redirect(url_for("home"))
    session["data_request"] = request.form["data_request"]
    return redirect(url_for("item_current", data_request=session["data_request"], item=session["item"]))

@app.route("/average_item_price/<data_request>/<item>")
def item_current(data_request,item):
    return render_template("item.html", data_request=data_request, item=item, project=False)


@app.route("/submit_projection", methods=["POST"])
def submit_projection():
    if 'item' not in session:
        return redirect(url_for("home"))
    session["year"] = request.form["year"]
    # THESE NEED TO BE BACK IN!
    # if session["item"] == "" or session["data_request"] == "" or session["date"] == "":
    #     return redirect(url_for("home"))
    return redirect(url_for("item_projection", data_request=session["data_request"], item=session["item"], year = session["year"]))

@app.route("/average_item_price/<data_request>/projection/<item>")
def item_projection(data_request, item):
    print(session["year"])
    return render_template("item.html", data_request=data_request, item=item, project=True, year=session["year"])


@app.route("/fig/<data_request>/<item>/<project>")
def fig(data_request, item,project):
    if "year" not in session:
        print("actual data")
        fig = create_figure(data_request, item, project)
    else:
        print("projected data")
        fig = item_projection(data_request, item)

    # img = io.BytesIO()
    # fig.savefig(img, format='png')
    # img.seek(0)
    # w = FileWrapper(img)
    # # w = werkzeug.wsgi.wrap_file(img)
    # return Response(w, mimetype="text/plain", direct_passthrough=True)

    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)

    return send_file(img, mimetype="image/png")

@app.route('/<path:path>')
def catch_all(path):
    return redirect(url_for("home"))

def db_get_items():
    conn = sl.connect('average_prices.db')
    # create a cursor to execute commends
    curs = conn.cursor()
    # Create Item List
    dfg = pd.read_sql_query("SELECT year, period, value, item_name, area_name FROM prices ORDER BY value", conn)
    #get unique items
    items = list(dfg["item_name"].unique())
    conn.close()
    return items

def db_get_name(item):
    #only use the actual name of the item
    item = item.split(",")
    return item[0]

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
def create_figure(data_request, item, project):
    df = db_create_dataframe(item)
    look_up_item_name = db_get_name(item)

    if "year" not in session:
        # list of years
        years_list = list(df["year"].unique())
        odd_years_list = []
        for year in years_list:
            if year % 2 != 0:
                odd_years_list.append(year)
        odd_years_list = sorted(odd_years_list)

        # # making a column for month using string slicing: allows us to group by month
        df["month"] = df["period"].str[-2:]
        list_of_month = list(df["period"].unique())

        # --------  Area Graph -------------------------------------------------
        fig = plt.Figure()
        fig.set_canvas(plt.gcf().canvas)
        fig, ax = plt.subplots(1, 1,figsize = (12,6))

        # add overall title
        fig.suptitle('Change in ' + look_up_item_name + ' price Across the U.S. from 1995 to 2023')

        x = df["year"]
        y = df["value"]

        # use ax[0] for your axis for top graph
        # set your other axis attributes using set() title, xlabel, ylabel
        ax.set(title ="Change of Price over Time By Area")
        ax.set( xlabel = "Years")
        ax.set( ylabel = "price($/unit)")
        pd.set_option('display.width', 500)


        # group df by 'YEAR'
        df_grouped = df.groupby("area_name")

        # loop through the years and plot observed temperature (TOBS) vs every day (MONTH_DAY)
        for area, group_data in df_grouped:
            # get year group

            # plot year group TOBS (y-axis) vs year group month day (x axis)
            ax.scatter(group_data["year"], group_data["value"], label = area, alpha = 0.5)

        # plot gridlines using axis.grid(...)
        ax.set_xticks(np.arange(min(x), max(x) + 1, 2))  # Set x-axis ticks for each year
        ax.set_xticklabels(odd_years_list,
                           rotation=45)  # Set the x-axis labels using years_list with rotation for better readability

        ax.grid()
        # show legend (make sure to add label attributes to your plot() call for this to work
        ax.legend()


        # -------- Average GRAPH -------------------------------------------------
        # be sure to use ax[1] going forward now
        #ax[1].plot(x, y)
        # set your attributes and xaxis ticks and labels as you did before
        # make the xtick list from a range from 0 to 365 and step by 31 days per month
        # xtick = np.arange(0, 29)
        # ax[1].set_xticks(xtick)
        # ax[1].set_xticklabels(years_list, rotation='vertical')
        fig2 = plt.Figure()
        fig2.set_canvas(plt.gcf().canvas)
        fig2, ax2 = plt.subplots(1, 1,figsize = (12,6))
        conn = sl.connect('average_prices.db')
        # add overall title
        fig2.suptitle('Change in ' + look_up_item_name + ' price from 1995 to 2023')

        m = 0
        grouped_list = []
        for m in range(len(list_of_month)):
            grouped_list.append([])
            sql = "SELECT value FROM prices WHERE area_name = 'U.S. city average' AND item_name = '" + item + "' AND period = '" + str(list_of_month[m]) + "' ORDER BY year"
            df1_grouped = pd.read_sql_query(sql, conn)
            price_list = pd.read_sql_query(sql, conn)
            grouped_list[m].append(price_list.values.tolist())
            m+=1

        i=0
        while i < len(grouped_list):
            ax2.boxplot(grouped_list[i][0])
            i +=1

        # ax[1].bar(years_list, avg_list, color = "blue", label = "average price each year", alpha = 0.5)


        # to avoid warnings, use axis.xaxis.set_xticks() and .set_ticklabels()
        xtick = np.arange(0, 29)
        ax2.set_xticks(xtick[::2])
        ax2.set_xticklabels(odd_years_list, rotation=45, horizontalalignment= 'left')

        ax2.set(title=" U.S. Average Change of Price over Time with Yearly Fluctuation")
        ax2.set(xlabel="Years")
        ax2.set(ylabel="price($/unit)")
        # add gridlines w/ .grid()
        ax2.grid()
        #make tight layout and finally show your plot
        fig2.tight_layout()
        if data_request == 'area':
            return fig
        if data_request == 'average':
            return fig2
    else:
        return item_projection(data_request,item)

# projection function
def item_projection(data_request,item):
    figp = plt.Figure()
    figp.set_canvas(plt.gcf().canvas)
    figp, ax = plt.subplots(1, 1, figsize=(12, 6))
    df = db_create_dataframe(item)
    # Define features (X) and target (y)
    X = df[['year']]  # input value
    y = df['value'] # prediction value
    print(X)
    if len(df) > 0:  # Ensure enough data for splitting
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    else:
        print("Insufficient data for splitting into training and testing sets.")

    # Modeling
    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("X_test")
    print(X_test)

    # new data
    years = []
    year_passed = int(session["year"])
    month_passed = 1
    print(df["year"].max())
    max_year = df["year"].max()
    years = list(range(max_year + 1, year_passed + 1))
    new_data = pd.DataFrame({'year': years})  # Assuming you have this input
    # Predict on test data or new inputs
    predictions = model.predict(new_data)  # Or use model.predict(new_data)

    # Visualize the prediction graph
    ax.scatter(df['year'], df['value'], color='blue', label='Original Data')  # Original data
    ax.plot(new_data['year'], predictions, color='red', label='Predicted Data')  # Predicted data
    # set labels
    plt.xlabel('Year')
    plt.ylabel('Price')
    plt.title(f'Price Prediction up to {year_passed}')
    plt.legend()
    figp.show()

    return figp

if __name__ == "__main__":
    # print(db_get_item())
    app.secret_key = os.urandom(12)
    app.run(debug=True, port = 5001)
