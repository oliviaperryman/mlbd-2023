#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

import pandas as pd

def person_a(data_path=None):
    """Reads the data in data/school_performance.csv
    and returns a dataframe with the first 5,000 rows.

    Returns:
    dataframe: containing first 5,000 rows of school_performace.csv
    """
    data = pd.read_csv(os.path.join(data_path, "school_performance.csv"))
    return data.head(5000)

def person_b(df):
    """Keeps only the data from the female students. 
    Where the column "gender" equals "female". 

    Parameters:
    df (dataframe): First 5,000 rows of school_performace.csv

    Returns:
    dataframe: Data from the female students
    """
    # Code goes over here.
    new_df = df.drop(df[df.gender != "female"].index)
    return new_df

def person_c(df):
    """Calculates the mean from the column "grade"

    Parameters:
    df (dataframe): Data from female students

    Returns:
    float: Mean grade
    """
    return df["grade"].mean()

def main(data_path="data/"):
    """ Main program """
    # Code goes over here.
    df = person_a(data_path)
    df = person_b(df)
    res = person_c(df)

    print(f"Mean grade of female students is {res}")

if __name__ == "__main__":
    main()
