"""
Downloads the LendingClub dataset from Kaggle and saves it as an artifact.
First you must configure your Kaggle API credentials following the instructions
here: https://github.com/Kaggle/kaggle-api

"""
import requests
import tempfile
import os
import zipfile
import pyspark
import mlflow
import argparse
import dbutils

def load_raw_data(url):
    '''
    with mlflow.start_run() as mlrun:
        local_dir = tempfile.mkdtemp()
        local_filename = os.path.join(local_dir, "loans.csv.zip")
        print("Downloading %s to %s" % (url, local_filename))
        r = requests.get(url, stream=True)
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

        extracted_dir = os.path.join(local_dir, 'ml-20m')
        print("Extracting %s into %s" % (local_filename, extracted_dir))
        with zipfile.ZipFile(local_filename, 'r') as zip_ref:
            zip_ref.extractall(local_dir)
        
        ratings_file = os.path.join(extracted_dir, 'loans.csv')

        print("Uploading ratings: %s" % ratings_file)
        mlflow.log_artifact(ratings_file, "ratings-csv-dir")
    '''

    # Pull data path and version from notebook params
    dbutils.widgets.text(name="deltaVersion", defaultValue="",
                         label="Table version, default=latest")
    dbutils.widgets.text(name="deltaPath", defaultValue="", label="Table path")

    data_version = None if dbutils.widgets.get("deltaVersion") == "" else int(
        dbutils.widgets.get("deltaVersion"))
    DELTA_TABLE_DEFAULT_PATH = "/ml/loan_stats.delta"
    data_path = DELTA_TABLE_DEFAULT_PATH if dbutils.widgets.get(
        "deltaPath") == "" else dbutils.widgets.get("deltaPath")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--url',
                        default="https://www.kaggle.com/wendykan/"
                                "lending-club-loan-data?select=loan.csv")
    args = parser.parse_args()

    load_raw_data(args.url)
