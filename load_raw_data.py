"""
Downloads the LendingClub dataset from Kaggle and saves it as an artifact.
First you must configure your Kaggle API credentials following the instructions
here: https://github.com/Kaggle/kaggle-api

"""
import mlflow
import argparse
import spark
from pyspark.sql.functions import *

def load_raw_data(data_path):

    with mlflow.start_run() as mlrun:
        loans = spark.read.parquet(data_path)
        print("Create bad loan label, this will include charged off, defaulted,"
              "and late repayments on loans...")
        loans = loans.filter(
            loans.loan_status.isin(["Default", "Charged Off", "Fully Paid"]))\
            .withColumn("bad_loan",
                        (~(loans.loan_status == "Fully Paid")).cast("int"))

        print("Casting numeric columns into the appropriate types...")
        loans = loans.withColumn('issue_year',
                                 substring(loans.issue_d, 5, 4).cast('double'))\
            .withColumn('earliest_year',
                        substring(loans.earliest_cr_line, 5, 4).cast('double'))\
            .withColumn('total_pymnt', loans.total_pymnt.cast('double'))
        loans = loans.withColumn('credit_length_in_years',
                                 (loans.issue_year - loans.earliest_year))
        loans.write.parquet("loans_processed.parquet")
        mlflow.log_artifact("/dbfs/loans_processed.parquet/",
                            "loans-processed-parquet")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--data_path',
                        default="/databricks-datasets/samples/"
                                "lending_club/parquet/")
    args = parser.parse_args()

    load_raw_data(args.data_path)
