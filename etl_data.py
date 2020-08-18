"""
Converts the raw CSV form to a Parquet form with just the columns we want
"""
import tempfile
import os
import mlflow
import argparse
import pandas as pd
import numpy as np


def etl_data(loans_csv_uri):
    with mlflow.start_run(nested=True) as mlrun:
        tmpdir = tempfile.mkdtemp()
        loans_parquet_dir = os.path.join(tmpdir, 'loans.parquet')
        print("Processing ratings CSV %s to Parquet %s" % (loans_csv_uri,
                                                           loans_parquet_dir))

        loans = pd.read_csv(os.path.join(loans_csv_uri, 'loans.csv'))

        # dropping columns with all NaNs
        nan_columns = [column for column in loans.columns if
                       loans[column].isnull().all()]
        loans.drop(
            columns=nan_columns + ['id', 'next_pymnt_d', 'settlement_status',
                                   'settlement_date', 'settlement_amount',
                                   'settlement_term', 'settlement_percentage',
                                   'debt_settlement_flag_date',
                                   'debt_settlement_flag'], inplace=True)

        print("Create bad loan label eg: charged off repayments on loans...")
        loans = loans[(loans.loan_status == "Fully Paid")
                      | (loans.loan_status == "Charged Off")]
        loans.loc[:, 'loan_status'] = loans.loan_status.apply(
                                            lambda x: int(x == "Charged Off"))
        loans.rename(columns={'loan_status': 'bad_loan'}, inplace=True)

        print("Dropping columns that have the same value for every loan...")
        uniques = loans.apply(lambda x: x.nunique())
        loans.drop(uniques[uniques == 1].index, axis=1, inplace=True)

        print("One hot encoding categorical columns...")
        loans = pd.concat([loans, pd.get_dummies(loans['home_ownership'])],
                          axis=1)
        loans = pd.concat([loans, pd.get_dummies(loans['purpose'])], axis=1)
        loans = pd.concat([loans, pd.get_dummies(loans['verification_status'])],
                          axis=1)
        loans = pd.concat([loans, pd.get_dummies(loans['addr_state'])], axis=1)
        loans = pd.concat([loans, pd.get_dummies(loans['term'])], axis=1)
        loans = pd.concat([loans, pd.get_dummies(loans['grade'])], axis=1)
        loans = pd.concat([loans, pd.get_dummies(loans['sub_grade'])], axis=1)
        loans = pd.concat([loans, pd.get_dummies(loans['emp_length'])], axis=1)

        loans.rename(columns={' 36 months': '36 months'}, inplace=True)
        loans.rename(columns={' 60 months': '60 months'}, inplace=True)

        print("Casting numeric columns into the appropriate types...")
        loans.loc[:, 'earliest_cr_line'] = loans.earliest_cr_line.apply(
                                                        lambda x: int(x[-4:]))
        loans.loc[:, 'issue_d'] = loans.issue_d.apply(lambda x: int(x[-4:]))
        loans.loc[:, 'int_rate'] = loans.int_rate.apply(lambda x: float(x[:6]))

        loans['credit_length'] = loans.issue_d - loans.earliest_cr_line

        obj_cols = [column for column in loans.columns if
                    loans[column].dtype == np.object]
        loans.drop(columns=obj_cols, inplace=True)

        loans.to_parquet(path=loans_parquet_dir)
        print("Uploading Parquet loans: %s" % loans_parquet_dir)
        mlflow.log_artifacts(tmpdir, "loans-processed-parquet-dir")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--loans_csv_uri")
    args = parser.parse_args()

    etl_data(args.loans_csv_uri)
