"""
Downloads the MovieLens dataset and saves it as an artifact
"""
import requests
import tempfile
import os
import zipfile
import mlflow
import argparse


def load_raw_data(url):
    with mlflow.start_run() as mlrun:
        local_dir = tempfile.mkdtemp()
        local_filename = os.path.join(local_dir, "LoanStats3a.csv.zip")
        print("Downloading %s to %s" % (url, local_filename))
        r = requests.get(url, verify=False, stream=True)
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

        extracted_dir = os.path.join(local_dir, 'data/raw')
        print("Extracting %s into %s" % (local_filename, extracted_dir))
        with zipfile.ZipFile(local_filename, 'r') as zip_ref:
            zip_ref.extractall(extracted_dir)

        loans_file = os.path.join(extracted_dir, 'LoanStats3a.csv')
        print(os.path.exists(loans_file))
        print("Uploading loans: %s" % loans_file)

        # remove first line from file to make it easy to read
        with open(loans_file, 'r') as f:
            new_file = os.path.join(extracted_dir, 'loans.csv')
            with open(new_file, 'w') as f1:
                next(f)  # skip header line
                for line in f:
                    f1.write(line)
        mlflow.log_artifact(new_file, "loans-raw-csv-dir")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--url',
                        default="https://resources.lendingclub.com/LoanStats3a.csv.zip")
    args = parser.parse_args()

    load_raw_data(args.url)
