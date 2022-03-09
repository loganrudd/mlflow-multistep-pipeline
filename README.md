Multistep Workflow Example
--------------------------
This multistep workflow was modified from the original [MLflow multistep workflow 
example](https://github.com/mlflow/mlflow/tree/master/examples/multistep_workflow)
to predict loan default using 5-fold cross validation and also was made to be compatible 
to run inside of a Databricks notebook.


There are three steps to this workflow:

- **load_raw_data.py**: Downloads the LendingClub dataset as CSV format

- **etl_data.py**: Converts the LendingClub CSV data from the
  previous step into Parquet, dropping unnecessary columns along the way,
  and engineering new columns.

- **train_lgbm.py**: Trains a LightGBM model and logs various things such as
  the best parameters chosen by RandomizedSearchCV, the best model fitted in
  pickle format, roc_auc metric, etc.

While we can run each of these steps manually, here we have a driver
run, defined as **main** (main.py). This run will run
the steps in order, passing the results of one to the next. 
Additionally, this run will attempt to determine if a sub-run has
already been executed successfully with the same parameters and, if so,
reuse the cached results.

Running this Example
^^^^^^^^^^^^^^^^^^^^

**In your local machine**:

First create a [Conda](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) 
environment with python=3.7 from conda.yaml file and activate:

.. code-block:: bash

    conda env create -n <name_of_env> -f conda.yaml python=3.7
    conda activate <name_of_env>

In order for the multistep workflow to find the other steps, you must
execute ``mlflow run`` from this directory.

.. code-block:: bash
    
    git clone git@github.com:loganrudd/mlflow-multistep-pipeline.git
    cd mlflow-multistep-pipeline/
    mlflow run .


This downloads and transforms the LendingClub dataset, processes features, and trains an LGBMClassifier 
model -- you can look at the details of the model, params, metrics, etc. that result from this pipeline by running ``mlflow ui``.

You can also try changing the parameter grids in the config.yaml to search through when tuning hyperparameters.
    
**In Databricks community edition** (developed on Databricks runtime version 7.0 ML, tested with 7.3 LTS ML):

First you need to setup your credentials

.. code-block:: python

    token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
    dbutils.fs.put("file:///root/.databrickscfg","[DEFAULT]\nhost=https://community.cloud.databricks.com\ntoken = "+token,overwrite=True)
    
then you can go ahead and execute the project with the MLflow api:

.. code-block:: python

    import mlflow
    mlflow.run('git://github.com/loganrudd/mlflow-multistep-pipeline.git')

then take a look at the 'Experiment' tab to see the logged results of the run!
