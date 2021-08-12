Multistep Workflow Example
--------------------------
This multistep workflow was modified from the original example to run inside of a Databricks notebook.
Most of this readme remains from the original example, but includes added support for Databricks.


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

In order for the multistep workflow to find the other steps, you must
execute ``mlflow run`` from this directory. So, in order to find out if
the Keras model does in fact improve upon the ALS model, you can simply
run:

.. code-block:: bash

    cd examples/multistep_workflow
    mlflow run .


This downloads and transforms the MovieLens dataset, trains an ALS 
model, and then trains a Keras model -- you can compare the results by 
using ``mlflow ui``.

You can also try changing the number of ALS iterations or Keras hidden
units:

.. code-block:: bash

    mlflow run . -P als_max_iter=20 -P keras_hidden_units=50
    
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