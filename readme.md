# Simple Linear Regression With Azure ML Workbench

#### Author(s): Dave Voyles | [@DaveVoyles](http://www.twitter.com/DaveVoyles)
#### URL: [www.DaveVoyles.com](http://www.davevoyles.com)

This sample creates a simple linear regression model from [Scikit-Learn Boston dataset](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html) hosts it in Azure as web service.
----------
My goal with this was create a bare-bones example of how to deploy a model to Azure from ML Workbench. I couldnt find another example which did only that. 

Before going any further, I'd recommend watching this video, [operationalize your models with AML](https://www.youtube.com/watch?v=hsU2rUYYc4o). This will walk you through nearly every step of the work below.

Then read the [conceptual overview of Azure ML model management](https://docs.microsoft.com/en-us/azure/machine-learning/preview/model-management-overview). A glance at the [Key Concepts page](https://docs.microsoft.com/en-us/azure/machine-learning/preview/overview-general-concepts) will help as well. 

## Overview
You'll need three (3) key **files** to deploy a model:
 1. Your model, saved as *model.pkl*
 2. A scoring script 
 3. *service_schema.json* for web-service input data

This is explained in more detail in the [deploy a web service page](https://docs.microsoft.com/en-us/azure/machine-learning/preview/model-management-service-deploy).

To deploy the web service along with the model file, you also need a scoring script. Optionally, you need a schema for the web-service input data. The scoring script loads your model and returns the prediction result(s) using the model. It must include two functions: **init** and **run.**

The model and the scoring file upload to the storage account you created as part of the environment setup. The deployment process builds a Docker image with your model, schema, and scoring file in it, and then pushes it to the Azure container registry.

The compute environment is based on Azure Container Services. Azure Container Services provides automatic exposure of Machine Learning APIs as REST API endpoints with the following features:

* Authentication
* Load balancing
* Automatic scale-out
* Encryption

**Azure Machine Learning model management uses the following information:**
* Model file or a directory with the model files
* User created Python file implementing a model scoring function
* Conda dependency file listing runtime dependencies
* Runtime environment choice (spark, python, etc) 
* Schema file for API parameters

## Do I need to make all of these things myself?
The model.pkl, scoring file, and service_schema.json you'll need to create.

That was my initial concern. When you start with the blank ML workbench project, you'll receive a folder marked *aml_config* with the config and compute dependencies you need to get a project working. The only thing you'll need to add to these are the specific libraries or dependencies your project will require. 


## Generating the schema 
The [model management overview page](https://docs.microsoft.com/en-us/azure/machine-learning/preview/model-management-service-deploy#2-create-a-schemajson-file) is the best resource I've found for understanding how to generate the score and schema files. 

A schema will automatically validate the input and output of your web service. The CLIs also use the schema to generate a Swagger document for your web service.

I've tried exporting it directly into the root directory, but it never actually gets saved. It seems that you must place it in the *./outputs* folder, which will then appear in the in the **outputs** section of ML Workbench after each run. 

## Creating a scoring.py file
Instructions on how to do that are [here.](https://docs.microsoft.com/en-us/azure/machine-learning/preview/model-management-service-deploy#3-create-a-scorepy-file)

## Conda Dependencies
The only thing I've added to the default *aml_config/conda_dependencies.yml* is *scikit-learn* to the dependencies section. 

```
dependencies:
  # The python interpreter version.
  # Currently Azure ML Workbench only supports 3.5.2.
  - python=3.5.2
  - matplotlib
  - scikit-learn
```

## Logging & Printing output
At the top of *linear_reg.py* I've imported the two files required for logging:

``` 
from azureml.logging          import get_azureml_logger
from azureml.dataprep.package import run
```

Along the way, I'm making use of both Python's default *print()* function, and also Azure's *run_logger.log()* function. Print() will return values to console, which makes your code easy to debug.

run_logger.log() will return text and graphs to ML Workbench, which is great for comparing runs.

The code used to generate the images below look like this:

```
print('MSE w/ TRAIN data: ',  mse_train)
print('MSE w/ TEST data:  ',  mse_test )
print('R-Square:           ', r_square )

# These results will appear in the Run Properties: Output in ML Workbench
run_logger.log('MSE w/ TRAIN data:', mse_train)
run_logger.log('MSE w/ TEST data: ', mse_test )
run_logger.log('R-Square:         ', r_square )
```

![ml-workbench-logger-1](https://www.dropbox.com/s/5qpsj3d0aljukdt/ml-workbench-logger-1.png?raw=1)

![ml-workbench-logger-2](https://www.dropbox.com/s/1ebv6k91egjouu4/ml-workbench-logger-2.png?raw=1)

And I can output a scatter plot with:

```plt.savefig("./outputs/scatter.png", bbox_inches='tight'      )```

## About the project
It shows how to use `matplotlib` to plot the data and the fitted line, and save a plot file (png format) to view it in the **Runs** view  in Azure Machine Learning Workbench.

Once your script is executed, you can see your plot as part of your run history in Azure ML Workbench by navigating to the **Runs** section in your project and clicking on your run. 

![](./docs/simplelrplot.png)

# Running the app
[There are several ways we can run this app.](https://docs.microsoft.com/en-us/azure/machine-learning/preview/data-prep-supported-runtime-data-environments) 

1. Locally
2. Locally in a Docker container
3. Azure DSVM
4. Azure in a Docker container
5. Azure HDInsight PySpark
6. Azure HDInsight Python

Any of the Azure options allow you to host this as a web service.

## Instructions for running the script from CLI window
You can run your scripts from the Workbench app. However, we use the command-line window to watch the feedback in real time.

## Running your linear regression script locally
Open the command-line window by clicking on **File** --> **Open Command Prompt** and install the `matplotlib` using the following command.

```
conda install matplotlib
```

Once matplotlib is installed, you can run the following command to run this sample. 

```
$ az ml experiment submit -c local linear_reg.py
```

## Running your linear regression script on local or remote Docker
If you have a Docker engine running locally, you can run `linear_reg.py` in a local docker container. Since Docker-based runs are managed by `conda_dependencies.yml` file, it needs to have a reference to the `matplotlib` library. This sample already has that reference. 

```
dependencies:
  - matplotlib
```

Run the following command for executing your script on local Docker:
```
# submit the experiment to local Docker container for execution
$ az ml experiment submit -c docker linear_reg.py
```

You can also execute your script on Docker on a remote machine. Similar to local Docker execution, `conda_dependencies.yml` needs to have the following reference:
```
dependencies:
  - matplotlib

```
If you have a compute target named _myvm_ for a remote VM, you can run the following command to execute your script:

```
$ az ml experiment submit -c myvm linear_reg.py
```

You can use this command to create a compute target.
```
$ az ml computetarget attach --name myvm --address <ip address or FQDN> --username <username> --password <pwd> --type remotedocker
```
>Note: Your first execution on docker-based compute target automatically downloads a base Docker image. For that reason, it takes a few minutes before your job starts to run. Your environment is then cached to make subsequent runs faster. 

## Deploying your app as a remote web service
1. Run the *linear_reg.py* file
2. Open the **run** panel in workbench and download the model.pkl file
3. Move the file to the root directory of this project
4. Run the *score.py* file
5. This generates our *service_schema.json* file in the outputs folder
6. Move the file to the root directory of this project

Now we have all three required files, and your directory should look like this:

* LinearRegression
    * model.pkl
    * score.py
    * service_schema.json
    * ....All-other-files

### Azure Steps
There's a lot going on here, so take a look at this image to get a feel for what we are about to do, as we'll be working from left to right. We have several things you'll need to do before we can deploy this: 

![overview-general-concepts](https://docs.microsoft.com/en-us/azure/machine-learning/preview/media/overview-general-concepts/hierarchy.png)

1. [Set up your experimentation account.](https://docs.microsoft.com/en-us/azure/machine-learning/preview/deployment-setup-configuration#environment-setup) (orange)
    * This is everything involved with the actual code behind our model.

2. Create a [model management account.](https://docs.microsoft.com/en-us/azure/machine-learning/preview/deployment-setup-configuration#create-a-model-management-account) (green)
    *  You need to do this once per subscription, and can reuse the same account in multiple deployments.

3. [Deploy it as a web service](https://docs.microsoft.com/en-us/azure/machine-learning/preview/model-management-service-deploy#4-register-a-model) (yellow)
    * This will create a *manifest*, *image*, and *service*.


# Resources
* [Model Management overview](https://docs.microsoft.com/en-us/azure/machine-learning/preview/model-management-overview)
* [Deploy a web service](https://docs.microsoft.com/en-us/azure/machine-learning/preview/model-management-service-deploy)
* [What's new in Azure ML? Ignite 2017 [VIDEO]]()
* [Step-by-step instructions to deploy a model from ML workbench](https://www.microsoft.com/developerblog/2017/10/24/bird-detection-with-azure-ml-workbench/#depl_link)
* [VS Code extension - VS Code Tools for AI](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.vscode-ai)


