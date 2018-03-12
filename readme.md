# Simple Linear Regression With Azure ML Workbench

#### Author(s): Dave Voyles | [@DaveVoyles](http://www.twitter.com/DaveVoyles)
#### URL: [www.DaveVoyles.com](http://www.davevoyles.com)

This sample creates a simple linear regression model from [Scikit-Learn Boston dataset](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html) and deploys to Azure as web service.
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

```generate_schema(run_func=run, inputs=inputs, filepath='./outputs/service.schema.json)```

You pass in your run function and an example of the input your model should expect, and it will generate a schema for you.

The model and the scoring file upload to the storage account you created as part of the environment setup. The deployment process builds a Docker image with your model, schema, and scoring file in it, and then pushes it to the Azure container registry.

The compute environment is based on Azure Container Services. Azure Container Services provides automatic exposure of Machine Learning APIs as REST API endpoints with the following features:

* Authentication
* Load balancing
* Automatic scale-out
* Encryption


**What is a "managed model?"**

A model is the output of a training process and is the application of a machine learning algorithm to training data. Model Management enables you to deploy models as web services, manage various versions of your models, and monitor their performance and metrics. "Managed" models have been registered with an Azure Machine Learning Model Management account. As an example, consider a scenario where you are trying to forecast sales. 

During the experimentation phase, you generate many models by using different data sets or algorithms. You have generated four models with varying accuracies but choose to register only the model with the highest accuracy. The model that is registered becomes your first managed model.

**What is a "deployment?"**

Model Management allows you to deploy models as packaged web service containers in Azure. These web services can be invoked using REST APIs. Each web service is counted as a single deployment, and the total number of active deployments are counted towards your plan. 

Using the sales forecasting example, when you deploy your best performing model, your plan is incremented by one deployment. If you then retrain and deploy another version, you have two deployments. If you determine that the newer model is better, and delete the original, your deployment count is decremented by one.

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

## Deploying your app as a (local) web service
This part can be confusing, because the terminology is a bit mixed here. It looks like we're doing a lot with Azure and the CLI, but in reality we setting up our *local* environment for testing, building, and deploying models. So in the end, this will give us a *localhost* endpoint to preview how our model would work, should we deploy it to Azure.  

### ML Workbench steps
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

### Environment Prep Steps
There's a lot going on here, so take a look at this image to get a feel for what we are about to do, as we'll be working from left to right. We have several things you'll need to do before we can deploy (create an endpoint): 

![overview-general-concepts](https://docs.microsoft.com/en-us/azure/machine-learning/preview/media/overview-general-concepts/hierarchy.png)

1. [Set up your experimentation account.](https://docs.microsoft.com/en-us/azure/machine-learning/preview/deployment-setup-configuration#environment-setup) (orange)
    * This is everything involved with the actual code behind our model.

2. Create a [model management account.](https://docs.microsoft.com/en-us/azure/machine-learning/preview/deployment-setup-configuration#create-a-model-management-account) (green)
    *  You need to do this once per subscription, and can reuse the same account in multiple deployments.

3. [Deploy it as a web service](https://docs.microsoft.com/en-us/azure/machine-learning/preview/model-management-service-deploy#4-register-a-model) (yellow)
    * This will create a *manifest*, *image*, and *service*.
    
    
#### You can actually do all of this with one step

Deploy the model as a web service:

```az ml service create realtime --model-file model.pkl -f score.py -n dvmodelmgmt -s  service_schema -r python -c ./aml_config/conda_dependencies.yml```

Alternatively, if you wanted to do it piece-by-piece, here is how:

#### Create an environment

Create a resource group to keep all of this stored

```az group create -l eastus2 -n dvmodelmgmt```

Local Deployment, using the newly created resource group:

```az ml env setup -l eastus2 -n dvmodelmgmt -g dvmodelmgmt```

To see more information for your environment, run:

```az ml env show -g dvmodelmgmt -n dvmodelmgmt```

You can set the new environment as your target context using:

```az ml env set -g dvmodelmgmt -n dvmodelmgmt```


#### Create a model management account

Create a new model management account

```az ml account modelmanagement create -l  eastus2 -n dvmodelmgmt -g dvmodelmgmt --sku-instances 1 --sku-name  S1```

Deploy the model as a web service (locally, for tetsing from *localhost*):

**NOTE:** I removed *.json* extension from the service_schema.json, otherwise the CLI will throw an error.

```az ml service create realtime --model-file model.pkl -f score.py -n dvmodelmgmt -s  service_schema -r python -c ./aml_config/conda_dependencies.yml```


We still need to deploy to production (Azure) though! Right now we can only call this from localhost. 


You'll notice that I use the name *dvmodelmgmt* for both my app name **and** resource group. I do that to keep it simple. Alternatively, you could have done this all through the web portal. You'll see that it worked for me, as I now have a resource group titled *dvmodelmgmt*. There also one with the same name followed by random digits. I never quite understand why that gets created though.

![aml-resource-groups-portal](https://www.dropbox.com/s/9e28i0u04bcjfe2/aml-resource-groups-portal.png?raw=1)
![aml-workbench-portal-resource-group](https://www.dropbox.com/s/82jiesitauauqry/aml-workbench-portal-resource-group.png?raw=1)

If all of your commands went through, you should see this in the console:

![aml-workbench-web-deploy-success](https://www.dropbox.com/s/2py4lmd1pkpsdf9/aml-workbench-web-deploy-success.png?raw=1)


### How would you consume (or call) this endpoint now?

You can connect to a Machine Learning Web service using any programming language that supports HTTP request and response.  Whether it is a web app, deskptop app, or another ML script, you're going to want to call that endpoint (currently at the localhost address), and pass in a pandas dataframe, as we specified that type in the main() function from score.py. You'll also need to paas the API key from your model management account.

I prefer to keep it simple and use something like [postman](https://www.getpostman.com/) to make a simple API call.  Set the body to *raw*, and pass in the data like so:

**TODO:** Replace this with example from linear regression. Right now I passed in dummy data from iris-classification as an example.

```{"input_df": [{"sepal length": 3.0, "petal width": 0.25, "sepal width": 3.6, "petal length": 1.3}]}```

Detailed instructions on how to do that are in this documentation, [How to consume an Azure ML web serivce.](https://docs.microsoft.com/en-us/azure/machine-learning/studio/consume-web-services)


# Resources
* [Operationalize your models with AML[VIDEO]](https://www.youtube.com/watch?v=hsU2rUYYc4o)
* [Model Management overview](https://docs.microsoft.com/en-us/azure/machine-learning/preview/model-management-overview)
* [Deploy a web service](https://docs.microsoft.com/en-us/azure/machine-learning/preview/model-management-service-deploy)
* [What's new in Azure ML? Ignite 2017 [VIDEO]]()
* [Step-by-step instructions to deploy a model from ML workbench](https://www.microsoft.com/developerblog/2017/10/24/bird-detection-with-azure-ml-workbench/#depl_link)
* [VS Code extension - VS Code Tools for AI](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.vscode-ai)
* [What is Azure ML? - FAQ](https://docs.microsoft.com/en-us/azure/machine-learning/preview/frequently-asked-questions)
[How to consume an Azure ML web serivce](https://docs.microsoft.com/en-us/azure/machine-learning/studio/consume-web-services)


