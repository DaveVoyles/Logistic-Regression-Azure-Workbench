# This script generates the scoring and schema files
# Creates the schema, and holds the init and run functions needed to 
# operationalize the Linear Regression sample

# Import data collection library. Only supported for docker mode.
# Functionality will be ignored when package isn't found
try:
    from azureml.datacollector import ModelDataCollector
except ImportError:
    print("Data collection is currently only supported in docker mode. May be disabled for local mode.")
    # Mocking out model data collector functionality
    class ModelDataCollector(object):
        def nop(*args, **kw): pass
        def __getattr__(self, _): return self.nop
        def __init__(self, *args, **kw): return None
    pass

import os

# Prepare the web service definition by authoring
# init() and run() functions. Test the functions
# before deploying the web service.
def init():  
    from sklearn.externals import joblib
    global model, inputs_dc, prediction_dc
    # The model we created in our linear_reg.py file is now a model.pkl
    model = joblib.load('model.pkl')
    inputs_dc     = ModelDataCollector('model.pkl', identifier="inputs"    )
    prediction_dc = ModelDataCollector('model.pkl', identifier="prediction")


#  Uses the model and the input data to return a prediction
def run(input_df):
    global clf2, inputs_dc, prediction_dc
    try:
        prediction = model.predict(input_df)
        # Archive model inputs and predictions from a web service. View the collected data from your storage account
        # More info here: https://docs.microsoft.com/en-us/azure/machine-learning/preview/how-to-use-model-data-collection
        inputs_dc.collect    (input_df)
        prediction_dc.collect(prediction)
        return prediction
    except Exception as e:
        return (str(e))


# Generate service_schema.json
def main():
    from azureml.api.schema.dataTypes        import DataTypes
    from azureml.api.schema.sampleDefinition import SampleDefinition
    from azureml.api.realtime.services       import generate_schema
    import pandas
  
    # Turn on data collection debug mode to view output in stdout
    os.environ["AML_MODEL_DC_DEBUG"] = 'true'

    inputs = {"input_df": SampleDefinition(DataTypes.PANDAS, yourinputdataframe)}
    generate_schema(run_func=run, inputs=inputs, filepath='service_schema.json')
    print("Schema generated")

    if __name__ == "__main__":
        main()
