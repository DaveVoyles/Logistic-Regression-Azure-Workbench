# Please make sure matplotlib is included in the conda_dependencies.yml file.
import pickle
import sys
import os
import numpy             as np
import pandas            as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# sklearn
from sklearn.metrics         import confusion_matrix, mean_squared_error, explained_variance_score
from sklearn.linear_model    import LinearRegression
from sklearn.datasets        import load_boston
from sklearn.model_selection import train_test_split

# Azure specific 
from azureml.logging          import get_azureml_logger
from azureml.dataprep.package import run

# Initialize the logger
run_logger = get_azureml_logger() 
# Create the outputs folder where the scatter plot & model.pkl will export to
os.makedirs('./outputs', exist_ok=True)
# Retrieve Boston data set
boston = load_boston()
# Boston data set is a series of keys
# RETURNS: dict_keys(['feature_names', 'target', 'data', 'DESCR'])
print('KEYS:')
print(boston.keys())
bos        = pd.DataFrame(boston.data)
# Convert the index to the column names.
bos.columns = boston.feature_names
print('\nDATA SET: ')
print(bos.head())
# 'Target' key is actually the list of home prices
# Create a column called 'PRICE' and add target prices to bos data frame
bos['PRICE'] = boston.target
# Remove target prices from our features, as this is what we will be trying to predict
X = bos.drop('PRICE', axis=1)


print('\n==========================================')
print('Training 1')
lr1 = LinearRegression()
lr1.fit(X, bos.PRICE)

print('ESTIMATED INTERCEPT COEFFICIENT:',    round(lr1.intercept_, 2))
print('NUMBER OF COEFFICIENTS:         ',    len(lr1.coef_          ))

# Print first 5 results
lr1.predict(X)[0:5]
print('First 5 home values: ', boston.target [0:5])
print('First 5 predictions: ', lr1.predict(X)[0:5])

# serialize the model on disk in the special 'outputs' folder
print('\n==========================================')
print ("Export the model to model.pkl")
f = open('./outputs/model.pkl', 'wb')
pickle.dump(lr1, f)
f.close()

print('\n==========================================')
print('Training 2: train_test_split')

# load the model back from the 'outputs' folder into memory
print("Import the model from model.pkl")
f2 = open('./outputs/model.pkl', 'rb')
lr2 = pickle.load(f2)

X_train, X_test, y_train, y_test = train_test_split(X, bos.PRICE, test_size=0.33, random_state=5)
lr2.fit(X_train, y_train)
pred_train = lr2.predict(X_train)
pred_test  = lr2.predict(X_test )
mse_train  = round(mean_squared_error      (y_train, lr2.predict(X_train)), 4)
mse_test   = round(mean_squared_error      (y_test,  lr2.predict(X_test )), 4)
r_square   = round(explained_variance_score(y_test,  pred_test          ),  4)

print('MSE w/ TRAIN data: ',  mse_train)
print('MSE w/ TEST data:  ',  mse_test )
print('R-Square:           ', r_square )

# These results will appear in the Run Properties: Output in ML Workbench
run_logger.log('MSE w/ TRAIN data:', mse_train)
run_logger.log('MSE w/ TEST data: ', mse_test )
run_logger.log('R-Square:         ', r_square )

# Draw a plot. This will also appear in the Run Properties: Output in ML Workbench
# Ideally, scatter plot should create a line. Model doesn't fit 100%, so scatter plot is not creating a line
plt.scatter(y_test, pred_test                                 )
plt.xlabel ('Prices: $Y_i$'                                   )
plt.ylabel ('Predicted prices: $\hat{Y}_i$'                   )
plt.title  ('Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$')
plt.savefig("./outputs/scatter.png", bbox_inches='tight'      )



