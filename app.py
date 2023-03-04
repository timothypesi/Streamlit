# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the data set into a Pandas dataframe
data = pd.read_csv("C:\\Users\\timothy.pesi\\Downloads\\train.csv")

# Define a preprocessor for the categorical feature 'SaleCondition'
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Define a column transformer to apply the preprocessor to the categorical feature
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, ['SaleCondition'])
    ])

# Define a Ridge regression model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', Ridge(alpha=1.0))
])

# Train the model on the entire data set
X = data[['WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 'MiscVal', 'SaleCondition']]
y = data['SalePrice']
model.fit(X, y)

# Define a function to make predictions
def predict_price(WoodDeckSF, OpenPorchSF, EnclosedPorch, MiscVal, SaleCondition):
    input_data = {'WoodDeckSF': WoodDeckSF,
                  'OpenPorchSF': OpenPorchSF,
                  'EnclosedPorch': EnclosedPorch,
                  'MiscVal': MiscVal,
                  'SaleCondition': SaleCondition}
    input_df = pd.DataFrame(input_data, index=[0])
    prediction = model.predict(input_df)[0]
    return prediction

# Define the Streamlit app
st.title('Predict Sales Price')

# Define the input fields
WoodDeckSF = st.slider('WoodDeckSF', min_value=0, max_value=1000, value=500, step=10)
OpenPorchSF = st.slider('OpenPorchSF', min_value=0, max_value=500, value=250, step=10)
EnclosedPorch = st.slider('EnclosedPorch', min_value=0, max_value=500, value=250, step=10)
MiscVal = st.slider('MiscVal', min_value=0, max_value=10000, value=5000, step=100)
SaleCondition = st.selectbox('SaleCondition', ['Normal', 'Abnorml', 'Partial', 'AdjLand', 'Alloca', 'Family'])

# Make predictions and display the result
prediction = predict_price(WoodDeckSF, OpenPorchSF, EnclosedPorch, MiscVal, SaleCondition)
st.write('Predicted Sales Price:', prediction)
                     
