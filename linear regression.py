# Supressing the warning messages
import warnings
warnings.filterwarnings('ignore')
#Importing Libraries
import numpy as np
import pickle
import pandas as pd

#import matplotlib.pyplot as plt
#import seaborn as sns
#from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
#from sklearn.metrics import classification_report
#from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
#reading the dataset
zomato_real=pd.read_csv(r"D:\Capgemini Training\sprint1\Project_SPrint1\hi\zomato.csv")
zomato_real.head()
# Function to count the number of cuisines
def cuisine_counter(inpStr):
    NumCuisines=len(str(inpStr).split(','))
    return(NumCuisines)

# Creating a new feature in data
# We will further explore the new feature just like other features
zomato_real['CuisineCount']=zomato_real['cuisines'].apply(cuisine_counter)
# Deleting those columns which are not useful in predictive analysis because these variables are qualitative
UselessColumns = ['url', 'address', 'name', 'phone', 'listed_in(city)','location', 'cuisines', 'menu_item']
zomato = zomato_real.drop(UselessColumns,axis=1)
#Removing the Duplicates
zomato.drop_duplicates(inplace=True)
#Remove the NaN values from the dataset
zomato.dropna(how='any',inplace=True)
#Changing the column names
zomato = zomato.rename(columns={'approx_cost(for two people)':'cost','listed_in(type)':'type'})
#Removing '/5' from Rates
zomato['cost'] = zomato['cost'].astype(str) #Changing the cost to string
zomato['cost'] = zomato['cost'].apply(lambda x: x.replace(',','')) #Using lambda function to replace ',' from cost
zomato['cost'] = zomato['cost'].astype(float) # Changing the cost to Float

zomato = zomato.loc[zomato.rate !='NEW']
zomato = zomato.loc[zomato.rate !='-'].reset_index(drop=True)
remove_slash = lambda x: x.replace('/5', '') if type(x) == np.str else x
zomato.rate = zomato.rate.apply(remove_slash).str.strip().astype('float')
# Replacing outliers with nearest possibe value
zomato['votes'][zomato['votes']>10000] =9300
SelectedColumns=['votes','cost','book_table', 'online_order', 'CuisineCount']

# Selecting final columns
DataForML=zomato[SelectedColumns]
#print(DataForML.head())
# Saving this final data for reference during deployment
#DataForML.to_pickle('DataForML.pkl')
# Converting the binary nominal variable sex to numeric
DataForML['book_table'].replace({'Yes':1, 'No':0}, inplace=True)
DataForML['online_order'].replace({'Yes':1, 'No':0}, inplace=True)
# Treating all the nominal variables at once using dummy variables
print(DataForML.head())
DataForML_Numeric=DataForML
print(DataForML_Numeric.head())



# Adding Target Variable to the data
DataForML_Numeric['rate']=zomato['rate']
#print(DataForML_Numeric.head())
TargetVariable='rate'
Predictors=['votes', 'cost', 'book_table','online_order', 'CuisineCount']
# Predictors=['votes', 'cost', 'CuisineCount']


X=DataForML_Numeric[Predictors].values
y=DataForML_Numeric[TargetVariable].values

# Split the data into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=428)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# Choose either standardization or Normalization
# On this data Min Max Normalization produced better results

# Choose between standardization and MinMAx normalization
#PredictorScaler=StandardScaler()
#PredictorScaler=MinMaxScaler()

# Storing the fit object for later reference
#PredictorScalerFit=PredictorScaler.fit(X)

# Generating the standardized values of X
#X=PredictorScalerFit.transform(X)

# Split the data into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#Prepare a Linear Regression Model
from sklearn.metrics import mean_squared_error
model0 =LinearRegression()
model0.fit(X_train,y_train)
print(X_test)
y_pred=model0.predict(X_test)
print(y_pred)
from sklearn.metrics import r2_score
print("Coeff:", model0.coef_)
print("Intercept:", model0.intercept_)
print('Mean squared error: %.3f' % mean_squared_error(y_test, y_pred))
print('Coefficient of determination: %.3f' % r2_score(y_test, y_pred))
print("The model testing score is" , model0.score(X_test, y_test))  
# print("The model training score is" , model.score(X_train, y_train))
# Saving model to disk
pickle.dump(model0, open('model1.pkl','wb'))

# Loading model to compare the results
model1 = pickle.load(open('model1.pkl','rb'))
print(model1.predict([[775,800, 1,1,3]]))
