from django.http import HttpResponse
from django.shortcuts import render
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotnine import *
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
import pickle
# Create your views here.

def home(request):
  return render(request, "home.html")

def graph(request):
  return render(request, "graph.html")

def about(request):
  return render(request, "about.html")

def predict(request):
  df_yield = pd.read_csv('yield.csv')
  # df_yield.shape

  # df_yield.head()

  # df_yield.tail(10)

  df_yield = df_yield.rename(index=str, columns={"Value": "hg/ha_yield"})
  df_yield = df_yield.drop(['Year Code','Element Code','Element','Year Code','Area Code','Domain Code','Domain','Unit','Item Code'], axis=1)
  # df_yield.head()

  # df_yield.describe()

  # df_yield.info()

  # Rain dataset
  df_rain = pd.read_csv('rainfall.csv')
  # df_rain.head()

  # df_rain.tail()

  df_rain = df_rain.rename(index=str, columns = {' Area':'Area'})
  # df_rain.head()

  # df_rain.info()

  df_rain['average_rain_fall_mm_per_year'] = pd.to_numeric(df_rain['average_rain_fall_mm_per_year'],errors = 'coerce')
  # df_rain.info()

  df_rain = df_rain.dropna()

  # df_rain.describe()

  yield_df = pd.merge(df_yield, df_rain, on=['Year','Area'])
  # yield_df.head()

  # yield_df.describe()

  # Pesticides DataSet

  df_pes = pd.read_csv('pesticides.csv')
  # df_pes.head()

  df_pes = df_pes.rename(index=str, columns={"Value": "pesticides_tonnes"})
  df_pes = df_pes.drop(['Element','Domain','Unit','Item'], axis=1)
  # df_pes.head()

  # df_pes.describe()

  # df_pes.info()

  yield_df = pd.merge(yield_df, df_pes, on=['Year','Area'])
  # yield_df.shape

  # yield_df.head()

  # Average Temprature

  avg_temp=  pd.read_csv('temp.csv')
  # avg_temp.head()

  # avg_temp.describe()

  avg_temp = avg_temp.rename(index=str, columns={"year": "Year", "country":'Area'})
  # avg_temp.head()

  yield_df = pd.merge(yield_df,avg_temp, on=['Area','Year'])
  # yield_df.head()

  # yield_df.shape


  # yield_df.describe()

  # yield_df.isnull().sum()

  # yield_df.info()

  # yield_df.nunique()

  yield_df.groupby(['Area'],sort=True)['hg/ha_yield'].sum().nlargest(10)

  
  # yield_df['Area'].value_counts()[:10].plot(kind='pie')
  # plt.show()

  # yield_df['Item'].value_counts()[:10].plot(kind='pie')
  # plt.show()

  
  # df_py=yield_df.groupby(['Year'])['pesticides_tonnes'].aggregate('sum').reset_index(name='pesticides_tonnes') 

  # ggplot(df_py, aes('Year','pesticides_tonnes'))+ geom_line(colour='purple')+theme_minimal()+labs(title = "Usage of Pesticides over the year",x = "Years", y = "Pesticides(ton)")

  # df2 = yield_df.groupby(['Item','Year'])['hg/ha_yield'].aggregate('sum').reset_index(name='hg/ha_yield') 
  # ggplot(df2,aes(x='Year', y='hg/ha_yield', colour='Item')) + geom_line()+theme_minimal()+ labs(title = "Yearwise crop-yield",x = "Year", y = "Yield(hg/ha)")

  features=yield_df.loc[:, yield_df.columns != 'hg/ha_yield']
  features = features.drop(['Year'], axis=1)
  label=yield_df['hg/ha_yield']
  
  
  ct1=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder='passthrough')
  features=np.array(ct1.fit_transform(features))


  le=LabelEncoder()
  features[:,10]=le.fit_transform(features[:,10])

  yield_df_onehot=pd.DataFrame(features)
  yield_df_onehot["hg/ha_yield"]=label
  # yield_df_onehot.head()
  
  # sns.heatmap(yield_df.corr())
  # plt.show()

  
  scaler=MinMaxScaler()
  features=scaler.fit_transform(features)
  
  train_data, test_data, train_labels, test_labels = train_test_split(features, label, test_size=0.3, random_state=42)
  
  test_df=pd.DataFrame(test_data,columns=yield_df_onehot.loc[:, yield_df_onehot.columns != 'hg/ha_yield'].columns) 
  
  # dt = DecisionTreeRegressor()
  # dt.fit(train_data,train_labels)
  # y_pred = dt.predict(test_data)
  # score = r2_score(test_labels,y_pred)

  rf = RandomForestRegressor()
  rf = rf.fit(train_data,train_labels)
  test_df["yield_predicted"]= rf.predict(test_data)
  test_df["yield_actual"]=pd.DataFrame(test_labels)["hg/ha_yield"].tolist()
  y_pred = rf.predict(test_data)
  score = r2_score(test_labels,y_pred)

  #Boxplot that shows yield for each item 
  # a4_dims = (16.7, 8.27)

  # fig, ax = plt.subplots(figsize=a4_dims)
  # sns.boxplot(x="Item",y="hg/ha_yield",palette="vlag",data=yield_df,ax=ax)

  # fig, ax = plt.subplots() 

  # ax.scatter(test_df["yield_actual"], test_df["yield_predicted"],edgecolors=(0, 0, 0))

  # ax.set_xlabel('Actual')
  # ax.set_ylabel('Predicted')
  # ax.set_title("Actual vs Predicted")
  # plt.show()

  Area=request.POST.get("state")
  Item=request.POST.get("crop")
  Year=2040
  average_rain_fall_mm_per_year=request.POST.get("rainfall")
  pesticides_tonnes=request.POST.get("pesticides")
  avg_temp=request.POST.get("season")

  inputs=np.array([[Area,Item,average_rain_fall_mm_per_year,pesticides_tonnes,avg_temp]])
  inputs=np.array(ct1.transform(inputs))
  inputs[:,10]=le.transform(inputs[:,10])
  inputs=scaler.transform(inputs)

  prediction=rf.predict(inputs)
  # print("Predicted yield is",prediction[0])
  
  # pickle.dump(rf,open("crop_model.pkl", "wb"))
  # pickle.dump(le,open("crop_le.sav", "wb"))
  # pickle.dump(scaler,open("crop_scaler.sav", "wb"))
  return render(request, "predict.html", context={
    "country" : Area,
    "crop" : Item,
    "rainfall" : average_rain_fall_mm_per_year,
    "pesticide" : pesticides_tonnes,
    "temperature" : avg_temp,
    "year" : Year,
    "result" : prediction[0]
  })









