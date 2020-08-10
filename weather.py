import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
production = pd.read_csv("time_series_15min_singleindex.csv",usecols=(lambda s: s.startswith('utc')|s.startswith('DE')),parse_dates=[0],index_col=0)
production.head(3)
production.tail(3)
production = production.loc[production.index.year == 2016,:]
production.head(3)
production.tail(3)
production.info()
#create plot
plt.plot(production.index,production['DE_wind_generation_actual'],c='black')
plt.title('Actual wind generation in Germany in MW')
plt.xlim(pd.Timestamp('2016-01-01'),pd.Timestamp('2017-01-01'))
plt.ylim(0,35000)
plt.show()
#create plot
plt.plot(production.index,production['DE_solar_generation_actual'], c='black')
plt.title('Actual solar generation in Germany in MW')
plt.xlim(pd.Timestamp('2016-01-01'),pd.Timestamp('2017-01-01'))
plt.ylim(0,27000)
plt.show()
production_wind_solar = production[[ 'DE_solar_generation_actual','DE_wind_generation_actual']]
weather = pd.read_csv("weather_data_GER_2016.csv", parse_dates=[0], index_col=0)
weather.head(3)
weather.tail(3)
weather.info()
weather.loc[weather.index == '2016-01-01 00:00:00', :]
#Averaging over all the 'chuncks':
2248704/256
weather_by_day = weather.groupby(weather.index).mean()
weather_by_day.head(24)
# create plot
plt.plot(weather_by_day.index, weather_by_day['v2'],c='black')
plt.title('Wind velocity 10 m above displacement height (m/s)')
plt.xlim(pd.Timestamp('2016-01-01'), pd.Timestamp('2017-01-01'))
plt.ylim(0, 12)
plt.show()
weather_by_day['T (C)'] = weather_by_day['T'] - 273.15
# create plot
plt.plot(weather_by_day.index, weather_by_day['T (C)'], c='black')
plt.title('Temperature (ºC)')
plt.xlim(pd.Timestamp('2016-01-01'), pd.Timestamp('2017-01-01'))
plt.ylim(-10, 30)
plt.show()
# create plot
plt.plot(weather_by_day.index, weather_by_day['rho'],c='black')
plt.title('Air density at the surface (kg/m³)')
plt.xlim(pd.Timestamp('2016-01-01'), pd.Timestamp('2017-01-01'))
plt.show()
# create plot
plt.plot(weather_by_day.index, weather_by_day['p'],c='black')
plt.title('Air pressure at the surface (Pa)')
plt.xlim(pd.Timestamp('2016-01-01'), pd.Timestamp('2017-01-01'))
plt.show()
# merge production_wind_solar and weather_by_day DataFrames
combined = pd.merge(production_wind_solar, weather_by_day, how='left', left_index=True, right_index=True)
# drop redundant 'T (C)' column
combined = combined.drop('T (C)', axis=1)
combined.head()
#for wind generation
sns.pairplot(combined, x_vars=['v1', 'v2', 'v_50m', 'z0'], y_vars=['DE_wind_generation_actual'],)
sns.pairplot(combined, x_vars=['SWTDN', 'SWGDN', 'T', 'rho', 'p'], y_vars=['DE_wind_generation_actual'],)
#for solar Generation
sns.pairplot(combined, x_vars=['v1', 'v2', 'v_50m', 'z0'], y_vars=['DE_solar_generation_actual'])
sns.pairplot(combined, x_vars=['SWTDN', 'SWGDN', 'T', 'rho', 'p'], y_vars=['DE_solar_generation_actual'])
#for wind 
sns.jointplot(x='v1', y='DE_wind_generation_actual', data=combined, kind='reg')
sns.jointplot(x='v2', y='DE_wind_generation_actual', data=combined, kind='reg')
sns.jointplot(x='v_50m', y='DE_wind_generation_actual', data=combined, kind='reg')
#for solar
sns.jointplot(x='SWTDN', y='DE_solar_generation_actual', data=combined, kind='reg')
sns.jointplot(x='SWGDN', y='DE_solar_generation_actual', data=combined, kind='reg')
 #predicting
#predicting the solar generation
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
# Importing the dataset
dataset = pd.read_csv('weather_data_GER_2016.csv')

X = dataset.iloc[:, 12:14].values
y = dataset.iloc[:, 14].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# Fitting Multiple Linear Regression to the Training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)
Z = regressor.predict(X_test)
print(regressor.intercept_)
print(regressor.coef_)
print('Mean Squared Error:',mean_squared_error(Z,y_test))
print('Root Mean Squared Error:',np.sqrt(mean_squared_error(Z,y_test)))
print('r_2 Statistic : %.2f' % r2_score(Z,y_test))
print(Z.round(2))
df = pd.DataFrame({'Actual': Z, 'Predicted': y_test})
print(df)
plt.scatter(Z,y_test,color='red')
#plt.plot(X_test,y_test,color='blue')
plt.show()
#predicting
#predicting the wind generation
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
# from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
# Importing the dataset
dataset = pd.read_csv('weather_data_GER_2016.csv')

X = dataset.iloc[:, 4:7].values
y = dataset.iloc[:, 9].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/5, random_state = 0)
# Fitting Multiple Linear Regression to the Training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)
Z = regressor.predict(X_test)
print(regressor.intercept_)
print(regressor.coef_)
print('Mean Squared Error:',mean_squared_error(Z,y_test))
print('Root Mean Squared Error:',np.sqrt(mean_squared_error(Z,y_test)))
print('r_2 Statistic : %.2f' % r2_score(Z,y_test))
#print(Z.round(2))
#print (X_train)
#print(y_train)
#Z = Z.round(2)
#m=np.array([X_test])
#n=np.array([y_test])
#print(m.shape)
#print(n.shape)
df = pd.DataFrame({'Actual': Z, 'Predicted': y_test})
print(df)
plt.scatter(Z,y_test,color='red')
#plt.plot(X_test,y_test,color='blue')
plt.show()