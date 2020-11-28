## Executive summary- Predict the price of homes at sale for the Ames Iowa Housing dataset
### Regression Challenge Project


The Ames Housing Data set contains information regarding the houses sold in the years 2006 to 2010 in Ames, Iowa. This data set includes property values with about 82 features of the property. There is a data dictionary containing 23 nomial (non-numerical such as: Neighborhood), 23 ordinal (Categorical with clear rates such as the overall condition of the house), 14 discrete (numerical such as year built) and 20 continous variables (numerical with any value such as floor square feet) and 2 additional observation identifiers such as ID. There are also three files provided which are train.csv file, test.csv file and sample_sub_reg.csv file. The main purpose of this project is to model the home prices by using the train data set which contains all of the training data for the model and the target variable is Sale Price. Then predicting the home prices by using the test data set. This data should be feed into the regression model to make predictions (The target variable, sale price has been removed from the test set!).


A final regression model should be used to make predictions on any new data. In this project, I used different techniques using regression (of course !) and some feature engineering. I first used the traning data set for fitting my model and created a model based on the features and prices in the training set and then used the test data set to test and predict the housing prices. Then I decided to make a better (in my opinion?!) model by keeping the model trained on the traning data set meaning the model will likely perform better when trained on all of the available data (larger number of data set)than just the subset (spliting train data set to train and test data sets) used to estimaate the performance of the model.

### Data Dictionary

| Feature | Data Type |Description
| --- | --- | --- |
| Order| int64 |Observation Number.
| PID | int64 |Parcel identification number - can be used with city web site for parcel review.
| MS SubClass | int64 |Identifies the type of dwelling involved in the sale.
| MS Zoning | object |Identifies the general zoning classification of the sale.
| Lot Frontage | float64 |Linear feet of street connected to property.
| Lot Area | int64 |Lot size in square feet.
| Street | object |Type of road access to property.
| Alley | object |Type of alley access to property.
| Lot Shape | object |General shape of property.
| Land Contour | object |Flatness of the property.
| Utilities | object |Type of utilities available.
| Lot Config | object |Lot configuration.
| Land Slope | object |Slope of property.
| Neighborhood | object |Physical locations within Ames city limits (map available).
| Condition 1 | object |Proximity to various conditions.
| Condition 2 | object |Proximity to various conditions (if more than one is present).
| Bldg Type | object |Type of dwelling.
| House Style | object |Style of dwelling.
| Overall Qual | int64 |Rates the overall material and finish of the house.
| Overall Cond | int64 |Rates the overall condition of the house.
| Year Built | int64 |Original construction date.
| Year Remod/Add | int64 |Remodel date (same as construction date if no remodeling or additions).
| Roof Style | object |Type of roof
| Roof Matl | object |Roof material.
| Exterior 1st | object |Exterior covering on house.
| Exterior 2nd | object |Exterior covering on house (if more than one material).
| Mas Vnr Type | object |Masonry veneer type.
| Mas Vnr Area | float64 |Masonry veneer area in square feet.
| Exter Qual | object |Evaluates the quality of the material on the exterior.
| Exter Cond | object |Evaluates the present condition of the material on the exterior.
| Foundation | object |Type of foundation.
| Bsmt Qual | object |Evaluates the height of the basement.
| Bsmt Cond | object |Evaluates the general condition of the basement.
| Bsmt Exposure | object |Refers to walkout or garden level walls.
| BsmtFin Type 1 | object |Rating of basement finished area.
| BsmtFin Type 2 | object |Rating of basement finished area (if multiple types).
| Total Bsmt SF | float64 |Total square feet of basement area.
| Heating | object |Type of heating.
| Heating QC | object |Heating quality and condition.
| Central Air | object |Central air conditioning.
| Electrical | object |Electrical system.
| Gr Liv Area | int64 |Above grade (ground) living area square feet.
| Bsmt Full Bath | float64 |Basement full bathrooms.
| Bsmt Half Bath | float64 |Basement half bathrooms.
| Full Bath | int64 |Full bathrooms above grade.
| Half Bath | int64 |Half baths above grade.
| Bedroom AbvGr | int64 |Number of bedrooms above basement level.
| Kitchen AbvGr | int64 |Number of kitchens.
| Kitchen Qual | object |Kitchen quality.
| TotRms AbvGrd | int64 |Total rooms above grade (does not include bathrooms).
| Functional | object |Home functionality rating.
| Fireplaces | int64 |Number of fireplaces.
| Fireplace Qu | object |Fireplace quality.
| Garage Type | object |Garage location.
| Garage Yr Blt | float64 |Year garage was built.
| Garage Finish | object |Interior finish of the garage Fin Finished.
| Garage Cars | float64 |Size of garage in car capacity.
| Garage Area | float64 |Size of garage in square feet.
| Garage Qual | object |Garage quality.
| Garage Cond | object |Garage condition.
| Paved Drive | object |Paved driveway.
| Wood Deck SF | int64 |Wood deck area in square feet.
| Open Porch SF | int64 |Open porch area in square feet.
| Enclosed Porch | int64 |Enclosed porch area in square feet.
| 3-Ssn Porch | int64 |Three season porch area in square feet.
| Screen Porch | int64 |Screen porch area in square feet.
| Pool Area | int64 |Pool area in square feet.
| Fence | object |Fence quality.
| Misc Feature | object |Miscellaneous feature not covered in other categories.
| Misc Val | int64 |Value of miscellaneous feature.
| Mo Sold | int64 |Month Sold.
| Yr Sold | int64 |Year Sold.
| Sale Type | object |Type of sale.
| SalePrice | int64 | the property's sale price in dollars.


### Processing and Modeling

Real world data can be messy and this project data set contains attributes that need severe modifications before they can be used in a predictive modeling. The predictions for house sale prices are more reliable when the target variable is normally distributed. Based on the graph presented in the slides for house price, the distribution is right skewed and by getting the log transformaation of them, the skewness will be removed. This can be easily done by Numpy, calling (log())  function on the target variable and the target column is ready for the next step. It is a a very good idea to also check the distribution of all the features (the ones that are going to be used in the model)  fitted in the model to see if they are normally distributed or skewed. By implementing that we will not lose the predictive power in the model. 

Once the train and test splits created, standardization is needed in order to scale all the features (variables). By doing StandardScaler, variables will have a standard z-score and mean of zero. Both train data set (except the houses price data) and test data set have been transformed with StandardScaler to match and then training data set has been fitted. Since we split the train set into train and test data sets, the performnace of the model can vary from one fit to another that is why Cross validation (cv) has been used in the final model which is a very important tool in model evaluation. CV=5 means there will be 5 fits and one for each fold of test section. Then cv will return the mean of the five- R2 scores which is more accurate. For regularization of the model ElasticNet() has been used which is a regularized regression method that linearly combines the L1 and L2 penalties of the lasso and ridge methods. Hyperparameters for ElasticNet regularization are alpha (alpha = a + b) which is regularization strength and L1 ratio (l1_ratio = a / (a + b)) which is the balance of Lasso and Ridge in the regularization. GridSearch CV with the estimator of ElasticCV has been used to fit the model. In the model by doing cv, the model will choose the best performing model with different range of hyperparameters that have been given to it. The next step is to use the score method on GridSearch cv to find the R2 score on the test data set. Don't forget to reverse the log transformation once you want to submit the house price predictions.


### Conclusion

The final model (**GridSearchCV with ElasticNet**) whith the best parameters of **alpha: 0.01** and **l1_ratio:  0.30** which has a little small penalty strength (alpha) and the ratio is so close to zero which means the model is close to Ridge regression. The R2 score on the training data is close to **94%** and **92%** on the unseen data (testing data set) which means the model explains 92% of variation in the house sale price in the unseen data. The scores are so close to each other, so there is no evidence of overfitting in the model. By using feature engineering, feature cleaning and regularizaton and target value transformation, the performance of the regression model has been greatly improved. 

