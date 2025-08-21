![image](https://github.com/stellartran/retail-sales-analysis/blob/main/retail_sales_cover.png)
# Unlocking Revenue: Which Factors Impact Retail Sales?

## Part I: Introduction

### I.1. Project purpose
This project aims to **analysis available dataset, develop a robust predictive model for retail sales** and **provide actionable insights for optimizing sales revenue**. By leveraging historical sales, marketing, and discount data, I seek to understand the key drivers of sales performance.

### I.2. Expected key outcomes
* Sales evaluation: Analyze how features like marketing, discounts, holidays affect, etc. impact sales
* Sales prediction: Build model to predict retail sales
* Actionable insights: Suggest reasonable actions to drive growth

### I.3. Dataset description
This dataset provides detailed insights into retail sales, featuring a range of factors that may influence sales performance: units sold, discount, marketing, and holiday effect.
* Year of dataset: 2022 and 2023
* Data volume: 30,000 rows
* Numbers of column: 11
* Data structure
01. Store_ID: Identifier for the retail store. (Categorical)
02. Product_ID: Identifier for the product. (Numerical)
03. Date: The date when the sale occurred. (Temporal - Key column)
04. Units_Sold: Quantity of items sold. (Numerical - Secondary Target Variable)
05. Sales_Revenue_USD: Total revenue generated from sales. (Numerical - Primary
Target Variable)
06. Discount_Percentage: The percentage discount applied to products. (Numerical)
07. Marketing_Spend_USD: Budget allocated to marketing efforts. (Numerical)
08. Store_Location: Geographic location of the store. (Categorical) => Continent
09. Product_Category: The category to which the product belongs. (Categorical)
10. Day_Week: Day when the sale took place. (Categorical) => Weekday/Weekend
11. Holiday_Effect: Indicator of whether the sale happened during a holiday period.
(Categorical/Binary)

### I.4. Tools
* Google Colab: Handle data, test prediction models
* PowerBI: EDA, visualize data

## Part II: Data cleaning

### II.1. Import libraries & data
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

df = pd.read_csv('retail_sales_original.csv')
df.head(10)
```
| Store_ID   |   Product_ID | Date      |   Units_Sold |   Sales_Revenue_USD |   Discount_Percentage |   Marketing_Spend_USD | Store_Location            | Product_Category   | Day_Week   | Holiday_Effect   |
|:-----------|-------------:|:----------|-------------:|--------------------:|----------------------:|----------------------:|:--------------------------|:-------------------|:-----------|:-----------------|
| Spearsland |     52372247 | 1/1/2022  |            9 |             2741.69 |                    20 |                    81 | Tanzania                  | Furniture          | Saturday   | False            |
| Spearsland |     52372247 | 1/2/2022  |            7 |             2665.53 |                     0 |                     0 | Mauritania                | Furniture          | Sunday     | False            |
| Spearsland |     52372247 | 1/3/2022  |            1 |              380.79 |                     0 |                     0 | Saint Pierre and Miquelon | Furniture          | Monday     | False            |
| Spearsland |     52372247 | 1/4/2022  |            4 |             1523.16 |                     0 |                     0 | Australia                 | Furniture          | Tuesday    | False            |
| Spearsland |     52372247 | 1/5/2022  |            2 |              761.58 |                     0 |                     0 | Swaziland                 | Furniture          | Wednesday  | False            |
| Spearsland |     52372247 | 1/6/2022  |            8 |             3046.32 |                     0 |                    41 | Bhutan                    | Furniture          | Thursday   | False            |
| Spearsland |     52372247 | 1/7/2022  |            6 |             2284.74 |                     0 |                     0 | Suriname                  | Furniture          | Friday     | False            |
| Spearsland |     52372247 | 1/8/2022  |            9 |             3427.11 |                     0 |                    83 | Taiwan                    | Furniture          | Saturday   | False            |
| Spearsland |     52372247 | 1/9/2022  |            7 |             2665.53 |                     0 |                     0 | Papua New Guinea          | Furniture          | Sunday     | False            |
| Spearsland |     52372247 | 1/10/2022 |            1 |              380.79 |                     0 |                   164 | Canada                    | Furniture          | Monday     | False            |


```python
df.info()
```
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 30000 entries, 0 to 29999
Data columns (total 11 columns):
 #   Column               Non-Null Count  Dtype  
---  ------               --------------  -----  
 0   Store_ID             30000 non-null  object 
 1   Product_ID           30000 non-null  int64  
 2   Date                 30000 non-null  object 
 3   Units_Sold           30000 non-null  int64  
 4   Sales_Revenue_USD    30000 non-null  float64
 5   Discount_Percentage  30000 non-null  int64  
 6   Marketing_Spend_USD  30000 non-null  int64  
 7   Store_Location       30000 non-null  object 
 8   Product_Category     30000 non-null  object 
 9   Day_Week             30000 non-null  object 
 10  Holiday_Effect       30000 non-null  bool   
dtypes: bool(1), float64(1), int64(4), object(5)
memory usage: 2.3+ MB
```

```python
print("* UNIQUE VALUE COUNT")
print("Store ID: ", df['Store_ID'].nunique())
print("Date: ", df['Date'].nunique())
print("Discount Percentage: ", df['Discount_Percentage'].nunique())
print("Product ID: ", df['Product_ID'].nunique())
print("Store Location: ", df['Store_Location'].nunique())
print("Product Category: ", df['Product_Category'].nunique())
print("Day of Week: ", df['Day_Week'].nunique())

print("* UNIQUE VALUE")
print("Discount Percentage: ", df['Discount_Percentage'].unique())
print("Product Category: ", df['Product_Category'].unique())
```
```
* UNIQUE VALUE COUNT
Store ID:  1
Date:  731
Discount Percentage:  5
Product ID:  42
Store Location:  243
Product Category:  4
Day of Week:  7
* UNIQUE VALUE
Discount Percentage:  [20  0 15 10  5]
Product Category:  ['Furniture' 'Electronics' 'Groceries' 'Clothing']
```

### II.2. Clean and re-structure dataset
***Initial data exploration***
* The dataset consists of 30,000 rows and 11 columns without missing values.
* Data type: Multiple types such as integer, float, object, boolean.
* Key column: Date
* There is only 1 value in "Store_ID" column (Spearsland), 42 unique products divided into 4 categories, 4 discount levels, and 243 unique store locations.


***Remove redundant column(s)***

There is only 1 value in "Store_ID" column (Spearsland). Thus, we can conclude that the data is from Sprersland store only and will also exclude this column to avoid affect the analysis.
```python
df = df.drop(['Store_ID'], axis=1)
```

***Add "Continent" column to group "Store_Location" values***

The "Store_Location" column has 243 unique values, which is too granular for meaningful statistical analysis. To simplify the data and conduct a more effective, high-level analysis of consumer behavior, I will add a new "Continent" column. This allows us to analyze sales trends and behaviors across broader geographical regions instead of 243 individual locations.
```python
#Create file with 243 unique Store_Location values
unique_store_locations = df['Store_Location'].unique()
unique_store_locations_df = pd.DataFrame(unique_store_locations, columns=['Unique Store Locations'])
unique_store_locations_df.to_csv('unique_store_locations.csv', index=False)

#Add Continent column
continent_df = pd.read_csv('continent.csv')

df['Continent'] = df['Store_Location'].map(
    continent_df.set_index('Location')['Continent']
).fillna('Unknown')
```

***Define weekend***

Although the "Day_Week" column has only 7 values, sales revenue is typically higher on weekends. To simplify the data and focus on key trends, I will group days into two categories: "Weekday" and "Weekend".
```python
day_type_map = {
    'Monday': 'Weekday', 'Tuesday': 'Weekday',
    'Wednesday': 'Weekday', 'Thursday': 'Weekday',
    'Friday': 'Weekday',
    'Saturday': 'Weekend', 'Sunday': 'Weekend'
}
df['Is_Weekend'] = df['Day_Week'].map(day_type_map)
```

***Correct data types***
* Date: Covert to datetime type
* Product_ID: Convert to object type
* All object type: Covert to category type
```python
df['Date'] = pd.to_datetime(df['Date'])
df['Product_ID'] = df['Product_ID'].astype('object')
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].astype('category')
df.info()
```
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 30000 entries, 0 to 29999
Data columns (total 12 columns):
 #   Column               Non-Null Count  Dtype         
---  ------               --------------  -----         
 0   Product_ID           30000 non-null  category      
 1   Date                 30000 non-null  datetime64[ns]
 2   Units_Sold           30000 non-null  int64         
 3   Sales_Revenue_USD    30000 non-null  float64       
 4   Discount_Percentage  30000 non-null  int64         
 5   Marketing_Spend_USD  30000 non-null  int64         
 6   Store_Location       30000 non-null  category      
 7   Product_Category     30000 non-null  category      
 8   Day_Week             30000 non-null  category      
 9   Holiday_Effect       30000 non-null  bool          
 10  Continent            30000 non-null  category      
 11  Is_Weekend           30000 non-null  category      
dtypes: bool(1), category(6), datetime64[ns](1), float64(1), int64(3)
memory usage: 1.4 MB
```

## Part III. EDA

This part presents an satistical overview. For more in-depth analysis, please refer to the detailed dashboard in PowerBI.
```python
# Download data to import to Power BI
df.to_csv('retail_sales_cleaned.csv', index=False)
```
View at [Full PowerBI Dashboard](https://drive.google.com/file/d/1wBAj0ui3IjlV0kb1wDdPYONLrEFNp-lS/view?usp=sharing) and [Presentation](https://drive.google.com/file/d/150nF6DFAmfez-Gq4jF12Vd3FfQYsP5mM/view?usp=sharing).

***Preliminary conclusion 1***
* Total Sales Revenue: $82.40M.
* $1.5M spent on marketing, which is relatively small compared to total sales revenue.
* The average discount percentage is only 2.97%, suggesting discounts are used sparingly.
* The second half of year had better sales performance. There was a noticeable dip in February both years. Units sold and revenue tend to move together, showing a healthy correlation.
* Product Categories: Furniture led with 13 unique products, following by Electronics and Clothing. Furniture and Electronics sold the most units, while Clothing had fewer units but a higher average price. Groceries sold in large volumes but at lower prices, which explains their smaller revenue share.
* Holiday: 99.45% of the cases, which is expected, as holidays are not daily occurrences. The 4 holidays counted were Thanksgiving and Christmas (2 years).
* Higher sales performance on weekends and holidays.
* Potential positive impact from both discount and marketing spend on sales revenue.
* The seasonal peak in late fall & winter and high revenue within Africa and America stores suggest some key insights for sales growth.

## Part IV. Sales prediction

### IV.1. Process
* Remove redundant columns: As presented above, the "Continent" and "Is_Weekend" columns were created to simplify the granular data in the "Store_Location" and "Day_Week" columns. When building a predictive model, it's best to remove the original columns to avoid data redundancy and potential multicollinearity.
* Encode data: Using one-hot coding to create dummy variances for category data type columns: Product_Category, Continent, Is_Weekend,Holiday_Effect.
* Draw heatmap to illustrate correlation among variances.
* Check multicollinearity by VIF.
* Run 7 prediction models including: OLS Linear Regression, Ridge, Lasso, Elastic Net, SGDRegressor, Random Forest Regressor, Gradient Boosting Regressor.
* Remove Units Sold variance and run prediction models again since we do not have this variance in real-world scenerio.
* Conclude the findings.

### IV.2. Preliminary conclusion 2
* Prediction models: Gradient Boosting provides the best predicted sales. However, it has not much difference from the other linear models and the predicted sales is not highly reliable.
* Key predictors: Units_Sold, Discount Percentage, and Categories.
* The model only accounts for a small portion of the variance in sales revenue. However, the conclusion that Units Sold is a primary driver of revenue is robust and can be trusted.

## Part V. Suggestions

### V.1. Amending the Dataset
* Invest in Data Collection Infrastructure
* Add New Data Points to Existing Records

### V.2. Boosting Sales Revenue
* Optimize Discounts Strategically
* Marketing on Top Categories and Periods
* Improve Cross-Selling Opportunities

## Part VI. The ending
Sincere gratitude is extended to Mr. Cường from practice class, Ms. Thảo from lecture class, all my valued classmates, and Swiss Coding Academy for your enthusiasm and support.
Should you need any further discussion, please feel free to contact me at:
* thuydungtran.dtt@freepik.com
* +84 903 949 561
