# Predict Future Sales
## Project Background
### Challenge Description
This competition consist of challenging time-series dataset consisting of daily sales data, provided by one of the largest Russian software firms - 1C Company. 
Challenge is to predict total sales for every product a store will sell in the next month.
### Data Description
Daily historical sales data of 1C Company is provided as training data. The task is to forecast the total amount of products sold per month in every shop for the test set. List of shops and products slightly changes every month. 
Data fields
ID - an Id that represents a (Shop, Item) tuple within the test set
shop_id - unique identifier of a shop
item_id - unique identifier of a product
item_category_id - unique identifier of item category
item_cnt_day - number of products sold. You are predicting a monthly amount of this measure
item_price - current price of an item
date - date in format dd/mm/yyyy
date_block_num - a consecutive month number, used for convenience. January 2013 is 0, February 2013 is 1,..., October 2015 is 33
item_name - name of item
shop_name - name of shop
item_category_name - name of item category
### File descriptions
sales_train.csv - the training set. Daily historical data from January 2013 to October 2015.
test.csv - the test set. need to forecast the sales for these shops and products for November 2015.
sample_submission.csv - a sample submission file in the correct format.
items.csv - supplemental information about the items/products.
item_categories.csv  - supplemental information about the items categories.
shops.csv- supplemental information about the shops.
As we analysed the dataset we found out the following facts
Item_cnt_data in sales_train.csv file contains both positive and negative values and it does not have any zero values. This lead us to conclusion that this file contains only sales and returns per day for a particular item at a particular shop.
Also we found out that shops, items and item categories are not consistent throughout the time period. They does not appear for some months.

## Data Analysis
1_DataAnalysis
2.2_AnalysingDailySales
3.2_AnalysingMonthlySales
