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
