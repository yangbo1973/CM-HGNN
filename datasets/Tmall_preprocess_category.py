# -*- coding: utf-8 -*-
import argparse
import time
import csv
import pickle
import operator
import datetime
import os
import pandas as pd

with open('user_log_format1.csv',"r") as f:              #读出商品的item_ID对应的种类category_Id
    reader = csv.DictReader(f)
    item_category = {}
    for row in reader:
        item = row['item_id']
        if item not in item_category:
            item_category[item] = row['cat_id']
            
with open('Tmall_category.csv', 'w',newline='') as csvfile:     
    writer = csv.DictWriter(csvfile,fieldnames=['item_id','category_id'])
    writer.writeheader()
    for k,v in item_category.items():
        writer.writerow({'item_id':k,'category_id':v})
