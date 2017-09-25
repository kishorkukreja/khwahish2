#!/usr/bin/env python
# Connection.py
# Importing the Required Libraries.
import time
import datetime
import pymongo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pymongo import MongoClient
from sklearn.cross_validation import train_test_split
# Connecting with Local MongoDB Database.
MONGO_HOST = "139.59.66.172"
MONGO_PORT = 27017
MONGO_DB = "sodelhi"
# MONGO_USER = "Username"
# MONGO_PASS = "password"
connection = MongoClient(MONGO_HOST, MONGO_PORT)
db = connection[MONGO_DB]
# Data_Retrival.py
# Retriving Data from users collection of the MongoDb Database.
final_data=[]
for obj in db.users.find({"feed._id":{'$exists':True},"feed.data.like.status":{'$exists':True},"interests":{'$exists':True},"feed.data.share.status":{'$exists':True},"feed.data.bookmark.status":{'$exists':True}},{"_id":1,"interests":1,"username":1,"feed.unique_id":1,"feed.data.read.duration":1,"feed.data.share.status":1,"feed.data.like.status":1,"feed.data.bookmark.status":1}):#.skip(3741 - 500):
    for feed_id in obj['feed']:
            final_obj = {}
            final_obj['_id']=obj['_id']
            final_obj['users_name'] = obj['username']
            final_obj['interests']=obj['interests']
######## test on original dataset 
#             if 'read' in feed_id['data'] and  len(feed_id['data']['read'])>0 and 'duration' in feed_id['data']['read'][0]:
#                 final_obj['timestamp']= feed_id['data']['read'][0]['duration']
            if 'bookmark' in feed_id['data'] and 'status' in feed_id['data']['bookmark']:
                final_obj['bookmark']= feed_id['data']['bookmark']['status']
            if 'share' in feed_id['data'] and  len(feed_id['data']['share'])>0 and 'status' in feed_id['data']['share'][0]:
                final_obj['share']= feed_id['data']['share'][0]['status']
            if 'like' in feed_id['data'] and 'status' in feed_id['data']['like']:
                final_obj['like']= feed_id['data']['like']['status']
            if 'unique_id' in feed_id:
                final_obj['article_id'] = feed_id['unique_id']
            final_data.append(final_obj)
# print final_data 
Dataset=pd.DataFrame(final_data)
Dataset.set_index('_id', inplace=True)
# Filtering out the a_users_name,unique_id from the Dataset.
Final_Dataset=Dataset.filter(['_id','users_name','article_id','interests'])
#Convert the millisecond to date and time.
# Final_Dataset[''] = pd.to_datetime(Final_Dataset[''], unit='ms')
#Getting the  date of the user
# Final_Dataset[''] = [d.date() for d in Final_Dataset['']]
# Filtering out the timestamp,bookmark_status,like_status,share_status from the Dataset.
rating=Dataset.filter(['bookmark','like','share'])#Add Timestamp
# Repalcing the NaN value with zero's.
rating=rating.fillna(0)
# Asigning Weight to the Timestamp,Like,Share,Bookmark.
# like=0.75
# share=3
# bookmarks=1.25
#Timestamp=?
Weight=(1.25,0.75,3)
# Finding the weighted Ratings.
weighted_ratings=rating * Weight
# weighted_ratings.
# weighted_ratings.head()
# Adding Ratings Field in the Final_Dataset.
Final_Dataset['rating']=weighted_ratings.sum(1)
# Final_Dataset.
#Getting Feed_id from the Unique_id
Final_Dataset["article_id"] = Final_Dataset["article_id"].map(lambda d: d.split("-",2)[2:][0])
# Final_Dataset['article_id'] = Final_Dataset['article_id'].str.extract('(......-\d\d\d\d)', expand=True)
Final_Dataset['interests']=Final_Dataset.interests.apply(' | '.join)
users = Final_Dataset['users_name'].unique()
train_data, test_data = train_test_split(Final_Dataset, test_size = 0.20, random_state=0)
Final_Dataset.to_csv('/home/shaktisocity/Desktop/khwahish2/data.csv',encoding='utf-8')
#test_data.to_csv('/home/shaktisocity/Desktop/khwahish2/test_rating_data.csv',encoding='utf-8')
#train_data.to_csv('/home/shaktisocity/Desktop/khwahish2/train_rating_data.csv',encoding='utf-8')

