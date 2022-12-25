from pymongo import MongoClient
from bson.objectid import ObjectId
import os

mongo_host = 'mongodb+srv://cluster0.bp0a9tr.mongodb.net/?retryWrites=true&w=majority'
mongo_authMechanism= 'SCRAM-SHA-256'

db_name = 'kaggle_connectx'
collection_name = 'history'

def initialize_db():
    client = MongoClient(mongo_host,
                            username=os.getenv("kaggleconnectx_username"),
                            password=os.getenv("kaggleconnectx_password"))
    db = client[db_name]
    collection = db[collection_name]

    return collection

def insert_payload(payload):
    collection = initialize_db()
    insert_id = collection.insert_one(payload).inserted_id
    return insert_id

def empty_collection():
    collection = initialize_db()
    collection.delete_many({})