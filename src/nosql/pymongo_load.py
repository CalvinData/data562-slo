"""
This module is a *one-time-use-only* utility that reads the records in the SLO
noSQL coding and tag datasets and loads them into MongoDB database collections.

See `pymongo_utilities.py` for details.
"""
import pymongo_utilities
import pandas as pd


def write_collection(database, collection_name, df):
    """ Write the given DataFrame to the given MongoDB collection. """
    collection = database[collection_name]
    # Delete all documents if they're there so that we can re-run this script.
    collection.delete_many({})
    collection.insert_many([dict(zip(df.columns, list(x))) for x in df.values])


client, database = pymongo_utilities.get_mongo_client_db("data562")


for dataset_name in ["slo_code", "slo_tags"]:
    df = pd.read_csv(f'../../data/nosql/{dataset_name}.csv')
    write_collection(database, dataset_name, df)

client.close()
