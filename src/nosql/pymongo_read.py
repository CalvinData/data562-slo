"""
This module demonstrates some basic MongoDB queries on the SLO
noSQL coding datasets.

See `pymongo_utilities.py` for details.
"""
import pymongo_utilities
from pandas import DataFrame

client, database = pymongo_utilities.get_mongo_client_db("data562")
code_collection = database["slo_code"]
tags_collection = database["slo_tags"]

# SELECT *
# FROM slo_code
# Notes:
# - This gets all the records in the SLO tweets to be coded collection.
# - Converting to a dataframe makes the results easier to manage,
#       see: https://www.mongodb.com/languages/python
# records = code_collection.find()
# records_df = DataFrame(records)
# print(records_df)

# SELECT id, tweet_url
# FROM project
# Notes:
# - This gets a list of tweet IDs and direct URLs; the URLs allow us to see
#       the tweets in context, which is useful in manual coding.
# - There are two "IDs" in the results:
#   - the original dataset `id`
#   - the MongoDB surrogate `_id`
# - Selecting on:
#   - `"id": 1` returns the title "id".
#   - `"_id": 0` gets rid of the default inclusion of the surrogate `_id`.
# records = code_collection.aggregate([
#     { "$project": { "id": 1, "tweet_url": 1, "_id": 0 } }
# ])
# records_df = DataFrame(records)
# print(records_df)

# SELECT id, company, tweet_norm
# FROM slo_code
# WHERE company = 'adani'
# Notes:
# - This gets all the tweets for Adani.
# - The two arguments to find() set up an *implicit* aggregation pipeline:
#       1. select (i.e., get records for company Adani)
#       2. project (i.e., get columns: id, company, tweet_norm)
# records = code_collection.aggregate([
#     { "$match": { "company": { "$eq": "adani" } } },
#     { "$project": { "id": 1, "company": 1, "tweet_norm": 1, "_id": 0} }
# ])
# records_df = DataFrame(records)
# print(records_df)

# SELECT company, count(id)
# FROM slo_code
# GROUP BY company
# ORDER BY count(id) DESC
# Notes:
# - Get a list of companies with the number of tweets for each,
#       sorted by count descending.
# - The three arguments to find() set up an *explicit* aggregation pipeline:
#       1. project (company, count is included by default)
#       2. group (by company, and compute aggregate sum)
#       3. sort (by the aggregate sum, in descending order)
# records = code_collection.aggregate([
#     { "$project": { "company": 1, "_id": 0} },
#     { "$group": { "_id": "$company", "count": { "$sum": 1 } } },
#     { "$sort": { "count": -1 } }
# ])
# records_df = DataFrame(records)
# print(records_df)

# SELECT id, company, stance
# FROM slo_code, slo_tags
# WHERE slo_code.id = slo_tags.id
# Notes:
# - Get a list of coded tweets, with ID, tweet text, and stance code.
# - This feels pretty manual, cf.
#       [DBTG](https://en.wikipedia.org/wiki/Data_Base_Task_Group), circa 1967
records = code_collection.aggregate([
    { "$lookup": {
        "from": "slo_tags",
        "localField": "id",
        "foreignField": "id",
        "as": "tag_document"
    } },
    { "$project": {
        "id": 1,
        "company": 1,
        "tag_document.stance": 1,
        "_id": 0
    } }
])
records_df = DataFrame(records)
print(records_df)

client.close()
