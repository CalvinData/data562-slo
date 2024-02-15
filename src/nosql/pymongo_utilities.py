"""
Utilities for PyMongoDB operations - Reference:
- https://www.mongodb.com/docs/guides/

Notes:
- Build the collections using `pymongo_load.py` and
    read them using `pymongo_read.py`.
- Installed 'pymongo[srv]' as specified in the guide.
- TODO: Auto-set the user ID/password in the production environment.
    See: https://docs.github.com/en/codespaces/managing-your-codespaces/managing-your-account-specific-secrets-for-github-codespaces
- I had to add my IP address to the MongoDB cloud database's "Network Access"
    settings, which worked for multiple machines.
"""
import os
from pymongo import MongoClient


def get_mongo_client_db(database_name):
    """ Return the MongoDB client and database objects for the given
    MongoDB cloud database name.
    """

    # Read the username/password/url from the environment, set by
    # running a gitignored .env file (in the shell running this script).
    username = os.environ.get("MONGODB_USERNAME")
    password = os.environ.get("MONGODB_PASSWORD")
    url = os.environ.get("MONGODB_URL")
    connection_string = f"mongodb+srv://{username}:{password}@{url}"

    client = MongoClient(connection_string)
    database = client[database_name]

    return client, database
