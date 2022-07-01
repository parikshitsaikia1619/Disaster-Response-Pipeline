# Disaster Response Pipeline Project
https://github.com/parikshitsaikia1619/Disaster-Response-Pipeline


### Project's Goal
To analyze disaster data from Appen (formally Figure 8) to build a model for an API that classifies disaster messages.
<br>
### Project Components
There are three components you'll need to complete for this project.

1. ETL Pipeline
In a Python script, `process_data.py`, write a data cleaning pipeline that:

* Loads the `messages` and `categories` datasets
* Merges the two datasets
* Cleans the data
* Stores it in a SQLite database
<br>

2. ML Pipeline
In a Python script, `train_classifier.py`, write a machine learning pipeline that:

* Loads data from the SQLite database
* Splits the dataset into training and test sets
* Builds a text processing and machine learning pipeline
* Trains and tunes a model using GridSearchCV
* Outputs results on the test set
* Exports the final model as a pickle file


3. Flask Web App

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click on the URL
