# DisasterResponse

Description

This is Udacity's Data Science Nanodegree program's project.

You will find a data set containing real messages that were sent druing disaster events. 
the project contains:
1. ETL pipeline to process the data,
2. ML pipeline to model the data,
3. Web app where an emergency worker can input a new message and get calssification results in several categories.


Installation

Clone this GIT repository:

git clone https://github.com/idjuven/DisasterResponse

Running the Program

1. to run the ETL pipeline (root directory): `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

2. to run the ML pipeline (root directory):  `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3. Web APP (in the APP directory): python run.py`

Source Code

1. the source code for ETL pipline is in the "data" folder: "ETL Pipeline Preparation.ipynb".

2. the source code for ML pipeline is in the "model" folder: "ML Pipeline Preparation.ipynb".



