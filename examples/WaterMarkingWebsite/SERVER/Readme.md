# Installation Instructions

Install the following programs in sequence - 

## Mongo DB

To Visualize the data, you can use Compass (Optional)

    https://www.mongodb.com/try/download/compass

## Python Libraries

    pip install -r requirements.txt

## How to start a new project

In EvaluationUI/setup/setup.sh change the variable TASK_NAME and set it to the name you want.
Modify the files' directory with the files you want, and you are good to go.
You can change the files/tasks.json file and remove the tasks which are not necessary for you.
The instructions for creating the json files are mentioned below

To start the server, run the setup.sh file in the UI code.

(Optional) It is not necessary to run python manage.py migrate, 
but you can run it to suppress the warnings

# Instructions for creating the json files

## Path of the samples

The audio files should be put in the files directory, and 
the path of the samples in the JSON files should be api/files + relative path from the files directory. 

Path for an audio file which is inside files and has name hello.wav
- api/files/hello.wav

## File formats

### files/originals.json

    This will contain one real sample corresponding to each speaker.

Example - 

{
    "SPK1": "/spk1_original.wav",
    "SPK2": "/spk2_original.wav",
    "SPK3": "/spk3_original.wav"
}

### files/model/*.json

    This will contain predictions of experiments in the following format.

Example - 

{
    "modelName": "testingModel1",
    "meta": "Some information Here",
    "predicted": {
        "SOURCE_SPK1": {
            "TARGET_SPK1": "/spk1pred/1.wav",
            "TARGET_SPK2": "/spk1pred/2.wav",
            "TARGET_SPK3": "/spk1pred/3.wav"
        },
        "SOURCE_SPK1": {
            "TARGET_SPK1": "/spk1pred/1.wav",
            "TARGET_SPK2": "/spk1pred/2.wav",
            "TARGET_SPK3": "/spk1pred/3.wav"
        },
        "SOURCE_SPK1": {
            "TARGET_SPK1": "/spk1pred/1.wav",
            "TARGET_SPK2": "/spk1pred/2.wav",
            "TARGET_SPK3": "/spk1pred/3.wav"
        }
    }
}

### files/abx.json

    This will contain the model combinations which need to be compared in the abx testing
    modelName in files/model/*.json should match with the name mentioned in combinations
Example - 

{
    "combinations": [
        ["testingModel1", "testingModel2"],
        ["testingModel2", "testingModel3"]
    ]
}

### files/qualityComp.json

    This will contain the model combinations which need to be compared in the quality comparision testing
    modelName in files/model/*.json should match with the name mentioned in combinations
    
Example - 

{
    "combinations": [
        ["testingModel1", "testingModel2"],
        ["testingModel2", "testingModel3"]
    ]
}

## General Information

    A common seed number is generated for each user, which is used for pseudo random generation.

## MOS

### Files Used - 

    1) files/originals.json
    2) files/mos.json
    3) files/model/*.json

### Method - 

    I mixed up all the songs of files/model/*.json and the files/originals.json.
    The songs are shuffled randomly and the evaluator is asked to score the quality.

## Speaker Identity

### Files Used - 

    1) files/originals.json
    2) files/model/*.json
    
### Method -

    I mixed up all the songs of files/model/*.json and the files/originals.json.
    The songs are shuffled randomly and the evaluator is asked to score the speaker similarity with the song of the target speaker found in files/originals.json.

## ABX Testing

### Files Used - 

    1) files/originals.json
    2) files/model/*.json
    3) files/abx.json
    
### Method - 

    I created combinations of same source speaker and target speaker from different models along with a reference sample of the target speaker.
    The combination was shuffled randomly and the evaluator is asked to choose which model sample is closer to the reference sample in terms of speaker identity.
    
    
## Quality Comparison Testing

### Files Used - 

    1) files/originals.json
    2) files/model/*.json
    3) files/qualityComp.json
    
### Method - 

    I created combinations of same source speaker and target speaker from different models.
    This was shuffled randomly and the evaluator is asked to choose which model sample has better sound quality.
    
    

    