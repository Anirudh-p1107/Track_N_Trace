# Track & Trace: Animal species classification based on footprints and sounds
- This is a GUI based desktop application that combines both the Pre-trained models of animal images and sounds into one tool for wildlife tracking.
- It is built by using Tkinter in Python. 
- The app classifies animal species from their footprint images and audio recordings and then displays the animal's image along with info/status as the output. 
- It features a simple user friendly interface that supports 2 different types of input and categorizes them into safe, unsafe and unknown animals.

## Wild-Paw and Wild-Echo
To checkout the model training process done in google colab. It trained on 10 animal classes for both image and audio 
- [Click here to visit Animal footprint model](https://github.com/Anirudh-p1107/Wild-Paw)
- [Click here to visit Animal sound model](https://github.com/Anirudh-p1107/Wild_Echo)

## How to use!
here's a step by step process.
- step 1: Download or fork this entire repository in your local device.
- step 2: Now download the best saved model that are `.keras` and `.pth` files from the google drive links
- step 3: Make sure your project structure should matches to the current repo, and move these files into your main directory
- step 4: Open cmd or terminal in the your directory and run the `MAIN.py`
- step 5: Upload images or audio files from your device to check the output.<br>

  (**Note:** Both the models are achieved less than 90% accuracy. So that the model may predicts the wrong output sometimes, because it has the small dataset which is not enough for model training)

## The saved models
No need to train the entire model from scratch. Here is the saved models with the better accuracy that are stored in google drive. <br>
  [These models are well trained and predicts the animals with 95% confidence]
- [Download the Footprint model](https://drive.google.com/file/d/1oVcUUvS6Z7fSpfek2QS-iCZvivQTMWjS/view?usp=drive_link)
- [Download the audio model](https://drive.google.com/file/d/1PjnpZWPOznmeC1sZ4ZZPDLW8YHHdPcfe/view?usp=drive_link)

## Documentation 
There is a complete information about this project in the PPT. You can refer the presentaion for better understanding. <br>
And also make sure to follow the instrructions from the `requirements.txt` file to run the application in your device. 
