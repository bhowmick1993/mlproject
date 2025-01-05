# End to End Machine Learing Project

This project is a simple ML where we need to predict math score of the students depending on the some other paramteres.
The dataset **data** folder.

### Video
Here you can find a small demo video of the Flask web application. It is very simple. The main purpose is to demonstrate how the notebook experiments gets converte into a finall application.

![Demo_Flask_Video](demo_video/prediction_demo.mkv)

### Image
![Demo_Flask_Image](demo_image/flask_web_app.PNG)

## There are a few interesting features that have been displayed in this project:

- How to orgainze an end-to-end ML project.
- how to create a simple Web UI using Flask.
- how to deploy this app in Amazon ElasticBeanStalk (EBS).

## How to Run?

- Clone this github Repo
   ```sh 
   git clone https://github.com/bhowmick1993/mlproject.git`

- First launch a VSCode instance.
- Open the project folder
- Create a new conda environment :
    ```sh
    conda create -p venv python==3.8 -y
- Activate the environment:
    ```sh
    conda activate venv

    *or you might have to provide the full venv path which is inside the project folder*
- Install the requirements
    ```sh
    pip install -r requirements.py

- Now finally run the flask application
    ```sh
    python application.py
    
- Once previous step is running, go to the browser and in the address bar type:

    ```sh
    http://127.0.0.1:5000/predict_data
