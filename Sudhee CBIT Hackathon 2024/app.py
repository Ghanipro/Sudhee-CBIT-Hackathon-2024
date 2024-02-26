from flask import Flask, render_template, request, redirect
from pytube import YouTube
import os
from pipeline import *
from classifiers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator

app = Flask(__name__)

# Load the classifier model
classifier = Meso4()
classifier.load('weights/Meso4_DF.h5')

# Function to download video from YouTube
def download_video(url):
    # Download the video
    yt = YouTube(url)
    # Specify the filename
    filename = "How To Spot a Deepfake_ Here are Two Simple Tricks #shorts #deepfake.mp4"
    yt.streams.get_highest_resolution().download(filename=filename)
    return filename

# Function to move the downloaded video to test_videos folder and rename it
def move_and_rename_video(video_path):
    # Move the video to test_videos folder
    new_path = os.path.join("test_videos", os.path.basename(video_path))
    os.rename(video_path, new_path)
    return new_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=["POST"])
def upload():
    file = request.files['browse']
    file_path = 'How To Spot a Deepfake_ Here are Two Simple Tricks #shorts #deepfake.mp4'
    file.save(f'test_videos/{file_path}')  
    return redirect('/')

@app.route('/delete', methods=["POST"])
def delete():
    file_path = 'test_videos/How To Spot a Deepfake_ Here are Two Simple Tricks #shorts #deepfake.mp4'
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"{file_path} has been successfully deleted.")
    else:
        print(f"{file_path} does not exist.")
    return redirect('/')


@app.route("/analyze", methods=["POST"])
def analyze():
    file_path = 'test_videos/How To Spot a Deepfake_ Here are Two Simple Tricks #shorts #deepfake.mp4'
    if os.path.exists(file_path):
        classifier = Meso4()
        classifier.load('weights/Meso4_DF.h5')

# 2 - Minimial image generator
# We did use it to read and compute the prediction by batchs on test videos
# but do as you please, the models were trained on 256x256 images in [0,1]^(n*n)

        dataGenerator = ImageDataGenerator(rescale=1./255)
        generator = dataGenerator.flow_from_directory(
                'test_images',
                target_size=(256, 256),
                batch_size=1,
                class_mode='binary',
                subset='training')

# 3 - Predict
        X, y = generator.next()
        print('Predicted :', classifier.predict(X), '\nReal class :', y)
        predictions = compute_accuracy(classifier, 'test_videos')
        for video_name in predictions:
            print('`{}` video class prediction :'.format(video_name), predictions[video_name][0])
        percentage = predictions[video_name][0] * 100 
        result = "is not deep fake"
        if percentage > 90:
            result = "highly deep fake"
        elif percentage > 80 and percentage <90:
            result = "likely to be deep fake"
        elif percentage >50 and percentage <80:
            result = "not likely to be deep fake"
        else:
            result = "is not deep fake"
        text_sent = "the given video is : "+str(percentage)+"%   likely to be a deep fake."
        return render_template('prediction.html',per = percentage,result=result)

    else:
        url = request.form["url"]
        try:
            # Download the video from YouTube
            downloaded_video_path = download_video(url)
            # Move and rename the video
            moved_video_path = move_and_rename_video(downloaded_video_path)
            # Analyze the video for deep fake
            classifier = Meso4()
            classifier.load('weights/Meso4_DF.h5')

        # 2 - Minimial image generator
        # We did use it to read and compute the prediction by batchs on test videos
        # but do as you please, the models were trained on 256x256 images in [0,1]^(n*n)

            dataGenerator = ImageDataGenerator(rescale=1./255)
            generator = dataGenerator.flow_from_directory(
                    'test_images',
                    target_size=(256, 256),
                    batch_size=1,
                    class_mode='binary',
                    subset='training')

        # 3 - Predict
            X, y = generator.next()
            print('Predicted :', classifier.predict(X), '\nReal class :', y)
            predictions = compute_accuracy(classifier, 'test_videos')
            for video_name in predictions:
                print('`{}` video class prediction :'.format(video_name), predictions[video_name][0])
            percentage = predictions[video_name][0] * 100 
            result = "is not deep fake"
            if percentage > 90:
                result = "highly deep fake"
            elif percentage > 80 and percentage <90:
                result = "likely to be deep fake"
            elif percentage >50 and percentage <80:
                result = "not likely to be deep fake"
            else:
                result = "is not deep fake"
            text_sent = "the given video is : "+str(percentage)+"%   likely to be a deep fake."
            return render_template('prediction.html',per = percentage,result=result)

        except Exception as e:
            print(f"Error occurred: {str(e)}")
            return str(e), 500

if __name__ == "__main__":
    app.run(debug=True)
