import requests
import os
import yaml
import logging
import statistics
from pathlib import Path
import DeepfakeBench.preprocessing.preprocess as prep
import DeepfakeBench.preprocessing.rearrange as rearr
from DeepfakeBench.training.metrics.utils import get_test_metrics
import DeepfakeBench.training.fakedetector as dfdetector
from flask import Flask, redirect, url_for, render_template, request
from werkzeug.utils import secure_filename
from datetime import timedelta
from dotenv import load_dotenv

# Load environent variables
load_dotenv()

# Initialize flask app
app = Flask(__name__, static_folder='static')
app.config['SECRET_KEY'] = os.getenv("SECRET_KEY")
app.config['UPLOAD_FOLDER'] = os.getenv("UPLOAD_FOLDER")


@app.route("/", methods = ['GET', 'POST'])
def index():
    err = None
    if request.args.get('fail'):
        if request.args.get('fail') == '1':
            err = "Not a '.mp4' file"

    return render_template("index.html", err=err)

@app.route("/file_upload", methods = ['GET', 'POST'])
def file_upload():
    results = None
    stats={}
    if request.method == 'POST':
        try:
            file = request.files['file']
        except:
            err = "Not a '.mp4' file"
            return render_template('file_upload.html', results=err)
        if file.filename.split(".")[-1] == 'mp4':
            file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'], secure_filename(file.filename)))
            preprocess_step()
            rearrange_step()
            results = dfdetector.main()
            stats['Accuracy'] = statistics.mean(results)
            if stats['Accuracy'] > 0.70:
                results = "REAL"
            else:
                results = "FAKE"

            delete_temp()

            return render_template('file_upload.html', results=results, stats=stats)
        else:
            return redirect(url_for('index', fail=1))
    else:
        return redirect(url_for('index'))




    return render_template("file_upload.html", results=results)

# from config.yaml load parameters
yaml_path = './preprocess_config.yaml'

def preprocess_step():
    # open the yaml file
    try:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.parser.ParserError as e:
        print("YAML file parsing error:", e)

    # Get the parameters
    dataset_name = 'temp'
    dataset_root_path = config['preprocess']['dataset_root_path']['default']
    comp = config['preprocess']['comp']['default']
    mode = config['preprocess']['mode']['default']
    stride = config['preprocess']['stride']['default']
    num_frames = config['preprocess']['num_frames']['default']

    # use dataset_name and dataset_root_path to get dataset_path
    dataset_path = Path(os.path.join(dataset_root_path, dataset_name))

    # Create logger
    log_path = f'./DeepfakeBench/preprocessing/logs/{dataset_name}.log'
    logger = prep.create_logger(log_path)

    # Define dataset path based on the input arguments
    ## temp
    if dataset_name == 'temp':
        sub_dataset_names = ['video']
        sub_dataset_paths = [Path(os.path.join(dataset_path, name)) for name in sub_dataset_names]

    # Check if dataset path exists
    if not Path(dataset_path).exists():
        logger.error(f"Dataset path does not exist: {dataset_path}")
        return redirect(url_for("/", err="Error in preprocessing"))

    if len(sub_dataset_paths) != 0:
        # Check if sub_dataset path exists
        for sub_dataset_path in sub_dataset_paths:
            if not Path(sub_dataset_path).exists():
                logger.error(f"Sub Dataset path does not exist: {sub_dataset_path}")
                return redirect(url_for("/", err="Error in preprocessing"))
        # preprocess each sub_dataset
        for sub_dataset_path in sub_dataset_paths:
            prep.preprocess(sub_dataset_path, None, mode, num_frames, stride, logger)
    else:
        logger.error(f"Sub Dataset path does not exist: {sub_dataset_paths}")
        return redirect(url_for("/", err="Error in preprocessing"))
    logger.info("Face cropping complete!")
    return


def rearrange_step():
    # open the yaml file
    try:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.parser.ParserError as e:
        print("YAML file parsing error:", e)

    dataset_name = config['rearrange']['dataset_name']['default']
    dataset_root_path = config['rearrange']['dataset_root_path']['default']
    output_file_path = config['rearrange']['output_file_path']['default']
    comp = config['rearrange']['comp']['default']
    perturbation = config['rearrange']['perturbation']['default']
    # Call the generate_dataset_file function
    rearr.generate_dataset_file(dataset_name, dataset_root_path, output_file_path, comp, perturbation)
    return

def delete_temp():
    dir_path = Path('./DeepfakeBench/datasets/temp/video/')
    for f in dir_path.rglob('*'):
        try:
            f.unlink()
        except OSError as e:
            print("Error: %s : %s" % (f, e.strerror))
    for f in dir_path.rglob('*'):
        try:
            f.rmdir()
        except OSError as e:
            print("Error: %s : %s" % (f, e.strerror))

if __name__ == '__main__':
    app.run(host=os.getenv("HOST"), port=os.getenv("PORT"))
