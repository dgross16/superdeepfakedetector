import os
import yaml
import statistics
from pathlib import Path

import DeepfakeBench.preprocessing.preprocess as prep
import DeepfakeBench.preprocessing.rearrange as rearr

from fakedetector import DFDetector

from flask import Flask, redirect, url_for, render_template, request
from werkzeug.utils import secure_filename
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
    stats = {}
    if request.method == 'POST':
        try:
            file = request.files['file']
        except:
            err = "Not a '.mp4' file"
            return render_template('file_upload.html', results=err)

        if file.filename.split(".")[-1] == 'mp4':
            # Save the video file properly to the server.
            file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'], secure_filename(file.filename)))


            # Split video into frames to prepare for inference.
            preprocess_video()

            # Inference each model, averaging the accuracies of each of them.
            skip_please = ['iid.yaml', 'meso4.yaml', 'pcl_xception.yaml', 'altfreezing.yaml', 'sbi.yaml', 'lsda.yaml',
                           'xclip.yaml', 'videomae.yaml', 'i3d.yaml', 'sladd_detector.yaml', 'tall.yaml',
                           'sia.yaml', 'uia_vit.yaml', 'multi_attention.yaml', 'timesformer.yaml', 'lrl.yaml',
                           'sta.yaml', 'resnet34.yaml', 'f3net.yaml', 'ftcn.yaml', 'clip.yaml', 'spsl.yaml',
                           'facexray.yaml', 'ucf.yaml', 'efficientnetb4.yaml', 'recce.yaml']
            results = []

            # parse options and load config
            for detector_config in os.scandir('./DeepfakeBench/training/config/detector/'):
                if detector_config.name in skip_please:
                    print(f"skipping {detector_config.name}")
                    continue

                # Create DFDetector object
                detector = DFDetector(detector_config)

                # start testing
                best_metric = detector.test_epoch()
                results += [best_metric['temp']['acc']]

            print('===> Test Done!')

            stats['Accuracy'] = statistics.mean(results)
            if stats['Accuracy'] > 0.70:
                results = "REAL"
            else:
                results = "FAKE"

            delete_video()

            return render_template('file_upload.html', results=results, stats=stats)
        else:
            return redirect(url_for('index'))
    else:
        return redirect(url_for('index'))

    return render_template("file_upload.html", results=results)

def preprocess_video():
    # from config.yaml load parameters
    yaml_path = './preprocess_config.yaml'

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

    dataset_name = config['rearrange']['dataset_name']['default']
    dataset_root_path = config['rearrange']['dataset_root_path']['default']
    output_file_path = config['rearrange']['output_file_path']['default']
    comp = config['rearrange']['comp']['default']
    perturbation = config['rearrange']['perturbation']['default']
    # Call the generate_dataset_file function
    rearr.generate_dataset_file(dataset_name, dataset_root_path, output_file_path, comp, perturbation)
    return

def delete_video():
    dir_path = Path(os.getenv("UPLOAD_FOLDER"))
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
