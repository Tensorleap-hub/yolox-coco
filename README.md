# YOLO-NAS Object Detection
The repository below is simulating integration to your local directory.
To understand the integration work that is required for that, please look at `leap_binder.py`

# Quick start
## Class Mapping
YOLO requires all labels to be consecutive, i.e. for 10 classes the model expects tha labels 0-9.
To generate a mapping between your custom classes and consecutive ids, please run the script `generate_labels_mappping.sh`: <br>
```bash
./generate_labels_mapping.sh
```
This script will generate a mapping file between the model logits (consecutive ids in range [0, num classes]) to classes names in the following format: <br>
```yaml
{
  "0": "person",
  "1": "bicycle",
  "2": "car",
  "3": "motorcycle",
  "4": "airplane",
  etc.
}
```
The file will be saved to `yolonas/label_id_to_name.json`
If there is any mismatch between model logits and classes names, e.g. 'car' should be logit '1' rather than '2', <br>
please modify the file `yolonas/label_id_to_name.json` accordingly. <br>
## Dataset Path and Model Import
Set the `dataset_path` value in `yolonas/object_detection_config.yaml` to your local dataset path.

Simply run:
```bash
leap project push model/yolo_nas_s_c_1.onnx
```
* Please replace the model path with your model

Now, we can open http://localhost:4589 and evaluate our model.

We also made another version that adds permute layers to the end of the model:

```bash
./upload_permuted_model.sh model/yolo_nas_s_c_1.onnx
```

* This option is preferable if the model layer name is different since the `leap_mapping.yaml` file won't find the layers it expects.
* Please replace the model path with your model


## Prerequisites

### Pyenv
We recommend on managing your python versions via `pyenv`. <br>
To install pyenv on your specific OS, please refer to this link: **[pyenv-installation](https://github.com/pyenv/pyenv#installation)** <br>
After installing and setting up the shell environment you can install any python version, for this project it is recommended to install python `3.9.16`
```
pyenv install 3.9.16
```
After installation run the following from a shell within the project root `yolo-nas-coco`
```
pyenv local 3.9.16
```
this will set your local python to the specified version
### Poetry 
We recommend using poetry as your python dependency manager, and we supplied an environment defined by the `poetry.lock` file. <br>
To install poetry on your specific OS please refer to this link: **[poetry-installation](https://python-poetry.org/docs/#installing-with-the-official-installer)** <br>
when poetry is installed, run the following commands from within the project root folder `yolo-nas-coco` to set the environment python version and create the environment:
```
poetry env use 3.9.16
poetry install
```
## Configure Dataset Path
The integration script expect a pth to a coco format dataset directory. <br>
Set the `dataset_path` value in `yolonas/object_detection_config.yaml` to your local dataset path.<br>
The dataset directory should include the following files and folders:
```
- train.json
- val.json
- test.json
- images/
```
## Model Import
To import the model into `Tensorleap`, first place it under the `model` directory.
Then run the following commands from a shell within the project root directory `yolo-nas-coco`:
```
sudo chmod +x upload_permuted_model.sh
```
make sure your poetry env is activated by running:
```commandline
poetry shell
```
and then run the script
```
./upload_permuted_model.sh model/<your-model-name>
```
the `upload_permuted_model.sh` script will import your raw model with permuted 
outputs as well as the corresponding mapping so that you are ready to start evaluating 


# Getting Started with Tensorleap Project

This quick start guide will walk you through the steps to get started with this example repository project.

## Prerequisites

Before you begin, ensure that you have the following prerequisites installed:

- **[Python](https://www.python.org/)** (version 3.7 or higher).
- **[Poetry](https://python-poetry.org/)**.
- **[Tensorleap](https://tensorleap.ai/)** platform access. To request a free trial click [here](https://meetings.hubspot.com/esmus/free-trial).
- **[Tensorleap CLI](https://github.com/tensorleap/leap-cli)**.


## Tensorleap **CLI Installation**

with `curl`:

```
curl -s https://raw.githubusercontent.com/tensorleap/leap-cli/master/install.sh | bash
```

## Tensorleap CLI Usage

### Tensorleap **Login**

To login to Tensorleap:

```
leap auth login [api key] [api url].
```

Go to resource management and copy the command from the `generate cli token` button at the bottom of the page.

<br>

**How To Generate CLI Token from the UI**

1. Login to the platform in 'CLIENT_NAME.tensorleap.ai'
2. Scroll down to the bottom of the **Resources Management** page, then click `GENERATE CLI TOKEN` in the bottom-left corner.
3. Once a CLI token is generated, just copy the whole text and paste it into your shell.


## Tensorleap **Project Deployment**

To deploy your local changes:

```
leap project push
```

### **Tensorleap files**

Tensorleap files in the repository include `leap_binder.py` and `leap.yaml`. The files consist of the  required configurations to make the code integrate with the Tensorleap engine:

**leap.yaml**

leap.yaml file is configured to a dataset in your Tensorleap environment and is synced to the dataset saved in the environment.

For any additional file being used, we add its path under `include` parameter:

```
include:
    - leap_binder.py
    - cityscapes_od/configs.py
    - [...]
```

**leap_binder.py file**

`leap_binder.py` configures all binding functions used to bind to Tensorleap engine. These are the functions used to evaluate and train the model, visualize the variables, and enrich the analysis with external metadata variables

## Testing

To test the system we can run `leap_test.py` file using poetry:
