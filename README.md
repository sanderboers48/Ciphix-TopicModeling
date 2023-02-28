# Ciphix-TopicModeling

This repository is made for the Ciphix Machine Learning Case. It consists out of two parts. The first is a Jupyter notebook (`CiphixCase.ipynb`) which show how the trained LDA model is made. The second part is a Flask web app (`main.py`), which hosts a webapp where you can type in your own sentence and the application will infer the sentence to one of the 10 learned topics.

## Installation and usage

There are two requirements files, `requirements.txt` is used for the Flask webapp and `requirements_trainedmodel.txt` is more extensive and also has the packages for the jupyter Notebook.

```bash
git clone https://github.com/sanderboers48/Ciphix-TopicModeling.git
cd Ciphix-TopicModeling

```

```bash
python -m venv venv
.\venv\Scripts\activate

pip install -r requirements.txt

python main.py

```
Or, if you want to run the notebook:
```bash
python -m venv venv
.\venv\Scripts\activate

pip install -r requirements_trainingmodel.txt

'Open the jupyter notebook "CiphixCase.ipynb"'

```

### Deployment
The webapp is also hosted on a free-tier of the Google Cloud services using App Engine. This site can be visited on:

http://ciphix-ml-case.ew.r.appspot.com


## Explantions
The Jupyter notebook has some markdown cells, which explain the various preprocessing and training steps. So check them out!


