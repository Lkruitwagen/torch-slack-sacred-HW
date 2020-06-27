# torch-slack-sacred-HW
A template repository for deep learning experimentation combining [PyTorch](https://pytorch.org/), [Sacred](https://sacred.readthedocs.io/en/stable/index.html), and [Slack](https://slack.com/intl/en-gb/). This repo uses MNIST as an example deep learning problem.

## Installation

### Repo
Create a custom environment using Conda or venv. Then, clone this repository and install requirements:

    git clone https://github.com/Lkruitwagen/torch-slack-sacred-HW.git
    cd torch-slack-sacred-HW
    pip install -r requirements.txt
    
### Oberservers - GCP and FileStorage
We monitor and output experiment results to both local storage and to a Google Cloud Platform (GCP) storage bucket using Sacred Observers. The FileStorageObserver will save results to `/experiments/` without further configuration. To configure the GCP observer, please do the following:
1. create a GCP storage bucket and a folder within the bucket to store experiment results. Save these as `{"bucket": "<your-bucket-name>", "basedir": "<your/base/dir>"}` in a json: `/credentials/gcp_bucket.json` 
2. [Obtain a GCP Service Account Key](https://cloud.google.com/docs/authentication/getting-started/) as a json and save it as `credentials/gcp_credentials.json`

### Slackbot
We also implement a simple callback that communicates with a Slack app to push updates in messages to a channel or directly to a user. To use slack messages:
1. [create a Slack app](https://api.slack.com/apps) in your Workspace
2. Create a token for the bot, give it permissions for messaging users and channels, and install it in your workspace.
3. Copy the bot token and message target to `credentials/slack.json` as the following: `{"token": "<xoxb-your-bot-token>", "target": "<#yourchannel-or-Uyouruser>"}`

## Useage
Use `runner.py` as the entrypoint for experiments. 

    python runner.py
