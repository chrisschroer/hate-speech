# NOHATE related Snorkel NLP project

Course project related to the NOHATE project to overcome the issue of expensive and rare gold labels

## General Information

### NOHATE Project
- Research project funded by the Federal Ministry of Education and Research of Germany to overcome crises in public communication
- Focusses on hate speeach about refugees, migration and foreigners on social media platforms or in online forums
- 3 project partners: Freie Universität Berlin, Beuth University of Applied Sciences Berlin and VICO Research & Consulting
- Main goal: Analyse and understand hatespeech in order to subsequently develop software for recognition of hateful comments
- Further information: https://www.polsoz.fu-berlin.de/en/kommwiss/v/bmbf-nohate/index.html

### Snorkel

- A system for quickly generating training data with weak supervision https://snorkel.org
- Related Github page: https://github.com/snorkel-team/snorkel
- Handcrafted rules are leveraged to train a generative model which is subsequently applied to create silver labels for an unlabeled dataset of the same domain.


## Course Project

### Outlines

- Address the problem of too few gold labels with Snorkel to replace the exhaustive gold labeling task with an automated process using handcrafted rules
- ~12k comments in German, different sources, unlabeled
- 500 gold labels (at all, not per label)
- Consider different classes/types of hatespeech (e.g. Intelligenz, Beschimpfung, Entmenschlichung)


### Goals

- Silver label the 12k comments with the use of Snorkel
- Assess the effect of adding silver labeled data to a gold label dataset on the model performance
- Set up an easily extendable infrastructure to facilitate incorporation of new Snorkel candidates or label functions

### Use of Snorkel

The following examples give a quick overview how we leveraged the Snorkel functionality for our course project:

#### Example of a Candidate Dictionary
- Handcrafted dictionaries are used to assess on a high level whether a document is assigned to a certain candidate class
- Focus is here drawn on assigning a broad range of documents to a certain candidate class
```python
# If a document contains one of the dictionary entries, it will be assigned to the Candidate class Implikation
implikation_signal_words = {"gehör", "soll", "müss", "sieht man", "!", }
```

#### Example of a Label Function
- Label functions create a label for each of the extracted candidate based on a handcrafted rule
- These labels are used to train the generative model
```python
# Imperatives often used in Implikation
def LF_imperative_in_sentence(candidate):

    if [item for item in candidate.get_parent().text if item in imperative_tags]:
        return 1

    return 0

```

### FastText Baseline Model Performance

- Accuracy: 0.286
- Precision per class [hate, no hate]:	[0.99, 0.09]
- Recall per class [hate, no hate]:	[0.232, 0.971]
- F1 score per class [hate, no hate]:	[0.376, 0.165]


## Results

Two major results were found during our work on the course project:

### Challenging Error Classes

(Challenges for Toxic Comment Classification: An In-Depth Error Analysis' (https://arxiv.org/abs/1809.07572))
- Out of vocabulary: Germoney (ID: 1, 196)



- Self-addressed signal words:
    - "weil wir dumm und träge sind" (Because we are stupid and lazy) (ID 212)
    - Tried to capture these with a labeling function and Spacy Tags -> Too little examples in dataset



- Hard classification in codebook:
    - DUMME Panzen ("Dumb Fucks") is active = Correct Intelligenz Candidate
    - keine gewalttätige Dummköpfe (no violent fools) slightly passive = Incorrect Intelligenz Candidate



- Sentence classified twice: One with, one without candidate gold label:
    - Das zeigt einfach nur das die Regierung langsam kalte Füße bekommt. Da wurden mal auf die schnelle Steuergelder locker gemacht um diese Demo's von gehirntoten Deutschen auf die Beine zu stellen. Es hat damals schon in Sachsen begonnen und es wird wieder so sein. (ID 61 - No Intelligenz-Label)
    - Da wurden mal auf die schnelle Steuergelder locker gemacht um diese Demo's von gehirntoten Deutschen auf die Beine zu stellen. (ID 62 - Intelligenz-Label)



- Generative Model
    - Not completely comprehensive: We printed out the predicted label of the model and the result of our labeling functions for the candidates to get a feeling for which labeling functions have an impact and which not

### Baseline Model (fastText) and Silver Snorkel-Label Model

- Accuracy remains the same
- Recall and F1-Score increase roughly +20% - 40% with the use of silver labels



| Model        | Baseline | Snorkel Labels |
| ------------ | -------- | -------------- |
|F1 imbalanced | 0.376    | 0.768          |
|F1 balanced   | 0.435    | 0.632          |
## Docker Container

**Disclaimer:**
Built with a "get it running"-approach.

Contains all needed dependencies for this project. It starts per default a `JupyterLab` with the directory `/opt/project`. If you want to run your own code mount it to this directory like:
``` bash
# for Linux and macOS
docker run -p 8888:8888 -v <path-to-project-code>:/opt/project snorkel

# for Windows
# IMPORTANT: path has to start with lower case: c:/ instead of C:/
# IMPORTANT: replace \ with /
# Example: <path-to-project-code> = c:/Users/Example_User/project-code
docker run -p 8888:8888 -v <path-to-project-code>:/opt/project snorkel

```

It is necessary to forward the containers port `8888` to a local port to access the notebook.


### Notes

- `pytorch` package is removed because of its size (~3.5 gb with dependencies). This shrinked the container of half its size


### Known issues

-
