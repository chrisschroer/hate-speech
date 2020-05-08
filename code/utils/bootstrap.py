# import libraries
import re
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# import snorkel stuff
from snorkel.lf_helpers import *
from snorkel import SnorkelSession
from snorkel.matchers import RegexMatchEach
from snorkel.learning import GenerativeModel
from snorkel.parser.spacy_parser import Spacy
from snorkel.viewer import SentenceNgramViewer
from snorkel.parser import DocPreprocessor, CorpusParser
from snorkel.candidates import CandidateSpace, CandidateExtractor, Ngrams
from snorkel.models import candidate_subclass, Document, Sentence, Candidate
from snorkel.annotations import LabelAnnotator, Annotator, save_marginals, load_marginals


# import own code
from .helper import  *
from .dataset import *
from .dictionaries import *
from .candidates_regex import *
from .labeling_functions import *
from .preprocessing import HateSpeechPreprocessor
from .preprocessing import RegexMatchEachSearch

# initialize snorkel session and connect to the database
session = SnorkelSession()