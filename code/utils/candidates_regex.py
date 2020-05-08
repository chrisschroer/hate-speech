from snorkel.models import candidate_subclass

from .dictionaries import *
from .helper import make_regex

candidate_indices = [8, 14, 15, 20, 24] # Indices for reading the corresponding column in the "comments_bewertungen_new_ids.csv" File

# Candidates
Intelligenz_BB3c      = candidate_subclass('Intelligenz', ['signal_word'])
WirkungaufBRD_BB3i    = candidate_subclass('WirkungaufBRD', ['signal_word'])
Implikation_BB4       = candidate_subclass('Implikation', ['signal_word']) # Gehören, sollten, etc. Abfrage mit Reflexiven Personalpronomen, Indefinite Pronomen, Imperative, kein Verb
Beschimpfung_BB6a     = candidate_subclass('Beschimpfung', ['signal_word'])
Entmenschlichung_BB6e = candidate_subclass('Entmenschlichung', ['signal_word'])


# Regexes
intelligenz_regex      = make_regex(intelligenz_signal_words)
wirkungaufbrd_regex    = make_regex(wirkungaufbrd_signal_words)
implikation_regex      = make_regex(implikation_signal_words)
beschimpfung_regex     = make_regex(beschimpfung_signal_words, offense_signal_words, refugee_related_signal_words, negative_signal_words)
entmenschlichung_regex = make_regex(entmenschlichung_signal_words, animal_signal_words)


candidate_classes = [
    Intelligenz_BB3c,
    WirkungaufBRD_BB3i,
    Implikation_BB4,
    Beschimpfung_BB6a,
    Entmenschlichung_BB6e
]

regex_classes = [
    intelligenz_regex,
    wirkungaufbrd_regex,
    implikation_regex,
    beschimpfung_regex,
    entmenschlichung_regex
]

# create (candidate, regex) tuples, combines the above lists
candidate_regex_tupels = [(candidate_class, candidate_regex) for candidate_class, candidate_regex in zip(candidate_classes, regex_classes)]
