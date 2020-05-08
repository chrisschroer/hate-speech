import re

from .helper import  *

# Spacy POS (part-of-speech) tags for pronouns (https://spacy.io/api/annotation#pos-de)
pronoun_tags = {"PDAT", "PDS", "PIAT", "PIDAT", "PIS", "PPER", "PPOSAT", "PPOSS", "PRELAT", "PRELS", "PRF", "PWAT", "PWAV", "PWS"}

# Some often used forms of address 
form_of_address = {"sie", "du", "ihr", "die", "alle", "alles", "den"}

# Verb tags in general including imperatives and infinitives
verb_tags = ["VAFIN","VAIMP","VAINF", "VMFIN", "VVFIN", "VMINF", "VVIMP"]

# Spacy POS (part-of-speech) tags for finite verbs (https://spacy.io/api/annotation#pos-de)
# Finite verbs containing (among others) information about the grammatical person
finite_verb_tags = ["VAFIN", "VMFIN", "VVFIN"]

# Verbforms of 'sein'. Often used to offense people.
conjugated_sein = ["bin", "bist", "ist", "sind", "seid"]

# Possessive pronouns
wirkung_auf_brd_words = ["uns", "mein", "wir","eigen", "gefahr", "stör", "vernicht", "schad", "ruin", "medium", "medien", "presse", "steuer", "sozial"]
wirkung_auf_brd_regex = make_regex(wirkung_auf_brd_words)

# Spacy POS (part-of-speech) tags for proper nouns (https://spacy.io/api/annotation#pos-de)
# Proper noun, often used for direct offense
proper_noun_tags = ["NE", "NNE"]

# UPPERCASE comments are indicators for negative vibe
def LF_upper_case(candidate):

    if(re.search('[A-Z]{3}', candidate.get_parent().text)):
        return 1

    return 0


# "Beschimpfung" and person entity in sentence -> chance of offensive meaning
def LF_Beschimpfung_and_Person(candidate):

    if candidate.type == "beschimpfung":
        if "PERSON" in candidate.get_parent().ner_tags:
            return 1

    return 0


# elipsis (..) has negative vibe
def LF_contain_ellipsis(candidate):

    if ".." in candidate.get_parent().text:
        return 1
    
    return 0


# pronoun before signal word indicates a personal attack
def LF_pronoun_before_signal_word(candidate):

    words_left = get_attribute_to_left(candidate, window=5)
    pos_tags_left = get_attribute_to_left(candidate, window=5, attribute="pos_tags")
    
    # TODO: maybe better than just "ich"?
    # ref: 'Challenges for Toxic Comment Classification: An In-Depth Error Analysis' (https://arxiv.org/abs/1809.07572)
    if "ich" in words_left or "wir" in words_left:
        return -1

    elif [item for item in pos_tags_left if item in pronoun_tags]:
        return 1

    return 0


# quotes are often indications for offensive comments
def LF_quotation_marks(candidate):

    if re.search("['\"].*['\"]", candidate.get_parent().text):
        return 1

    return 0


# colloquial forms of address and signal words
def LF_form_of_address_and_signale_word(candidate):

    words_left = get_attribute_to_left(candidate, window=5)

    if [item for item in words_left if item in form_of_address]:
        return 1

    return 0


# finite verb 'sein' near to a signal word is used to offense people
def LF_sein_is_finite_verb_near_signal_word(candidate):

    words_left = get_attribute_to_left(candidate, window=5)
    pos_tags_left = get_attribute_to_left(candidate, window=5, attribute="pos_tags")

    words_right = get_attribute_to_right(candidate, window=5)
    pos_tags_rigth = get_attribute_to_right(candidate, window=5, attribute="pos_tags")

    if [False for word_left, pos_tag_left, word_right, pos_tag_rigth in zip(words_left, pos_tags_left, words_right, pos_tags_rigth) if (pos_tag_left in finite_verb_tags and word_left in conjugated_sein) or (pos_tag_rigth in finite_verb_tags and word_left in conjugated_sein)]:
        return 1

    return 0


def LF_proper_noun_after_signal_word(candidate):

    pos_tags_right = get_attribute_to_right(candidate, window=5, attribute="pos_tags")

    if [item for item in pos_tags_right if item in proper_noun_tags]:
        return 1

    return 0

def LF_no_form_of_verb_in_sentence(candidate):
    
    if not [item for item in candidate.get_parent().text if item in verb_tags]:
        return 1
    
    return 0


def LF_wirkung_auf_brd_in_sentence(candidate):
    
    if re.search(wirkung_auf_brd_regex, candidate.get_parent().text):
        return 1

    return -1

# ------------------------------------
# Set appropriate labeling functions for each of the candidates
labeling_functions_all_candidates = [
    LF_upper_case,
    LF_quotation_marks,
    LF_contain_ellipsis,
    LF_pronoun_before_signal_word,
    LF_form_of_address_and_signale_word,
    LF_sein_is_finite_verb_near_signal_word,
]

labeling_functions_wirkungaufbrd = [
    # add general labeling functions
    *labeling_functions_all_candidates,
    LF_wirkung_auf_brd_in_sentence,
    LF_Beschimpfung_and_Person,
]

labeling_functions_implikation = [
    # add general labeling functions
    *labeling_functions_all_candidates,
    LF_contain_ellipsis,
    LF_no_form_of_verb_in_sentence,
    LF_Beschimpfung_and_Person,
]

labeling_functions_beschimpfung = [
    # add general labeling functions
    *labeling_functions_all_candidates,

    LF_Beschimpfung_and_Person,
    LF_proper_noun_after_signal_word,
]

labeling_functions_intelligenz = [
    # add general labeling functions
    *labeling_functions_all_candidates,
]

labeling_functions_entmenschlichung = [
    # add general labeling functions
    *labeling_functions_all_candidates,
]

# Store all labeling functions list for each candidate in one nested list
labeling_functions_all = [
    labeling_functions_intelligenz,
    labeling_functions_wirkungaufbrd,
    labeling_functions_implikation,
    labeling_functions_beschimpfung,
    labeling_functions_entmenschlichung
]