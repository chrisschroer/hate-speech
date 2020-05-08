# helper functions to make RegEx more robust
def extend_umlauts(regex: str) -> str:
    regex = regex.replace("ä", "(ä|a|ae)")
    regex = regex.replace("ö", "(ö|o|oe)")
    regex = regex.replace("ü", "(ü|u|ue)")
    regex = regex.replace("ß", "(ß|s|ss)")
    
    return regex

# unfold lists into one
def unfold_lists(list_with_lists: list) -> list:
    # https://stackoverflow.com/a/952952/6453788
    return [item for sublist in list_with_lists for item in sublist]

def make_regex(*args) -> str:
    dictionaries = [i for i in args]
    signal_words_list = unfold_lists(dictionaries)
    regex = "|".join(signal_words_list)
    
    # hack because snorkel adds a '$' to the end of the regex string.
    # This means the string has to be at the end of a word to be matched.
    regex += "|äääöööüüüßßß"
    
    return extend_umlauts(regex)


# returns the 'window' given 'attribute' to the 'candidate' left
def get_attribute_to_left(candidate, window=3, attribute='lemmas'):
    span = candidate[0]
    index = span.get_word_start()

    return span.get_parent()._asdict()[attribute][max(0, index-window):index]


# returns the 'window' given 'attribute' to the 'candidate' right
def get_attribute_to_right(candidate, window=3, attribute='lemmas'):
    span = candidate[0]
    index = span.get_word_end()

    return span.get_parent()._asdict()[attribute][index+1:index+1+window]