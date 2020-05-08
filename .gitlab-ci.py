import os
import subprocess

from code.utils.bootstrap import *


### Some Helper Functions

def pipeline():

    if os.environ['CI_JOB_STAGE'] == "parse_corpus":

        parse_corpus('data/vico_user_comments.csv')


    if os.environ['CI_JOB_STAGE'] == "extract_candidates":

        extract_candidates()


    if os.environ['CI_JOB_STAGE'] == "label_candidates":

        label_candidates()


    if os.environ['CI_JOB_STAGE'] == "train_generative_model":

        train_generative_model()


    if os.environ['CI_JOB_STAGE'] == "test_gold_labels":

        test_gold_labels()


    if os.environ['CI_JOB_STAGE'] == "train_external_model":

        train_external_model()


def parse_corpus(file):

    hate_speech_preprocessor = HateSpeechPreprocessor(file)
    corpus_parser = CorpusParser(parser=Spacy(lang="de"))
    corpus_parser.apply(hate_speech_preprocessor)


def extract_candidates():

    candidate_extractors = []

    for candidate_class, candidate_regex in candidate_regex_tupels:

        n_grams             = Ngrams(n_max = 5)
        matcher             = RegexMatchEachSearch(rgx = candidate_regex)
        candidate_extractor = CandidateExtractor(candidate_class, [n_grams], [matcher])

        candidate_extractors.append((candidate_class, candidate_extractor))


    documents = session.query(Document).order_by(Document.name).all()

    train_sentences = set()
    dev_sentences   = set()
    test_sentences  = set()

    for index, document in enumerate(documents):
        for sentence in document.sentences:
            if index % 10 == 8:
                dev_sentences.add(sentence)
            elif index % 10 == 9:
                test_sentences.add(sentence)
            else:
                train_sentences.add(sentence)

    # delete all candidates
    print("Clearing database ...\n")

    if session.query(Candidate).count() != 0:
        session.query(Candidate).delete()

    if session.query(Intelligenz_BB3c).count() != 0:
        session.query(Intelligenz_BB3c).delete()

    if session.query(Beschimpfung_BB6a).count() != 0:
        session.query(Beschimpfung_BB6a).delete()

    if session.query(Entmenschlichung_BB6e).count() != 0:
        session.query(Entmenschlichung_BB6e).delete()


    print("Extracting Candidates ...\n")

    for candidate_class, candidate_extractor in candidate_extractors:
        for split, sentence in enumerate([train_sentences, dev_sentences, test_sentences]):

            candidate_extractor.apply(sentence, split=split, clear=False)
            print(f"Number of {candidate_class.__name__} candidates in split {split}: {session.query(candidate_class).filter(candidate_class.split == split).count()}\n\n")


def label_candidates():

    L_train = []

    ## clear existing labels
    for index in range(len(candidate_classes)):
        LabelAnnotator(lfs=[]).clear(session, split = 0, key_group = index)

    # hack to unlock database
    session.commit()

    for index, candidate_class in enumerate(candidate_classes):
        L_train.append(LabelAnnotator(lfs=labeling_functions_all[index]).apply(split=0, key_group = index, clear = False, cids_query = session.query(candidate_class.id).filter(candidate_class.split == 0)))

    print(L_train)


def train_generative_model(output = True):

    L_train = []
    marginals = []
    generativ_models = []

    for index, candidate_class in enumerate(candidate_classes):
        L_train.append(LabelAnnotator(lfs = []).load_matrix(session, split=0, key_group = index, cids_query = session.query(candidate_class.id).filter(candidate_class.split == 0)))

    print(L_train, "\n")

    for index, candidate_class in enumerate(candidate_classes):
        generativ_models.append(GenerativeModel())

        decay=0.95
        epochs=100
        reg_param=1e-6
        step_size=0.1 / L_train[index].shape[0]

        generativ_models[index].train(L_train[index], epochs = epochs, decay = decay, step_size = step_size, reg_param = reg_param)
        marginals.append(generativ_models[index].marginals(L_train[index]))

        print(f"Class {index + 1} {candidate_class.__name__} -> {len(marginals[index])}")


    if output:
        create_binary_fasttext_snorkel_dataset(session, marginals, candidate_classes, "data/dataset_binary_classification_gesnorkelt.txt")

    return marginals


def test_gold_labels():

    # run the standard pipeline with the gold label set
    parse_corpus("data/comments_bewertungen_ohne_label.csv")
    extract_candidates()
    label_candidates()
    marginals = train_generative_model(False)

    # compute some metrics and outputs

    gold_labels = pd.read_csv("data/comments_bewertungen_new_ids.csv")
    gold_labels = gold_labels.replace(r"^\s*$", 0.0, regex=True)
    gold_labels.iloc[:, 3:] = gold_labels.iloc[:, 3:].astype(int)

    dfs_candidates = []

    for index, candidate_class in enumerate(candidate_classes):
        candidates = session.query(candidate_class).all()

        # initalize with zeros => if one sentence is labeled as 1 the whole document is 1
        df_candidate = pd.DataFrame(index = range(1, 501), columns = ["SnorkelLabel"], data = np.zeros(shape = (500, 1), dtype = int))

        for candidate, marginal in zip(candidates, marginals[index]):
            if marginal > marginals[index].mean():

                # use the document_id as label of the rows
                # => use loc
                df_candidate.loc[candidate.get_parent().document_id] = 1

        df_candidate["GoldLabel"] = gold_labels.iloc[:, candidate_indices[index]].values

        dfs_candidates.append(df_candidate)


    # Confusion Matrices
    error_classes = {}

    for index, candidate_class in enumerate(candidate_classes):
        class_name = candidate_class.__name__

        print(f"Candidate {class_name}:\n")
        print(pd.crosstab(dfs_candidates[index].GoldLabel, dfs_candidates[index].SnorkelLabel))

        accuracy = accuracy_score(dfs_candidates[index].GoldLabel, dfs_candidates[index].SnorkelLabel)
        precision, recall, f_score, _ = precision_recall_fscore_support(dfs_candidates[index].GoldLabel, dfs_candidates[index].SnorkelLabel, labels = [1, 0])

        print(f"\nAccuracy: {float(format(accuracy, '.3f'))}\n")

        print(f"Precision per class [hate, no hate]:\t{[float(format(x, '.3f')) for x in precision]}")
        print(f"Recall per class [hate, no hate]:\t{[float(format(x, '.3f')) for x in recall]}")
        print(f"F1 score per class [hate, no hate]:\t{[float(format(x, '.3f')) for x in f_score]}")

        error_classes[candidate_class] = {"False Negative": list(dfs_candidates[index][(dfs_candidates[index].GoldLabel == 1) & (dfs_candidates[index].SnorkelLabel == 0)].index)}
        error_classes[candidate_class]["False Positive"] = list(dfs_candidates[index][(dfs_candidates[index].GoldLabel == 0) & (dfs_candidates[index].SnorkelLabel == 1)].index)

        print('\n------------------------------------------------------------------------------------------------\n')


    # print the error classes
    for candidate_class in error_classes.keys():

        print(f"Hate speech dimension: {candidate_class.__name__}\n")

        for error_class in error_classes[candidate_class].keys():

            documents = session.query(Document).filter(Document.id.in_(error_classes[candidate_class][error_class])).all()

            print(f"\n\tError Class: {error_class}\n\n")

            for document in documents:

                print(f"\t\t{' '.join(map(lambda x : x.text, document.get_children()))}\n")

        print('\n------------------------------------------------------------------------------------------------\n\n')


    # print not as candidatec extracted documents
    candidates = session.query(Candidate).all()
    candidates_id = set([candidate.get_parent().document_id for candidate in candidates])
    not_candidate_documents = session.query(Document).filter(Document.id.notin_(candidates_id)).all()

    print(f"Not as candidates extracted documents.")
    print("(Some are of course without any hate speech content.)\n\n")

    for document in not_candidate_documents:

        print(f"\t{' '.join(map(lambda x : x.text, document.get_children()))}\n")


    # print labeled candidates
    for index, candidate_class in enumerate(candidate_classes):
        candidates = session.query(candidate_class).all()

        print(f"Hate speech dimension: {candidate_class.__name__}\n")

        for candidate, marginal in zip(candidates, marginals[index]):

            print(f"\t{candidate.get_parent().text}\n")

            if marginal > marginals[index].mean():
                print(f"\t\tPrediction: Hatespeech")
            else:
                print(f"\t\tPrediction: No Hatespeech")

            print(f"\t\tLabels:     {candidate.labels}\n\n")

        print('------------------------------------------------------------------------------------------------\n\n')


def train_external_model():

    train_command = "fasttext supervised -input data/dataset_binary_classification_gesnorkelt.txt -output fasttext_model_binary_classification"
    subprocess.run(train_command, shell=True, check=True)

    test_set_description = {
        "gold_label_testset_binary.txt": "split by BB1 column (hate / no hate)",
        "gold_label_testset_binary_by_dimension.txt": "split by known (snorkeled) hate speech dimensions",
        "gold_label_testset_binary_balanced.txt": "split by BB1 column (hate / no hate) and balanced"
    }

    #formatting output
    print("\n")

    for testset in test_set_description.keys():
        print(f"Testing on Gold Label Set - {test_set_description[testset]}\n")

        test_command = f'python code/utils/fasttext_testing.py "$(cut -f 1 data/{testset})" "$(fasttext predict fasttext_model_binary_classification.bin data/{testset})"'
        completed_process = subprocess.run(test_command, shell=True, check=True, encoding="utf-8", stdout=subprocess.PIPE)

        print(completed_process.stdout)

        print('\n------------------------------------------------------------------------------------------------\n')


## Run the pipeline
pipeline()
