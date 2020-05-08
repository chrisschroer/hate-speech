import random

def create_binary_fasttext_snorkel_dataset(session, marginals, candidate_classes, dataset_path):
    dataset_lines = []

    for index, candidate_class in enumerate(candidate_classes):

        marginals_split_0  = marginals[index]
        candidates_split_0 = session.query(candidate_class).filter(candidate_class.split == 0).all()

        hatespeech_sentences = []
        no_hatespeech_sentences = []

        for candidate, marginal in zip(candidates_split_0, marginals_split_0):
            if marginal > marginals[index].mean():
                hatespeech_sentences.append(f"__label__hate\t{candidate.get_parent().text}")
            else:
                no_hatespeech_sentences.append(f"__label__nohate\t{candidate.get_parent().text}")

        dataset_lines.extend(hatespeech_sentences)
        dataset_lines.extend(no_hatespeech_sentences)

        # shuffel the lines
        random.shuffle(dataset_lines)
        
    # finally write dataset
    with open(dataset_path, "w") as file:
        file.writelines("\n".join(dataset_lines))