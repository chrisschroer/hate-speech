# bootstrapping -> imports and connect to database
from utils.bootstrap import *
import subprocess


# ------------------ Apply Labeling Functions -------------------------
L_train = []

## clear existing labels
for index in range(len(candidate_classes)):
    LabelAnnotator(lfs=[]).clear(session, split = 0, key_group = index)

# hack to unlock database
session.commit()

for index, candidate_class in enumerate(candidate_classes):
    L_train.append(LabelAnnotator(lfs=labeling_functions_all[index]).apply(split=0, key_group = index, clear = False, cids_query = session.query(candidate_class.id).filter(candidate_class.split == 0)))
    
print(L_train)

# ------------------ Train Generative Model -------------------------

marginals = []
generativ_models = []
    
for index, candidate_class in enumerate(candidate_classes):
    generativ_models.append(GenerativeModel())

    decay=0.95
    epochs=100
    reg_param=1e-6
    step_size=0.1 / L_train[index].shape[0]

    generativ_models[index].train(L_train[index], epochs = epochs, decay = decay, step_size = step_size, reg_param = reg_param)
    marginals.append(generativ_models[index].marginals(L_train[index]))

    #print(f"Class {index + 1} {candidate_class.__name__} -> {len(marginals[index])}")
    
    
    
# ------------------ Find Threshold with F1 score -------------------------
    
# Load Gold Labels, replace missing values, and cast to proper data types
gold_labels = pd.read_csv("../data/comments_bewertungen_new_ids.csv")
gold_labels = gold_labels.replace(r"^\s*$", 0.0, regex=True)
gold_labels.iloc[:, 3:] = gold_labels.iloc[:, 3:].astype(int)

from sklearn.metrics import precision_recall_curve

candidate_thresholds = []

for index, candidate_class in enumerate(candidate_classes[0:5]):
    
    #labels = gold_labels.iloc[:, candidate_indices[index]].values[[x.get_parent().get_parent().id - 1 for x in session.query(candidate_class).all()]]
    
    
    
    gold_labels_candidate = gold_labels.iloc[:,candidate_indices[0]]
    
    candidate_ids = [x.get_parent().get_parent().id - 1 for x in session.query(candidate_class).all()]
    
    snorkel_labels = np.zeros(500)
    
    snorkel_labels[candidate_ids] = marginals[index]
    
    precision, recall, thresholds = precision_recall_curve(gold_labels_candidate, snorkel_labels)
    
    thresholds_plot = np.append(thresholds, 1) 
    
    f1 = 2*precision*recall/(precision+recall) # Calculate F1 score
    candidate_thresholds.append(thresholds[np.where(f1 == max(f1[(recall>0) & (recall < 1)]))]) # store threshold for highest F1 score with recall > 0 

    if index == 0:
        thresholds_plot_presentation = thresholds_plot
        precision_presentation = precision
        recall_presentation = recall
        x_presentation = candidate_thresholds[index]
    #plt.plot(thresholds_plot, precision, color="red") 
    #plt.plot(thresholds_plot, recall, color="blue")
    #plt.axvline(x=candidate_thresholds[index])
    #plt.show()
    
    
# ------------------ Calculate Metrics -------------------------    

dfs_candidates = []

for index, candidate_class in enumerate(candidate_classes[0:2]):
    candidates = session.query(candidate_class).all()
    
    # initalize with zeros => if one sentence is labeled as 1 the whole document is 1
    df_candidate = pd.DataFrame(index = range(1, 501), columns = ["SnorkelLabel"], data = np.zeros(shape = (500, 1), dtype = int))

    for candidate, marginal in zip(candidates, marginals[index]):
        if marginal > candidate_thresholds[index]:
            
            # use the document_id as label of the rows
            # => use loc
            df_candidate.loc[candidate.get_parent().document_id] = 1

    df_candidate["GoldLabel"] = gold_labels.iloc[:, candidate_indices[index]].values
    
    dfs_candidates.append(df_candidate)
    
# Confusion Matrices and Precision/Recall/F1 Metric

from sklearn.metrics import precision_recall_fscore_support, accuracy_score

error_classes = {}

for index, candidate_class in enumerate(candidate_classes[0:1]):
    class_name = candidate_class.__name__
    
    #print(f"Candidate {class_name}:\n")
    #print(pd.crosstab(dfs_candidates[index].GoldLabel, dfs_candidates[index].SnorkelLabel))
    
    accuracy = accuracy_score(dfs_candidates[index].GoldLabel, dfs_candidates[index].SnorkelLabel)
    precision, recall, f_score, _ = precision_recall_fscore_support(dfs_candidates[index].GoldLabel, dfs_candidates[index].SnorkelLabel, labels = [1, 0])
    
    #print(f"\nAccuracy:\t{float(format(accuracy, '.3f'))}\n")
    
    #print(f"Precision per class [hate, no hate]:\t{[float(format(x, '.3f')) for x in precision]}")
    #print(f"Recall per class [hate, no hate]:\t{[float(format(x, '.3f')) for x in recall]}")
    #print(f"F1 score per class [hate, no hate]:\t{[float(format(x, '.3f')) for x in f_score]}")
    
    if index == 0:
        class_name_presentation = class_name
        cross_tab_presentation = pd.crosstab(dfs_candidates[index].GoldLabel, dfs_candidates[index].SnorkelLabel)
        candidate_accuracy_presentation = accuracy_score(dfs_candidates[index].GoldLabel, dfs_candidates[index].SnorkelLabel)
        candidate_precision_presentation, candidate_recall_presentation, candidate_f_score_presentation, _ = precision_recall_fscore_support(dfs_candidates[index].GoldLabel, dfs_candidates[index].SnorkelLabel, labels = [1, 0])
        
    error_classes[candidate_class] = {"False Negative": list(dfs_candidates[index][(dfs_candidates[index].GoldLabel == 1) & (dfs_candidates[index].SnorkelLabel == 0)].index)}
    error_classes[candidate_class]["False Positive"] = list(dfs_candidates[index][(dfs_candidates[index].GoldLabel == 0) & (dfs_candidates[index].SnorkelLabel == 1)].index)

    
    
gold_testset ="../data/gold_label_testset_binary.txt"
gold_balanced_testset = "../data/gold_label_testset_binary_balanced.txt"

base_model_trainset = "../data/base_model_train_fasttext.txt"
snorkelt_trainset = "../data/dataset_binary_classification_gesnorkelt.txt"
joint_trainset = "../data/dataset_binary_classification_joint_gesnorkelt+basemodel.txt"

base_model = "../model/base_model_arndt.bin"
iteration_1_model = "../model/iteration_1_model.bin"
trained_model = "../model/fasttext_model_binary_classification.bin"
trained_with_embeddings_model = "../model/fasttext_model_binary_classification_with_embeddings.bin"
joint_trained_model = "../model/fasttext_model_binary_classification_joint_gesnorkelt+basemodel.bin"

prediction_examples = "../data/for_example_predictions.txt"




def add_snorkel_set_to_arndt_set():
    # read both train sets
    snorkel_train_set_lines = []
    base_model_train_set_lines = []

    with open(base_model_trainset) as file:
        base_model_train_set_lines = file.readlines()

    with open(snorkelt_trainset) as file:
        snorkel_train_set_lines = file.readlines()


    hate = []
    nohate = []

    for train_set in [base_model_train_set_lines, snorkel_train_set_lines]:
        for train_data in train_set:
            if train_data.startswith("__label__nohate"):
                nohate.append(train_data)
            elif train_data.startswith("__label__hate"):
                hate.append(train_data)
            else:
                pass

    random.shuffle(nohate)
    random.shuffle(hate)

    # combine them
    joint_train_set_lines = hate + nohate[:len(hate)]
    random.shuffle(joint_train_set_lines)

    # write joint train set 
    with open(joint_trainset, "w") as file:
        file.writelines(joint_train_set_lines)