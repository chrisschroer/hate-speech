import argparse
import numpy as np

from sklearn.metrics import precision_recall_fscore_support, accuracy_score



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate and print some metrics for fasttext predictions.')
    parser.add_argument('test', help='Test Labels')
    parser.add_argument('predict', help='Predicted Labels')
    args = parser.parse_args()

    test_labels = np.array(args.test.splitlines())
    pred_labels = np.array(args.predict.splitlines())

    accuracy = accuracy_score(test_labels, pred_labels)
    precision, recall, f_score, _ = precision_recall_fscore_support(test_labels, pred_labels)

    print(f"Accuracy: {float(format(accuracy, '.3f'))}\n")
        
    print(f"Precision per class [hate, no hate]:\t{[float(format(x, '.3f')) for x in precision]}")
    print(f"Recall per class [hate, no hate]:\t{[float(format(x, '.3f')) for x in recall]}")
    print(f"F1 score per class [hate, no hate]:\t{[float(format(x, '.3f')) for x in f_score]}")
