"""
Split the data in a training and test set
"""
import argparse
from collections import Counter
import json


def main():
    parser = argparse.ArgumentParser(description='Split in train and test')

    parser.add_argument('--input',
                    default= r"D:\ProjectData\Uni\ltrs\data\classifier\classifier_data.json",
                    help='The training data for the classifier. If "None", the existing model is loaded')
    
    parser.add_argument('--training',
                    default= r"D:\ProjectData\Uni\ltrs\data\classifier\classifier_training_data.json",
                    help='Training data')

    parser.add_argument('--test',
                        default= r"D:\ProjectData\Uni\ltrs\data\classifier\classifier_test_data.json",
                        help='Test data')

    parser.add_argument('--filter_label',
                        default= "author",
                        help='The label for filtering by count')

    parser.add_argument('--filter_min_count',
                        default= 200,
                        help='A label which occurs less in the training data is not used')


    args = parser.parse_args()

    # Use all training lines (except each 1st) or fewer (for quicker training)
    #training_lines_of_10 = [0,1,2,3,4,5,6,7,8]
    training_lines_of_10 = [1]
    test_lines_of_10 = [9]

    training_data = []
    test_data = []
    label_count = Counter()
    with open(args.input, encoding="utf-8") as infile:
        i = 0
        for line in infile:
            line = line.strip()
            json_data = json.loads(line)
            i += 1
            if int(i%10) in training_lines_of_10:
                training_data.append(json_data)
                label_count[json_data[args.filter_label]] += 1
            if int(i%10) in test_lines_of_10:
                test_data.append(json_data)

    with open(args.training, "w", encoding="utf-8") as out:
        for json_data in training_data:
            if label_count[json_data[args.filter_label]] >= args.filter_min_count:
                out.write(json.dumps(json_data, ensure_ascii=False) + "\n")
    with open(args.test, "w", encoding="utf-8") as out:
        for json_data in test_data:
            if label_count[json_data[args.filter_label]] >= args.filter_min_count:
                out.write(json.dumps(json_data, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()