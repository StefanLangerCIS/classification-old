"""
Split the data in a training and test set
"""
import argparse
from collections import Counter
import json
import os


def main():
    parser = argparse.ArgumentParser(description='Split in train and test')

    parser.add_argument('--input',
                    default= r"D:\ProjectData\Uni\ltrs\data\classifier\classifier_data.json",
                    help='All data, the data_records to be split. One json record per line')
    
    parser.add_argument('--output_folder',
                    default= r"D:\ProjectData\Uni\ltrs\data\classifier",
                    help = 'Folder where the split data set will be put')

    parser.add_argument('--filter_label',
                        default= "author",
                        help='The label for filtering by count')

    parser.add_argument('--filter_min_count',
                        default= 1000,
                        help='A label which occurs less in the data is not used')


    args = parser.parse_args()

    # List of arbitrary length, distribute data in buckect (number of buckets len(list)) and assign to set designated by the string
    # 0 or "" for skipping the line
    input_split = ["train","train","eval","train","train","train","train","test","train","train"]
    # Small set
    #input_split = ["train", "eval", "train", "test",0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    split_data = {}
    label_count = Counter()
    with open(args.input, encoding="utf-8") as infile:
        i = 0
        for line in infile:
            line = line.strip()
            json_data = json.loads(line)
            output_set = input_split[i%len(input_split)]
            if isinstance(output_set, str) and len(output_set) > 0:
                if not output_set in split_data:
                    split_data[output_set] = []
                split_data[output_set].append(json_data)
                label_count[json_data[args.filter_label]] += 1
            else:
                # skip line
                pass
            i += 1

    for output_set in split_data:
        file_name = os.path.join(args.output_folder, "classifier_data_{0}.json".format(output_set))
        with open(file_name, "w", encoding="utf-8") as out:
            for json_data in split_data[output_set]:
                if label_count[json_data[args.filter_label]] >= args.filter_min_count:
                    out.write(json.dumps(json_data, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()