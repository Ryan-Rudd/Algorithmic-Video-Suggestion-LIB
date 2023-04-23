from collections import Counter
import uuid
import os
import json


class TrainData:
    def __init__(self, video_tags, video_publisher, video_topic):
        self.video_tags = video_tags
        self.video_publisher = video_publisher
        self.video_topic = video_topic

def train_avs_data(data: TrainData, user_id, train_data_folder="train_data"):
    print("="*7, "\tAVS TRAINING\t", "="*7, f"\nNew Data Stored @MAR: {id(data)}")
    
    # create the train data folder if it doesn't exist
    if not os.path.exists(train_data_folder):
        print("Train model directory doesn't exist: Creating")
        os.mkdir(train_data_folder)
        print("Successfully Created train model directory")

    # create the user's train data file if it doesn't exist
    file_path = f"{train_data_folder}/{user_id}.train_data"
    if not os.path.isfile(file_path):
        with open(file_path, "w") as file:
            json.dump({"publishers": [], "tags": [], "topics": [], "suggested_publishers": "", "suggested_tags": "", "suggested_topics": ""}, file)

    # load the user's train data file
    with open(file_path, 'r') as file_data:
        data_dict = json.load(file_data)

    # update the publishers, tags and topics lists
    data_dict["publishers"].append(data.video_publisher)
    data_dict["tags"].append(data.video_tags)
    data_dict["topics"].append(data.video_topic)
    
    # count the occurrences of each variable and get the most common variable for each category
    most_common = {}
    for key in ["publishers", "tags", "topics"]:
        counter = Counter(data_dict[key])
        most_common[f"suggested_{key}"] = counter.most_common(1)[0][0]
        data_dict[key] = list(set(data_dict[key]))
    
    # update the suggested variables
    data_dict.update(most_common)

    # save the updated train data file
    with open(file_path, 'w') as file_data:
        json.dump(data_dict, file_data)

    print("Most frequent variables for suggestion:")
    print(most_common)

data = TrainData(video_publisher="PUBLISHER", video_tags="hELLO", video_topic="SCIENCE")
train_avs_data(data, uuid.uuid4())
