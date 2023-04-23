from ctypes import cast, py_object
import uuid
import os
import re

class TrainData(object):
    def __init__(self, VIDEO_TAGS, VIDEO_PUBLISHER, VIDEO_TOPIC):
        self.VIDEO_TAGS = VIDEO_TAGS
        self.VIDEO_PUBLISHER = VIDEO_PUBLISHER
        self.VIDEO_TOPIC = VIDEO_TOPIC

class ModelTrainer:
    def trainAVSdata(data: TrainData, user_id, train_data_folder = "train_data"):
        print("="*7,"\tAVS TRAINING\t","="*7, f"\nNew Data Stored @MAR: {id(data)}")
        memory_data = cast(id(data), py_object).value
        if memory_data == data:
            print(f"Data store check: memory_data & model data are stored correctly")
        if not os.path.exists(train_data_folder):
            print("Train model directory doesn't exist: Creating")
            os.mkdir(train_data_folder)
            print("Succesfuly Create train model directory")
            pass
        if not os.path.isfile(f'{train_data_folder}/{user_id}.train_data'):
            with open(f'{train_data_folder}/{user_id}.train_data', "w") as file:
                FILE_DATA = """
PUBLISHERS:[]
TAGS:[]
TOPICS:[]

RES:{
    SUGGESTED_PUBLISHERS:[]
    SUGGESTED_TAGS:[]
    SUGGESTED_TOPICS:[]
}
                """
                file.write(FILE_DATA)
                pass

        with open(f'{train_data_folder}/{user_id}.train_data', 'r') as file_data:

            file_data_text = str(file_data.read())

            TARGET_PUBLISHERS_ARRAY = "PUBLISHERS:"
            TARGET_TAGS_ARRAY = "TAGS:"
            TARGET_TOPICS_ARRAY = "TOPICS:"

            ARRAY_DEFINITION_BEGIN_TOKEN = "["
            ARRAY_DEFINITION_END_TOKEN = "]"
            ARRAY_CONTINOUS_ITEM_TOKEN = ","

            # SEARCH FOR PUBLISHERS ARRAY INDEX
            publisher_res = re.search(TARGET_PUBLISHERS_ARRAY, file_data_text)
            print(f"Found publisher array index from training model: {publisher_res.span()}")


user_data = TrainData(
    VIDEO_PUBLISHER="PUBLISHER",
    VIDEO_TAGS=["TAG 1", "TAG 2"],
    VIDEO_TOPIC = "TOPIC")

ModelTrainer.trainAVSdata(user_data, uuid.uuid4())