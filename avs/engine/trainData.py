from ctypes import cast, py_object
import uuid
import os

class TrainData(object):
    def __init__(self, VIDEO_TAGS, VIDEO_PUBLISHER, VIDEO_TOPIC, VIDEO_STAMP):
        self.VIDEO_TAGS = VIDEO_TAGS
        self.VIDEO_PUBLISHER = VIDEO_PUBLISHER
        self.VIDEO_TOPIC = VIDEO_TOPIC
        self.VIDEO_STAMP = VIDEO_STAMP

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
        


        
        

user_data = TrainData(
    VIDEO_PUBLISHER="PUBLISHER",
    VIDEO_TAGS=["TAG 1", "TAG 2"],
    VIDEO_TOPIC = "TOPIC",
    VIDEO_STAMP="STAMP")

ModelTrainer.trainAVSdata(user_data, uuid.uuid4())