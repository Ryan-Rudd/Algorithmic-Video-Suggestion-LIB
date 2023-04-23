from ctypes import cast, py_object
import uuid
import os
import re
from collections import Counter

class TrainData(object):
    def __init__(self, VIDEO_TAGS, VIDEO_PUBLISHER, VIDEO_TOPIC):
        self.VIDEO_TAGS = VIDEO_TAGS
        self.VIDEO_PUBLISHER = VIDEO_PUBLISHER
        self.VIDEO_TOPIC = VIDEO_TOPIC

class ModelTrainer:
    def trainAVSdata(data: TrainData, user_id, train_data_folder="train_data"):
        print("="*7,"\tAVS TRAINING\t","="*7, f"\nNew Data Stored @MAR: {id(data)}")
        memory_data = cast(id(data), py_object).value
        if memory_data == data:
            print(f"Data store check: memory_data & model data are stored correctly")
        if not os.path.exists(train_data_folder):
            print("Train model directory doesn't exist: Creating")
            os.mkdir(train_data_folder)
            print("Successfully Created train model directory")
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

        with open(f'{train_data_folder}/{user_id}.train_data', 'r+') as file_data:
            file_data_text = str(file_data.read())

            TARGET_PUBLISHERS_ARRAY = "PUBLISHERS:"
            TARGET_TAGS_ARRAY = "TAGS:"
            TARGET_TOPICS_ARRAY = "TOPICS:"
            TARGET_SUGGESTED_PUBLISHERS_ARRAY = "SUGGESTED_PUBLISHERS:"
            TARGET_SUGGESTED_TAGS_ARRAY = "SUGGESTED_TAGS:"
            TARGET_SUGGESTED_TOPICS_ARRAY = "SUGGESTED_TOPICS:"

            ARRAY_DEFINITION_BEGIN_TOKEN = "\\["
            ARRAY_DEFINITION_END_TOKEN = "\\]"
            ARRAY_CONTINUOUS_ITEM_TOKEN = ","

            # SEARCH FOR PUBLISHERS ARRAY INDEX
            publisher_res = re.search(TARGET_PUBLISHERS_ARRAY, file_data_text)
            print(f"Found publisher array index from training model: {publisher_res.span()}")
            array_open_index = re.search(ARRAY_DEFINITION_BEGIN_TOKEN, file_data_text[publisher_res.span()[1]:])  # ERROR ON THIS LINE
            array_close_index = re.search(ARRAY_DEFINITION_END_TOKEN, file_data_text[publisher_res.span()[1]:])

            # Update Publishers List
            publisher_list = file_data_text[publisher_res.span()[1] + array_open_index.span()[0] : publisher_res.span()[1] + array_close_index.span()[0]]
            publisher_list = [i.strip() for i in publisher_list.split(ARRAY_CONTINUOUS_ITEM_TOKEN) if i.strip()]
            publisher_list.extend(data.VIDEO_PUBLISHER)
            publisher_list = list(set(publisher_list))
            file_data.seek(publisher_res.span()[1] + array_open_index.span()[0])
            file_data.write(ARRAY_CONTINUOUS_ITEM_TOKEN.join(publisher_list) + ARRAY_DEFINITION_END_TOKEN)

            # SEARCH FOR TAGS ARRAY INDEX
            tags_res = re.search(TARGET_TAGS_ARRAY, file_data_text)
            print(f"Found tags array index from training model: {tags_res.span()}")
            array_open_index = re.search(ARRAY_DEFINITION_BEGIN_TOKEN, file_data_text[tags_res.span()[1]:])
            array_close_index = re.search(ARRAY_DEFINITION_END_TOKEN, file_data_text[tags_res.span()[1]:])

            # Update Tags List
            tags_list = file_data_text[tags_res.span()[1] + array_open_index.span()[0] : tags_res.span()[1] + array_close_index.span()[0]]
            tags_list = [i.strip() for i in tags_list.split(ARRAY_CONTINUOUS_ITEM_TOKEN) if i.strip()]
            tags_list.extend(data.VIDEO_TAGS)
            tags_list = list(set(tags_list))
            file_data.seek(tags_res.span()[1] + array_open_index.span()[0])
            file_data.write(ARRAY_CONTINUOUS_ITEM_TOKEN.join(tags_list) + ARRAY_DEFINITION_END_TOKEN)

            # SEARCH FOR TOPICS ARRAY INDEX
            topics_res = re.search(TARGET_TOPICS_ARRAY, file_data_text)
            print(f"Found topics array index from training model: {topics_res.span()}")
            array_open_index = re.search(ARRAY_DEFINITION_BEGIN_TOKEN, file_data_text[topics_res.span()[1]:])
            array_close_index = re.search(ARRAY_DEFINITION_END_TOKEN, file_data_text[topics_res.span()[1]:])

            # Update Topics List
            topics_list = file_data_text[topics_res.span()[1] + array_open_index.span()[0] : topics_res.span()[1] + array_close_index.span()[0]]
            topics_list = [i.strip() for i in topics_list.split(ARRAY_CONTINUOUS_ITEM_TOKEN) if i.strip()]
            topics_list.append(data.VIDEO_TOPIC)
            topics_list = list(set(topics_list))
            file_data.seek(topics_res.span()[1] + array_open_index.span()[0])
            file_data.write(ARRAY_CONTINUOUS_ITEM_TOKEN.join(topics_list) + ARRAY_DEFINITION_END_TOKEN)

            # CREATE SUGGESTED VARIABLES
            suggested_variables = {
                TARGET_SUGGESTED_PUBLISHERS_ARRAY: Counter(publisher_list).most_common(1)[0][0],
                TARGET_SUGGESTED_TAGS_ARRAY: Counter(tags_list).most_common(1)[0][0],
                TARGET_SUGGESTED_TOPICS_ARRAY: Counter(topics_list).most_common(1)[0][0]
            }

            # SEARCH FOR SUGGESTED PUBLISHERS ARRAY INDEX
            suggested_publisher_res = re.search(TARGET_SUGGESTED_PUBLISHERS_ARRAY, file_data_text)
            print(f"Found suggested publisher array index from training model: {suggested_publisher_res.span()}")
            array_open_index = re.search(ARRAY_DEFINITION_BEGIN_TOKEN, file_data_text[suggested_publisher_res.span()[1]:])
            array_close_index = re.search(ARRAY_DEFINITION_END_TOKEN, file_data_text[suggested_publisher_res.span()[1]:])
            file_data.seek(suggested_publisher_res.span()[1] + array_open_index.span()[0])
            file_data.write(f"[{suggested_variables[TARGET_SUGGESTED_PUBLISHERS_ARRAY]}]{ARRAY_DEFINITION_END_TOKEN}")

                    # Update Tags List
        tags_list = file_data_text[tags_res.span()[1] + array_open_index.span()[0] : tags_res.span()[1] + array_close_index.span()[0]]
        tags_list = [i.strip() for i in tags_list.split(ARRAY_CONTINUOUS_ITEM_TOKEN) if i.strip()]
        tags_list.extend(data.VIDEO_TAGS)
        tags_list = list(set(tags_list))
        file_data.seek(tags_res.span()[1] + array_open_index.span()[0])
        file_data.write(ARRAY_CONTINUOUS_ITEM_TOKEN.join(tags_list) + ARRAY_DEFINITION_END_TOKEN)

        # SEARCH FOR TOPICS ARRAY INDEX
        topics_res = re.search(TARGET_TOPICS_ARRAY, file_data_text)
        print(f"Found topics array index from training model: {topics_res.span()}")
        array_open_index = re.search(ARRAY_DEFINITION_BEGIN_TOKEN, file_data_text[topics_res.span()[1]:])
        array_close_index = re.search(ARRAY_DEFINITION_END_TOKEN, file_data_text[topics_res.span()[1]:])

        # Update Topics List
        topics_list = file_data_text[topics_res.span()[1] + array_open_index.span()[0] : topics_res.span()[1] + array_close_index.span()[0]]
        topics_list = [i.strip() for i in topics_list.split(ARRAY_CONTINUOUS_ITEM_TOKEN) if i.strip()]
        topics_list.append(data.VIDEO_TOPIC)
        topics_list = list(set(topics_list))
        file_data.seek(topics_res.span()[1] + array_open_index.span()[0])
        file_data.write(ARRAY_CONTINUOUS_ITEM_TOKEN.join(topics_list) + ARRAY_DEFINITION_END_TOKEN)

        # Update suggested variables
        suggested_data = {
            "SUGGESTED_PUBLISHERS": publisher_list,
            "SUGGESTED_TAGS": tags_list,
            "SUGGESTED_TOPICS": topics_list
        }
        suggested_res = re.search("RES:{", file_data_text)
        if suggested_res:
            suggested_res_end = re.search("}", file_data_text[suggested_res.span()[1]:])
            suggested_data_str = file_data_text[suggested_res.span()[1] + suggested_res_end.span()[0]:]
            suggested_data_str = "{ " + suggested_data_str.strip()[:-1] + " }"
            suggested_data_dict = eval(suggested_data_str)
            suggested_data_dict.update(suggested_data)
            suggested_data = suggested_data_dict

        file_data.seek(suggested_res.span()[1] + suggested_res_end.span()[0])
        file_data.write(str(suggested_data).replace("'", '"'))

        # Get the most frequent variables for suggestion
        variables = {
            "PUBLISHERS": publisher_list,
            "TAGS": tags_list,
            "TOPICS": topics_list
        }

        # count the occurrences of each variable
        count_vars = {}
        for key, value in variables.items():
            count_vars[key] = Counter(value)

        # get the most common variable for each category
        most_frequent = {
            "SUGGESTED_PUBLISHERS": count_vars["PUBLISHERS"].most_common(1)[0][0],
            "SUGGESTED_TAGS": count_vars["TAGS"].most_common(1)[0][0],
            "SUGGESTED_TOPICS": count_vars["TOPICS"].most_common(1)[0][0]
        }

        print("Most frequent variables for suggestion:")
        print(most_frequent)

data = TrainData(VIDEO_PUBLISHER="PUBLISHER", VIDEO_TAGS="hELLO", VIDEO_TOPIC="SCIENCE")
ModelTrainer.trainAVSdata(data, uuid.uuid4())