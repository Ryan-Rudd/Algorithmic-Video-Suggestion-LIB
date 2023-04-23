# Configuration file for the video suggestion library

# Path to the raw data file
RAW_DATA_FILEPATH = 'data/raw_data/video_data.csv'

# Path to the processed data file
PROCESSED_DATA_FILEPATH = 'data/processed_data/video_data_processed.csv'

# Default number of recommendations to generate
DEFAULT_N_RECOMMENDATIONS = 10

# Default parameters for collaborative filtering
DEFAULT_N_FACTORS = 20
DEFAULT_N_EPOCHS = 10
DEFAULT_CF_LR = 0.01
DEFAULT_CF_REG = 0.01

# Default parameter for hybrid approach
DEFAULT_ALPHA = 0.5
