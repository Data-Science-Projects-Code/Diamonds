import pandas as pd

print("Loading CSV file...")
df = pd.read_csv("../data/diamonds_raw.csv", index_col=[0])

print("Dropping unnecessary columns...")
df = df.drop(['date', 'diamond_id'], axis=1)

print("Renaming columns...")
df = df.rename(columns={'cut': 'cut_quality', 'shape': 'cut', 'size': 'carat_weight'})

print("Filling missing values...")
df = df.fillna({
    'color': 'unknown',
    'cut_quality': 'unknown',
    'eye_clean': 'unknown',
    'fancy_color_dominant_color': 'unknown', 
    'fancy_color_secondary_color': 'unknown',
    'fancy_color_overtone': 'unknown', 
    'fancy_color_intensity': 'unknown',  
    'girdle_min': 'unknown', 
    'girdle_max': 'unknown', 
    'culet_size': 'unknown', 
    'culet_condition': 'unknown',
    'fluor_color': 'unknown',  
    'fluor_intensity': 'unknown'
})

print("Selecting specific columns...")
df = df.loc[:, [
    'cut', 'color', 'clarity', 'carat_weight', 'cut_quality', 'lab', 'symmetry',
    'polish', 'eye_clean', 'culet_size', 'culet_condition', 'depth_percent',
    'table_percent', 'meas_length', 'meas_width', 'meas_depth', 'girdle_min', 'girdle_max', 
    'fluor_color', 'fluor_intensity', 'fancy_color_dominant_color',
    'fancy_color_secondary_color', 'fancy_color_overtone', 'fancy_color_intensity',
    'total_sales_price'
]]

print("Removing problematic cases...")
df = df.loc[lambda df_: ~df_["cut_quality"].str.contains("None")]

print("Saving to Parquet file...")
df.to_parquet("../data/diamonds.parquet")

print("Done!")

