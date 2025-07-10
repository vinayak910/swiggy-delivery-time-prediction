import numpy as np 
import pandas as pd 


def change_column_names(data: pd.DataFrame):
    return (
        data.rename(str.lower,axis=1)
        .rename({
            "delivery_person_id" : "rider_id",
            "delivery_person_age": "age",
            "delivery_person_ratings": "ratings",
            "delivery_location_latitude": "delivery_latitude",
            "delivery_location_longitude": "delivery_longitude",
            "time_orderd": "order_time",
            "time_order_picked": "order_picked_time",
            "weatherconditions": "weather",
            "road_traffic_density": "traffic",
            "city": "city_type",
            "time_taken(min)": "time_taken"},axis=1)
    )

def time_of_day(ser):

    return(

        pd.cut(ser, bins = [0 , 6 , 12 , 17 , 20 , 24],
               right = True , 
               labels = ['after_midnight', 'morning' , 'afternoon' , 'evening', 'night'])


    )

def data_cleaning(data: pd.DataFrame):
    minor_index = data.loc[data['age'].astype(float)<18].index.tolist()
    six_star_index = data.loc[data['ratings'] == '6'].index.tolist()

    return (
        data
        .drop(columns= 'id')
        .drop(index = minor_index)
        .drop(index = six_star_index)
        .replace("NaN " , np.nan)
        .assign(
            city_name = lambda x: x['rider_id'].str.split('RES').str.get(0),

            age = lambda x : x['age'].astype(float),

            ratings = lambda x : x['ratings'].astype(float), 

            restaurant_latitude = lambda x : (x['restaurant_latitude'].abs()),

            restaurant_longitude = lambda x : x['restaurant_longitude'].abs(),

            delivery_latitude = lambda x : x['delivery_latitude'].abs(),

            delivery_longitude = lambda x : x['delivery_longitude'].abs(),

            order_date = lambda x: pd.to_datetime(x['order_date'], dayfirst = True),

            order_day = lambda x : x['order_date'].dt.day, 

            order_month = lambda x: x['order_date'].dt.month, 

            order_day_of_week = lambda x: x['order_date'].dt.day_name().str.lower(),

            is_weekend = lambda x : x['order_date'].dt.day_name().isin(['Saturday',"Sunday"]).astype(int),

            order_time = lambda x : pd.to_datetime(x['order_time'],format = 'mixed'),

            order_picked_time = lambda x : pd.to_datetime(x['order_picked_time'], format = 'mixed'),

            pickup_time_minutes = lambda x: (
                (x['order_picked_time'] - x['order_time']).dt.seconds/60
            ),

            order_time_hour = lambda x : x['order_time'].dt.hour,

            order_time_of_day = lambda x: x['order_time_hour'].pipe(time_of_day),

            weather = lambda x: (
                x['weather']
                .str.replace("conditions ", "")
                .str.lower()
                .replace("nan",np.nan)
            ),

            traffic = lambda x : x['traffic'].str.rstrip().str.lower(),

            type_of_order = lambda x: x['type_of_order'].str.rstrip().str.lower(),

            type_of_vehicle = lambda x: x['type_of_vehicle'].str.rstrip().str.lower(),

            festival = lambda x: x['festival'].str.rstrip().str.lower(),

            city_type = lambda x : x['city_type'].str.rstrip().str.lower(),

            multiple_deliveries = lambda x: x['multiple_deliveries'].astype(float),

            time_taken = lambda x : (x['time_taken']
                                     .str.replace("(min) ", "")
                                     .astype(int)
                                     )
        ).drop(columns = ['order_time', 'order_picked_time'])

    )

def clean_lat_long(data: pd.DataFrame,threshold = 1):
    location_cols = ['restaurant_latitude',
                    'restaurant_longitude',
                    'delivery_latitude',
                    'delivery_longitude']

    return (
        data.assign(**{
            col:(
                np.where(data[col]<threshold,np.nan,data[col].values)
            )
            for col in location_cols
        })
    )


def extract_datetime_features(col):
    date_col = pd.to_datetime(col , dayfirst=True)

    return (

        pd.DataFrame(

        {
            "day":date_col.dt.day,
            "month": date_col.dt.month,
            "year":date_col.dt.year,
            "day_of_week":date_col.dt.day_name(),
            "is_weekend":date_col.dt.day_name().isin(["Saturday", "Sunday"]).astype(int)
        }
    )
    )


def calculate_haversine_distance(df):
    location_columns = ['restaurant_latitude',
                    'restaurant_longitude',
                    'delivery_latitude',
                    'delivery_longitude']
    lat1 = df[location_columns[0]]
    lon1 = df[location_columns[1]]
    lat2 = df[location_columns[2]]
    lon2 = df[location_columns[3]]

    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(
        dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    distance = 6371 * c

    return (
        df.assign(
            distance = distance)
    )


def create_distance_type(data: pd.DataFrame):
    return(
        data
        .assign(
                distance_type = pd.cut(data["distance"],bins=[0,5,10,15,25],
                                        right=False,labels=["short","medium","long","very_long"])
    ))

def perform_data_cleaning(data: pd.DataFrame, saved_data_path = "swiggy_cleaned.csv"):

    cleaned_data = (
        data
        .pipe(change_column_names)
        .pipe(data_cleaning)
        .pipe(clean_lat_long)
        .pipe(calculate_haversine_distance)
        .pipe(create_distance_type)
    )
    cleaned_data.to_csv(saved_data_path, index = False)



if __name__ == "__main__":

    DATA_PATH = "swiggy.csv"

    df = pd.read_csv(DATA_PATH)

    print("swiggy data loaded successfuly")


    perform_data_cleaning(df)
