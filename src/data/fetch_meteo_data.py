#%%
import pandas as pd
import cdsapi

c = cdsapi.Client()

#%%
c.retrieve(
    'reanalysis-era5-land',
    {
        'variable': [
            '2m_temperature', 'total_precipitation',
        ],
        'year': [
            '1970', '1971', '1972',
        ],
        'month': [
            '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12',
        ],
        'day': [
            '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13',
            '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26',
            '27', '28', '29', '30', '31',
        ],
        'time': '12:00',  # Request daily aggregated data
        'area': [
            52.7, -1.1, 52.6, -1.0,  # North, West, South, East - approximate bounding box for Leicester
        ],
        'format': 'netcdf',
    },
    'daily_weather_data.nc')
