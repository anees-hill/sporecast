# FORECAST PROJECT - forecast_retro_meteo_preprocess. METEO PRE-PROCESSING

# SOURCE SECTION # =====================================================
library(DBI)
library(Hmisc)
library(lubridate)
library(circular)
library(tidyverse)
library(dotenv)
dotenv()
postgres_details <- Sys.getenv("POSTGRES_DETAILS")

# CONNECT TO DB
con <- dbConnect(odbc::odbc(), .connection_string = postgres_details, 
                 timeout = 10)
# Extract data ---------------------------------------------------------------------
# Derby Central
db_cen <- dbGetQuery(con, "
SELECT st.src_id, st.observation_station, st.distance_der_cen AS station_distance_km, st.angle_der_cen AS station_angle_deg,
  CASE WHEN st.distance_der_cen >=30 THEN 'extra'
  WHEN st.distance_der_cen <30 THEN 'intra' END
  AS distance_group_30km,
  temp.date, temp.max_air_temp, temp.min_air_temp, temp.min_grss_temp, temp.min_conc_temp,
  rain.prcp_amt, wea.end_period AS weather_period, wea.cs_24hr_sun_dur, wea.wmo_24hr_sun_dur, wea.conc_state_id,
  wea.lying_snow_flag, wea.snow_depth, wea.frsh_snw_amt, wea.snow_day_id, wea.hail_day_id, wea.thunder_day_flag,
  wea.gale_day_flag, wea.frsh_mnt_snwfall_flag, rad.glbl_irad_amt, rad.difu_irad_amt, rad.direct_irad, rad.irad_bal_amt, rad.glbl_s_lat_irad_amt,
  rad.glbl_horz_ilmn,
  EXTRACT (YEAR FROM temp.date) AS YEAR, EXTRACT (MONTH FROM temp.date) AS MONTH, EXTRACT (DAY FROM temp.date) AS DAY 
FROM stations AS st
  FULL JOIN daily_temperature AS temp
  USING (src_id)
  FULL JOIN daily_rainfall AS rain
  ON temp.date = rain.date AND temp.src_id = rain.src_id
  FULL JOIN daily_weather AS wea
  ON temp.date = wea.date AND temp.src_id = wea.src_id
  FULL JOIN daily_radiation AS rad
  ON temp.date = rad.date AND temp.src_id = rad.src_id
WHERE (wea.end_period = 'pm' AND st.distance_der_cen <= 60 AND temp.date BETWEEN '19700101' AND '19891231')  
ORDER BY st.distance_der_cen, temp.date;
                     ")

db_mik <- dbGetQuery(con, "
  SELECT st.src_id, st.observation_station, st.distance_der_mik AS station_distance_km, st.angle_der_cen AS station_angle_deg,
  CASE WHEN st.distance_der_mik >=30 THEN 'extra'
  WHEN st.distance_der_mik <30 THEN 'intra' END
  AS distance_group_30km,
  temp.date, temp.max_air_temp, temp.min_air_temp, temp.min_grss_temp, temp.min_conc_temp,
  rain.prcp_amt, wea.end_period AS weather_period, wea.cs_24hr_sun_dur, wea.wmo_24hr_sun_dur, wea.conc_state_id,
  wea.lying_snow_flag, wea.snow_depth, wea.frsh_snw_amt, wea.snow_day_id, wea.hail_day_id, wea.thunder_day_flag,
  wea.gale_day_flag, wea.frsh_mnt_snwfall_flag, rad.glbl_irad_amt, rad.difu_irad_amt, rad.direct_irad, rad.irad_bal_amt, rad.glbl_s_lat_irad_amt,
  rad.glbl_horz_ilmn,
  EXTRACT (YEAR FROM temp.date) AS YEAR, EXTRACT (MONTH FROM temp.date) AS MONTH, EXTRACT (DAY FROM temp.date) AS DAY 
  FROM stations AS st
  FULL JOIN daily_temperature AS temp
  USING (src_id)
  FULL JOIN daily_rainfall AS rain
  ON temp.date = rain.date AND temp.src_id = rain.src_id
  FULL JOIN daily_weather AS wea
  ON temp.date = wea.date AND temp.src_id = wea.src_id
  FULL JOIN daily_radiation AS rad
  ON temp.date = rad.date AND temp.src_id = rad.src_id
  WHERE (wea.end_period = 'pm' AND st.distance_der_mik <= 60 AND temp.date BETWEEN '19900101' AND '20051231')  
  ORDER BY st.distance_der_mik, temp.date;
                     ")

db_lei <- dbGetQuery(con, "
SELECT st.src_id, st.observation_station, st.distance_lei AS station_distance_km, st.angle_der_cen AS station_angle_deg,
  CASE WHEN st.distance_lei >=30 THEN 'extra'
  WHEN st.distance_lei <30 THEN 'intra' END
  AS distance_group_30km,
  temp.date, temp.max_air_temp, temp.min_air_temp, temp.min_grss_temp, temp.min_conc_temp,
  rain.prcp_amt, wea.end_period AS weather_period, wea.cs_24hr_sun_dur, wea.wmo_24hr_sun_dur, wea.conc_state_id,
  wea.lying_snow_flag, wea.snow_depth, wea.frsh_snw_amt, wea.snow_day_id, wea.hail_day_id, wea.thunder_day_flag,
  wea.gale_day_flag, wea.frsh_mnt_snwfall_flag, rad.glbl_irad_amt, rad.difu_irad_amt, rad.direct_irad, rad.irad_bal_amt, rad.glbl_s_lat_irad_amt,
  rad.glbl_horz_ilmn,
  EXTRACT (YEAR FROM temp.date) AS YEAR, EXTRACT (MONTH FROM temp.date) AS MONTH, EXTRACT (DAY FROM temp.date) AS DAY 
FROM stations AS st
  FULL JOIN daily_temperature AS temp
  USING (src_id)
  FULL JOIN daily_rainfall AS rain
  ON temp.date = rain.date AND temp.src_id = rain.src_id
  FULL JOIN daily_weather AS wea
  ON temp.date = wea.date AND temp.src_id = wea.src_id
  FULL JOIN daily_radiation AS rad
  ON temp.date = rad.date AND temp.src_id = rad.src_id
WHERE (wea.end_period = 'pm' AND temp.date BETWEEN '20060101' AND '20211231')  
ORDER BY st.distance_lei, temp.date;
                     ")

# Wind direction data - incorporate all (only one station has long-term wind data)
db_wind <- dbGetQuery(con, "
SELECT date, src_id, mean_wind_dir, max_gust_dir, mean_wind_speed, max_gust_speed
  FROM mean_wind
  WHERE mean_wind_dir NOTNULL OR max_gust_dir NOTNULL;
                     ")

# Tidy data --------------------------------------------------------------------
db_cen$sampling_site <- "derby_cen"
db_mik$sampling_site <- "derby_mik"
db_lei$sampling_site <- "leicester"

db <- db_cen %>% 
  bind_rows(db_mik) %>% 
  bind_rows(db_lei)

# Generate consensus data cols --------------------------------------------------
# Generate guide
db_c <- db %>% 
  filter(distance_group_30km == "intra") %>% 
  group_by(year) %>% 
  distinct(observation_station) %>% 
  arrange(year)
intra_weather_stations <- as.data.frame(table(db_c$year)); colnames(intra_weather_stations) <- c("year","intra")

db_c <- db %>% 
  filter(distance_group_30km == "extra") %>% 
  group_by(year) %>% 
  distinct(observation_station) %>% 
  arrange(year)
extra_weather_stations <- as.data.frame(table(db_c$year)); colnames(extra_weather_stations) <- c("year","extra")

weather_stations <- intra_weather_stations %>% 
  full_join(extra_weather_stations, by = "year") %>% 
  mutate(opt_in_or_ext = ifelse((intra < 2)|is.na(intra), "extra","intra"))

# function to incorporate weather stations based on generated df above
years_vec <- sort(unique(weather_stations$year))


pull_indv_weather_station_data <- function(year_selection){
  category <- weather_stations[which(weather_stations$year==year_selection),"opt_in_or_ext"]
  filtered_db <- db %>% 
    filter(year == year_selection) %>% 
    filter(distance_group_30km == category)
  return(filtered_db)
}

db_map <- map_df(years_vec, pull_indv_weather_station_data)

db_c <- db_map %>%
  group_by(date) %>%
  summarise(c_max_air_temp = max(max_air_temp, na.rm=T),
            c_min_air_temp = min(min_air_temp, na.rm=T),
            c_min_grss_temp = min(min_grss_temp, na.rm=T),
            c_min_conc_temp = min(min_conc_temp, na.rm=T),
            c_prcp_amt = mean(prcp_amt, na.rm=T),
            c_cs_24hr_sun_dur = mean(cs_24hr_sun_dur, na.rm=T),
            c_wmo_24hr_sun_dur = mean(wmo_24hr_sun_dur, na.rm=T),
            c_thunder_day_flag = ifelse(any(thunder_day_flag>0),1,0),
            c_gale_day_flag =  ifelse(any(gale_day_flag>0),1,0),
            c_glbl_irad_amt = mean(glbl_irad_amt, na.rm=T),
            c_difu_irad_amt = mean(difu_irad_amt, na.rm=T),
            c_direct_irad = mean(direct_irad, na.rm=T),
            c_irad_bal_amt = mean(irad_bal_amt, na.rm=T),
            c_glbl_s_lat_irad_amt = mean(glbl_s_lat_irad_amt, na.rm=T),
            c_glbl_horz_ilmn = mean(glbl_horz_ilmn, na.rm=T))
            
db_c[sapply(db_c, is.infinite)] <- NA

# Process wind --------------------------------------------------------

# Correct NaNs for NAs (in max_gust_speed col)
db_wind <- db_wind %>% 
  mutate(max_gust_speed = ifelse(max_gust_speed == "NaN", NA, max_gust_speed))

# wind direction - circular means!
# Aggregate hourly observations to 24hr
db_wind_mean <- db_wind %>% 
  select(date, mean_wind_dir) %>% 
  mutate(mean_wind_dir = ifelse(mean_wind_dir > 360, NA, mean_wind_dir)) %>% 
  # mean of 'mean_wind_dir'
  group_by(date) %>%
  drop_na(mean_wind_dir) %>%
  summarise(
    c_mean_wind_dir =
      mean_wind_dir %>%
      circular(units = 'degrees', rotation = 'clock') %>%
      mean.circular()%%360) %>%
  ungroup() %>% 
  mutate(c_mean_wind_dir = as.double(c_mean_wind_dir))

db_wind_max <- db_wind %>% 
  select(date, max_gust_dir) %>% 
  mutate(max_gust_dir = ifelse(max_gust_dir > 360, NA, max_gust_dir)) %>% 
  # mean of 'mean_wind_dir'
  group_by(date) %>%
  drop_na(max_gust_dir) %>%
  summarise(
    c_max_gust_dir =
      max_gust_dir %>%
      circular(units = 'degrees', rotation = 'clock') %>%
      mean.circular()%%360) %>%
  ungroup() %>% 
  mutate(c_max_gust_dir = as.double(c_max_gust_dir))

db_wind_speed <- db_wind %>% 
  select(date, mean_wind_speed, max_gust_speed) %>% 
  group_by(date) %>% 
  summarise(c_mean_wind_speed = mean(mean_wind_speed, na.rm = T),
            c_max_gust_speed = max(max_gust_speed, na.rm = T))
db_wind_speed[sapply(db_wind_speed, is.infinite)] <- NA


db_c <- db_c %>% 
  full_join(db_wind_mean, by = "date") %>% 
  full_join(db_wind_max, by = "date") %>% 
  full_join(db_wind_speed, by = "date")

final_stations <- db_c %>%
  mutate(year = year(date)) %>% 
  group_by(year) %>% 
  distinct(observation_station) %>% 
  arrange(year)
table(final_stations$year) # shows coverage of weather stations for each year

# Check for ANY infinite values
map(colnames(db_c), ~{any(is.infinite(pluck(db_c, .)))})

# Export ------------------------------------------


source("config.R")
site_meteo_file <- "meteo_all.R"
consensus_meteo_file <- "meteo_consensus.R"
saveRDS(db, site_meteo_file)
saveRDS(db_c, consensus_meteo_file)
