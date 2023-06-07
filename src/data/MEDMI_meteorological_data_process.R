# FORECAST PROJECT - METEO PRE PROCESSING

library(tidyverse)
library(lubridate)
library(janitor)
library(geosphere)
library(circular)
library(dotenv)
dotenv()
source("config.R")


scaffold <- data.frame(date = seq.Date(from = dmy("01-01-1970"), to = dmy("31-12-2021"), by = 1))
midas_open_dat <- "meteo_consensus.R"
midas_open_dat <- midas_open_dat

# Aero sampler lat/lon
lei_coords <- c(52.62, -1.12)
der_coords <- c(52.93, -1.48)

# IMPORT & MERGE-----------------------------------------------
# PRECIPITATION (daily 09:00)
dat_precip1 <- read_csv("midas.rain_drnl_ob.prcp_amt_1.DER.01011970-31121990.csv")
dat_precip2 <- read_csv("midas.rain_drnl_ob.prcp_amt_1.DER.01011991-31122005.csv")
dat_precip3 <- read_csv("midas.rain_drnl_ob.prcp_amt_1.LEI.01012006-03112016.csv")
dat_precip4 <- read_csv("midas.rain_drnl_ob.prcp_amt_1.LEI.04112016-31122021.csv")
dat_precip <- bind_rows(dat_precip1, dat_precip2, dat_precip3, dat_precip4)
dat_precip <- clean_names(dat_precip)
dat_precip$variable <- "c_prcp_amt"
rm(dat_precip1, dat_precip2, dat_precip3, dat_precip4)

# AIR PRESSURE (hourly)
dat_ap1 <- read_csv("midas.weather_hrly_ob.msl_pressure.DER.01011970-31122005.csv")
dat_ap2 <- read_csv("midas.weather_hrly_ob.msl_pressure.LEI.01012006-31122021.csv")
dat_ap <- bind_rows(dat_ap1, dat_ap2)
dat_ap <- clean_names(dat_ap)
dat_ap$variable <- "h_airpress"
rm(dat_ap1, dat_ap2)

# DEW POINT (hourly)
dat_dp1 <- read_csv("midas.weather_hrly_ob.dewpoint.DER.01011970-31122005.csv")
dat_dp2 <- read_csv("midas.weather_hrly_ob.dewpoint.LEI.01012006-31122021.csv")
dat_dp <- bind_rows(dat_dp1, dat_dp2)
dat_dp <- clean_names(dat_dp)
dat_dp$variable <- "h_dewpoint"
rm(dat_dp1, dat_dp2)

# RH (hourly)
dat_rh1 <- read_csv("midas.weather_hrly_ob.rltv_hum.DER.01011970-31122005.csv")
dat_rh2 <- read_csv("midas.weather_hrly_ob.rltv_hum.LEI.01012006-31122021.csv")
dat_rh <- bind_rows(dat_rh1, dat_rh2)
dat_rh <- clean_names(dat_rh)
dat_rh$variable <- "h_relhum"
# pull missing data from wider search
dat_rh <- dat_rh %>% 
  mutate_at(.vars = "time", .funs = as.Date)
dat_rh_missing <- dat_rh %>% 
  select(time, value) %>% 
  mutate_at(.vars = "time", .funs = as.Date) %>% 
  right_join(scaffold, by = c("time" = "date")) %>% 
  filter(is.na(value)) %>% 
  select(time) %>%
  pull()
dat_rh_wide <- read_csv("midas.weather_hrly_ob.rltv_hum.WIDE.01011970-31122021.csv")
dat_rh_wide <- clean_names(dat_rh_wide)
dat_rh <- dat_rh_wide %>% 
  mutate(variable = "h_relhum") %>% 
  mutate_at(.vars = "time", .funs = as.Date) %>% 
  filter(time %in% dat_rh_missing) %>% 
  bind_rows(dat_rh)

rm(dat_rh1, dat_rh2, dat_rh_missing, dat_rh_wide)

# MAX TEMP (daily)
dat_mt1 <- read_csv("midas.temp_drnl_ob.max_air_temp_24.DER.01011970-31122005.csv")
dat_mt2 <- read_csv("midas.temp_drnl_ob.max_air_temp_24.LEI.01012006-31122021.csv")
dat_mt <- bind_rows(dat_mt1, dat_mt2)
dat_mt <- clean_names(dat_mt)
dat_mt$variable <- "c_max_air_temp"
# pull missing data from wider search
dat_mt <- dat_mt %>% 
  mutate_at(.vars = "time", .funs = as.Date)
dat_mt_missing <- dat_mt %>% 
  select(time, value) %>% 
  mutate_at(.vars = "time", .funs = as.Date) %>% 
  right_join(scaffold, by = c("time" = "date")) %>% 
  filter(is.na(value)) %>% 
  select(time) %>%
  pull()
dat_mt_wide <- read_csv("midas.temp_drnl_ob.max_air_temp_24.WIDE.01011970-31122021.csv")
dat_mt_wide <- clean_names(dat_mt_wide)
dat_mt <- dat_mt_wide %>% 
  mutate(variable = "c_max_air_temp") %>% 
  mutate_at(.vars = "time", .funs = as.Date) %>% 
  filter(time %in% dat_mt_missing) %>% 
  bind_rows(dat_mt)
# Get 1996 from MIDAS OPEN dataset
dat_mt_missing <- dat_mt %>% 
  select(time, value) %>% 
  right_join(scaffold, by = c("time" = "date")) %>% 
  filter(is.na(value)) %>% 
  select(time) %>%
  pull()
dat_mt_midasopen <- midas_open_dat %>% 
  select(date, c_max_air_temp) %>% 
  filter(date %in% dat_mt_missing) %>% 
  rename("value" = "c_max_air_temp", "time" = "date") %>% 
  mutate(site_name = "MIDAS_OPEN", variable = "c_max_air_temp")
dat_mt <- dat_mt %>% 
  bind_rows(dat_mt_midasopen)

rm(dat_mt1, dat_mt2, dat_mt_missing, dat_mt_wide, dat_mt_midasopen)

# MIN TEMP (daily)
dat_mint1 <- read_csv("midas.temp_drnl_ob.min_air_temp_24.DER.01011970-31122005.csv")
dat_mint2 <- read_csv("midas.temp_drnl_ob.min_air_temp_24.LEI.01012006-31122021.csv")
dat_mint <- bind_rows(dat_mint1, dat_mint2)
dat_mint <- clean_names(dat_mint)
dat_mint$variable <- "c_min_air_temp"
# pull missing data from wider search
dat_mint <- dat_mint %>% 
  mutate_at(.vars = "time", .funs = as.Date)
dat_mint_missing <- dat_mint %>% 
  select(time, value) %>% 
  mutate_at(.vars = "time", .funs = as.Date) %>% 
  right_join(scaffold, by = c("time" = "date")) %>% 
  filter(is.na(value)) %>% 
  select(time) %>%
  pull()
dat_mint_wide <- read_csv("midas.temp_drnl_ob.min_air_temp_24.WIDE.01011970-31122021.csv")
dat_mint_wide <- clean_names(dat_mint_wide)
dat_mint <- dat_mint_wide %>% 
  mutate(variable = "c_min_air_temp") %>% 
  mutate_at(.vars = "time", .funs = as.Date) %>% 
  filter(time %in% dat_mint_missing) %>% 
  bind_rows(dat_mint)
# Get 1996 from MIDAS OPEN dataset
dat_mint_missing <- dat_mint %>% 
  select(time, value) %>% 
  right_join(scaffold, by = c("time" = "date")) %>% 
  filter(is.na(value)) %>% 
  select(time) %>%
  pull()
dat_mint_midasopen <- midas_open_dat %>% 
  select(date, c_min_air_temp) %>% 
  filter(date %in% dat_mint_missing) %>% 
  rename("value" = "c_min_air_temp", "time" = "date") %>% 
  mutate(site_name = "MIDAS_OPEN", variable = "c_min_air_temp")
dat_mint <- dat_mint %>% 
  bind_rows(dat_mint_midasopen)

rm(dat_mint1, dat_mint2, dat_mint_missing, dat_mint_wide, dat_mint_midasopen)

# CLOUD TOTAL OKTAS (hourly)
dat_cc1 <- read_csv("midas.weather_hrly_ob.cld_ttl_amt_id.DER.01011970-31122005.csv")
dat_cc2 <- read_csv("midas.weather_hrly_ob.cld_ttl_amt_id.LEI.01012006-31122021.csv")
dat_cc <- bind_rows(dat_cc1, dat_cc2)
dat_cc <- clean_names(dat_cc)
dat_cc$variable <- "h_cloudcov"
rm(dat_cc1, dat_cc2)

# THUNDER FLAG (daily)
dat_tf1 <- read_csv("midas.weather_drnl_ob.thunder_day_flag_24.DER.01011970-31122005.csv")
dat_tf2 <- read_csv("midas.weather_drnl_ob.thunder_day_flag_24.LEI.01012006-31122021.csv")
dat_tf <- bind_rows(dat_tf1, dat_tf2)
dat_tf <- clean_names(dat_tf)
dat_tf$variable <- "c_thunder_day_flag"
# pull missing data from wider search
dat_tf <- dat_tf %>% 
  mutate_at(.vars = "time", .funs = as.Date)
dat_tf_missing <- dat_tf %>% 
  select(time, value) %>% 
  mutate_at(.vars = "time", .funs = as.Date) %>% 
  right_join(scaffold, by = c("time" = "date")) %>% 
  filter(is.na(value)) %>% 
  select(time) %>%
  pull()
dat_tf_wide <- read_csv("midas.weather_drnl_ob.thunder_day_flag_24.WIDE.01011970-31122021.csv")
dat_tf_wide <- clean_names(dat_tf_wide)
dat_tf <- dat_tf_wide %>% 
  mutate(variable = "c_thunder_day_flag") %>% 
  mutate_at(.vars = "time", .funs = as.Date) %>% 
  filter(time %in% dat_tf_missing) %>% 
  bind_rows(dat_tf)
dat_tf <- dat_tf %>% mutate(value = ifelse(value > 1, NA, value)) # fix > 1 values
rm(dat_tf1, dat_tf2, dat_tf_missing, dat_tf_wide)


# MEAN WIND SPEED (HOURLY)
dat_ws1 <- read_csv("midas.wind_mean_ob.mean_wind_speed_1.DER.01011970-31122005.csv")
dat_ws2 <- read_csv("midas.wind_mean_ob.mean_wind_speed_1.LEI.01012006-31122021.csv")
dat_ws <- bind_rows(dat_ws1, dat_ws2)
dat_ws <- clean_names(dat_ws)
dat_ws$variable <- "h_mean_wind_speed"
rm(dat_ws1, dat_ws2)

# WIND DIR (HOURLY)
dat_wd1 <- read_csv("midas.wind_mean_ob.mean_wind_dir_1.DER.01011970-31122005.csv")
dat_wd2 <- read_csv("midas.wind_mean_ob.mean_wind_dir_1.LEI.01012006-31122021.csv")
dat_wd <- bind_rows(dat_wd1, dat_wd2)
dat_wd <- clean_names(dat_wd)
dat_wd$variable <- "h_mean_wind_dir"
rm(dat_wd1, dat_wd2)

# GLOBAL IRRAD (hourly)
dat_gi1 <- read_csv("midas.radt_ob_v2.glbl_irad_amt_1.DER.01011970-31122005.csv")
dat_gi2 <- read_csv("midas.radt_ob_v2.glbl_irad_amt_1.LEI.01012006-31122021.csv")
dat_gi <- bind_rows(dat_gi1, dat_gi2)
dat_gi <- clean_names(dat_gi)
dat_gi$variable <- "h_global_irrad"
rm(dat_gi1, dat_gi2)

dat <- bind_rows(dat_precip, dat_ap, dat_dp, dat_rh, dat_mt, dat_mint, dat_cc, dat_tf, dat_ws, dat_wd, dat_gi)
rm(dat_precip, dat_ap, dat_dp, dat_rh, dat_mt, dat_mint, dat_cc, dat_tf, dat_ws, dat_wd, dat_gi)
gc()




# Extract time components
dat <- dat %>% 
  mutate(year = year(time), month = month(time), day = day(time))

# Separate data sets by processing stream
dat_hourly <- dat %>% 
  filter(variable %in% c("h_airpress", "h_dewpoint", "h_relhum", "h_cloudcov", "h_mean_wind_speed", "h_global_irrad"))

dat_daily <- dat %>% 
  anti_join(dat_hourly, by = "variable") %>% 
  filter(variable != "h_mean_wind_dir")

dat_circular <- dat %>% 
  filter(variable %in% c("h_mean_wind_dir"))

rm(dat); gc()

# Aggregate hourly to daily
dat_airpress_c <- dat_hourly %>% 
  filter(variable == "h_airpress") %>% 
  group_by(site_name, year, month, day) %>%
  summarise(c_max_airpress = max(value, na.rm = T),
            c_min_airpress = min(value, na.rm = T),
            c_avg_airpress = mean(value, na.rm = T),
            c_med_airpress = median(value, na.rm = T),
            latitude = latitude, longitude = longitude, altitude = altitude) %>% 
  pivot_longer(cols = starts_with("c_"), names_to = "variable", values_to = "value")

dat_dewpoint_c <- dat_hourly %>% 
  filter(variable == "h_dewpoint") %>% 
  group_by(site_name, year, month, day) %>%
  summarise(c_max_dewpoint = max(value, na.rm = T),
            c_min_dewpoint = min(value, na.rm = T),
            c_avg_dewpoint = mean(value, na.rm = T),
            c_med_dewpoint = median(value, na.rm = T),
            latitude = latitude, longitude = longitude, altitude = altitude) %>% 
  pivot_longer(cols = starts_with("c_"), names_to = "variable", values_to = "value")

dat_relhum_c <- dat_hourly %>% 
  filter(variable == "h_relhum") %>% 
  group_by(site_name, year, month, day) %>%
  summarise(c_max_relhum = max(value, na.rm = T),
            c_min_relhum = min(value, na.rm = T),
            c_avg_relhum = mean(value, na.rm = T),
            c_med_relhum = median(value, na.rm = T),
            latitude = latitude, longitude = longitude, altitude = altitude) %>% 
  pivot_longer(cols = starts_with("c_"), names_to = "variable", values_to = "value")

dat_cloudcov_c <- dat_hourly %>% 
  filter(variable == "h_cloudcov") %>% 
  group_by(site_name, year, month, day) %>%
  summarise(c_max_cloudcov = max(value, na.rm = T),
            c_min_cloudcov = min(value, na.rm = T),
            c_avg_cloudcov = mean(value, na.rm = T),
            c_med_cloudcov = median(value, na.rm = T),
            latitude = latitude, longitude = longitude, altitude = altitude) %>% 
  pivot_longer(cols = starts_with("c_"), names_to = "variable", values_to = "value")

dat_mean_wind_speed_c <- dat_hourly %>% 
  filter(variable == "h_mean_wind_speed") %>% 
  group_by(site_name, year, month, day) %>%
  summarise(c_max_mean_wind_speed = max(value, na.rm = T),
            c_min_mean_wind_speed = min(value, na.rm = T),
            c_avg_mean_wind_speed = mean(value, na.rm = T),
            c_med_mean_wind_speed = median(value, na.rm = T),
            latitude = latitude, longitude = longitude, altitude = altitude) %>% 
  pivot_longer(cols = starts_with("c_"), names_to = "variable", values_to = "value")

dat_global_irrad_c <- dat_hourly %>% 
  filter(variable == "h_global_irrad") %>% 
  group_by(site_name, year, month, day) %>%
  summarise(c_max_global_irrad = max(value, na.rm = T),
            c_min_global_irrad = min(value, na.rm = T),
            c_avg_global_irrad = mean(value, na.rm = T),
            c_med_global_irrad = median(value, na.rm = T),
            latitude = latitude, longitude = longitude, altitude = altitude) %>% 
  pivot_longer(cols = starts_with("c_"), names_to = "variable", values_to = "value")

rm(dat_hourly); gc()

# Process wind daily averages
dat_mean_wind_dir_c <- dat_circular %>% 
  filter(variable == "h_mean_wind_dir") %>% 
  mutate(value = ifelse(value > 360, NA, value)) %>% 
  group_by(site_name, year, month, day) %>%
  drop_na(value) %>%
  summarise(
    c_mean_wind_dir =
      value %>%
      circular(units = 'degrees', rotation = 'clock') %>%
      mean.circular()%%360,
    latitude = latitude, longitude = longitude, altitude = altitude) %>%
  ungroup() %>% 
  mutate(c_mean_wind_dir = as.double(c_mean_wind_dir)) %>% 
  rename("value" = "c_mean_wind_dir") %>% 
  mutate(variable = "c_mean_wind_dir") %>% 
  distinct()

# Merge ALL daily aggregated data
dat_hourly <- bind_rows(dat_airpress_c, dat_dewpoint_c, dat_relhum_c, dat_cloudcov_c, dat_mean_wind_speed_c, dat_global_irrad_c, dat_mean_wind_dir_c)
rm(dat_circular, dat_airpress_c, dat_dewpoint_c, dat_relhum_c, dat_cloudcov_c, dat_mean_wind_speed_c, dat_global_irrad_c, dat_mean_wind_dir_c); gc()
dat_daily <- dat_daily %>% 
  bind_rows(dat_hourly) %>% 
  select(-easting_grid_reference, -grid_reference_type, -northing_grid_reference, -postcode_sector, -site_identifier)
rm(dat_hourly); gc()


# Lat lon distance lookup table
latlon_tbl <- dat_daily %>% 
  select(site_name, latitude, longitude) %>% 
  distinct() %>% 
  rowwise() %>% 
  mutate(dist_lei = distm(c(longitude, latitude), c(lei_coords[2], lei_coords[1]), fun = distHaversine)) %>% 
  mutate(dist_der = distm(c(longitude, latitude), c(der_coords[2], der_coords[1]), fun = distHaversine)) %>% 
  select(-latitude, -longitude) %>% 
  mutate_at(.vars = c("dist_lei", "dist_der"), .funs = as.double)
# Attach to data
dat_daily <- dat_daily %>% 
  left_join(latlon_tbl, by = "site_name") %>% 
  mutate(date = paste0(day,"-",month,"-",year)) %>%
  mutate_at(.vars = "date", .funs = dmy) %>% 
  select(-time)

# IMPORT MISSING VARIABLES FROM MIDAS-OPEN DATASET (Note: variables with missing data imports have already been performed above) -----------
dat_daily <- midas_open_dat %>% 
  select(date, c_cs_24hr_sun_dur, c_max_gust_dir) %>%
  pivot_longer(cols = c("c_cs_24hr_sun_dur", "c_max_gust_dir"), names_to = "variable", values_to = "value") %>% 
  bind_rows(dat_daily)
  
# CREATE TEMPERATURE AVERAGE
dat_daily <- dat_daily %>%
  filter(variable %in% c("c_max_air_temp", "c_min_air_temp")) %>% 
  group_by(site_name, date) %>% 
  summarise(value = mean(value),
            variable = "c_avg_air_temp", altitude = altitude, latitude = latitude, longitude = longitude,
            year = year, month = month, day = day, dist_lei = dist_lei, dist_der = dist_der) %>% 
  distinct() %>% 
  bind_rows(dat_daily) %>% 
  distinct()


# CREATE DATASETS ================================================
# (1) MEDMI_nearest_station  --------------------------------------
# # FILTER DATA DAILY by NEAREST LOCATION (TOO SLOW WITH WIDER DATASET - too many locations)
# Note: package dtplyr used to hasten process
library(dtplyr)
library(data.table)
dat_daily_lazy <- lazy_dt(dat_daily)

dat_sliced_der <- dat_daily_lazy %>%
  ungroup() %>% 
  filter(date < dmy("01-01-2006")) %>% 
  rename("dist_sampler" = "dist_der") %>%
  group_by(variable, date) %>% 
  slice_min(dist_sampler, n = 1) %>% 
  as_tibble()
dat_sliced_lei <- dat_daily_lazy %>%
  ungroup() %>% 
  filter(date >= dmy("01-01-2006")) %>% 
  rename("dist_sampler" = "dist_lei") %>%
  group_by(variable, date) %>% 
  slice_min(dist_sampler, n = 1) %>% 
  as_tibble()

dat_sliced <- dat_sliced_der %>% 
  bind_rows(dat_sliced_lei) %>% 
  mutate(dataset = "MEDMI_nearest_station")

# (2) MEDMI_mean_averaged  --------------------------------------
# AVERAGE ACROSS STATIONS (non-weighted)
dat_mean_cont <- dat_daily %>% 
  # Filter to applicable variables
  filter(variable != "c_mean_wind_dir", variable != "c_max_gust_dir") %>% 
  group_by(variable, date) %>% 
  summarise(value = mean(value, na.rm = T))

dat_mean_thunder <- dat_mean_cont %>% 
  filter(variable == "c_thunder_day_flag") %>% 
  group_by(date) %>% 
  mutate(value = ifelse(value > 0, 1, 0))

dat_mean_wind_dir <- dat_daily %>% 
  filter(variable == "c_mean_wind_dir") %>% 
  group_by(date) %>%
  drop_na(value) %>%
  summarise(
    value =
      value %>%
      circular(units = 'degrees', rotation = 'clock') %>%
      mean.circular()%%360,
    variable = "c_mean_wind_dir") %>% 
  ungroup() %>% 
  mutate(value = as.double(value)) %>% 
  distinct()

dat_max_gust_dir <- dat_daily %>% 
  filter(variable == "c_max_gust_dir")

dat_mean <- dat_mean_cont %>% 
  filter(variable != "c_thunder_day_flag") %>% 
  bind_rows(dat_mean_thunder, dat_mean_wind_dir, dat_max_gust_dir) %>% 
  mutate(dataset = "MEDMI_30km_mean")

# Integrate number of station per observation col
station_count_tbl <- dat_daily %>% 
  group_by(variable, date) %>% 
  summarise(n_stations = n())
dat_mean <- dat_mean %>% 
  left_join(station_count_tbl, by = c("variable", "date"))
  

rm(dat_mean_cont, dat_mean_thunder); gc()

# (3) MEDMI_median_averaged (except wind dir (mean))  --------------------------------------
# AVERAGE ACROSS STATIONS (non-weighted)
dat_median_cont <- dat_daily %>% 
  # Filter to applicable variables
  filter(variable != "c_mean_wind_dir") %>% 
  group_by(variable, date) %>% 
  summarise(value = median(value, na.rm = T))

# dat_median_thunder <- dat_median_cont %>% 
#   filter(variable == "c_thunder_day_flag") %>% 
#   group_by(date) %>% 
#   mutate(value = ifelse(value > 0, 1, 0))

dat_median <- dat_median_cont %>% 
  bind_rows(dat_mean_wind_dir, dat_max_gust_dir) %>% 
  mutate(dataset = "MEDMI_30km_median")
dat_median <- dat_median %>% 
  left_join(station_count_tbl, by = c("variable", "date"))

rm(dat_median_cont, dat_mean_wind_dir); gc()

# Combine all datasets
dat <- bind_rows(dat_sliced, dat_mean, dat_median)
rm(dat_sliced, dat_mean, dat_median); gc()


beepr::beep()

nearest_station_avail <- c("c_avg_air_temp","c_avg_airpress","c_avg_cloudcov","c_avg_dewpoint","c_avg_global_irrad",
                           "c_avg_mean_wind_speed","c_avg_relhum","c_max_air_temp","c_max_airpress",
                           "c_max_cloudcov","c_max_dewpoint","c_max_global_irrad","c_max_mean_wind_speed","c_max_relhum",
                           "c_mean_wind_dir","c_med_airpress","c_med_cloudcov","c_med_dewpoint","c_med_global_irrad",
                           "c_med_mean_wind_speed","c_med_relhum","c_min_air_temp","c_min_airpress","c_min_cloudcov",
                           "c_min_dewpoint","c_min_global_irrad","c_min_mean_wind_speed","c_min_relhum","c_prcp_amt",
                           "c_thunder_day_flag")

# Get n_samplers across years
stations_month <- dat %>%
  filter(variable %in% c("c_avg_air_temp")) %>% 
  filter(dataset == "MEDMI_30km_mean") %>% 
  mutate(month = month(date), year = year(date)) %>% 
  group_by(variable, year, month) %>% 
  summarise(n_stations = max(n_stations))

stations_year <- dat %>%
  filter(variable %in% c("c_avg_air_temp")) %>% 
  filter(dataset == "MEDMI_30km_mean") %>% 
  mutate(year = year(date)) %>% 
  group_by(variable, year) %>% 
  summarise(n_stations = max(n_stations))


write_csv(stations_month, "n_stations_month.csv")
write_csv(stations_year, "n_stations_year.csv")


# Visualise output data
vis_output <- function(var_selection, dataset_selection){
  print(paste0("Plotting ", var_selection))
  if((dataset_selection == "MEDMI_nearest_station") & (var_selection %in% nearest_station_avail)){
    plot <- dat %>% 
      filter(variable == var_selection, dataset == dataset_selection) %>% 
    ggplot(aes(date, value, colour = dist_sampler, group = 1)) +
      geom_line() +
      geom_vline(xintercept = dmy("01-01-2006"), alpha = 0.3, colour = "red") +
      theme_minimal() +
      labs(title = var_selection)
  } else if((dataset_selection != "MEDMI_nearest_station") & (var_selection %in% nearest_station_avail)){
    plot <- dat %>% 
      filter(variable == var_selection, dataset == dataset_selection) %>% 
      ggplot(aes(date, value, colour = n_stations, group = 1)) +
      geom_line() +
      geom_vline(xintercept = dmy("01-01-2006"), alpha = 0.3, colour = "red") +
      theme_minimal() +
      scale_color_viridis_c() +
      labs(title = var_selection)
    # For imported variables
  } else if(var_selection %in% c("c_cs_24hr_sun_dur", "dat_max_gust_dir")){
    plot <- dat %>% 
      filter(variable == var_selection, dataset == dataset_selection) %>% 
      ggplot(aes(date, value)) +
      geom_line() +
      geom_vline(xintercept = dmy("01-01-2006"), alpha = 0.3, colour = "red") +
      theme_minimal() +
      labs(title = var_selection)
  }
  return(plot)
}

plots1 <- map(soun(dat$variable), vis_output,
             dataset_selection = "MEDMI_nearest_station")
plots2 <- map(soun(dat$variable), vis_output,
              dataset_selection = "MEDMI_30km_mean")
plots3 <- map(soun(dat$variable), vis_output,
              dataset_selection = "MEDMI_30km_median")


beepr::beep(); Sys.sleep(0.25); beepr::beep()

dat_nearest_wide <- dat %>% 
  filter(dataset == "MEDMI_nearest_station") %>% 
  select(date, variable, value) %>% 
  na.omit() %>% 
  pivot_wider(names_from = "variable", values_from = "value") %>% 
  select(-c_avg_global_irrad, -c_max_global_irrad, c_min_global_irrad, c_thunder_day_flag)

dat_mean_wide <- dat %>% 
  filter(dataset == "MEDMI_30km_mean") %>% 
  select(date, variable, value) %>% 
  pivot_wider(names_from = "variable", values_from = "value") %>% 
  select(-c_avg_global_irrad, -c_max_global_irrad, c_min_global_irrad, c_thunder_day_flag)

dat_median_wide <- dat %>% 
  filter(dataset == "MEDMI_30km_median") %>% 
  select(date, variable, value) %>% 
  pivot_wider(names_from = "variable", values_from = "value") %>% 
  select(-c_avg_global_irrad, -c_max_global_irrad, c_min_global_irrad, c_thunder_day_flag)

midas_nearest_station <- "meteo_midasfull_nearestStation.csv"
midas_30km_mean <- "meteo_midasfull_30kmMean.csv"
midas_30km_median <- "meteo_midasfull_30kmMedian"

write_csv(dat_nearest_wide, midas_nearest_station)

write_csv(dat_mean_wide, midas_30km_mean)

write_csv(dat_median_wide, midas_30km_median)
