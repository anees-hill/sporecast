<img src="https://github.com/anees-hill/sporecast/blob/main/images/spore_cast_logo_v1.png?raw=true" width="400" height="150">

# Introduction
**SporeCast** is a Python-based tool designed to forecast airborne concentrations of various outdoor fungal spore types, many of which are known triggers for asthma and allergic rhinitis. This robust pipeline leverages the power of Scikit-learn, imbalanced-learn, and TensorFlow, and is orchestrated by Luigi for AWS-based deployment. The work was presented at the Royal Meteorological Society's Atmospheric Science Conference 2023 ([link](https://www.atmosphericscienceconference.uk/21-march-2023)).

This ongoing project is an extension of my ongoing PhD work at the University of Leicester and only includes code (no data). The original work features model training that utilised the world’s longest continuous time-series of fungal spore monitoring data (up to 50 years), including more taxa than those in published studies.

## Major Features
- **Multi-decade Fungal Spore Time-Series Analysis**: Specifically designed to handle long-term fungal spore time-series data. Given the sensitivity of fungal growth and sporulation patterns to changes in climate and land use, we've implemented novel feature generation that leverages the Kats time-series package.

- **Imputation Procedures**: Custom imputation procedures designed to maintain the integrity of fungal spore seasonality trends.

- **Time-Series Aware Cross-Validation**: Cross-validation takes the temporal nature of the data into account, ensuring more reliable model performance.

- **CI/CD Equipped**: Built for continuous deployment, with scripts structured around the Luigi module. Includes prompts for passive hyperparameter tuning in each fungal group model.

Please note that this project is still under active development as time allows alongside my PhD commitments.

## Training pipeline:
<img src="https://github.com/anees-hill/sporecast/blob/main/images/sporecast_training_flowchart_v1_dark.png?raw=true" width="1650" height="600">

## Note
This project is part of my ongoing PhD research. The ideas and approaches presented here are the product of my original research and should not be used for publication elsewhere without explicit permission and appropriate attribution.
