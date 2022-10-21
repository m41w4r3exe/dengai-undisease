# Project
[DengAI: Predicting Disease Spread](https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/)

# Timeline
## Thursday
3 PM: project start
4 PM: Define what kind of problem we're dealing with
5 PM: Data cleaning
    - 4:10 PM columns analyzed to get sense of robustness
    - cleaning ideas
6 PM: Data visualization
- every overlapping feature (precipitation, temperature) plotted together to see diff
7 PM: Baseline models ready and scored

## Friday
- 11 AM: separate files
- 12 PM entire pipeline functional: model pipe, cross-validate, visualize
  - 12 PM: visualization function ready
  - 12 PM: cross-validation ready
- 2PM visualize every feature as a line chart with cities as colors
  - also visualization of effect of transformers
- make one model per city and one model for both
  - use warm_start parameter in regressor object to train model on two cities sequentially
- feature engineering (see feature engineering)
- create new pipeline (time-series forecasting)
- 8 PM: First draft ready

## Saturday
3 PM: presentation

# Observations
- Regression model is overfitting and flatlining: can be due to inertia of the mosquito growth
  - tweak parameters to have less overfitting
  - reduce dimensionality and increase time element 
  - must try time series prediction

- project improvements
  - separate files into a script
  - make function to make the pipeline: everytime the pipeline is tested for improvement we need the following to happen in encapsulated functions
    - pipeline
    - cross-validate
    - visualize
      - training preds vs y
      - test preds vs y
      - official preds


# Feature Engineering Ideas
- feature engineering/model tuning
  - ndvi data
    - make model for each city
    - try average of all ndvis
    - interpolation of the ndvi data over time to create higher def grid
  - generic data
    - filling null values
    - rescaling of data
    - everything 0 to 1
    - standardized data
    - one hot encoder for cities
    - dropping null values (for the baseline)
    - filling null values afterwards
    - overlapping data average as a way of filling data, and feature reduction
      - weighted average 
  - reanalysis (might be just recompute of forecast/real obs)
    - fuse similar data points
      - mean avg temp
      - max vs min temperature 
        - --> extract difference as a feature
        - compare that diff with diurn (tdtr)
      - relative vs specific humidity --> average
  - weather station (real observations)
  - dates
    - drop week start date (redundant with weekofyear)
    - drop all dates
    - encode dates as cos function
    - create seasons
    - create months feature
  - city encoding
    - try mean encoding
