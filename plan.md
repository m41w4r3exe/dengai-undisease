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

## Friday
8 PM: First draft ready

## Saturday
3 PM: presentation