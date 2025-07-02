# Time-Series Features Bug Fix

## Problem
The original code had a bug in the `add_time_series_features` method where it tried to calculate time-series features (slopes and FFT) on individual scalar values instead of complete time series. This happened because the method was called before the data was aggregated by sequence.

## Root Cause
The pipeline flow was:
1. `load_and_preprocess` - loads raw data (each row = single timestamp)
2. `add_time_series_features` - **BUG HERE** - tried to apply `map_elements` to scalar values
3. `aggregate_sequence_features` - groups by sequence_id to create one row per sequence

The `map_elements` function expected arrays/lists of time series data, but received individual float values.

## Solution
1. **Removed** the problematic `add_time_series_features` method
2. **Integrated** time-series feature calculation into `aggregate_sequence_features` method
3. **Added** proper time-series features during the aggregation step when we have access to complete sequences:
   - **Slopes**: Linear regression slopes for accelerometer and rotation sensors (acc_x, acc_y, acc_z, rot_x, rot_y, rot_z)
   - **Ranges**: Max - min values as trend indicators
   - **FFT Features**: Mean and standard deviation of first 5 FFT components for accelerometer data

## Implementation Details
- Time-series features are now calculated correctly using `pl.col(col).map_elements()` within the group_by context
- Added proper error handling (minimum sequence length checks)
- Consistent implementation in both training pipeline and API inference code
- Added return type specifications for better type safety

## Features Added
- `{sensor}_{axis}_slope`: Linear regression slope for IMU sensors
- `{sensor}_{axis}_range`: Range (max-min) for trend detection  
- `acc_{axis}_fft_mean`: Mean of first 5 FFT components
- `acc_{axis}_fft_std`: Standard deviation of first 5 FFT components

## Benefits
- **No more skip messages**: Time-series features are now properly calculated
- **Better model performance**: More informative features for sequence classification
- **Consistent pipeline**: Same feature engineering in training and inference
- **Robust implementation**: Proper handling of edge cases and minimum sequence lengths

## Testing
The fix has been applied to both the main training pipeline and the API inference code to ensure consistency between training and production environments. 