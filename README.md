# Statistical Downscaling
This repository holds the necessary codes and information to reproduce the statistical downscaling of precipitation data using Augmented ConvLSTM. *(Currently includes for India, will be extended to include US shortly)*

## File Structure
<ul>
    <li> <b>01_Setting_Up_conda</b>:  <i>(Follow these steps to create a similar conda environment as to what was used here. This will help in installing all the vital packages whilst avoiding any conflicts)</i>
    <li> <b>02_Raw_Data</b>: <i>After setting up, we get our Raw data in order (not uploaded here due to size constraints). The <b>README</b> there contains the information regarding the manual/automated fetching of these data. Codes for the ones automated has been provided.</i>
    <li> <b>03_Preprocess_Data</b>: Here, the data is clipped for India region for the time period 1948-2005. Then, it is regridded to match the IMD resolution. Finally, we create the stacked datasets for each GCM taking care about the time data type (including datetime64, 365Days, 360Days formats). To ensure compatibility, we drop some dates, including leap days and non-standard days (30-Feb in 360Days format).
    <li> <b>04_Testing_Stacks</b>: 

* Testing.ipynb: The trained weights are used to analyze the stacks to give us Y_hat and the RMSE plots.
* Datasets.ipynb: The results are exported in netCDF4 format.
<ul>
