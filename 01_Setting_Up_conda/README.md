# conda_tf

## Windows (For Initial Data Fetching and check)
* conda create -n conda_tf python=3.7.12 matplotlib -c conda-forge
* conda activate conda_tf
* conda install tensorflow=2.1.0 -c conda-forge
* conda install tensorflow-gpu -c conda-forge
* conda install ipykernel
* conda install scikit-learn seaborn networkx xarray
* conda install netcdf4 -c conda-forge
* conda install -c iamsaswata imdlib
* conda install -c conda-forge elevation
* conda install -c conda-forge esgf-pyclient
* conda install geopandas gdal rasterio basemap -c conda-forge
* conda install cartopy -c conda-forge
* conda install dask

<hr>

## Linux (For Model Testing)
* conda create -n conda_tf tensorflow=2.7.0 tensorflow-gpu geopandas gdal rasterio basemap -c conda-forge
* conda install ipykernel
* conda install xarray seaborn dask
* conda install netcdf4 cartopy -c conda-forge

# conda_esmf

## Linux (For Regridding and Stacking)
* conda create -n conda_esmf python=3.7 xesmf esmpy=8.0.0 -c conda-forge
* conda activate conda_esmf
* conda install geopandas gdal rasterio basemap -c conda-forge
* conda install ipykernel
* conda install dask seaborn
* conda install netcf4 cartopy -c conda-forge