import numpy as np
import xarray as xr

CH_CENTER = [46.818, 8.228]
CH_BOUNDING_BOX = [45.66, 47.87, 5.84, 10.98]

# Data extraction functions for ERA5


def rename_dimensions_variables(ds):
    """Rename dimensions and attributes of the given dataset to homogenize data."""
    if 'latitude' in ds.dims:
        ds = ds.rename({'latitude': 'lat'})
    if 'longitude' in ds.dims:
        ds = ds.rename({'longitude': 'lon'})

    return ds


def get_era5_data(files, start, end):
    """Extract ERA5 data for the given file(s) pattern/path."""
    print('Extracting data for the period {} - {}'.format(start, end))
    ds = xr.open_mfdataset(files, combine='by_coords')
    ds = rename_dimensions_variables(ds)
    ds = ds.sel(
        time=slice(start, end)
    )

    return ds


def extract_nearest_point_data(ds, lat, lon):
    """Return the time series data for the nearest grid point.

    Arguments:
        ds -- the dataset (xarray Dataset) to extract the data from
        lat -- the latitude coordinate of the point of interest
        lon -- the longitude coordinate of the point of interest

    Example:
    z = xr.open_mfdataset(DATADIR + '/ERA5/geopotential/*.nc', combine='by_coords')
    a = extract_nearest_point_data(z, CH_CENTER[0], CH_CENTER[1])
    """
    return ds.sel({'lat': lat, 'lon': lon}, method="nearest")


def extract_points_around(ds, lat, lon, step_lat, step_lon, nb_lat, nb_lon):
    """Return the time series data for a grid point mesh around the provided coordinates.

    Arguments:
    ds -- the dataset (xarray Dataset) to extract the data from
    lat -- the latitude coordinate of the center of the mesh
    lon -- the longitude coordinate of the center of the mesh
    step_lat -- the step in latitude of the mesh
    step_lon -- the step in longitude of the mesh
    nb_lat -- the total number of grid points to extract for the latitude axis (the mesh will be centered)
    nb_lon -- the total number of grid points to extract for the longitude axis (the mesh will be centered)

    Example:
    z = xr.open_mfdataset(DATADIR + '/ERA5/geopotential/*.nc', combine='by_coords')
    a = extract_points_around(z, CH_CENTER[0], CH_CENTER[1], step_lat=1, step_lon=1, nb_lat=3, nb_lon=3)
    """
    lats = np.arange(lat - step_lat * (nb_lat - 1) / 2,
                     lat + step_lat * nb_lat / 2, step_lat)
    lons = np.arange(lon - step_lon * (nb_lon - 1) / 2,
                     lon + step_lon * nb_lon / 2, step_lon)
    xx, yy = np.meshgrid(lats, lons)
    xx = xx.flatten()
    yy = yy.flatten()
    xys = np.column_stack((xx, yy))

    data = []
    for xy in xys:
        data.append(extract_nearest_point_data(ds, xy[0], xy[1]))

    return data


def get_data_mean_over_box(ds, lats, lons, level=0):
    """Extract data from points within a bounding box and process the mean.

    Arguments:
    ds -- the dataset (xarray Dataset) to extract the data from
    lats -- the min/max latitude coordinates of the bounding box
    lons -- the min/max longitude coordinates of the bounding box
    level -- the desired vertical level
    """
    if len(lats) != 2:
        raise Exception('An array of length 2 is expected for the lats.')
    if len(lons) != 2:
        raise Exception('An array of length 2 is expected for the lons.')

    lat_start = min(lats)
    lat_end = max(lats)

    if (ds.lat[0] > ds.lat[1]):
        lat_start = max(lats)
        lat_end = min(lats)

    if 'level' in ds.dims:
        ds_box = ds.sel(
            lat=slice(lat_start, lat_end), lon=slice(min(lons), max(lons)), level=level
        )
    else:
        ds_box = ds.sel(
            lat=slice(lat_start, lat_end), lon=slice(min(lons), max(lons))
        )

    return ds_box.mean(['lat', 'lon'])


def get_data_mean_over_CH_box(ds, level=0):
    """Extract data over the bounding box of Switzerland and return the mean time series.

    Arguments:
    level -- the desired vertical level
    """
    return get_data_mean_over_box(ds, [CH_BOUNDING_BOX[0], CH_BOUNDING_BOX[1]], [CH_BOUNDING_BOX[2], CH_BOUNDING_BOX[3]], level)
