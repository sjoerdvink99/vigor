def is_spatial(key):
    spatial_keywords = {
        'lat', 'lon', 'latitude', 'longitude', 'location', 'address', 
        'geolocation', 'coord', 'coordinates'
    }
    return any(keyword in key.lower() for keyword in spatial_keywords)


def is_temporal(key):
    temporal_keywords = {
        'time', 'date', 'timestamp', 'datetime', 'year', 'month', 'day', 
        'hour', 'minute', 'second', 'duration', 'epoch'
    }
    return any(keyword in key.lower() for keyword in temporal_keywords)