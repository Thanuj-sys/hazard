"""Interpret source tags in HOT/OSM exposure exports without inventing risk labels."""
EXPOSURE_FEATURE_LIMIT = 5000
SUPPORTED_KINDS = ('population','building','hospital','clinic','healthcare','school','college','road','shelter','infrastructure')


def infer_feature_kind(properties):
    explicit = properties.get('kind')
    if explicit is not None:
        if not isinstance(explicit,str) or explicit not in SUPPORTED_KINDS:
            raise ValueError(f'Unsupported feature kind: {explicit}')
        return explicit
    def tag(name):
        value = properties.get(name)
        return value.strip().lower() if isinstance(value,str) else ''
    amenity,healthcare,building = tag('amenity'),tag('healthcare'),tag('building')
    if amenity=='hospital' or healthcare=='hospital' or building=='hospital':
        return 'hospital'
    if amenity in ('clinic','doctors') or healthcare in ('clinic','doctor','doctors'):
        return 'clinic'
    if healthcare not in ('','no','false','0'):
        return 'healthcare'
    if amenity in ('school','kindergarten') or building in ('school','kindergarten'):
        return 'school'
    if amenity in ('college','university') or building in ('college','university'):
        return 'college'
    if amenity=='shelter':
        return 'shelter'
    if tag('highway') not in ('','no','false','0'):
        return 'road'
    if building not in ('','no','false','0'):
        return 'building'
    return 'infrastructure'


def source_feature_id(feature,index):
    properties = feature.get('properties') or {}
    identifier = feature.get('id')
    if identifier is None:
        identifier = properties.get('id')
    if identifier is None:
        identifier = properties.get('@id')
    if identifier is None and properties.get('osm_id') is not None:
        # HOT uid is a contributor ID shared by many features, not an object ID.
        identifier = f"osm/{properties.get('osm_type') or feature.get('geometry',{}).get('type','object')}/{properties['osm_id']}"
    if identifier is None:
        identifier = f'feature_{index}'
    if not isinstance(identifier,(str,int)) or isinstance(identifier,bool):
        raise ValueError('Feature IDs must be strings or integers.')
    return str(identifier)
