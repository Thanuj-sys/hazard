"""Software fixtures only. These tests do not establish real disaster accuracy."""
import copy
import json
import math
import pytest
from evaluation.output_assessment import validate_output, generate_assessment, assess_vulnerability, _llm_paragraphs


@pytest.fixture
def prediction():
    return {'hazard_type':'flood','hazard_probabilities':{'flood':.8,'wildfire':.1,'landslide':.1},
            'predicted_coordinates':[10.,75.], 'radius_km':2., 'affected_area_km2':math.pi*4,
            'data_source':'software test fixture','inference_mode':'test_fixture',
            'source_acquired_at':'2020-01-01T00:00:00Z'}


def feature(identifier,kind,geometry,**properties):
    return {'type':'Feature','id':identifier,'geometry':geometry,'properties':{'kind':kind,**properties}}


def collection(*features):
    return {'type':'FeatureCollection','features':list(features)}


SOURCE={'name':'software fixture','reference':'test protocol only; no real source claim'}
POINT={'type':'Point','coordinates':[75.,10.]}
POLYGON={'type':'Polygon','coordinates':[[[74.999,9.999],[75.001,9.999],[75.001,10.001],[74.999,10.001],[74.999,9.999]]]}


def test_valid_output_is_not_claimed_as_verified_accuracy(prediction):
    value=validate_output(prediction)
    assert value['failure_count']==0
    assert value['prediction_correctness']=='not_evaluated'
    assert any(row['name']=='independent_accuracy' and row['status']=='not_evaluated' for row in value['checks'])


@pytest.mark.parametrize('field,value', [('radius_km',-1),('radius_km',True),('predicted_coordinates',[95,75]),('hazard_probabilities',{'flood':1.2,'wildfire':0,'landslide':0})])
def test_invalid_predictions_fail_integrity(prediction,field,value):
    prediction[field]=value
    assert validate_output(prediction)['failure_count']>0
    result=generate_assessment(prediction)
    assert result['vulnerability']['status']=='not_evaluated'
    assert result['map_svg'] is None


def test_missing_exposure_does_not_invent_zero_population(prediction):
    result=generate_assessment(prediction)
    exposure=result['vulnerability']
    assert exposure['status']=='data_unavailable'
    assert exposure['intersecting_feature_count'] is None
    assert exposure['population_total_exposed'] is None
    assert exposure['vulnerability_score'] is None
    assert '<svg' in result['map_svg']
    assert 'not a verified disaster bulletin' in result['report']['text']


def test_point_exposure_count(prediction):
    layer=collection(feature('hospital1','hospital',POINT),feature('school1','school',{'type':'Point','coordinates':[76.,11.]}))
    result=assess_vulnerability(prediction,layer,SOURCE)
    assert result['intersecting_feature_count']==1
    assert result['feature_counts']=={'hospital':1}
    assert result['population_fully_contained_units'] is None
    assert result['road_length_inside_km'] is None


def test_population_subset_and_source_classes(prediction):
    layer=collection(feature('cell1','population',POLYGON,population=100,vulnerability_class='high'))
    result=assess_vulnerability(prediction,layer,SOURCE)
    assert result['population_fully_contained_units']==100
    assert result['population_partial_units']==0
    assert result['population_total_exposed'] is None
    assert result['source_vulnerability_classes']=={'high':1}
    assert result['vulnerability_score'] is None


def test_partial_population_not_apportioned(prediction):
    large={'type':'Polygon','coordinates':[[[74.99,9.99],[75.04,9.99],[75.04,10.04],[74.99,10.04],[74.99,9.99]]]}
    layer=collection(feature('cell1','population',large,population=100))
    result=assess_vulnerability(prediction,layer,SOURCE)
    assert result['population_fully_contained_units']==0
    assert result['population_partial_units']==1
    assert result['population_total_exposed'] is None


def test_overlapping_population_units_rejected(prediction):
    layer=collection(feature('a','population',POLYGON,population=100),feature('b','population',POLYGON,population=100))
    with pytest.raises(ValueError,match='overlap'):
        assess_vulnerability(prediction,layer,SOURCE)


def test_duplicate_ids_rejected(prediction):
    layer=collection(feature('a','hospital',POINT),feature('a','school',POINT))
    with pytest.raises(ValueError,match='Duplicate'):
        assess_vulnerability(prediction,layer,SOURCE)


def test_roads_are_clipped_before_length_measurement(prediction):
    road={'type':'LineString','coordinates':[[74.9,10.],[75.1,10.]]}
    value=assess_vulnerability(prediction,collection(feature('r','road',road)),SOURCE)
    assert 3.8 < value['road_length_inside_km'] < 4.2
    assert value['feature_counts']['road']==1


def test_source_required_and_non_wgs84_rejected(prediction):
    layer=collection(feature('a','hospital',POINT))
    with pytest.raises(ValueError,match='source'):
        assess_vulnerability(prediction,layer,None)
    layer['crs']={'type':'name','properties':{'name':'EPSG:3857'}}
    with pytest.raises(ValueError,match='CRS'):
        assess_vulnerability(prediction,layer,SOURCE)


def test_html_report_escapes_source_text(prediction):
    source={'name':'<script>alert(1)</script>','reference':'<img src=x onerror=alert(1)>'}
    result=generate_assessment(prediction,collection(feature('a','hospital',POINT)),source)
    assert '<script>' not in result['report']['html']
    assert '&lt;script&gt;' in result['report']['html']
    assert result['report']['input_sha256']


def test_synthetic_input_stays_demonstration(prediction):
    prediction['inference_mode']='demo_synthetic'
    result=generate_assessment(prediction)
    assert result['report']['scope']=='generated_input_demonstration'
    assert result['validation']['warning_count']>=1


def test_llm_unconfigured_never_creates_fake_text(monkeypatch):
    monkeypatch.delenv('OLLAMA_MODEL',raising=False)
    result=_llm_paragraphs([{'id':'F1','value':'test'}])
    assert result['status']=='not_configured'
    assert result['paragraphs']==[]


def test_llm_remote_endpoint_rejected(monkeypatch):
    monkeypatch.setenv('OLLAMA_MODEL','fixture')
    monkeypatch.setenv('OLLAMA_BASE_URL','https://remote.example.test')
    assert _llm_paragraphs([])['status']=='configuration_error'


@pytest.mark.parametrize('content,expected', [({'paragraphs':[{'text':'A model estimate.','fact_ids':['F1']}]},'generated'),({'paragraphs':[{'text':'Invented citation.','fact_ids':['F999']}]},'invalid_response'),([], 'invalid_response')])
def test_llm_references_checked_without_network(monkeypatch,content,expected):
    import requests
    monkeypatch.setenv('OLLAMA_MODEL','fixture')
    monkeypatch.setenv('OLLAMA_BASE_URL','http://127.0.0.1:11434')
    class Response:
        status_code=200
        def json(self):
            return {'done':True,'message':{'content':json.dumps(content)}}
    class Session:
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def post(self,*args,**kwargs): return Response()
    monkeypatch.setattr(requests,'Session',Session)
    result=_llm_paragraphs([{'id':'F1','value':'software fixture'}])
    assert result['status']==expected
    if expected=='invalid_response':
        assert not result['paragraphs']


def test_invalid_footprint_fails_output_validation(prediction):
    prediction['footprint']={'type':'Point','coordinates':[75.,10.]}
    result=generate_assessment(prediction)
    assert result['validation']['failure_count']>0
    assert result['vulnerability']['status']=='not_evaluated'


def test_zero_radius_is_reported_without_inventing_footprint(prediction):
    prediction['radius_km']=0
    prediction['affected_area_km2']=0
    result=generate_assessment(prediction)
    assert result['validation']['failure_count']==0
    assert result['vulnerability']['status']=='not_evaluated'
    assert 'positive radius' in result['vulnerability']['reason']
    assert result['map_svg'] is None

# HOT/OSM import tests below also use synthetic software geometries only.
@pytest.mark.parametrize('properties,expected', [
    ({'amenity':'hospital','building':'yes'},'hospital'),
    ({'amenity':'clinic'},'clinic'),
    ({'healthcare':'centre'},'healthcare'),
    ({'amenity':'college'},'college'),
    ({'highway':'residential'},'road'),
    ({'building':'yes'},'building'),
    ({'building':'no'},'infrastructure'),
    ({'kind':'school','building':'yes'},'school'),
])
def test_osm_tags_classified_without_risk_invention(properties,expected):
    from evaluation.exposure_import import infer_feature_kind
    assert infer_feature_kind(properties)==expected
    assert 'vulnerability_class' not in properties


def test_hot_uid_is_not_used_as_feature_id():
    from evaluation.exposure_import import source_feature_id
    first={'properties':{'uid':42,'osm_id':100,'osm_type':'nodes'},'geometry':POINT}
    second={'properties':{'uid':42,'osm_id':101,'osm_type':'nodes'},'geometry':POINT}
    assert source_feature_id(first,0)=='osm/nodes/100'
    assert source_feature_id(first,0)!=source_feature_id(second,1)


def test_hot_roads_and_hospitals_import_directly(prediction):
    road={'type':'Feature','geometry':{'type':'LineString','coordinates':[[74.9,10.],[75.1,10.]]},
          'properties':{'osm_id':10,'osm_type':'ways_line','uid':1,'highway':'residential'}}
    hospital={'type':'Feature','geometry':POINT,'properties':{'osm_id':20,'osm_type':'nodes','uid':1,'amenity':'hospital','name':'Fixture hospital'}}
    result=assess_vulnerability(prediction,collection(road,hospital),SOURCE)
    assert result['feature_counts']=={'road':1,'hospital':1}
    assert 3.8 < result['road_length_inside_km'] < 4.2
    assert result['feature_exposure']['hospital']['fraction']==1
    assert result['population_total_exposed'] is None


def test_import_can_exceed_old_2000_feature_limit(prediction):
    entries=[{'type':'Feature','id':str(i),'geometry':POINT,'properties':{'amenity':'hospital'}} for i in range(2001)]
    result=assess_vulnerability(prediction,collection(*entries),SOURCE)
    assert result['features_supplied']==2001
    assert result['feature_counts']['hospital']==2001


def test_building_overlap_union_avoids_duplicate_area(prediction):
    one=feature('building-a','building',POLYGON)
    two=feature('building-b','building',POLYGON)
    single=assess_vulnerability(prediction,collection(one),SOURCE)
    doubled=assess_vulnerability(prediction,collection(one,two),SOURCE)
    assert doubled['feature_counts']['building']==2
    assert doubled['mapped_building_overlap_km2']==pytest.approx(single['mapped_building_overlap_km2'])
    assert doubled['vulnerability_score'] is None
