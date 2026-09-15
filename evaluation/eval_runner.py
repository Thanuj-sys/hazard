from typing import List, Dict, Any
from collections import defaultdict
from .metrics import MultiHazardEvaluator

class EvaluationRunner:
    """
    Runs evaluation on a batch of test data.
    Computes average metrics per hazard type and overall averages.
    """
    def __init__(self):
        self.evaluator = MultiHazardEvaluator()

    def evaluate_batch(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate a batch of predictions against ground truth.
        
        Args:
            test_cases: List of test pairs (each is a dict containing 'true' and 'pred' info, and 'hazard_type')
            
        Returns:
            dict: Summary metrics broken down by hazard type and overall.
        """
        hazard_metrics = defaultdict(lambda: defaultdict(list))
        
        for case in test_cases:
            hazard = case.get("hazard_type", "unknown")
            pred = case.get("pred", {})
            true = case.get("true", {})
            
            # Regression metrics
            hav_dist = self.evaluator.calculate_haversine_distance(
                pred.get("lat", 0), pred.get("lon", 0),
                true.get("lat", 0), true.get("lon", 0)
            )
            
            rad_errs = self.evaluator.calculate_radius_error(
                pred.get("radius", 0), true.get("radius", 0)
            )
            
            # Language Reasoner metrics
            dap_ddr = self.evaluator.calculate_dap_ddr(
                pred.get("claims", []), true.get("claims", [])
            )
            
            fc = self.evaluator.calculate_factual_correctness(
                pred.get("geo_claims", {}), true.get("geo_attributes", {})
            )
            
            rn = self.evaluator.calculate_recovery_necessity(
                pred.get("interventions", []), true.get("interventions", [])
            )
            
            sc = self.evaluator.calculate_strategic_completeness(
                pred.get("interventions", []), true.get("standard_checklist", [])
            )
            
            app = self.evaluator.calculate_action_priority_precision(
                pred.get("priorities", []), true.get("priorities", [])
            )
            
            avg_score = self.evaluator.calculate_avg_score(
                dap_ddr["dap"], dap_ddr["ddr"], fc, rn, sc, app
            )
            
            # Store in hazard breakdown
            metrics_dict = {
                "haversine_dist_km": hav_dist,
                "radius_abs_err_km": rad_errs["absolute_error"],
                "radius_rel_err": rad_errs["relative_error"],
                "dap": dap_ddr["dap"],
                "ddr": dap_ddr["ddr"],
                "factual_correctness": fc,
                "recovery_necessity": rn,
                "strategic_completeness": sc,
                "action_priority_precision": app,
                "average_score": avg_score
            }
            
            for k, v in metrics_dict.items():
                hazard_metrics[hazard][k].append(v)
                hazard_metrics["overall"][k].append(v)
                
        # Calculate averages
        summary = {}
        for hazard_type, metrics in hazard_metrics.items():
            summary[hazard_type] = {}
            for metric_name, values in metrics.items():
                avg_val = sum(values) / len(values) if values else 0.0
                summary[hazard_type][metric_name] = avg_val
                
        return summary

if __name__ == '__main__':
    # Mock report runner
    mock_test_cases = [
        {
            "hazard_type": "flood",
            "pred": {
                "lat": 34.05, "lon": -118.25, "radius": 10.5,
                "claims": ["Water level rising", "Evacuation needed"],
                "geo_claims": {"elevation": 100.0, "slope": 5.0},
                "interventions": ["Boats", "Sandbags"],
                "priorities": ["Evacuation", "Sandbags", "Boats"]
            },
            "true": {
                "lat": 34.00, "lon": -118.20, "radius": 12.0,
                "claims": ["Water level rising", "Evacuation needed", "Power outage"],
                "geo_attributes": {"elevation": 95.0, "slope": 5.5},
                "interventions": ["Boats", "Sandbags", "Shelters"],
                "standard_checklist": ["Boats", "Sandbags", "Shelters", "Food"],
                "priorities": ["Evacuation", "Shelters", "Sandbags", "Boats"]
            }
        },
        {
            "hazard_type": "wildfire",
            "pred": {
                "lat": 39.73, "lon": -104.98, "radius": 25.0,
                "claims": ["High winds", "Dry vegetation"],
                "geo_claims": {"ndvi": 0.2, "slope": 15.0},
                "interventions": ["Air drops", "Firelines"],
                "priorities": ["Evacuation", "Air drops", "Firelines"]
            },
            "true": {
                "lat": 39.75, "lon": -105.00, "radius": 20.0,
                "claims": ["High winds", "Dry vegetation", "Structures threatened"],
                "geo_attributes": {"ndvi": 0.18, "slope": 16.0},
                "interventions": ["Air drops", "Firelines", "Evacuation centers"],
                "standard_checklist": ["Air drops", "Firelines", "Evacuation centers", "Medical"],
                "priorities": ["Evacuation", "Structures", "Air drops", "Firelines"]
            }
        }
    ]
    
    runner = EvaluationRunner()
    results = runner.evaluate_batch(mock_test_cases)
    
    import json
    print(json.dumps(results, indent=2))
