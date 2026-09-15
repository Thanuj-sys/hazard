import math
from typing import Dict, List, Tuple, Any
from scipy.stats import kendalltau
import torch

class MultiHazardEvaluator:
    """
    Evaluator for MultiHazard-VLM pipeline.
    
    Provides custom metrics for multi-hazard mapping:
    - Haversine Distance (regression)
    - Radius Error (regression)
    - DAP and DDR (language extraction)
    - Factual Correctness (grounding)
    - Recovery Necessity & Strategic Completeness (decision making)
    - Action Priority Precision (ranking)
    """

    @staticmethod
    def calculate_haversine_distance(pred_lat: float, pred_lon: float, true_lat: float, true_lon: float) -> float:
        """
        Calculate Haversine distance in kilometers between predicted and true coordinates.
        
        Args:
            pred_lat: Predicted latitude.
            pred_lon: Predicted longitude.
            true_lat: True latitude.
            true_lon: True longitude.
            
        Returns:
            float: Distance in kilometers.
        """
        R = 6371.0 # Earth radius in km

        lat1 = math.radians(pred_lat)
        lon1 = math.radians(pred_lon)
        lat2 = math.radians(true_lat)
        lon2 = math.radians(true_lon)

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = R * c
        return distance

    @staticmethod
    def calculate_radius_error(pred_radius: float, true_radius: float) -> Dict[str, float]:
        """
        Calculate absolute and relative radius errors.
        
        Args:
            pred_radius: Predicted hazard radius in km.
            true_radius: True hazard radius in km.
            
        Returns:
            dict: Containing 'absolute_error' and 'relative_error'.
        """
        abs_err = abs(pred_radius - true_radius)
        rel_err = abs_err / true_radius if true_radius > 0 else 0.0
        return {
            "absolute_error": abs_err,
            "relative_error": rel_err
        }

    @staticmethod
    def calculate_dap_ddr(predicted_claims: List[str], ground_truth_claims: List[str]) -> Dict[str, float]:
        """
        Calculate Direct Assessment Precision (DAP) and Direct Discovery Recall (DDR).
        
        Note: In practice, claims are extracted from text via LLM-as-a-judge or entity extraction.
        
        Args:
            predicted_claims: List of hazard claims predicted by the model.
            ground_truth_claims: List of ground truth hazard claims.
            
        Returns:
            dict: Containing 'dap' (precision) and 'ddr' (recall).
        """
        predicted_set = set(predicted_claims)
        gt_set = set(ground_truth_claims)
        
        matching = len(predicted_set.intersection(gt_set))
        dap = matching / len(predicted_set) if len(predicted_set) > 0 else 0.0
        ddr = matching / len(gt_set) if len(gt_set) > 0 else 0.0
        
        return {"dap": dap, "ddr": ddr}

    @staticmethod
    def calculate_factual_correctness(predicted_geo_claims: Dict[str, float], actual_geo_attributes: Dict[str, float], tolerance: float = 0.1) -> float:
        """
        Verify if report claims about DEM/slope/NDVI match the actual attributes.
        
        Args:
            predicted_geo_claims: Extracted quantitative claims from the language reasoner.
            actual_geo_attributes: Actual values for DEM, slope, NDVI, etc.
            tolerance: Allowed relative error tolerance.
            
        Returns:
            float: Ratio of correct claims to total evaluated claims.
        """
        if not predicted_geo_claims:
            return 0.0
            
        correct = 0
        total = 0
        for key, pred_val in predicted_geo_claims.items():
            if key in actual_geo_attributes:
                total += 1
                actual_val = actual_geo_attributes[key]
                # Compare with tolerance
                if actual_val == 0.0:
                    if abs(pred_val) < 1e-5:
                        correct += 1
                else:
                    if abs(pred_val - actual_val) / abs(actual_val) <= tolerance:
                        correct += 1
                        
        return correct / total if total > 0 else 0.0

    @staticmethod
    def calculate_recovery_necessity(predicted_interventions: List[str], required_interventions: List[str]) -> float:
        """
        Calculate precision of recommended recovery actions.
        
        Args:
            predicted_interventions: List of predicted recovery actions.
            required_interventions: List of actual required actions.
            
        Returns:
            float: Precision (Recovery Necessity score).
        """
        pred_set = set(predicted_interventions)
        req_set = set(required_interventions)
        
        matching = len(pred_set.intersection(req_set))
        return matching / len(pred_set) if len(pred_set) > 0 else 0.0

    @staticmethod
    def calculate_strategic_completeness(predicted_interventions: List[str], standard_checklist: List[str]) -> float:
        """
        Calculate recall of predicted interventions against a standard checklist.
        
        Args:
            predicted_interventions: List of predicted recovery actions.
            standard_checklist: Standard checklist of required actions.
            
        Returns:
            float: Recall (Strategic Completeness score).
        """
        pred_set = set(predicted_interventions)
        chk_set = set(standard_checklist)
        
        matching = len(pred_set.intersection(chk_set))
        return matching / len(chk_set) if len(chk_set) > 0 else 0.0

    @staticmethod
    def calculate_action_priority_precision(predicted_priorities: List[str], true_priorities: List[str]) -> float:
        """
        Compute Kendall's Tau ranking correlation between predicted and true priorities.
        
        Args:
            predicted_priorities: List of items ranked by predicted priority.
            true_priorities: List of items ranked by expert ground truth priority.
            
        Returns:
            float: Kendall's Tau correlation normalized to [0, 1].
        """
        # Assign rank based on position in true priorities
        true_ranks = {item: rank for rank, item in enumerate(true_priorities)}
        
        # We need overlapping items to calculate correlation.
        # If predicted item is missing in true priorities, penalize it by putting it at the end.
        max_rank = len(true_priorities)
        
        pred_scores = []
        true_scores = []
        
        # Only evaluate overlapping elements for simplicity
        common_items = set(predicted_priorities).intersection(set(true_priorities))
        if len(common_items) < 2:
            return 0.0
            
        # Re-filter to preserve order
        filtered_pred = [item for item in predicted_priorities if item in common_items]
        filtered_true = [item for item in true_priorities if item in common_items]
        
        pred_ranks_dict = {item: rank for rank, item in enumerate(filtered_pred)}
        true_ranks_dict = {item: rank for rank, item in enumerate(filtered_true)}
        
        for item in common_items:
            pred_scores.append(pred_ranks_dict[item])
            true_scores.append(true_ranks_dict[item])
            
        tau, p_value = kendalltau(pred_scores, true_scores)
        if math.isnan(tau):
            tau = 0.0
            
        # Normalize tau from [-1, 1] to [0, 1]
        return (tau + 1.0) / 2.0

    @staticmethod
    def calculate_avg_score(dap: float, ddr: float, fc: float, rn: float, sc: float, app: float) -> float:
        """
        Calculate average evaluation score across all 6 language/reasoning metrics.
        
        Args:
            dap: Direct Assessment Precision.
            ddr: Direct Discovery Recall.
            fc: Factual Correctness.
            rn: Recovery Necessity.
            sc: Strategic Completeness.
            app: Action Priority Precision.
            
        Returns:
            float: Average score.
        """
        return (dap + ddr + fc + rn + sc + app) / 6.0

    @staticmethod
    def evaluate(predictions: Any, targets: Any) -> Dict[str, float]:
        """Compute standard classification metrics for hazard predictions.

        This method provides a real evaluator API for the training loop and tests,
        while preserving the more specialized geospatial and reasoning metrics above.
        """
        if isinstance(predictions, torch.Tensor):
            preds = predictions.detach().cpu().reshape(-1)
        else:
            preds = torch.as_tensor(predictions).reshape(-1)

        if isinstance(targets, torch.Tensor):
            true = targets.detach().cpu().reshape(-1)
        else:
            true = torch.as_tensor(targets).reshape(-1)

        if preds.numel() == 0 or true.numel() == 0:
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

        if preds.dtype.is_floating_point:
            preds = preds.round().long()

        valid = true.numel()
        correct = torch.sum(preds == true).item()
        acc = correct / valid if valid else 0.0

        classes = torch.unique(torch.cat([preds, true]))
        precision_total = 0.0
        recall_total = 0.0
        f1_total = 0.0

        for cls in classes:
            cls = int(cls.item())
            pred_mask = preds == cls
            true_mask = true == cls

            tp = torch.sum(pred_mask & true_mask).item()
            fp = torch.sum(pred_mask & (~true_mask)).item()
            fn = torch.sum((~pred_mask) & true_mask).item()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            precision_total += precision
            recall_total += recall
            f1_total += f1

        num_classes = len(classes) if len(classes) > 0 else 1
        precision = precision_total / num_classes
        recall = recall_total / num_classes
        f1 = f1_total / num_classes

        return {
            'accuracy': float(acc),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
        }
