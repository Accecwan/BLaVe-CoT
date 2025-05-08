import json
import os
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, recall_score

current_dir_path = os.path.dirname(os.path.realpath(__file__))

def GroundingDifference(annFile, resFile, dataset_type="vqa"):
    """
    Evaluation function to compare ground truth annotations and predicted results.
    
    :param annFile: Path to the annotation file.
    :param resFile: Path to the result file with predictions.
    :param dataset_type: Type of the dataset ("vqa" or "vizwiz").
    :return: Evaluation results as a dictionary.
    """
    Anns_vqa = []
    Anns_vizwiz = []
    Ress_vqa = []
    Ress_vizwiz = []

    # Open the annotation and result files and load the data
    with open(annFile, 'r') as annF:
        with open(resFile, 'r') as resF:
            anns = json.load(annF)
            ress = json.load(resF)

            # Map labels based on dataset type (VQA or VizWiz)
            if dataset_type == "vqa":
                label_map = {"single": 1, "multiple": 0}
                Anns_labels = np.array([label_map.get(ann["binary_label"], 0) for ann in anns])
                Res_labels = np.array([res["single_grounding"] for res in ress])
            elif dataset_type == "vizwiz":
                label_map = {"single": 1, "multiple": 0}
                Anns_labels = np.array([label_map.get(ann["binary_label"], 0) for ann in anns])
                Res_labels = np.array([res["single_grounding"] for res in ress])

            # Separate annotations and results based on image_id for VQA and VizWiz datasets
            for ann in anns:
                if ann["image_id"].startswith("Viz"):
                    Anns_vizwiz.append(ann)
                else:
                    Anns_vqa.append(ann)
            for res in ress:
                if res["question_id"].startswith("Viz"):
                    Ress_vizwiz.append(res)
                else:
                    Ress_vqa.append(res)

            # Get corresponding labels for VQA and VizWiz datasets
            if dataset_type == "vqa":
                Anns_vqa_labels = np.array([label_map.get(ann["binary_label"], 0) for ann in Anns_vqa])
                Ress_vqa_labels = np.array([res["single_grounding"] for res in Ress_vqa])
            elif dataset_type == "vizwiz":
                Anns_vizwiz_labels = np.array([ann["binary_label"] for ann in Anns_vizwiz])
                Ress_vizwiz_labels = np.array([res["single_grounding"] for res in Ress_vizwiz])

            # Ensure that the number of annotations matches the number of predictions
            if len(Anns_labels) != len(Res_labels):
                print("Unsuccessful submission! The number of generated files does not match the number of ground-truth files.")
                return {}
            else:
                results = {}

                # Calculate overall evaluation metrics
                results["overall_f1"] = round(100 * f1_score(Anns_labels, Res_labels > 0.5), 2)
                results['overall_precision'] = round(100 * average_precision_score(Anns_labels, Res_labels > 0.5), 2)
                results['overall_recall'] = round(100 * recall_score(Anns_labels, Res_labels > 0.5), 2)

                # Calculate evaluation metrics for the VQA dataset
                if dataset_type == "vqa":
                    results['vqav2_f1'] = round(100 * f1_score(Anns_vqa_labels, Ress_vqa_labels > 0.5), 2)
                    results['vqa_precision'] = round(100 * average_precision_score(Anns_vqa_labels, Ress_vqa_labels > 0.5), 2)
                    results['vqa_recall'] = round(100 * recall_score(Anns_vqa_labels, Ress_vqa_labels > 0.5), 2)

                # Calculate evaluation metrics for the VizWiz dataset
                if dataset_type == "vizwiz":
                    results['vizwiz_f1'] = round(100 * f1_score(Anns_vizwiz_labels, Ress_vizwiz_labels > 0.5), 2)
                    results['vizwiz_precision'] = round(100 * average_precision_score(Anns_vizwiz_labels, Ress_vizwiz_labels > 0.5), 2)
                    results['vizwiz_recall'] = round(100 * recall_score(Anns_vizwiz_labels, Ress_vizwiz_labels > 0.5), 2)

                return results

# Mapping of phases to dataset splits
phase_splits = {
    "test-dev2023": ["test-dev"],
    "test-standard2023": ["test"],
    "test-challenge2023": ["test"]
}

def evaluate(resFile, phase_codename, dataset_type="vqa", **kwargs):
    """
    Main evaluation function to assess the predictions on the VQA or VizWiz dataset.
    
    :param resFile: Path to the prediction results file.
    :param phase_codename: Name of the evaluation phase.
    :param dataset_type: Type of the dataset ("vqa" or "vizwiz").
    :param kwargs: Additional evaluation parameters.
    :return: Evaluation results as a dictionary.
    """
    result = []
    splits = phase_splits[phase_codename]
    
    for split in splits:
        # Define the annotation file path based on dataset type
        if dataset_type == "vqa":
            # Example: VQA annotation path
            annFile = "/root/dataset-split/split/vqa_val_annotations.json"
        elif dataset_type == "vizwiz":
            # Example: VizWiz annotation path
            annFile = "/root/dataset-split/split/vizwiz_val_annotations.json"

        print(f"Starting evaluation for phase: {phase_codename}, dataset type: {dataset_type}")
        
        # Compare the annotations with the predictions for the current split
        result.append({split: GroundingDifference(annFile, resFile, dataset_type=dataset_type)})
        
        output = {"result": result}
        output["submission_result"] = result
        print(result)
        print("Evaluation phase completed.")
        return output

if __name__ == "__main__":
    # Example: Evaluate VQA dataset predictions
    evaluate("/root/autodl-tmp/output/submission_vqa_token.json", "test-standard2023", dataset_type="vqa")

    # Example: Evaluate VizWiz dataset predictions
    # evaluate("/root/autodl-tmp/output/submission_BLIP.json", "test-standard2023", dataset_type="vizwiz")
