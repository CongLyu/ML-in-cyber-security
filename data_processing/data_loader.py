# data_loader.py
from xml.etree import ElementTree as ET
import pandas as pd


def load_dataset(xml_file_path):
    """
    Load an XML dataset and parse it into a pandas DataFrame.

    Parameters:
    - xml_file_path: str, the path to the XML file.

    Returns:
    - findings_df: DataFrame, contains the parsed data from the XML.
    """

    # Load and parse the XML file
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    # Define a nested function to extract the relevant data from each finding
    def extract_finding_data(finding):
        data = {
            'id': finding.get('id'),
            'detection-method': finding.get('detection-method'),
            'first-seen': finding.get('first-seen'),
            'last-updated': finding.get('last-updated'),
            'severity': finding.get('severity'),
            'status': finding.get('status'),
            'cwe-id': finding.find('.//cwe').get('id') if finding.find(
                './/cwe') is not None else None,
            'cwe-href': finding.find('.//cwe').get('href') if finding.find(
                './/cwe') is not None else None,
            'location-type': finding.find('.//location').get(
                'type') if finding.find('.//location') is not None else None,
            'location-path': finding.find('.//location').get(
                'path') if finding.find('.//location') is not None else None,
            'rule-code': finding.find('.//rule').get('code') if finding.find(
                './/rule') is not None else None,
            'rule-name': finding.find('.//rule').get('name') if finding.find(
                './/rule') is not None else None
        }
        return data

    # Extract data for each finding
    findings_data = [extract_finding_data(finding) for finding in
                     root.iter('finding')]

    # Convert the list of dictionaries into a DataFrame
    findings_df = pd.DataFrame(findings_data)

    return findings_df
