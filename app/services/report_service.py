import pandas as pd
import os
import logging
from typing import Dict, List, Tuple, Optional
from app.models.requests import StoreMetadata

logger = logging.getLogger(__name__)


class DeDupReportGenerator:
    """Generate deduplication reports from grouped similarity results"""

    def __init__(self):
        pass

    def _add_groups_to_df(self, kyc_df: pd.DataFrame, grouped_results: Dict) -> Tuple[pd.DataFrame, Dict]:
        """Add group information to the KYC dataframe based on similarity results"""
        path_mapping = {}

        # Create mapping from image paths to group information
        for group_name, data in grouped_results.items():
            group_count = len(data['paths'])
            for path_info in data['paths']:
                path_parts = path_info['path'].split('/')
                for i, part in enumerate(path_parts):
                    if part.startswith('KYC1') or part.startswith('KYC2'):
                        # Extract the key that matches with dataframe
                        key = '/'.join(path_parts[i:i+6])
                        path_mapping[key] = {
                            'group_name': group_name,
                            'group_count': group_count,
                            'similarity_score': path_info['similarity_score'],
                            'image_path': path_info['path']
                        }
                        break

        # Initialize columns
        kyc_df = kyc_df.copy()  # Avoid SettingWithCopyWarning
        kyc_df['group_name'] = 'Not Identified'
        kyc_df['group_count'] = 0
        kyc_df['similarity_score'] = 0.0

        # Map groups to dataframe rows
        for idx, row in kyc_df.iterrows():
            try:
                # Create match key based on image_path and transaction_id
                match_key = os.path.join(row["image_path"], row["kyctransactionid"])
                if match_key in path_mapping:
                    match_info = path_mapping[match_key]
                    kyc_df.at[idx, 'group_name'] = match_info['group_name']
                    kyc_df.at[idx, 'group_count'] = match_info['group_count']
                    kyc_df.at[idx, 'similarity_score'] = match_info['similarity_score']
            except Exception as e:
                logger.warning(f"Error processing row {idx}: {str(e)}")
                continue

        # Reorder columns to put group info first
        columns = ['group_name', 'group_count', 'similarity_score'] + \
                  [col for col in kyc_df.columns if col not in ['group_name', 'group_count', 'similarity_score']]
        kyc_df = kyc_df[columns]

        return kyc_df, path_mapping

    def generate_dedup_report(self, kyc_df: pd.DataFrame, grouped_results: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate the main deduplication report with fraud detection"""

        # Add group information to the dataframe
        report_df, path_mapping = self._add_groups_to_df(kyc_df, grouped_results)

        # Filter for records that are part of identified groups
        fraud_df = report_df[report_df.group_name != 'Not Identified'].copy()

        # Fraud Condition: Groups with more than 1 unique ID number
        fraud_df = fraud_df[fraud_df.groupby("group_name")["idnum"].transform("nunique") > 1]

        # Select relevant columns for the fraud report
        selected_columns = [
            "group_name", "group_count", "similarity_score", "msisdn",
            "createdon", "updatedon", "retailernumber", "kitid",
            "userid", "idnum", "idtype", "nationality"
        ]

        # Create summary report
        summary_report = self.create_summary_report(report_df, fraud_df)

        return fraud_df[selected_columns], summary_report

    def create_summary_report(self, report_df: pd.DataFrame, fraud_df: pd.DataFrame) -> pd.DataFrame:
        """Create a summary report with key statistics"""

        total_unique_records = report_df.shape[0]
        total_unique_person_groups = len(report_df[report_df.group_name != 'Not Identified'].group_name.unique())
        number_of_fraudulent_groups = len(fraud_df.group_name.unique()) if not fraud_df.empty else 0
        total_records_in_fraudulent_groups = fraud_df.shape[0]
        unique_id_numbers_in_fraudulent_groups = len(fraud_df.userid.unique()) if not fraud_df.empty else 0

        # Calculate percentages
        percentage_fraudulent_records = round(
            total_records_in_fraudulent_groups / total_unique_records * 100, 2
        ) if total_unique_records > 0 else 0

        percentage_fraudulent_person_groups = round(
            number_of_fraudulent_groups / total_unique_person_groups * 100, 2
        ) if total_unique_person_groups > 0 else 0

        # Create summary data
        summary_data = {
            "Metric": [
                "Total Unique Records",
                "Total Number of Unique Person Groups",
                "Fraudulent Groups (More than 1 ID Number for a particular group)",
                "Total Records falling in Fraudulent Groups",
                "Unique ID Numbers in Fraudulent Groups",
                "Percentage Fraudulent Records",
                "Percentage Fraudulent Person Groups"
            ],
            "Value": [
                total_unique_records,
                total_unique_person_groups,
                number_of_fraudulent_groups,
                total_records_in_fraudulent_groups,
                unique_id_numbers_in_fraudulent_groups,
                f"{percentage_fraudulent_records}%",
                f"{percentage_fraudulent_person_groups}%"
            ]
        }

        return pd.DataFrame(summary_data)

    def analyze_group_patterns(self, fraud_df: pd.DataFrame) -> Dict:
        """Analyze patterns in fraudulent groups"""
        if fraud_df.empty:
            return {"message": "No fraudulent groups found"}

        analysis = {}

        # Group size distribution
        group_sizes = fraud_df.groupby('group_name')['group_count'].first()
        analysis['group_size_stats'] = {
            'mean_size': group_sizes.mean(),
            'median_size': group_sizes.median(),
            'max_size': group_sizes.max(),
            'min_size': group_sizes.min()
        }

        # ID type distribution in fraud cases
        id_type_distribution = fraud_df['idtype'].value_counts().to_dict()
        analysis['id_type_distribution'] = id_type_distribution

        # Retailer analysis
        retailer_fraud_count = fraud_df.groupby('retailernumber').size().sort_values(ascending=False)
        analysis['top_retailers_with_fraud'] = retailer_fraud_count.head(10).to_dict()

        # Kit ID analysis
        kit_fraud_count = fraud_df.groupby('kitid').size().sort_values(ascending=False)
        analysis['top_kits_with_fraud'] = kit_fraud_count.head(10).to_dict()

        # Time-based analysis
        fraud_df_copy = fraud_df.copy()
        fraud_df_copy['createdon'] = pd.to_datetime(fraud_df_copy['createdon'], errors='coerce')
        if not fraud_df_copy['createdon'].isna().all():
            fraud_df_copy['creation_hour'] = fraud_df_copy['createdon'].dt.hour
            hourly_distribution = fraud_df_copy['creation_hour'].value_counts().sort_index().to_dict()
            analysis['hourly_fraud_distribution'] = hourly_distribution

        return analysis

    def export_reports(self, fraud_df: pd.DataFrame, summary_df: pd.DataFrame,
                       output_dir: str = "./reports") -> Dict[str, str]:
        """Export reports to CSV files"""

        os.makedirs(output_dir, exist_ok=True)

        file_paths = {}

        # Export fraud report
        fraud_report_path = os.path.join(output_dir, "fraud_report.csv")
        fraud_df.to_csv(fraud_report_path, index=False)
        file_paths['fraud_report'] = fraud_report_path

        # Export summary report
        summary_report_path = os.path.join(output_dir, "summary_report.csv")
        summary_df.to_csv(summary_report_path, index=False)
        file_paths['summary_report'] = summary_report_path

        # Export group analysis if fraud data exists
        if not fraud_df.empty:
            analysis = self.analyze_group_patterns(fraud_df)
            analysis_df = pd.DataFrame([
                {"Analysis_Type": k, "Data": str(v)} for k, v in analysis.items()
            ])
            analysis_path = os.path.join(output_dir, "fraud_analysis.csv")
            analysis_df.to_csv(analysis_path, index=False)
            file_paths['fraud_analysis'] = analysis_path

        logger.info(f"Reports exported to {output_dir}")
        return file_paths

    def generate_group_detail_report(self, fraud_df: pd.DataFrame) -> pd.DataFrame:
        """Generate detailed report showing all records in each fraudulent group"""

        if fraud_df.empty:
            return pd.DataFrame()

        # Sort by group name and similarity score
        detailed_report = fraud_df.sort_values(['group_name', 'similarity_score'],
                                               ascending=[True, False])

        # Add group statistics
        group_stats = fraud_df.groupby('group_name').agg({
            'idnum': 'nunique',
            'msisdn': 'nunique',
            'similarity_score': ['mean', 'max', 'min']
        }).round(3)

        group_stats.columns = ['unique_ids', 'unique_msisdns', 'avg_similarity', 'max_similarity', 'min_similarity']
        group_stats = group_stats.reset_index()

        # Merge with detailed report
        detailed_report = detailed_report.merge(group_stats, on='group_name', how='left')

        return detailed_report


# Global instance
report_generator = DeDupReportGenerator()