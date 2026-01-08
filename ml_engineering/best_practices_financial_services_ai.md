# Best Practices for Financial Services AI on Google Cloud

## Overview

Financial Services AI on Google Cloud provides specialized machine learning and AI services for fraud detection, risk assessment, anti-money laundering (AML), credit scoring, algorithmic trading, and regulatory compliance. These services help financial institutions automate decision-making, reduce fraud, ensure compliance, and improve customer experiences.

## 1. Fraud Detection

### 1.1 Transaction Fraud Detection

```python
from google.cloud import bigquery
from google.cloud import aiplatform
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from datetime import datetime, timedelta

class FraudDetectionManager:
    """Manager for fraud detection operations."""
    
    def __init__(
        self,
        project_id: str,
        location: str = 'us-central1',
        model_endpoint: str = None
    ):
        """
        Initialize Fraud Detection Manager.
        
        Args:
            project_id: GCP project ID
            location: GCP region
            model_endpoint: Vertex AI model endpoint
        """
        self.project_id = project_id
        self.location = location
        self.model_endpoint = model_endpoint
        self.bq_client = bigquery.Client(project=project_id)
        
        if model_endpoint:
            aiplatform.init(project=project_id, location=location)
    
    def extract_transaction_features(
        self,
        transaction: Dict[str, Any],
        user_history: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Extract features from transaction.
        
        Args:
            transaction: Transaction data
            user_history: Historical transactions for user
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Basic transaction features
        features['amount'] = float(transaction['amount'])
        features['hour_of_day'] = datetime.fromisoformat(
            transaction['timestamp']
        ).hour
        features['day_of_week'] = datetime.fromisoformat(
            transaction['timestamp']
        ).weekday()
        
        # Merchant features
        features['merchant_category'] = hash(transaction.get('merchant_category', '')) % 100
        features['is_international'] = int(transaction.get('is_international', False))
        features['is_online'] = int(transaction.get('is_online', False))
        
        # User behavioral features
        if not user_history.empty:
            # Average transaction amount
            features['avg_amount_30d'] = user_history['amount'].mean()
            features['std_amount_30d'] = user_history['amount'].std()
            
            # Transaction frequency
            features['tx_count_24h'] = len(
                user_history[
                    user_history['timestamp'] >= 
                    (datetime.now() - timedelta(hours=24))
                ]
            )
            features['tx_count_7d'] = len(
                user_history[
                    user_history['timestamp'] >= 
                    (datetime.now() - timedelta(days=7))
                ]
            )
            
            # Velocity features
            features['amount_ratio'] = features['amount'] / (features['avg_amount_30d'] + 1e-6)
            
            # Location features
            if 'location' in transaction and not user_history.empty:
                common_locations = user_history['location'].mode()
                features['is_common_location'] = int(
                    transaction['location'] in common_locations.values
                )
        else:
            # New user - use defaults
            features['avg_amount_30d'] = features['amount']
            features['std_amount_30d'] = 0
            features['tx_count_24h'] = 1
            features['tx_count_7d'] = 1
            features['amount_ratio'] = 1.0
            features['is_common_location'] = 0
        
        return features
    
    def calculate_fraud_score(
        self,
        transaction: Dict[str, Any],
        model_type: str = 'isolation_forest'
    ) -> Dict[str, Any]:
        """
        Calculate fraud score for transaction.
        
        Args:
            transaction: Transaction data
            model_type: Type of model to use
            
        Returns:
            Dictionary with fraud score and reasons
        """
        # Get user history
        user_id = transaction['user_id']
        query = f"""
        SELECT *
        FROM `{self.project_id}.transactions.history`
        WHERE user_id = '{user_id}'
        AND timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
        ORDER BY timestamp DESC
        """
        
        user_history = self.bq_client.query(query).to_dataframe()
        
        # Extract features
        features = self.extract_transaction_features(transaction, user_history)
        
        # Calculate risk factors
        risk_factors = []
        fraud_score = 0.0
        
        # Rule-based scoring
        if features['amount'] > features['avg_amount_30d'] * 5:
            risk_factors.append('Unusually high amount')
            fraud_score += 0.3
        
        if features['is_international'] and features['amount'] > 1000:
            risk_factors.append('Large international transaction')
            fraud_score += 0.2
        
        if features['tx_count_24h'] > 10:
            risk_factors.append('High transaction frequency')
            fraud_score += 0.15
        
        if not features['is_common_location']:
            risk_factors.append('Uncommon location')
            fraud_score += 0.1
        
        if features['hour_of_day'] < 5:  # 12am - 5am
            risk_factors.append('Unusual time')
            fraud_score += 0.05
        
        # ML model scoring (if available)
        if self.model_endpoint and model_type == 'ml_model':
            try:
                endpoint = aiplatform.Endpoint(self.model_endpoint)
                feature_vector = [[v for v in features.values()]]
                predictions = endpoint.predict(instances=feature_vector)
                ml_score = predictions.predictions[0][0]
                fraud_score = max(fraud_score, ml_score)
                
                if ml_score > 0.7:
                    risk_factors.append(f'ML model high risk ({ml_score:.2f})')
            except Exception as e:
                print(f"ML model prediction error: {e}")
        
        # Cap score at 1.0
        fraud_score = min(fraud_score, 1.0)
        
        return {
            'transaction_id': transaction['transaction_id'],
            'fraud_score': fraud_score,
            'risk_level': self._get_risk_level(fraud_score),
            'risk_factors': risk_factors,
            'features': features
        }
    
    def _get_risk_level(self, score: float) -> str:
        """Get risk level from score."""
        if score >= 0.8:
            return 'CRITICAL'
        elif score >= 0.6:
            return 'HIGH'
        elif score >= 0.4:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def flag_suspicious_transactions(
        self,
        threshold: float = 0.6,
        time_window_hours: int = 24
    ) -> pd.DataFrame:
        """
        Flag suspicious transactions.
        
        Args:
            threshold: Fraud score threshold
            time_window_hours: Time window to check
            
        Returns:
            DataFrame of suspicious transactions
        """
        query = f"""
        SELECT
            transaction_id,
            user_id,
            amount,
            merchant_category,
            timestamp,
            is_international,
            location
        FROM
            `{self.project_id}.transactions.recent`
        WHERE
            timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {time_window_hours} HOUR)
            AND status = 'pending'
        """
        
        transactions = self.bq_client.query(query).to_dataframe()
        
        suspicious_tx = []
        for _, tx in transactions.iterrows():
            result = self.calculate_fraud_score(tx.to_dict())
            if result['fraud_score'] >= threshold:
                suspicious_tx.append(result)
        
        return pd.DataFrame(suspicious_tx)
    
    def create_fraud_alert(
        self,
        transaction_result: Dict[str, Any],
        alert_channel: str = 'email'
    ):
        """
        Create fraud alert.
        
        Args:
            transaction_result: Result from calculate_fraud_score
            alert_channel: Alert channel (email, sms, webhook)
        """
        from google.cloud import pubsub_v1
        import json
        
        publisher = pubsub_v1.PublisherClient()
        topic_path = publisher.topic_path(self.project_id, 'fraud-alerts')
        
        alert = {
            'transaction_id': transaction_result['transaction_id'],
            'fraud_score': transaction_result['fraud_score'],
            'risk_level': transaction_result['risk_level'],
            'risk_factors': transaction_result['risk_factors'],
            'alert_channel': alert_channel,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        message_data = json.dumps(alert).encode('utf-8')
        future = publisher.publish(topic_path, message_data)
        future.result()
        
        print(f"Alert created for transaction {alert['transaction_id']}")


# Example usage
fraud_detector = FraudDetectionManager(
    project_id='my-project',
    location='us-central1',
    model_endpoint='projects/123/locations/us-central1/endpoints/456'
)

# Analyze transaction
transaction = {
    'transaction_id': 'tx_123',
    'user_id': 'user_456',
    'amount': 5000.00,
    'merchant_category': 'electronics',
    'timestamp': datetime.now().isoformat(),
    'is_international': True,
    'is_online': True,
    'location': 'US-NY'
}

result = fraud_detector.calculate_fraud_score(transaction)
print(f"Fraud Score: {result['fraud_score']:.2f}")
print(f"Risk Level: {result['risk_level']}")
```

### 1.2 Account Takeover Detection

```python
class AccountSecurityManager:
    """Manager for account security operations."""
    
    def __init__(self, project_id: str):
        """
        Initialize Account Security Manager.
        
        Args:
            project_id: GCP project ID
        """
        self.project_id = project_id
        self.bq_client = bigquery.Client(project=project_id)
    
    def detect_account_takeover(
        self,
        user_id: str,
        login_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Detect potential account takeover.
        
        Args:
            user_id: User identifier
            login_data: Login attempt data
            
        Returns:
            Dictionary with takeover risk assessment
        """
        # Get user's login history
        query = f"""
        SELECT
            ip_address,
            device_fingerprint,
            location,
            user_agent,
            timestamp
        FROM
            `{self.project_id}.security.login_history`
        WHERE
            user_id = '{user_id}'
            AND timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY)
        ORDER BY timestamp DESC
        """
        
        history = self.bq_client.query(query).to_dataframe()
        
        risk_score = 0.0
        risk_factors = []
        
        # Check IP address
        if not history.empty:
            common_ips = history['ip_address'].mode()
            if login_data['ip_address'] not in common_ips.values:
                risk_score += 0.2
                risk_factors.append('Unknown IP address')
        
        # Check location
        if 'location' in login_data and not history.empty:
            common_locations = history['location'].mode()
            if login_data['location'] not in common_locations.values:
                risk_score += 0.2
                risk_factors.append('Unknown location')
                
                # Check for impossible travel
                if not history.empty:
                    last_location = history.iloc[0]['location']
                    last_timestamp = history.iloc[0]['timestamp']
                    time_diff = (datetime.now() - last_timestamp).total_seconds() / 3600
                    
                    if time_diff < 2 and last_location != login_data['location']:
                        risk_score += 0.3
                        risk_factors.append('Impossible travel detected')
        
        # Check device
        if 'device_fingerprint' in login_data and not history.empty:
            if login_data['device_fingerprint'] not in history['device_fingerprint'].values:
                risk_score += 0.15
                risk_factors.append('Unknown device')
        
        # Check failed login attempts
        failed_attempts_query = f"""
        SELECT COUNT(*) as failed_count
        FROM `{self.project_id}.security.login_attempts`
        WHERE user_id = '{user_id}'
        AND success = FALSE
        AND timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 HOUR)
        """
        
        failed_count = self.bq_client.query(failed_attempts_query).to_dataframe().iloc[0]['failed_count']
        if failed_count > 3:
            risk_score += 0.25
            risk_factors.append(f'{failed_count} failed login attempts')
        
        return {
            'user_id': user_id,
            'risk_score': min(risk_score, 1.0),
            'risk_level': self._get_risk_level(risk_score),
            'risk_factors': risk_factors,
            'recommended_action': self._get_recommended_action(risk_score)
        }
    
    def _get_risk_level(self, score: float) -> str:
        """Get risk level from score."""
        if score >= 0.7:
            return 'HIGH'
        elif score >= 0.4:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _get_recommended_action(self, score: float) -> str:
        """Get recommended action based on risk score."""
        if score >= 0.7:
            return 'BLOCK_AND_NOTIFY'
        elif score >= 0.4:
            return 'REQUIRE_MFA'
        else:
            return 'ALLOW'


# Example usage
security_manager = AccountSecurityManager(
    project_id='my-project'
)

login_data = {
    'ip_address': '192.168.1.100',
    'device_fingerprint': 'abc123',
    'location': 'US-CA',
    'user_agent': 'Mozilla/5.0...'
}

takeover_risk = security_manager.detect_account_takeover(
    user_id='user_456',
    login_data=login_data
)
```

## 2. Anti-Money Laundering (AML)

### 2.1 Transaction Monitoring

```python
from typing import List, Tuple

class AMLMonitoringManager:
    """Manager for AML monitoring operations."""
    
    def __init__(self, project_id: str, dataset_id: str):
        """
        Initialize AML Monitoring Manager.
        
        Args:
            project_id: GCP project ID
            dataset_id: BigQuery dataset ID
        """
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.bq_client = bigquery.Client(project=project_id)
    
    def detect_structuring(
        self,
        account_id: str,
        threshold: float = 10000.0,
        time_window_days: int = 7
    ) -> Dict[str, Any]:
        """
        Detect structuring (smurfing) patterns.
        
        Args:
            account_id: Account identifier
            threshold: Reporting threshold
            time_window_days: Time window to analyze
            
        Returns:
            Dictionary with structuring detection results
        """
        query = f"""
        SELECT
            transaction_id,
            amount,
            timestamp,
            transaction_type
        FROM
            `{self.project_id}.{self.dataset_id}.transactions`
        WHERE
            account_id = '{account_id}'
            AND timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {time_window_days} DAY)
            AND transaction_type IN ('DEPOSIT', 'WITHDRAWAL')
        ORDER BY timestamp
        """
        
        transactions = self.bq_client.query(query).to_dataframe()
        
        if transactions.empty:
            return {'structuring_detected': False}
        
        # Check for multiple transactions just below threshold
        suspicious_tx = transactions[
            (transactions['amount'] >= threshold * 0.8) & 
            (transactions['amount'] < threshold)
        ]
        
        # Check for rapid succession
        structuring_patterns = []
        if len(suspicious_tx) >= 3:
            # Calculate time between transactions
            suspicious_tx = suspicious_tx.sort_values('timestamp')
            time_diffs = suspicious_tx['timestamp'].diff().dt.total_seconds() / 3600  # hours
            
            # Flag if multiple transactions within short time
            rapid_succession = time_diffs[time_diffs <= 24].count()
            if rapid_succession >= 2:
                structuring_patterns.append({
                    'pattern': 'RAPID_BELOW_THRESHOLD',
                    'transaction_count': len(suspicious_tx),
                    'total_amount': suspicious_tx['amount'].sum(),
                    'time_span_hours': (
                        suspicious_tx['timestamp'].max() - 
                        suspicious_tx['timestamp'].min()
                    ).total_seconds() / 3600
                })
        
        return {
            'account_id': account_id,
            'structuring_detected': len(structuring_patterns) > 0,
            'patterns': structuring_patterns,
            'suspicious_transaction_count': len(suspicious_tx),
            'total_suspicious_amount': suspicious_tx['amount'].sum()
        }
    
    def detect_layering(
        self,
        account_id: str,
        min_transfers: int = 5
    ) -> Dict[str, Any]:
        """
        Detect layering patterns (complex transfer chains).
        
        Args:
            account_id: Account identifier
            min_transfers: Minimum transfers to flag
            
        Returns:
            Dictionary with layering detection results
        """
        query = f"""
        WITH transfer_chains AS (
            SELECT
                t1.account_id as source_account,
                t1.destination_account as intermediate_account,
                t2.destination_account as final_account,
                t1.amount as initial_amount,
                t2.amount as final_amount,
                t1.timestamp as start_time,
                t2.timestamp as end_time
            FROM
                `{self.project_id}.{self.dataset_id}.transfers` t1
            JOIN
                `{self.project_id}.{self.dataset_id}.transfers` t2
            ON
                t1.destination_account = t2.account_id
            WHERE
                t1.account_id = '{account_id}'
                AND t2.timestamp >= t1.timestamp
                AND t2.timestamp <= TIMESTAMP_ADD(t1.timestamp, INTERVAL 7 DAY)
        )
        SELECT *
        FROM transfer_chains
        """
        
        chains = self.bq_client.query(query).to_dataframe()
        
        if chains.empty:
            return {'layering_detected': False}
        
        # Analyze transfer patterns
        layering_patterns = []
        
        # Group by destination to find complex chains
        destination_groups = chains.groupby('final_account')
        
        for dest_account, group in destination_groups:
            if len(group) >= min_transfers:
                layering_patterns.append({
                    'pattern': 'COMPLEX_TRANSFER_CHAIN',
                    'transfer_count': len(group),
                    'total_amount': group['final_amount'].sum(),
                    'intermediate_accounts': group['intermediate_account'].unique().tolist(),
                    'final_destination': dest_account
                })
        
        return {
            'account_id': account_id,
            'layering_detected': len(layering_patterns) > 0,
            'patterns': layering_patterns
        }
    
    def calculate_aml_risk_score(
        self,
        account_id: str
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive AML risk score.
        
        Args:
            account_id: Account identifier
            
        Returns:
            Dictionary with AML risk assessment
        """
        risk_score = 0.0
        risk_factors = []
        
        # Check for structuring
        structuring_result = self.detect_structuring(account_id)
        if structuring_result['structuring_detected']:
            risk_score += 0.4
            risk_factors.append('Structuring pattern detected')
        
        # Check for layering
        layering_result = self.detect_layering(account_id)
        if layering_result['layering_detected']:
            risk_score += 0.3
            risk_factors.append('Layering pattern detected')
        
        # Check transaction volume
        volume_query = f"""
        SELECT
            COUNT(*) as tx_count,
            SUM(amount) as total_amount
        FROM
            `{self.project_id}.{self.dataset_id}.transactions`
        WHERE
            account_id = '{account_id}'
            AND timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
        """
        
        volume = self.bq_client.query(volume_query).to_dataframe().iloc[0]
        
        if volume['total_amount'] > 100000:
            risk_score += 0.2
            risk_factors.append('High transaction volume')
        
        # Check high-risk jurisdictions
        jurisdiction_query = f"""
        SELECT DISTINCT destination_country
        FROM `{self.project_id}.{self.dataset_id}.international_transfers`
        WHERE account_id = '{account_id}'
        AND timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY)
        """
        
        countries = self.bq_client.query(jurisdiction_query).to_dataframe()
        high_risk_countries = ['XX', 'YY']  # Example high-risk countries
        
        if not countries.empty:
            risky_destinations = countries[
                countries['destination_country'].isin(high_risk_countries)
            ]
            if not risky_destinations.empty:
                risk_score += 0.1
                risk_factors.append('Transfers to high-risk jurisdictions')
        
        return {
            'account_id': account_id,
            'aml_risk_score': min(risk_score, 1.0),
            'risk_level': self._get_risk_level(risk_score),
            'risk_factors': risk_factors,
            'requires_sar': risk_score >= 0.7  # Suspicious Activity Report
        }
    
    def _get_risk_level(self, score: float) -> str:
        """Get risk level from score."""
        if score >= 0.7:
            return 'HIGH'
        elif score >= 0.4:
            return 'MEDIUM'
        else:
            return 'LOW'


# Example usage
aml_monitor = AMLMonitoringManager(
    project_id='my-project',
    dataset_id='financial_transactions'
)

# Calculate AML risk
aml_risk = aml_monitor.calculate_aml_risk_score(
    account_id='account_789'
)

print(f"AML Risk Score: {aml_risk['aml_risk_score']:.2f}")
print(f"Requires SAR: {aml_risk['requires_sar']}")
```

## 3. Credit Risk Assessment

### 3.1 Credit Scoring

```python
from google.cloud import aiplatform
from sklearn.ensemble import GradientBoostingClassifier
import joblib

class CreditRiskManager:
    """Manager for credit risk assessment."""
    
    def __init__(
        self,
        project_id: str,
        location: str = 'us-central1'
    ):
        """
        Initialize Credit Risk Manager.
        
        Args:
            project_id: GCP project ID
            location: GCP region
        """
        self.project_id = project_id
        self.location = location
        self.bq_client = bigquery.Client(project=project_id)
        aiplatform.init(project=project_id, location=location)
    
    def extract_credit_features(
        self,
        applicant_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Extract features for credit scoring.
        
        Args:
            applicant_data: Applicant information
            
        Returns:
            Dictionary of credit features
        """
        features = {}
        
        # Demographics
        features['age'] = applicant_data.get('age', 0)
        features['income'] = applicant_data.get('annual_income', 0)
        features['employment_length'] = applicant_data.get('employment_length_months', 0)
        
        # Credit history
        features['credit_history_length'] = applicant_data.get('credit_history_months', 0)
        features['num_credit_accounts'] = applicant_data.get('num_credit_accounts', 0)
        features['num_open_accounts'] = applicant_data.get('num_open_accounts', 0)
        features['total_credit_limit'] = applicant_data.get('total_credit_limit', 0)
        features['credit_utilization'] = applicant_data.get('credit_utilization_ratio', 0)
        
        # Payment history
        features['num_late_payments'] = applicant_data.get('num_late_payments_12m', 0)
        features['num_defaults'] = applicant_data.get('num_defaults', 0)
        features['num_bankruptcies'] = applicant_data.get('num_bankruptcies', 0)
        
        # Debt
        features['total_debt'] = applicant_data.get('total_debt', 0)
        features['debt_to_income'] = features['total_debt'] / (features['income'] + 1e-6)
        
        # Loan details
        features['loan_amount'] = applicant_data.get('requested_loan_amount', 0)
        features['loan_to_income'] = features['loan_amount'] / (features['income'] + 1e-6)
        
        return features
    
    def calculate_credit_score(
        self,
        applicant_data: Dict[str, Any],
        model_endpoint: str = None
    ) -> Dict[str, Any]:
        """
        Calculate credit score.
        
        Args:
            applicant_data: Applicant information
            model_endpoint: Vertex AI model endpoint
            
        Returns:
            Dictionary with credit score and decision
        """
        features = self.extract_credit_features(applicant_data)
        
        # Rule-based scoring (FICO-like)
        score = 850  # Start with perfect score
        
        # Payment history (35%)
        score -= features['num_late_payments'] * 20
        score -= features['num_defaults'] * 100
        score -= features['num_bankruptcies'] * 150
        
        # Credit utilization (30%)
        if features['credit_utilization'] > 0.7:
            score -= 100
        elif features['credit_utilization'] > 0.5:
            score -= 50
        elif features['credit_utilization'] > 0.3:
            score -= 20
        
        # Credit history length (15%)
        if features['credit_history_length'] < 24:
            score -= 50
        elif features['credit_history_length'] < 60:
            score -= 25
        
        # Credit mix (10%)
        if features['num_credit_accounts'] < 3:
            score -= 30
        
        # New credit (10%)
        if features['num_open_accounts'] > 5:
            score -= 20
        
        # Debt-to-income ratio
        if features['debt_to_income'] > 0.43:
            score -= 50
        
        # Ensure score is in valid range
        score = max(300, min(850, score))
        
        # ML model scoring (if available)
        if model_endpoint:
            try:
                endpoint = aiplatform.Endpoint(model_endpoint)
                feature_vector = [[v for v in features.values()]]
                predictions = endpoint.predict(instances=feature_vector)
                ml_score = int(predictions.predictions[0][0])
                
                # Average rule-based and ML scores
                score = int((score + ml_score) / 2)
            except Exception as e:
                print(f"ML model prediction error: {e}")
        
        # Determine credit decision
        if score >= 740:
            decision = 'APPROVED'
            risk_level = 'LOW'
        elif score >= 670:
            decision = 'APPROVED'
            risk_level = 'MEDIUM'
        elif score >= 580:
            decision = 'MANUAL_REVIEW'
            risk_level = 'HIGH'
        else:
            decision = 'DECLINED'
            risk_level = 'VERY_HIGH'
        
        return {
            'applicant_id': applicant_data.get('applicant_id'),
            'credit_score': score,
            'decision': decision,
            'risk_level': risk_level,
            'key_factors': self._get_key_factors(features),
            'features': features
        }
    
    def _get_key_factors(self, features: Dict[str, float]) -> List[str]:
        """Get key factors affecting credit score."""
        factors = []
        
        if features['num_late_payments'] > 0:
            factors.append(f"Late payments: {int(features['num_late_payments'])}")
        
        if features['credit_utilization'] > 0.5:
            factors.append(f"High credit utilization: {features['credit_utilization']:.1%}")
        
        if features['debt_to_income'] > 0.4:
            factors.append(f"High debt-to-income: {features['debt_to_income']:.1%}")
        
        if features['credit_history_length'] < 24:
            factors.append("Short credit history")
        
        if features['num_defaults'] > 0:
            factors.append(f"Defaults: {int(features['num_defaults'])}")
        
        return factors
    
    def batch_credit_assessment(
        self,
        applicant_ids: List[str]
    ) -> pd.DataFrame:
        """
        Perform batch credit assessment.
        
        Args:
            applicant_ids: List of applicant IDs
            
        Returns:
            DataFrame with assessment results
        """
        query = f"""
        SELECT *
        FROM `{self.project_id}.credit.applicants`
        WHERE applicant_id IN UNNEST(@applicant_ids)
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("applicant_ids", "STRING", applicant_ids)
            ]
        )
        
        applicants = self.bq_client.query(query, job_config=job_config).to_dataframe()
        
        results = []
        for _, applicant in applicants.iterrows():
            result = self.calculate_credit_score(applicant.to_dict())
            results.append(result)
        
        return pd.DataFrame(results)


# Example usage
credit_manager = CreditRiskManager(
    project_id='my-project',
    location='us-central1'
)

# Assess credit application
applicant = {
    'applicant_id': 'app_123',
    'age': 35,
    'annual_income': 75000,
    'employment_length_months': 60,
    'credit_history_months': 120,
    'num_credit_accounts': 5,
    'num_open_accounts': 3,
    'total_credit_limit': 50000,
    'credit_utilization_ratio': 0.3,
    'num_late_payments_12m': 1,
    'num_defaults': 0,
    'num_bankruptcies': 0,
    'total_debt': 20000,
    'requested_loan_amount': 15000
}

credit_result = credit_manager.calculate_credit_score(applicant)
print(f"Credit Score: {credit_result['credit_score']}")
print(f"Decision: {credit_result['decision']}")
```

## 4. Document Processing (KYC/KYB)

### 4.1 Identity Verification

```python
from google.cloud import documentai_v1 as documentai
from google.cloud import vision

class DocumentProcessingManager:
    """Manager for financial document processing."""
    
    def __init__(
        self,
        project_id: str,
        location: str = 'us',
        processor_id: str = None
    ):
        """
        Initialize Document Processing Manager.
        
        Args:
            project_id: GCP project ID
            location: Processor location
            processor_id: Document AI processor ID
        """
        self.project_id = project_id
        self.location = location
        self.processor_id = processor_id
        self.docai_client = documentai.DocumentProcessorServiceClient()
        self.vision_client = vision.ImageAnnotatorClient()
    
    def process_identity_document(
        self,
        document_path: str,
        document_type: str = 'drivers_license'
    ) -> Dict[str, Any]:
        """
        Process identity document.
        
        Args:
            document_path: GCS path to document
            document_type: Type of document
            
        Returns:
            Dictionary with extracted information
        """
        # Read document
        with open(document_path, 'rb') as doc_file:
            document_content = doc_file.read()
        
        # Create processor name
        processor_name = self.docai_client.processor_path(
            self.project_id,
            self.location,
            self.processor_id
        )
        
        # Configure request
        request = documentai.ProcessRequest(
            name=processor_name,
            raw_document=documentai.RawDocument(
                content=document_content,
                mime_type='application/pdf'
            )
        )
        
        # Process document
        result = self.docai_client.process_document(request=request)
        document = result.document
        
        # Extract entities
        extracted_data = {
            'document_type': document_type,
            'entities': {}
        }
        
        for entity in document.entities:
            extracted_data['entities'][entity.type_] = entity.mention_text
        
        # Validate extracted data
        validation_result = self._validate_identity_data(extracted_data)
        extracted_data['validation'] = validation_result
        
        return extracted_data
    
    def _validate_identity_data(
        self,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate identity data.
        
        Args:
            data: Extracted identity data
            
        Returns:
            Dictionary with validation results
        """
        validation = {
            'is_valid': True,
            'errors': []
        }
        
        entities = data.get('entities', {})
        
        # Check required fields
        required_fields = ['full_name', 'date_of_birth', 'document_number']
        for field in required_fields:
            if field not in entities:
                validation['is_valid'] = False
                validation['errors'].append(f"Missing required field: {field}")
        
        # Validate date of birth format
        if 'date_of_birth' in entities:
            try:
                dob = datetime.strptime(entities['date_of_birth'], '%Y-%m-%d')
                age = (datetime.now() - dob).days / 365.25
                if age < 18:
                    validation['is_valid'] = False
                    validation['errors'].append("Applicant must be 18 or older")
            except ValueError:
                validation['is_valid'] = False
                validation['errors'].append("Invalid date of birth format")
        
        return validation
    
    def verify_document_authenticity(
        self,
        document_path: str
    ) -> Dict[str, Any]:
        """
        Verify document authenticity using Vision API.
        
        Args:
            document_path: Path to document image
            
        Returns:
            Dictionary with authenticity check results
        """
        with open(document_path, 'rb') as image_file:
            content = image_file.read()
        
        image = vision.Image(content=content)
        
        # Detect document features
        response = self.vision_client.document_text_detection(image=image)
        
        authenticity = {
            'is_authentic': True,
            'confidence': 1.0,
            'checks': []
        }
        
        # Check for text quality
        if response.full_text_annotation:
            text_confidence = sum(
                page.confidence 
                for page in response.full_text_annotation.pages
            ) / len(response.full_text_annotation.pages)
            
            if text_confidence < 0.7:
                authenticity['is_authentic'] = False
                authenticity['checks'].append('Low text quality')
                authenticity['confidence'] = text_confidence
        
        # Additional authenticity checks would go here
        # (e.g., security features, watermarks, holograms)
        
        return authenticity


# Example usage
doc_processor = DocumentProcessingManager(
    project_id='my-project',
    location='us',
    processor_id='abc123def456'
)

# Process identity document
identity_data = doc_processor.process_identity_document(
    document_path='/path/to/drivers_license.pdf',
    document_type='drivers_license'
)

# Verify authenticity
authenticity = doc_processor.verify_document_authenticity(
    document_path='/path/to/drivers_license.jpg'
)
```

## 5. Quick Reference Checklist

### Fraud Detection
- [ ] Implement real-time transaction monitoring
- [ ] Set up fraud scoring models
- [ ] Configure rule-based detection
- [ ] Enable ML-based anomaly detection
- [ ] Create fraud alert workflows
- [ ] Implement case management system
- [ ] Set up false positive feedback loop

### AML Compliance
- [ ] Implement transaction monitoring
- [ ] Detect structuring patterns
- [ ] Identify layering schemes
- [ ] Monitor high-risk jurisdictions
- [ ] Generate Suspicious Activity Reports (SARs)
- [ ] Maintain audit trails
- [ ] Implement customer due diligence (CDD)

### Credit Risk
- [ ] Build credit scoring models
- [ ] Extract relevant features
- [ ] Implement decision rules
- [ ] Set up batch assessment pipelines
- [ ] Monitor model performance
- [ ] Track default rates
- [ ] Implement model recalibration

### Document Processing
- [ ] Set up Document AI processors
- [ ] Implement identity verification
- [ ] Validate extracted data
- [ ] Check document authenticity
- [ ] Store documents securely
- [ ] Implement OCR quality checks
- [ ] Enable fraud detection on documents

### Security & Compliance
- [ ] Implement data encryption (at rest/in transit)
- [ ] Set up access controls (IAM)
- [ ] Enable audit logging
- [ ] Ensure PCI-DSS compliance
- [ ] Implement data retention policies
- [ ] Set up secure key management (KMS)
- [ ] Enable VPC Service Controls

### Performance Optimization
- [ ] Use batch prediction for bulk assessments
- [ ] Implement caching for frequent queries
- [ ] Optimize feature extraction
- [ ] Use streaming for real-time detection
- [ ] Monitor model latency
- [ ] Optimize database queries
- [ ] Use appropriate instance types

### Cost Management
- [ ] Use BigQuery flat-rate pricing
- [ ] Optimize query performance
- [ ] Implement data lifecycle policies
- [ ] Use preemptible instances where possible
- [ ] Monitor API usage
- [ ] Use batch processing for non-urgent tasks
- [ ] Implement cost alerts

### Monitoring & Alerting
- [ ] Set up Cloud Monitoring dashboards
- [ ] Configure alerting for high-risk events
- [ ] Monitor model performance metrics
- [ ] Track false positive/negative rates
- [ ] Monitor processing latency
- [ ] Set up SLA monitoring
- [ ] Create incident response playbooks
