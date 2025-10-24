import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime, timedelta
import asyncio
import json
from collections import deque
import os

logger = logging.getLogger(__name__)

class ModelMonitoringSystem:
    """
    Comprehensive monitoring system for ClashVision v3.0-PCI-RL
    
    Tracks:
    - prediction_accuracy_rolling_100
    - PCI_distribution_shift
    - model_confidence_drift
    - error_correlation_with_PCI
    
    Triggers retraining when:
    - accuracy_drop > 5%
    - PCI_correlation > 0.25
    - population_shift > 10%
    """
    
    def __init__(
        self,
        enhanced_predictor,
        monitoring_interval_minutes: int = 10,
        alert_thresholds: Dict[str, float] = None
    ):
        self.enhanced_predictor = enhanced_predictor
        self.monitoring_interval_minutes = monitoring_interval_minutes
        
        # Default alert thresholds
        self.alert_thresholds = alert_thresholds or {
            'accuracy_drop': 0.05,  # 5%
            'pci_correlation': 0.25,
            'population_shift': 0.10,  # 10%
            'confidence_drift': 0.15,  # 15%
            'error_rate_spike': 0.20   # 20%
        }
        
        # Monitoring data storage
        self.metrics_history = deque(maxlen=1000)  # Store last 1000 monitoring cycles
        self.prediction_logs = deque(maxlen=10000)  # Store last 10k predictions
        self.pci_distribution_history = deque(maxlen=100)  # PCI distribution snapshots
        self.alert_history = deque(maxlen=500)  # Alert history
        
        # Monitoring state
        self.is_monitoring = False
        self.last_monitoring_time = None
        self.baseline_metrics = None
        
        # Performance tracking
        self.rolling_accuracy = deque(maxlen=100)
        self.rolling_confidence = deque(maxlen=100)
        self.rolling_pci_values = deque(maxlen=100)
        self.rolling_errors = deque(maxlen=100)
        
    async def start_monitoring(self):
        """Start the monitoring system"""
        try:
            logger.info("Starting Model Monitoring System...")
            
            self.is_monitoring = True
            
            # Initialize baseline metrics
            await self._establish_baseline()
            
            # Start monitoring loop
            asyncio.create_task(self._monitoring_loop())
            
            logger.info("Model Monitoring System started successfully")
            
        except Exception as e:
            logger.error(f"Error starting monitoring system: {e}")
            self.is_monitoring = False
    
    async def stop_monitoring(self):
        """Stop the monitoring system"""
        self.is_monitoring = False
        logger.info("Model Monitoring System stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                await self._collect_metrics()
                await self._analyze_metrics()
                await self._check_alert_conditions()
                
                # Wait for next monitoring cycle
                await asyncio.sleep(self.monitoring_interval_minutes * 60)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    async def _establish_baseline(self):
        """Establish baseline metrics for comparison"""
        try:
            if self.enhanced_predictor.is_ready:
                current_metrics = self.enhanced_predictor.get_performance_metrics()
                
                self.baseline_metrics = {
                    'accuracy': current_metrics.get('current_accuracy', 0.5),
                    'confidence_avg': current_metrics.get('confidence_avg', 0.5),
                    'timestamp': datetime.now().isoformat()
                }
                
                logger.info(f"Baseline metrics established: {self.baseline_metrics}")
            
        except Exception as e:
            logger.error(f"Error establishing baseline: {e}")
    
    async def _collect_metrics(self):
        """Collect current performance metrics"""
        try:
            timestamp = datetime.now()
            
            # Get current performance metrics
            if self.enhanced_predictor.is_ready:
                performance_metrics = self.enhanced_predictor.get_performance_metrics()
            else:
                performance_metrics = {}
            
            # Calculate rolling metrics
            rolling_metrics = self._calculate_rolling_metrics()
            
            # Calculate PCI distribution
            pci_distribution = self._calculate_pci_distribution()
            
            # Combine all metrics
            current_metrics = {
                'timestamp': timestamp.isoformat(),
                'performance': performance_metrics,
                'rolling': rolling_metrics,
                'pci_distribution': pci_distribution
            }
            
            # Store metrics
            self.metrics_history.append(current_metrics)
            self.last_monitoring_time = timestamp
            
            logger.debug(f"Metrics collected at {timestamp}")
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
    
    def _calculate_rolling_metrics(self) -> Dict[str, float]:
        """Calculate rolling window metrics"""
        try:
            metrics = {}
            
            if self.rolling_accuracy:
                metrics['accuracy_rolling_100'] = np.mean(list(self.rolling_accuracy))
                metrics['accuracy_std'] = np.std(list(self.rolling_accuracy))
            
            if self.rolling_confidence:
                metrics['confidence_rolling_avg'] = np.mean(list(self.rolling_confidence))
                metrics['confidence_std'] = np.std(list(self.rolling_confidence))
            
            if self.rolling_pci_values:
                metrics['pci_rolling_avg'] = np.mean(list(self.rolling_pci_values))
                metrics['pci_std'] = np.std(list(self.rolling_pci_values))
            
            if self.rolling_errors:
                metrics['error_rate'] = np.mean(list(self.rolling_errors))
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating rolling metrics: {e}")
            return {}
    
    def _calculate_pci_distribution(self) -> Dict[str, Any]:
        """Calculate current PCI distribution"""
        try:
            if not self.rolling_pci_values:
                return {}
            
            pci_values = list(self.rolling_pci_values)
            
            distribution = {
                'mean': np.mean(pci_values),
                'std': np.std(pci_values),
                'min': np.min(pci_values),
                'max': np.max(pci_values),
                'quartiles': {
                    'q25': np.percentile(pci_values, 25),
                    'q50': np.percentile(pci_values, 50),
                    'q75': np.percentile(pci_values, 75)
                },
                'bins': {
                    'very_low': np.sum(np.array(pci_values) < 0.25) / len(pci_values),
                    'low': np.sum((np.array(pci_values) >= 0.25) & (np.array(pci_values) < 0.5)) / len(pci_values),
                    'medium': np.sum((np.array(pci_values) >= 0.5) & (np.array(pci_values) < 0.75)) / len(pci_values),
                    'high': np.sum(np.array(pci_values) >= 0.75) / len(pci_values)
                }
            }
            
            return distribution
            
        except Exception as e:
            logger.error(f"Error calculating PCI distribution: {e}")
            return {}
    
    async def _analyze_metrics(self):
        """Analyze collected metrics for trends and anomalies"""
        try:
            if len(self.metrics_history) < 2:
                return
            
            current_metrics = self.metrics_history[-1]
            previous_metrics = self.metrics_history[-2]
            
            # Analyze accuracy trend
            await self._analyze_accuracy_trend(current_metrics, previous_metrics)
            
            # Analyze PCI distribution shift
            await self._analyze_pci_shift(current_metrics, previous_metrics)
            
            # Analyze confidence drift
            await self._analyze_confidence_drift(current_metrics, previous_metrics)
            
            # Analyze error correlation with PCI
            await self._analyze_error_pci_correlation()
            
        except Exception as e:
            logger.error(f"Error analyzing metrics: {e}")
    
    async def _analyze_accuracy_trend(self, current: Dict, previous: Dict):
        """Analyze accuracy trends"""
        try:
            current_acc = current.get('rolling', {}).get('accuracy_rolling_100', 0)
            previous_acc = previous.get('rolling', {}).get('accuracy_rolling_100', 0)
            
            if current_acc > 0 and previous_acc > 0:
                accuracy_change = current_acc - previous_acc
                
                if abs(accuracy_change) > 0.02:  # 2% change threshold
                    trend = "increasing" if accuracy_change > 0 else "decreasing"
                    logger.info(f"Accuracy trend detected: {trend} by {abs(accuracy_change):.4f}")
                    
                    # Store trend information
                    trend_info = {
                        'type': 'accuracy_trend',
                        'direction': trend,
                        'magnitude': abs(accuracy_change),
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    self.alert_history.append(trend_info)
            
        except Exception as e:
            logger.error(f"Error analyzing accuracy trend: {e}")
    
    async def _analyze_pci_shift(self, current: Dict, previous: Dict):
        """Analyze PCI distribution shifts"""
        try:
            current_pci = current.get('pci_distribution', {})
            previous_pci = previous.get('pci_distribution', {})
            
            if current_pci and previous_pci:
                # Compare distribution means
                current_mean = current_pci.get('mean', 0.5)
                previous_mean = previous_pci.get('mean', 0.5)
                
                shift_magnitude = abs(current_mean - previous_mean)
                
                if shift_magnitude > 0.05:  # 5% shift threshold
                    logger.warning(f"PCI distribution shift detected: {shift_magnitude:.4f}")
                    
                    shift_info = {
                        'type': 'pci_distribution_shift',
                        'magnitude': shift_magnitude,
                        'direction': 'higher' if current_mean > previous_mean else 'lower',
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    self.alert_history.append(shift_info)
            
        except Exception as e:
            logger.error(f"Error analyzing PCI shift: {e}")
    
    async def _analyze_confidence_drift(self, current: Dict, previous: Dict):
        """Analyze model confidence drift"""
        try:
            current_conf = current.get('rolling', {}).get('confidence_rolling_avg', 0)
            previous_conf = previous.get('rolling', {}).get('confidence_rolling_avg', 0)
            
            if current_conf > 0 and previous_conf > 0:
                confidence_change = abs(current_conf - previous_conf)
                
                if confidence_change > 0.1:  # 10% confidence change
                    logger.warning(f"Model confidence drift detected: {confidence_change:.4f}")
                    
                    drift_info = {
                        'type': 'confidence_drift',
                        'magnitude': confidence_change,
                        'direction': 'increasing' if current_conf > previous_conf else 'decreasing',
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    self.alert_history.append(drift_info)
            
        except Exception as e:
            logger.error(f"Error analyzing confidence drift: {e}")
    
    async def _analyze_error_pci_correlation(self):
        """Analyze correlation between prediction errors and PCI values"""
        try:
            if len(self.rolling_errors) < 20 or len(self.rolling_pci_values) < 20:
                return
            
            errors = list(self.rolling_errors)[-20:]  # Last 20 errors
            pci_values = list(self.rolling_pci_values)[-20:]  # Last 20 PCI values
            
            if len(errors) == len(pci_values):
                correlation = np.corrcoef(errors, pci_values)[0, 1]
                
                if not np.isnan(correlation) and abs(correlation) > 0.3:
                    logger.warning(f"Strong error-PCI correlation detected: {correlation:.4f}")
                    
                    correlation_info = {
                        'type': 'error_pci_correlation',
                        'correlation': correlation,
                        'strength': 'strong' if abs(correlation) > 0.5 else 'moderate',
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    self.alert_history.append(correlation_info)
            
        except Exception as e:
            logger.error(f"Error analyzing error-PCI correlation: {e}")
    
    async def _check_alert_conditions(self):
        """Check if any alert conditions are met"""
        try:
            if not self.metrics_history:
                return
            
            current_metrics = self.metrics_history[-1]
            
            # Check accuracy drop
            await self._check_accuracy_drop_alert(current_metrics)
            
            # Check PCI correlation alert
            await self._check_pci_correlation_alert()
            
            # Check population shift alert
            await self._check_population_shift_alert()
            
        except Exception as e:
            logger.error(f"Error checking alert conditions: {e}")
    
    async def _check_accuracy_drop_alert(self, current_metrics: Dict):
        """Check for accuracy drop alert"""
        try:
            if not self.baseline_metrics:
                return
            
            current_accuracy = current_metrics.get('rolling', {}).get('accuracy_rolling_100', 0)
            baseline_accuracy = self.baseline_metrics.get('accuracy', 0)
            
            if current_accuracy > 0 and baseline_accuracy > 0:
                accuracy_drop = baseline_accuracy - current_accuracy
                
                if accuracy_drop > self.alert_thresholds['accuracy_drop']:
                    await self._trigger_alert('accuracy_drop', {
                        'current_accuracy': current_accuracy,
                        'baseline_accuracy': baseline_accuracy,
                        'drop_magnitude': accuracy_drop
                    })
            
        except Exception as e:
            logger.error(f"Error checking accuracy drop alert: {e}")
    
    async def _check_pci_correlation_alert(self):
        """Check for PCI correlation alert"""
        try:
            if len(self.rolling_errors) < 30 or len(self.rolling_pci_values) < 30:
                return
            
            errors = list(self.rolling_errors)
            pci_values = list(self.rolling_pci_values)
            
            correlation = np.corrcoef(errors, pci_values)[0, 1]
            
            if not np.isnan(correlation) and abs(correlation) > self.alert_thresholds['pci_correlation']:
                await self._trigger_alert('pci_correlation', {
                    'correlation': correlation,
                    'threshold': self.alert_thresholds['pci_correlation']
                })
            
        except Exception as e:
            logger.error(f"Error checking PCI correlation alert: {e}")
    
    async def _check_population_shift_alert(self):
        """Check for population shift alert"""
        try:
            if len(self.pci_distribution_history) < 2:
                return
            
            current_dist = self.pci_distribution_history[-1]
            baseline_dist = self.pci_distribution_history[0]  # Use first as baseline
            
            # Compare distribution bins
            shift_magnitude = 0
            for bin_name in ['very_low', 'low', 'medium', 'high']:
                current_prop = current_dist.get('bins', {}).get(bin_name, 0)
                baseline_prop = baseline_dist.get('bins', {}).get(bin_name, 0)
                shift_magnitude += abs(current_prop - baseline_prop)
            
            shift_magnitude /= 4  # Average shift across bins
            
            if shift_magnitude > self.alert_thresholds['population_shift']:
                await self._trigger_alert('population_shift', {
                    'shift_magnitude': shift_magnitude,
                    'threshold': self.alert_thresholds['population_shift']
                })
            
        except Exception as e:
            logger.error(f"Error checking population shift alert: {e}")
    
    async def _trigger_alert(self, alert_type: str, alert_data: Dict):
        """Trigger an alert and potentially initiate retraining"""
        try:
            alert = {
                'type': alert_type,
                'timestamp': datetime.now().isoformat(),
                'data': alert_data,
                'severity': self._determine_alert_severity(alert_type, alert_data)
            }
            
            self.alert_history.append(alert)
            
            logger.warning(f"ALERT TRIGGERED: {alert_type} - {alert_data}")
            
            # Check if retraining should be triggered
            if alert['severity'] in ['high', 'critical']:
                await self._trigger_retraining(alert)
            
        except Exception as e:
            logger.error(f"Error triggering alert: {e}")
    
    def _determine_alert_severity(self, alert_type: str, alert_data: Dict) -> str:
        """Determine alert severity level"""
        try:
            if alert_type == 'accuracy_drop':
                drop = alert_data.get('drop_magnitude', 0)
                if drop > 0.15:  # 15% drop
                    return 'critical'
                elif drop > 0.10:  # 10% drop
                    return 'high'
                else:
                    return 'medium'
            
            elif alert_type == 'pci_correlation':
                correlation = abs(alert_data.get('correlation', 0))
                if correlation > 0.5:
                    return 'high'
                else:
                    return 'medium'
            
            elif alert_type == 'population_shift':
                shift = alert_data.get('shift_magnitude', 0)
                if shift > 0.2:  # 20% shift
                    return 'high'
                else:
                    return 'medium'
            
            return 'low'
            
        except Exception as e:
            logger.error(f"Error determining alert severity: {e}")
            return 'low'
    
    async def _trigger_retraining(self, alert: Dict):
        """Trigger model retraining based on alert"""
        try:
            logger.info(f"Triggering retraining due to alert: {alert['type']}")
            
            # Get training pipeline from enhanced predictor
            if hasattr(self.enhanced_predictor, 'training_pipeline') and self.enhanced_predictor.training_pipeline:
                training_pipeline = self.enhanced_predictor.training_pipeline
                
                # Create dummy training data (in production, this would fetch real data)
                training_data = pd.DataFrame()  # Would be populated with real data
                
                if len(training_data) > 0:
                    # Trigger retraining
                    await training_pipeline.train_model(
                        training_data=training_data,
                        force_retrain=True,
                        hyperparameter_tuning=True
                    )
                    
                    logger.info("Retraining completed successfully")
                else:
                    logger.warning("Insufficient training data for retraining")
            
        except Exception as e:
            logger.error(f"Error triggering retraining: {e}")
    
    def log_prediction(
        self,
        prediction_data: Dict[str, Any],
        actual_outcome: Optional[bool] = None
    ):
        """Log a prediction for monitoring"""
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'win_probability': prediction_data.get('win_probability', 0.5),
                'confidence': prediction_data.get('confidence', 0.5),
                'pci_value': prediction_data.get('pci_value', 0.5),
                'actual_outcome': actual_outcome
            }
            
            self.prediction_logs.append(log_entry)
            
            # Update rolling metrics
            self.rolling_confidence.append(log_entry['confidence'])
            self.rolling_pci_values.append(log_entry['pci_value'])
            
            if actual_outcome is not None:
                predicted_win = log_entry['win_probability'] > 0.5
                was_correct = predicted_win == actual_outcome
                
                self.rolling_accuracy.append(1.0 if was_correct else 0.0)
                self.rolling_errors.append(0.0 if was_correct else 1.0)
            
        except Exception as e:
            logger.error(f"Error logging prediction: {e}")
    
    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get monitoring dashboard data"""
        try:
            if not self.metrics_history:
                return {"status": "no_data"}
            
            latest_metrics = self.metrics_history[-1]
            
            dashboard = {
                "status": "active" if self.is_monitoring else "inactive",
                "last_update": self.last_monitoring_time.isoformat() if self.last_monitoring_time else None,
                "current_metrics": {
                    "accuracy_rolling_100": latest_metrics.get('rolling', {}).get('accuracy_rolling_100', 0),
                    "confidence_avg": latest_metrics.get('rolling', {}).get('confidence_rolling_avg', 0),
                    "pci_avg": latest_metrics.get('rolling', {}).get('pci_rolling_avg', 0.5),
                    "error_rate": latest_metrics.get('rolling', {}).get('error_rate', 0)
                },
                "pci_distribution": latest_metrics.get('pci_distribution', {}),
                "recent_alerts": list(self.alert_history)[-10:],  # Last 10 alerts
                "alert_thresholds": self.alert_thresholds,
                "baseline_metrics": self.baseline_metrics
            }
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Error getting monitoring dashboard: {e}")
            return {"status": "error", "message": str(e)}
    
    def save_monitoring_data(self, filepath: str):
        """Save monitoring data to file"""
        try:
            monitoring_data = {
                "metrics_history": list(self.metrics_history),
                "prediction_logs": list(self.prediction_logs),
                "alert_history": list(self.alert_history),
                "baseline_metrics": self.baseline_metrics,
                "alert_thresholds": self.alert_thresholds
            }
            
            with open(filepath, 'w') as f:
                json.dump(monitoring_data, f, indent=2, default=str)
            
            logger.info(f"Monitoring data saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving monitoring data: {e}")
    
    def load_monitoring_data(self, filepath: str):
        """Load monitoring data from file"""
        try:
            with open(filepath, 'r') as f:
                monitoring_data = json.load(f)
            
            self.metrics_history = deque(monitoring_data.get('metrics_history', []), maxlen=1000)
            self.prediction_logs = deque(monitoring_data.get('prediction_logs', []), maxlen=10000)
            self.alert_history = deque(monitoring_data.get('alert_history', []), maxlen=500)
            self.baseline_metrics = monitoring_data.get('baseline_metrics')
            self.alert_thresholds.update(monitoring_data.get('alert_thresholds', {}))
            
            logger.info(f"Monitoring data loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading monitoring data: {e}")
