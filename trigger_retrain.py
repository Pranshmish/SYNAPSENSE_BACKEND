import requests
import json

BASE_URL = "http://localhost:8000"

def trigger_retrain():
    print("🚀 Triggering Model Retraining...")
    try:
        # Call the train endpoint
        resp = requests.post(f"{BASE_URL}/train")
        
        if resp.status_code == 200:
            result = resp.json()
            metrics = result.get('metrics', {})
            
            print("\n✅ Retraining Complete!")
            print(f"New Training Accuracy: {metrics.get('training_accuracy')}%")
            print(f"CV Accuracy: {metrics.get('cv_accuracy', 'N/A')}")
            print(f"OOB Accuracy: {metrics.get('oob_accuracy', 'N/A')}")
            print(f"Classes: {metrics.get('classes')}")
            
            if metrics.get('training_accuracy') < 100:
                print("\n📉 Accuracy has dropped from 100%, indicating reduced overfitting.")
            else:
                print("\n⚠️ Accuracy is still 100%. The synthetic data might be too easy.")
                
        else:
            print(f"❌ Training failed: {resp.text}")
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")

if __name__ == "__main__":
    trigger_retrain()
