
document.getElementById('churn-form').addEventListener('submit', async function(event) {
    event.preventDefault();
    await predictChurn();
});

async function predictChurn() {
    const button = document.getElementById('predict-button');
    button.disabled = true;
    button.textContent = 'Predicting...';

    const monthlyCharges = document.getElementById('monthly-charges').value;
    const tenure = document.getElementById('tenure').value;
    const totalCharges = document.getElementById('total-charges').value;
    const contractType = document.getElementById('contract-type').value;

    if (!monthlyCharges || !tenure || !totalCharges || !contractType) {
        alert('Please fill in all fields.');
        button.disabled = false;
        button.textContent = 'Predict Churn Risk';
        return;
    }

    const data = {
        monthly_charges: parseFloat(monthlyCharges),
        tenure: parseInt(tenure),
        total_charges: parseFloat(totalCharges),
        contract_type: contractType
    };

    if (data.monthly_charges < 0 || data.monthly_charges > 200) {
        alert('Monthly charges must be between 0 and 200.');
        button.disabled = false;
        button.textContent = 'Predict Churn Risk';
        return;
    }
    if (data.tenure < 0 || data.tenure > 72) {
        alert('Tenure must be between 0 and 72 months.');
        button.disabled = false;
        button.textContent = 'Predict Churn Risk';
        return;
    }
    if (data.total_charges < 0 || data.total_charges > 10000) {
        alert('Total charges must be between 0 and 10000.');
        button.disabled = false;
        button.textContent = 'Predict Churn Risk';
        return;
    }

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            throw new Error('Prediction request failed.');
        }

        const result = await response.json();
        const churnResult = document.getElementById('churn-result');
        const churnProbability = document.getElementById('churn-probability');
        const churnBar = document.getElementById('churn-bar');
        const churnRecommendation = document.getElementById('churn-recommendation');

        const probabilityPercent = (result.probability * 100).toFixed(0);
        churnProbability.textContent = `${probabilityPercent}%`;
        churnBar.style.width = `${probabilityPercent}%`;
        churnBar.classList.remove('bg-error', 'bg-warning', 'bg-success');
        if (result.probability >= 0.7) {
            churnBar.classList.add('bg-error');
        } else if (result.probability >= 0.5) {
            churnBar.classList.add('bg-warning');
        } else {
            churnBar.classList.add('bg-success');
        }
        churnRecommendation.innerHTML = `<strong>Recommendation:</strong> ${result.recommendation}`;
        churnResult.classList.remove('hidden');

        button.disabled = false;
        button.textContent = 'Predict Churn Risk';
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while predicting churn. Please try again.');
        button.disabled = false;
        button.textContent = 'Predict Churn Risk';
    }
}
