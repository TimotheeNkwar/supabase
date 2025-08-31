# Statistical Analysis in R
library(dplyr)
library(ggplot2)

# A/B Test Analysis Function
analyze_ab_test <- function(control, treatment) {
  control_mean <- mean(control)
  treatment_mean <- mean(treatment)
  test_result <- t.test(treatment, control)
  pooled_sd <- sqrt(((length(control)-1)*var(control) + 
                    (length(treatment)-1)*var(treatment)) / 
                   (length(control) + length(treatment) - 2))
  effect_size <- (treatment_mean - control_mean) / pooled_sd
  return(list(
    control_mean = control_mean,
    treatment_mean = treatment_mean,
    p_value = test_result$p.value,
    effect_size = effect_size,
    significant = test_result$p.value < 0.05
  ))
}







import rpy2.robjects as ro
from rpy2.robjects import r






















# Load R script
r.source("ab_test.R")

@app.route('/')
def index():
    return render_template('skills_tools.html')

@app.route('/api/run_ab_test', methods=['POST'])
def run_ab_test():
    """
    Run an A/B test using the analyze_ab_test R function.

    POST data should contain two lists: "control" and "treatment". If not provided, default values are used.

    Returns a JSON object with the means of the control and treatment groups, the lift as a percentage, and a boolean indicating whether the result is statistically significant.

    Example request:
    {
        "control": [2.3, 2.1, 2.4, 2.2, 2.5, 2.0, 2.3],
        "treatment": [2.8, 2.9, 2.7, 3.0, 2.6, 2.8, 2.9]
    }
    Example response:
    {
        "control_mean": 2.26,
        "treatment_mean": 2.81,
        "lift": 24.3,
        "significant": true
    }
    """

    try:
        # Get data from request (optional: allow custom input)
        data = request.get_json()
        control = data.get('control', [2.3, 2.1, 2.4, 2.2, 2.5, 2.0, 2.3])
        treatment = data.get('treatment', [2.8, 2.9, 2.7, 3.0, 2.6, 2.8, 2.9])

        # Convert lists to R vectors
        control_r = ro.FloatVector(control)
        treatment_r = ro.FloatVector(treatment)

        # Call R function
        results = r['analyze_ab_test'](control_r, treatment_r)
        lift = round((results.rx2('treatment_mean')[0] / results.rx2('control_mean')[0] - 1) * 100, 1)

        return jsonify({
            'control_mean': round(results.rx2('control_mean')[0], 2),
            'treatment_mean': round(results.rx2('treatment_mean')[0], 2),
            'lift': lift,
            'significant': results.rx2('significant')[0]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
