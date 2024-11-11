from flask import Flask, render_template, request, url_for, session
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats

app = Flask(__name__)
app.secret_key = "your_secret_key_here"  # Replace with your own secret key, needed for session management

def generate_data(N, mu, beta0, beta1, sigma2, S):
    # Generate data and initial plots
    X = np.random.rand(N)  # Generate random values for X
    Y = beta0 + beta1 * X + mu + np.random.normal(0, np.sqrt(sigma2), N)  # Generate Y

    # Fit a linear regression model to X and Y
    model = LinearRegression().fit(X.reshape(-1, 1), Y)
    slope = model.coef_[0]
    intercept = model.intercept_

    # Generate a scatter plot of (X, Y) with the fitted regression line
    plot1_path = "static/plot1.png"
    plt.figure()
    plt.scatter(X, Y, label="Data Points")
    plt.plot(X, model.predict(X.reshape(-1, 1)), color="red", label="Regression Line")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.savefig(plot1_path)
    plt.close()

    # Run S simulations to generate slopes and intercepts
    slopes = []
    intercepts = []
    for _ in range(S):
        X_sim = np.random.rand(N)  # Generate simulated X values
        Y_sim = beta0 + beta1 * X_sim + mu + np.random.normal(0, np.sqrt(sigma2), N)  # Generate simulated Y values
        sim_model = LinearRegression().fit(X_sim.reshape(-1, 1), Y_sim)
        slopes.append(sim_model.coef_[0])
        intercepts.append(sim_model.intercept_)

    # Plot histograms of slopes and intercepts
    plot2_path = "static/plot2.png"
    plt.figure()
    plt.hist(slopes, bins=20, alpha=0.5, label="Slopes")
    plt.hist(intercepts, bins=20, alpha=0.5, label="Intercepts")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(plot2_path)
    plt.close()

    # Calculate proportions of slopes and intercepts more extreme than observed
    slope_more_extreme = np.mean(np.abs(slopes) >= abs(slope))
    intercept_extreme = np.mean(np.abs(intercepts) >= abs(intercept))

    return (
        X,
        Y,
        slope,
        intercept,
        plot1_path,
        plot2_path,
        slope_more_extreme,
        intercept_extreme,
        slopes,
        intercepts,
    )

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get user input from the form
        N = int(request.form["N"])
        mu = float(request.form["mu"])
        sigma2 = float(request.form["sigma2"])
        beta0 = float(request.form["beta0"])
        beta1 = float(request.form["beta1"])
        S = int(request.form["S"])

        # Generate data and initial plots
        (
            X,
            Y,
            slope,
            intercept,
            plot1,
            plot2,
            slope_extreme,
            intercept_extreme,
            slopes,
            intercepts,
        ) = generate_data(N, mu, beta0, beta1, sigma2, S)

        # Store data in session
        session["X"] = X.tolist()
        session["Y"] = Y.tolist()
        session["slope"] = slope
        session["intercept"] = intercept
        session["slopes"] = slopes
        session["intercepts"] = intercepts
        session["slope_extreme"] = slope_extreme
        session["intercept_extreme"] = intercept_extreme
        session["N"] = N
        session["mu"] = mu
        session["sigma2"] = sigma2
        session["beta0"] = beta0
        session["beta1"] = beta1
        session["S"] = S

        return render_template(
            "index.html",
            plot1=plot1,
            plot2=plot2,
            slope_extreme=slope_extreme,
            intercept_extreme=intercept_extreme,
            N=N,
            mu=mu,
            sigma2=sigma2,
            beta0=beta0,
            beta1=beta1,
            S=S,
        )
        
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    return index()

@app.route("/hypothesis_test", methods=["POST"])
def hypothesis_test():
    # Retrieve data from session with default values
    N = int(session.get("N", 100))
    S = int(session.get("S", 1000))
    slope = float(session.get("slope", 0))
    intercept = float(session.get("intercept", 0))
    slopes = session.get("slopes", [])
    print(slopes)
    intercepts = session.get("intercepts", [])
    beta0 = float(session.get("beta0", 0))
    beta1 = float(session.get("beta1", 1))

    parameter = request.form.get("parameter")
    test_type = request.form.get("test_type")

    # Use the slopes or intercepts from the simulations
    if parameter == "slope":
        simulated_stats = np.array(slopes)
        observed_stat = slope
        hypothesized_value = beta1
    else:
        simulated_stats = np.array(intercepts)
        observed_stat = intercept
        hypothesized_value = beta0

    # Calculate p-value based on test type
    if test_type == ">":
        p_value = np.mean(simulated_stats >= observed_stat)
    elif test_type == "<":
        p_value = np.mean(simulated_stats <= observed_stat)
    else:  # Not equal
        p_value = np.mean(np.abs(simulated_stats - hypothesized_value) >= abs(observed_stat - hypothesized_value))

    # Fun message for small p-value
    fun_message = "Wow! You've encountered a rare event!" if p_value <= 0.0001 else None

    # Plot histogram of simulated statistics
    plot3_path = "static/plot3.png"
    plt.figure()
    plt.hist(simulated_stats, bins=20, alpha=0.7, label="Simulated Statistics")
    plt.axvline(observed_stat, color="red", linestyle="--", label="Observed Slope")
    plt.axvline(hypothesized_value, color="blue", linestyle="--", label="Hypothesized Slope")
    plt.xlabel("Statistic Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(plot3_path)
    plt.close()

    return render_template(
        "index.html",
        plot1="static/plot1.png",
        plot2="static/plot2.png",
        plot3=plot3_path,
        parameter=parameter,
        observed_stat=observed_stat,
        hypothesized_value=hypothesized_value,
        N=N,
        beta0=beta0,
        beta1=beta1,
        S=S,
        p_value=p_value,
        fun_message=fun_message,
    )

@app.route("/confidence_interval", methods=["POST"])
def confidence_interval():
    # Retrieve data from session
    slopes = np.array(session.get("slopes", []))
    intercepts = np.array(session.get("intercepts", []))
    parameter = request.form.get("parameter")
    confidence_level = float(request.form.get("confidence_level")) / 100

    # Select the appropriate statistics for confidence interval calculation
    if parameter == "slope":
        stats_array = slopes
        true_value = session.get("beta1", 0)  # Default to 0 if beta1 is None
    else:
        stats_array = intercepts
        true_value = session.get("beta0", 0)  # Default to 0 if beta0 is None
    
    parameter = request.form.get("parameter")
    confidence_level_input = request.form.get("confidence_level")

    if parameter is None or confidence_level_input is None:
        return "Error: Parameter or confidence level is not selected.", 400

    # Ensure confidence_level_input is a valid float
    try:
        confidence_level = float(confidence_level_input) / 100
    except ValueError:
        return "Error: Confidence level must be a valid number.", 400

    # Calculate the mean and standard error
    if len(stats_array) > 0:  # Ensure stats_array is not empty
        mean_estimate = np.mean(stats_array)
        se = np.std(stats_array) / np.sqrt(len(stats_array))
        # Calculate the confidence interval using the t-distribution
        ci_lower, ci_upper = stats.t.interval(confidence_level, len(stats_array) - 1, loc=mean_estimate, scale=se)
    else:
        mean_estimate, ci_lower, ci_upper = None, None, None  # Handle empty stats_array case


    # Check if the true parameter is within the confidence interval
    includes_true = (
        "True" if ci_lower is not None and ci_upper is not None and ci_lower <= true_value <= ci_upper else "False"
    )

    # Plot confidence interval
    plot4_path = "static/plot4.png"
    plt.figure()
    plt.scatter(stats_array, np.zeros_like(stats_array), alpha=0.5, color="gray", label="Simulated Estimates")
    if mean_estimate is not None:
        plt.axvline(mean_estimate, color="blue", linestyle="-", label="Mean Estimate")
    if ci_lower is not None and ci_upper is not None:
        plt.hlines(0, ci_lower, ci_upper, color="green", linestyle="-", label=f"{int(confidence_level * 100)}% Confidence Interval")
    plt.axvline(true_value, color="red", linestyle="--", label="True Slope")
    plt.xlabel("Statistic Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(plot4_path)
    plt.close()

    # Return results to template
    return render_template(
        "index.html",
        plot1="static/plot1.png",
        plot2="static/plot2.png",
        plot3="static/plot3.png",
        plot4=plot4_path,
        parameter=parameter,
        mean_estimate=mean_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        includes_true=includes_true,
        confidence_level=int(confidence_level * 100),
    )


if __name__ == "__main__":
    app.run(debug=True)
