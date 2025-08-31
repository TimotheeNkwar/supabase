document.addEventListener("DOMContentLoaded", () => {
    const button = document.querySelector(".btn-accent");
    const textInput = document.getElementById("sentiment-text");
    const languageSelect = document.getElementById("language-select");
    const resultDiv = document.getElementById("sentiment-result");
    const label = document.getElementById("sentiment-label");
    const posScore = document.getElementById("positive-score");
    const neuScore = document.getElementById("neutral-score");
    const negScore = document.getElementById("negative-score");
    const posBar = document.getElementById("positive-bar");
    const neuBar = document.getElementById("neutral-bar");
    const negBar = document.getElementById("negative-bar");
    const insights = document.getElementById("sentiment-insights");
    const errorDiv = document.getElementById("error-message");

    button.addEventListener("click", async () => {
        const text = textInput.value.trim();
        const language = languageSelect.value;

        if (!text) {
            errorDiv.innerText = "Please enter text to analyze.";
            errorDiv.classList.remove("hidden");
            return;
        }

        errorDiv.classList.add("hidden");
        resultDiv.classList.add("hidden");
        button.disabled = true;

        try {
            const response = await fetch("/api/analyze-sentiment", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text, language })
            });

            const data = await response.json();

            if (response.ok) {
                resultDiv.classList.remove("hidden");
                label.innerText = data.sentiment || "Unknown";
                posScore.innerText = data.positive != null ? data.positive + "%" : "0%";
                neuScore.innerText = data.neutral != null ? data.neutral + "%" : "0%";
                negScore.innerText = data.negative != null ? data.negative + "%" : "0%";

                posBar.style.width = data.positive != null ? data.positive + "%" : "0%";
                neuBar.style.width = data.neutral != null ? data.neutral + "%" : "0%";
                negBar.style.width = data.negative != null ? data.negative + "%" : "0%";

                insights.innerText = data.insights || "";
            } else {
                errorDiv.innerText = data.error || "An error occurred";
                errorDiv.classList.remove("hidden");
            }
        } catch (err) {
            errorDiv.innerText = "Network error: " + err.message;
            errorDiv.classList.remove("hidden");
        } finally {
            button.disabled = false;
        }
    });
});
