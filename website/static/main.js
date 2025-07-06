async function sendInput() {
    const input = document.getElementById("inputBox").value;

    const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ input: input })
    });

    const data = await response.text();
    document.getElementById("resultBox").innerText = data.result;
}