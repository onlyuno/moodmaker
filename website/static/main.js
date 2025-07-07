async function sendImage() {
    const fileInput = document.getElementById("file");
    const file = fileInput.files[0];

    if (!file) {
        alert("이미지를 선택해주세요.");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch("/predict", {
        method: "POST",
        body: formData
    });

    if (!response.ok) {
        document.getElementById("resultBox").innerHTML = "분석 실패. 서버 응답 오류.";
        return;
    }

    const result = await response.json();
    document.getElementById("resultBox").innerHTML = `분석된 장소는 <strong>${result.prediction}</strong>입니다.`;
}
