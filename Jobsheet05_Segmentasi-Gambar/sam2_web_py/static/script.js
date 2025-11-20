(() => {
  const video = document.getElementById("video");
  const canvas = document.getElementById("captureCanvas");
  const captureButton = document.getElementById("btnCapture");
  const statusEl = document.getElementById("status");
  const resultImage = document.getElementById("resultImage");

  let stream = null;
  let isProcessing = false;

  const setStatus = (text, isError = false) => {
    statusEl.textContent = `Status: ${text}`;
    statusEl.classList.toggle("error", isError);
  };

  const initCamera = async () => {
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user" },
        audio: false,
      });
      video.srcObject = stream;
      await video.play();
      setStatus("camera ready");
    } catch (err) {
      setStatus(`camera error: ${err.message}`, true);
    }
  };

  const captureFrame = () =>
    new Promise((resolve, reject) => {
      if (!video.videoWidth || !video.videoHeight) {
        reject(new Error("Camera not ready yet."));
        return;
      }

      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      canvas.toBlob(
        (blob) => {
          if (!blob) {
            reject(new Error("Failed to capture frame"));
            return;
          }
          resolve(blob);
        },
        "image/jpeg",
        0.92
      );
    });

  const sendFrame = async () => {
    if (isProcessing) return;
    isProcessing = true;
    captureButton.disabled = true;
    setStatus("processing with SAM2...");

    try {
      const frameBlob = await captureFrame();
      const formData = new FormData();
      formData.append("frame", frameBlob, "frame.jpg");

      const response = await fetch("/api/segment", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || "Server error");
      }

      if (data.image_base64) {
        resultImage.src = data.image_base64;
        setStatus("processed by SAM2");
      } else {
        throw new Error("Invalid response from server");
      }
    } catch (err) {
      setStatus(err.message || "processing failed", true);
    } finally {
      isProcessing = false;
      captureButton.disabled = false;
    }
  };

  captureButton.addEventListener("click", sendFrame);
  window.addEventListener("beforeunload", () => {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
    }
  });

  document.addEventListener("DOMContentLoaded", initCamera);
})();
