(() => {
  const secureHosts = new Set(["localhost", "127.0.0.1", "::1"]);
  const isSecure = window.isSecureContext || secureHosts.has(window.location.hostname);

  const ensureMediaDevices = () => {
    const nav = navigator;
    nav.mediaDevices = nav.mediaDevices || {};

    if (!nav.mediaDevices.getUserMedia) {
      const legacy =
        nav.getUserMedia || nav.webkitGetUserMedia || nav.mozGetUserMedia || nav.msGetUserMedia;
      if (legacy) {
        nav.mediaDevices.getUserMedia = (constraints) =>
          new Promise((resolve, reject) =>
            legacy.call(nav, constraints, resolve, reject)
          );
      }
    }

    if (!nav.mediaDevices.getUserMedia) {
      nav.mediaDevices.getUserMedia = () =>
        Promise.reject(
          new Error(
            isSecure
              ? "Camera API not supported in this browser. Please use a modern Chrome or Safari."
              : "Camera access requires HTTPS or localhost."
          )
        );
    }

    return nav.mediaDevices.getUserMedia;
  };

  const setup = () => {
    const video = document.getElementById("video");
    const canvas = document.getElementById("captureCanvas");
    const captureButton = document.getElementById("btnCapture");
    const statusEl = document.getElementById("status");
    const resultImage = document.getElementById("resultImage");

    if (!video || !canvas || !captureButton || !statusEl || !resultImage) {
      console.error("Required DOM elements are missing; aborting camera setup.");
      return;
    }

    let stream = null;
    let isProcessing = false;
    let cameraInitPromise = null;

    const setStatus = (text, isError = false) => {
      statusEl.textContent = `Status: ${text}`;
      statusEl.classList.toggle("error", isError);
    };

    const initCamera = async () => {
      if (stream) return stream;
      if (cameraInitPromise) return cameraInitPromise;

      ensureMediaDevices();
      const constraints = {
        video: {
          facingMode: { ideal: "environment" },
          width: { ideal: 640 },
          height: { ideal: 360 },
        },
        audio: false,
      };

      cameraInitPromise = navigator.mediaDevices.getUserMedia(constraints)
        .then((mediaStream) => {
          stream = mediaStream;
          video.srcObject = mediaStream;
          video.muted = true;
          video.playsInline = true;
          return video
            .play()
            .catch(() => Promise.resolve())
            .then(() => {
              setStatus("camera ready");
              return mediaStream;
            });
        })
        .catch((err) => {
          setStatus(`camera error: ${err.message}`, true);
          cameraInitPromise = null;
          throw err;
        });

      return cameraInitPromise;
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
        if (!ctx) {
          reject(new Error("Canvas is not supported in this browser."));
          return;
        }

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
        await initCamera();
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

    const shutdownCamera = () => {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
        stream = null;
      }
      cameraInitPromise = null;
    };

    window.addEventListener("pagehide", shutdownCamera);
    window.addEventListener("beforeunload", shutdownCamera);
    window.addEventListener("visibilitychange", () => {
      if (document.hidden) {
        shutdownCamera();
      }
    });

    initCamera();
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", setup);
  } else {
    setup();
  }
})();
