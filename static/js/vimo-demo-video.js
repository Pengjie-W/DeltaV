(function () {
  const demoVideoPath = "/static/media/workflow.mp4?v=20260710-video";

  function assetUrl(path) {
    if (typeof window.__deltavAsset === "function") {
      return window.__deltavAsset(path);
    }
    return path;
  }

  function ensurePlaybackAttributes(video) {
    video.muted = true;
    video.defaultMuted = true;
    video.autoplay = true;
    video.loop = true;
    video.playsInline = true;
    video.preload = "auto";
    video.setAttribute("muted", "");
    video.setAttribute("autoplay", "");
    video.setAttribute("loop", "");
    video.setAttribute("playsinline", "");
    video.setAttribute("webkit-playsinline", "");
  }

  function replaceFirstVideo() {
    const video = document.querySelector("video");
    if (!video) return false;

    ensurePlaybackAttributes(video);
    const demoVideoSrc = assetUrl(demoVideoPath);

    const source = video.querySelector("source");
    if (source) {
      source.src = demoVideoSrc;
    }
    if (video.src !== demoVideoSrc) {
      video.src = demoVideoSrc;
    }
    video.dataset.vimoDemoVideo = "true";
    video.load();

    const playPromise = video.play();
    if (playPromise && typeof playPromise.catch === "function") {
      playPromise.catch(() => {
        video.controls = true;
      });
    }
    return true;
  }

  function start() {
    if (replaceFirstVideo()) return;

    let attempts = 0;
    const timer = window.setInterval(() => {
      attempts += 1;
      if (replaceFirstVideo() || attempts > 100) {
        window.clearInterval(timer);
      }
    }, 100);

    const root = document.querySelector("#root");
    if (root) {
      const observer = new MutationObserver(() => replaceFirstVideo());
      observer.observe(root, { childList: true, subtree: true });
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start, { once: true });
  } else {
    start();
  }
})();
