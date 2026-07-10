(function () {
  const demoVideoSrc = "/static/media/workflow.mp4?v=20260709-deltav";

  function replaceFirstVideo() {
    const video = document.querySelector("video");
    if (!video) return false;
    if (video.dataset.vimoDemoVideo === "true") return true;

    const source = video.querySelector("source");
    if (source) {
      source.src = demoVideoSrc;
    }
    video.src = demoVideoSrc;
    video.dataset.vimoDemoVideo = "true";
    video.load();
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
