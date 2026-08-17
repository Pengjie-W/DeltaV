(function () {
  let tokenCountsByImage = {};

  function assetUrl(path) {
    return typeof window.__deltavAsset === "function"
      ? window.__deltavAsset(path)
      : path;
  }

  function getImagePath(image) {
    const source = image.currentSrc || image.getAttribute("src");
    if (!source) return "";

    try {
      return new URL(source, window.location.href).pathname;
    } catch (_error) {
      return source;
    }
  }

  function getTokenCount(pathname) {
    if (tokenCountsByImage[pathname]) return tokenCountsByImage[pathname];

    const matchedPath = Object.keys(tokenCountsByImage).find((path) =>
      pathname.endsWith(path)
    );
    return matchedPath ? tokenCountsByImage[matchedPath] : null;
  }

  function addTokenLabel(image) {
    const tokenCount = getTokenCount(getImagePath(image));
    if (!tokenCount || !image.parentElement) return false;

    const container = image.parentElement;
    container.classList.add("vimo-token-image-container");
    image.dataset.vimoVisualTokens = String(tokenCount);

    let label = container.querySelector(".vimo-image-token-label");
    if (!label) {
      label = document.createElement("span");
      label.className = "vimo-image-token-label";
      container.appendChild(label);
    }

    const labelText = `${tokenCount} visual tokens`;
    if (label.textContent !== labelText) label.textContent = labelText;
    label.setAttribute(
      "aria-label",
      `This image is represented by ${tokenCount} visual tokens`
    );
    return true;
  }

  function labelOutputImages(root) {
    const scope = root && root.querySelectorAll ? root : document;
    scope
      .querySelectorAll("#playground .collection-container img")
      .forEach(addTokenLabel);
  }

  function observePlayground() {
    const playground = document.querySelector("#playground");
    if (playground) {
      labelOutputImages(playground);
      const observer = new MutationObserver(() => labelOutputImages(playground));
      observer.observe(playground, { childList: true, subtree: true });
      return;
    }

    const root = document.querySelector("#root");
    if (!root) return;

    const rootObserver = new MutationObserver(() => {
      labelOutputImages(document);
      const mountedPlayground = document.querySelector("#playground");
      if (!mountedPlayground) return;

      rootObserver.disconnect();
      const playgroundObserver = new MutationObserver(() =>
        labelOutputImages(mountedPlayground)
      );
      playgroundObserver.observe(mountedPlayground, {
        childList: true,
        subtree: true
      });
    });
    rootObserver.observe(root, { childList: true, subtree: true });
  }

  async function loadTokenCounts() {
    const response = await fetch(assetUrl("/data/visual-token-counts.json"));
    if (!response.ok) {
      throw new Error(`Unable to load visual token counts: ${response.status}`);
    }
    const data = await response.json();
    tokenCountsByImage = data.images || {};
    window.__deltavVisualTokenCounts = tokenCountsByImage;
    window.dispatchEvent(new CustomEvent("deltav:visual-token-counts-ready", {
      detail: tokenCountsByImage
    }));
    return tokenCountsByImage;
  }

  function start() {
    window.__deltavVisualTokenCountsReady = loadTokenCounts()
      .then((counts) => {
        observePlayground();
        labelOutputImages(document);
        return counts;
      })
      .catch((error) => {
        console.error(error);
        return {};
      });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start, { once: true });
  } else {
    start();
  }
})();
