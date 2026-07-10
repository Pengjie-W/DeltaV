(function () {
  const categoryLabels = {
    "Image Restoration and Jigsaw Reasoning": "Jigsaw Restoration",
    "Logical and Abstract Reasoning": "Logic",
    "Mathematical and Algorithmic Reasoning": "Math",
    "Natural Science and Domain-Specific Reasoning": "Science",
    "Spatial Perception and Embodied Planning": "Spatial Planning",
    "Strategic Game and Planning": "Strategy Planning"
  };

  function clean(text) {
    return (text || "").replace(/\s+/g, " ").trim();
  }

  function renameNode(node) {
    const text = clean(node.textContent);
    if (!categoryLabels[text]) return false;
    node.textContent = categoryLabels[text];
    node.dataset.vimoShortLabel = "true";
    return true;
  }

  function renameCategoryLabels() {
    const targets = document.querySelectorAll(
      "#gallery-nav span, #playground .collection-container h2"
    );
    let changed = false;

    targets.forEach((node) => {
      if (renameNode(node)) changed = true;
    });

    return changed;
  }

  function start() {
    renameCategoryLabels();

    let attempts = 0;
    const timer = window.setInterval(() => {
      attempts += 1;
      const changed = renameCategoryLabels();
      if (changed || attempts > 100) {
        window.clearInterval(timer);
      }
    }, 100);

    const root = document.querySelector("#root");
    if (root) {
      const observer = new MutationObserver(() => renameCategoryLabels());
      observer.observe(root, { childList: true, subtree: true });
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start, { once: true });
  } else {
    start();
  }
})();
