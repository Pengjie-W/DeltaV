(function () {
  const fullTitle = "DeltaV: Thinking with Visual State Updates in Unified Large Multimodal Models";
  const footerSummary = "Thinking with Visual State Updates in Unified Large Multimodal Models.";
  const introSummary = "DeltaV is a unified large multimodal model (ULMM) designed to think with visual state updates during interleaved multimodal reasoning. Conditioned on historical visual states, it incrementally predicts compact visual update tokens that capture sparse but reasoning-critical changes across reasoning steps, avoiding repeated modeling of unchanged content. Token budgets are dynamically allocated by the TSIM Router according to temporal visual variation, and visual states are encoded by the TSIM-Tok tokenizer.";
  const examplesHeading = "Examples";
  const links = {
    github: "https://github.com/Pengjie-W/DeltaV",
    huggingFace: "https://huggingface.co/wpj20000/DeltaV-2B/tree/main",
    arxiv: "https://arxiv.org/abs/2607.08434",
    demo: "http://vlrlabmonkey.xyz:10088/"
  };

  function clean(text) {
    return (text || "").replace(/\s+/g, " ").trim();
  }

  function updateMeta(name, content) {
    const selector = `meta[name="${name}"], meta[property="${name}"]`;
    const node = document.querySelector(selector);
    if (node && node.getAttribute("content") !== content) {
      node.setAttribute("content", content);
    }
  }

  function updateProjectLinks() {
    document.querySelectorAll("a").forEach((link) => {
      const href = link.getAttribute("href") || "";
      if (href === "https://github.com/bytedance-seed/DeltaV") {
        link.href = links.github;
      } else if (
        href === "https://huggingface.co/ByteDance-Seed/DeltaV-7B-MoT" ||
        href === "https://huggingface.co/dle666/ViMo-2B/tree/main"
      ) {
        link.href = links.huggingFace;
      } else if (href === "https://arxiv.org/abs/2505.14683") {
        link.href = links.arxiv;
        link.removeAttribute("aria-disabled");
        link.removeAttribute("title");
      } else if (href === "https://demo.deltav-ai.org") {
        link.href = links.demo;
      }
    });
  }

  function updateHero() {
    const intro = document.querySelector("#introduction");
    if (!intro) return false;

    const titleNode = intro.querySelector("h1 span") || intro.querySelector("h1");
    if (titleNode) {
      if (clean(titleNode.textContent) !== fullTitle) {
        titleNode.textContent = fullTitle;
      }
      const h1 = intro.querySelector("h1");
      if (h1 && h1.dataset.vimoTitle !== "true") h1.dataset.vimoTitle = "true";
    }

    intro.querySelectorAll("p").forEach((node) => {
      const text = clean(node.textContent);
      if (
        text.startsWith("Released on") ||
        text.includes("Chaorui Deng") ||
        text.includes("Equal contribution")
      ) {
        node.remove();
      }
    });

    let introCopy = intro.nextElementSibling;
    while (introCopy && introCopy.tagName === "BR") {
      const next = introCopy.nextElementSibling;
      introCopy.remove();
      introCopy = next;
    }

    if (
      introCopy &&
      introCopy.tagName === "P" &&
      clean(introCopy.textContent).startsWith("Today we introduce DeltaV")
    ) {
      introCopy.textContent = introSummary;
      introCopy.dataset.vimoIntroCopy = "true";

      let trailingBreak = introCopy.nextElementSibling;
      while (trailingBreak && trailingBreak.tagName === "BR") {
        const next = trailingBreak.nextElementSibling;
        trailingBreak.remove();
        trailingBreak = next;
      }

      if (
        !introCopy.nextElementSibling ||
        introCopy.nextElementSibling.dataset.vimoExamplesHeading !== "true"
      ) {
        const heading = document.createElement("h2");
        heading.textContent = examplesHeading;
        heading.dataset.vimoExamplesHeading = "true";
        heading.className = "vimo-examples-heading";
        introCopy.insertAdjacentElement("afterend", heading);
      }
    }

    if (document.title !== fullTitle) document.title = fullTitle;
    updateMeta("twitter:title", fullTitle);
    updateMeta("og:title", fullTitle);
    updateMeta("og:site_name", "DeltaV");
    updateProjectLinks();
    return true;
  }

  function updateFooter() {
    const footer = document.querySelector("footer");
    if (!footer) return false;

    footer.querySelectorAll("span, p").forEach((node) => {
      const text = clean(node.textContent);
      if (text === "DeltaV") {
        node.textContent = "DeltaV";
      } else if (text === "A Scalable Unified Multimodal Model for next-generation AI systems.") {
        node.textContent = footerSummary;
      } else if (
        text === "\u00a9 2025 DeltaV Unified Multimodal Model. All rights reserved." ||
        text === "\u00a9 2025 ViMo. All rights reserved." ||
        text === "\u00a9 2025 DeltaV. All rights reserved."
      ) {
        node.remove();
      }
    });

    return true;
  }

  function updateBranding() {
    const heroReady = updateHero();
    const footerReady = updateFooter();
    return heroReady && footerReady;
  }

  function start() {
    if (updateBranding()) return;

    let attempts = 0;
    let observer = null;
    const stop = (timer) => {
      window.clearInterval(timer);
      if (observer) observer.disconnect();
    };
    const timer = window.setInterval(() => {
      attempts += 1;
      if (updateBranding() || attempts > 100) {
        stop(timer);
      }
    }, 100);

    const root = document.querySelector("#root");
    if (root) {
      observer = new MutationObserver(() => {
        if (updateBranding()) stop(timer);
      });
      observer.observe(root, { childList: true, subtree: true });
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start, { once: true });
  } else {
    start();
  }
})();
