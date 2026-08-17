(function () {
  const fullTitle = "DeltaV: Thinking with Visual State Updates in Unified Large Multimodal Models";
  const introSummary = "DeltaV is an update-centric unified large multimodal model. Instead of regenerating a complete image at every reasoning step, it preserves the visual history and generates only the compact, reasoning-critical changes.";
  const links = {
    github: "https://github.com/Pengjie-W/DeltaV",
    huggingFace: "https://huggingface.co/wpj20000/DeltaV-2B/tree/main",
    arxiv: "https://arxiv.org/abs/2607.08434",
    demo: "http://vlrlabmonkey.xyz:10088/"
  };

  function clean(text) {
    return (text || "").replace(/\s+/g, " ").trim();
  }

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function assetUrl(path) {
    return typeof window.__deltavAsset === "function"
      ? window.__deltavAsset(path)
      : path;
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

  function buildStats() {
    const stats = el("div", "deltav-stat-grid");
    stats.dataset.deltavStats = "true";

    [
      ["55.6%", "fewer newly generated visual tokens", "vs. full-image generation"],
      ["+3.3%", "multimodal reasoning", "over full-image modeling"],
      ["1.05M", "StructCoT samples", "across 44 task domains"]
    ].forEach(([value, label, detail]) => {
      const card = el("div", "deltav-stat-card");
      card.appendChild(el("strong", "deltav-stat-value", value));
      card.appendChild(el("span", "deltav-stat-label", label));
      card.appendChild(el("span", "deltav-stat-detail", detail));
      stats.appendChild(card);
    });

    return stats;
  }

  function buildTraceCard(options) {
    const card = el("article", "deltav-trace-card");
    const head = el("div", "deltav-trace-head");
    head.appendChild(el("span", "deltav-step-index", options.index));
    head.appendChild(el("span", "deltav-step-title", options.title));
    card.appendChild(head);

    const media = el("div", "deltav-trace-media");
    const image = el("img");
    image.src = assetUrl(options.src);
    image.alt = options.alt;
    image.loading = options.index === "01" ? "eager" : "lazy";
    media.appendChild(image);
    card.appendChild(media);

    const foot = el("div", "deltav-trace-foot");
    foot.appendChild(el("span", "deltav-step-action", options.action));
    const budget = el(
      "span",
      options.tokenPath ? "deltav-token-chip is-loading" : "deltav-base-chip",
      options.tokenPath ? "" : "base visual state"
    );
    if (options.tokenPath) {
      budget.dataset.deltavTokenPath = options.tokenPath;
      budget.setAttribute("aria-live", "polite");
    }
    foot.appendChild(budget);
    card.appendChild(foot);
    return card;
  }

  function buildSpotlight() {
    const section = el("section", "deltav-spotlight");
    section.id = "deltav-spotlight";
    section.dataset.deltavSpotlight = "true";

    const header = el("div", "deltav-spotlight-header");
    header.appendChild(el("p", "deltav-kicker", "Featured reasoning trace"));
    header.appendChild(el("h2", "", "A full image is expensive. A small update is enough."));
    header.appendChild(el(
      "p",
      "deltav-spotlight-copy",
      "In this Light Up puzzle, each step changes only a few cells. DeltaV carries the existing board forward and uses an adaptive, per-image token budget for each update."
    ));
    section.appendChild(header);

    const comparison = el("div", "deltav-comparison-strip");
    const full = el("div", "deltav-comparison-item deltav-comparison-muted");
    full.appendChild(el("span", "deltav-comparison-label", "Conventional ULMM"));
    full.appendChild(el("strong", "", "Regenerate the full visual state"));
    const fullBar = el("div", "deltav-budget-bar deltav-budget-full");
    fullBar.appendChild(el("span", "", "repeated content"));
    full.appendChild(fullBar);

    const update = el("div", "deltav-comparison-arrow", "→");
    update.setAttribute("aria-hidden", "true");

    const delta = el("div", "deltav-comparison-item deltav-comparison-active");
    delta.appendChild(el("span", "deltav-comparison-label", "DeltaV"));
    delta.appendChild(el("strong", "", "Generate only what changed"));
    const deltaBar = el("div", "deltav-budget-bar deltav-budget-delta");
    deltaBar.appendChild(el("span", "", "visual update"));
    delta.appendChild(deltaBar);

    comparison.appendChild(full);
    comparison.appendChild(update);
    comparison.appendChild(delta);
    section.appendChild(comparison);

    const trace = el("div", "deltav-trace-grid");
    [
      {
        index: "01",
        title: "Observe",
        src: "/input/Logic_light_up_struct_sample_001007_0.png",
        alt: "The initial Light Up puzzle board",
        action: "Read the puzzle"
      },
      {
        index: "02",
        title: "Update",
        src: "/images/Logic_light_up_struct_sample_001007_out_0.png",
        alt: "Light Up board after placing the first bulb",
        action: "Place bulb at (0, 4)",
        tokenPath: "/images/Logic_light_up_struct_sample_001007_out_0.png"
      },
      {
        index: "03",
        title: "Update",
        src: "/images/Logic_light_up_struct_sample_001007_out_1.png",
        alt: "Light Up board after placing the second bulb",
        action: "Place bulb at (1, 0)",
        tokenPath: "/images/Logic_light_up_struct_sample_001007_out_1.png"
      },
      {
        index: "04",
        title: "Resolve",
        src: "/images/Logic_light_up_struct_sample_001007_out_5.png",
        alt: "The completed Light Up puzzle board",
        action: "Complete the board",
        tokenPath: "/images/Logic_light_up_struct_sample_001007_out_5.png"
      }
    ].forEach((step) => trace.appendChild(buildTraceCard(step)));
    section.appendChild(trace);

    const takeaway = el("div", "deltav-takeaway");
    takeaway.appendChild(el("span", "deltav-takeaway-mark", "Δ"));
    const takeawayCopy = el("div");
    takeawayCopy.appendChild(el("strong", "", "The key idea"));
    takeawayCopy.appendChild(el(
      "p",
      "",
      "Reasoning advances through sparse state transitions. DeltaV concentrates tokens and supervision on those transitions instead of repeatedly modeling unchanged pixels."
    ));
    takeaway.appendChild(takeawayCopy);
    section.appendChild(takeaway);

    return section;
  }

  function updateHero() {
    const intro = document.querySelector("#introduction");
    if (!intro) return false;

    const h1 = intro.querySelector("h1");
    const titleNode = intro.querySelector("h1 span") || h1;
    if (titleNode && clean(titleNode.textContent) !== "Think with what changed.") {
      titleNode.textContent = "Think with what changed.";
    }
    if (h1) h1.dataset.vimoTitle = "true";

    if (h1 && !intro.querySelector("[data-deltav-eyebrow]")) {
      const eyebrow = el("p", "deltav-hero-eyebrow", "DeltaV · Update-centric multimodal reasoning");
      eyebrow.dataset.deltavEyebrow = "true";
      h1.insertAdjacentElement("beforebegin", eyebrow);
    }

    if (h1 && !intro.querySelector("[data-deltav-paper-title]")) {
      const paperTitle = el("p", "deltav-paper-title", fullTitle);
      paperTitle.dataset.deltavPaperTitle = "true";
      h1.insertAdjacentElement("afterend", paperTitle);
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

    if (introCopy && introCopy.tagName === "P") {
      introCopy.textContent = introSummary;
      introCopy.dataset.vimoIntroCopy = "true";

      if (!document.querySelector("[data-deltav-stats]")) {
        introCopy.insertAdjacentElement("afterend", buildStats());
      }
    }

    if (document.title !== fullTitle) document.title = fullTitle;
    updateMeta("twitter:title", fullTitle);
    updateMeta("og:title", fullTitle);
    updateMeta("og:site_name", "DeltaV");
    updateProjectLinks();
    return true;
  }

  function installSpotlight() {
    if (document.querySelector("[data-deltav-spotlight]")) return true;
    const playground = document.querySelector("#playground");
    if (!playground || !playground.parentNode) return false;

    const previous = playground.previousElementSibling;
    if (previous && previous.dataset && previous.dataset.vimoExamplesHeading === "true") {
      previous.remove();
    }

    playground.parentNode.insertBefore(buildSpotlight(), playground);

    const heading = el("div", "deltav-gallery-heading");
    heading.dataset.vimoExamplesHeading = "true";
    heading.appendChild(el("p", "deltav-kicker", "More examples"));
    heading.appendChild(el("h2", "", "Explore DeltaV reasoning traces"));
    heading.appendChild(el(
      "p",
      "",
      "Follow the interleaved text and visual updates across logic, mathematics, science, spatial planning, strategy, and visual search."
    ));
    playground.parentNode.insertBefore(heading, playground);
    return true;
  }

  function updateFooter() {
    const footer = document.querySelector("footer");
    if (!footer) return false;

    footer.querySelectorAll("span, p").forEach((node) => {
      const text = clean(node.textContent);
      if (text === "A Scalable Unified Multimodal Model for next-generation AI systems.") {
        node.textContent = "Compact visual updates for efficient interleaved multimodal reasoning.";
      } else if (/^© 2025 (DeltaV|ViMo)/.test(text)) {
        node.remove();
      }
    });
    return true;
  }

  function syncFeaturedTokenCounts(counts) {
    if (!counts) return;
    document.querySelectorAll("[data-deltav-token-path]").forEach((chip) => {
      const path = chip.dataset.deltavTokenPath;
      const matchedPath = Object.keys(counts).find((key) => path.endsWith(key));
      const tokenCount = counts[path] || (matchedPath ? counts[matchedPath] : null);
      if (!tokenCount) return;
      chip.textContent = `${tokenCount} visual tokens`;
      chip.classList.remove("is-loading");
      chip.setAttribute(
        "aria-label",
        `This output image is represented by ${tokenCount} visual tokens`
      );
    });
  }

  function updatePage() {
    const heroReady = updateHero();
    const spotlightReady = installSpotlight();
    updateFooter();
    return heroReady && spotlightReady;
  }

  function start() {
    window.addEventListener("deltav:visual-token-counts-ready", (event) => {
      syncFeaturedTokenCounts(event.detail);
    });
    if (window.__deltavVisualTokenCountsReady) {
      window.__deltavVisualTokenCountsReady.then(syncFeaturedTokenCounts);
    }

    if (updatePage()) return;

    let attempts = 0;
    let observer = null;
    const stop = (timer) => {
      window.clearInterval(timer);
      if (observer) observer.disconnect();
    };
    const timer = window.setInterval(() => {
      attempts += 1;
      if (updatePage() || attempts > 120) stop(timer);
    }, 100);

    const root = document.querySelector("#root");
    if (root) {
      observer = new MutationObserver(() => {
        if (updatePage()) stop(timer);
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
