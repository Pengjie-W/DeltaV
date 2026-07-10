(function () {
  const methodText = [
    "DeltaV treats multimodal reasoning as a sequence of text steps and visual state updates. The first image provides the base visual state; later steps add compact visual update tokens instead of regenerating full images.",
    "TSIM-Tok measures how much the visual state changes, and the TSIM Router assigns an adaptive token budget. Minor changes use fewer tokens, while complex transitions keep enough detail for reconstruction and reasoning."
  ];

  const reasoningAblation = {
    columns: ["Setting", "2D Reasoning", "3D Reasoning", "Scientific Reasoning", "Strategic Reasoning", "Overall"],
    rows: [
      ["Text-Only", "51.2", "64.1", "45.0", "40.4", "50.1"],
      ["FullMod 144", "46.6", "59.5", "46.1", "40.7", "48.2"],
      ["IncMod w/ Fixed Budget 144", "51.3", "59.7", "45.4", "42.0", "49.6"],
      ["IncMod w/ TSIM Router 64", "48.8", "64.9", "47.5", "43.9", "51.2"]
    ],
    best: new Set([
      "IncMod w/ Fixed Budget 144|2D Reasoning",
      "IncMod w/ TSIM Router 64|3D Reasoning",
      "IncMod w/ TSIM Router 64|Scientific Reasoning",
      "IncMod w/ TSIM Router 64|Strategic Reasoning",
      "IncMod w/ TSIM Router 64|Overall"
    ])
  };

  const benchmarkAblation = {
    columns: ["Setting", "MMVP", "MM-Vet", "MathVista", "ChartQA", "LogicVista", "BLINK", "EMMA"],
    rows: [
      ["Text-Only", "0.73/0.48", "47.3", "66.0", "80.4", "35.8", "48.9", "23.9"],
      ["IncMod w/ TSIM Router", "0.73/0.48", "51.6 (4.3\u2191)", "67.1 (1.1\u2191)", "81.2 (0.8\u2191)", "36.3 (0.5\u2191)", "52.0 (3.1\u2191)", "25.9 (2.0\u2191)"]
    ],
    best: new Set([
      "IncMod w/ TSIM Router|MM-Vet",
      "IncMod w/ TSIM Router|MathVista",
      "IncMod w/ TSIM Router|ChartQA",
      "IncMod w/ TSIM Router|LogicVista",
      "IncMod w/ TSIM Router|BLINK",
      "IncMod w/ TSIM Router|EMMA"
    ])
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

  function findHeadingBlock(title) {
    const heading = Array.from(document.querySelectorAll(".mdx-content h2"))
      .find((node) => clean(node.textContent) === title);
    if (!heading) return null;
    return heading.closest("p") || heading;
  }

  function replaceRange(start, end, replacement) {
    if (!start || !end || start.parentNode !== end.parentNode) return false;
    const parent = start.parentNode;
    let node = start;
    while (node && node !== end) {
      const next = node.nextSibling;
      parent.removeChild(node);
      node = next;
    }
    parent.insertBefore(replacement, end);
    return true;
  }

  function buildHeading(title) {
    const header = el("div", "vimo-paper-heading");
    header.appendChild(el("h2", "", title));
    return header;
  }

  function buildFigure(options) {
    const figure = el("figure", "vimo-paper-figure " + (options.className || ""));
    const img = el("img", "", "");
    img.src = options.src;
    img.alt = options.alt;
    img.loading = "lazy";
    figure.appendChild(img);

    if (options.caption) {
      figure.appendChild(el("figcaption", "", options.caption));
    }

    return figure;
  }

  function buildTable(data) {
    const table = el("table", "vimo-paper-table !mt-0");
    const thead = el("thead");
    const headRow = el("tr");

    data.columns.forEach((column, index) => {
      const th = el("th", index === 0 ? "vimo-paper-setting" : "", column);
      th.scope = "col";
      headRow.appendChild(th);
    });
    thead.appendChild(headRow);
    table.appendChild(thead);

    const tbody = el("tbody");
    data.rows.forEach((row) => {
      const tr = el("tr");
      data.columns.forEach((column, index) => {
        const td = el("td", index === 0 ? "vimo-paper-setting" : "", row[index]);
        if (index > 0 && data.best.has(row[0] + "|" + column)) {
          const span = el("span", "font-semibold", row[index]);
          td.textContent = "";
          td.appendChild(span);
        }
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);

    return table;
  }

  function buildTableBlock(title, data, note) {
    const block = el("div", "vimo-paper-table-block vimo-paper-wide");
    block.appendChild(el("h3", "vimo-paper-table-title", title));

    const scroll = el("div", "overflow-x-auto pb-4 scrollbar-thin");
    const container = el("div", "table-container !mt-0 vimo-paper-table-container");
    container.appendChild(buildTable(data));
    scroll.appendChild(container);
    block.appendChild(scroll);

    if (note) {
      block.appendChild(el("p", "vimo-paper-note", note));
    }
    return block;
  }

  function buildMethodSection() {
    const section = el("section", "vimo-paper-section vimo-method-section");
    section.id = "vimo-method-section";
    section.appendChild(buildHeading("Method"));

    const copy = el("div", "vimo-paper-copy");
    methodText.forEach((paragraph) => copy.appendChild(el("p", "", paragraph)));
    copy.appendChild(el("div", "vimo-paper-equation", "Y = {Z0, X1, \u0394Z1, X2, \u0394Z2, ...}"));
    section.appendChild(copy);

    section.appendChild(buildFigure({
      className: "vimo-method-figure",
      src: "/static/media/vimo-method.png",
      alt: "Overview of DeltaV visual state update modeling",
      caption: "DeltaV represents visual reasoning as a base visual state followed by compact updates. TSIM-Tok produces variation-aware tokens, and the TSIM Router assigns the token budget."
    }));

    return section;
  }

  function buildEmergingSection() {
    const section = el("section", "vimo-paper-section vimo-emerging-section");
    section.id = "vimo-emerging-section";
    section.appendChild(buildHeading("Analysis"));

    const copy = el("div", "vimo-paper-copy");
    copy.appendChild(el("p", "", "Compact visual updates preserve reconstruction quality with fewer visual tokens and improve multimodal reasoning over full-image modeling."));
    section.appendChild(copy);

    section.appendChild(buildFigure({
      className: "vimo-effectiveness-figure vimo-paper-wide",
      src: "/static/media/vimo-effectiveness.png",
      alt: "Comparison of full-image modeling and incremental modeling under different token budgets",
      caption: "Image reconstruction under different token budgets. FullMod generates full images; IncMod predicts incremental visual updates."
    }));

    section.appendChild(buildTableBlock(
      "Incremental Visual Modeling for Reasoning",
      reasoningAblation,
      "Evaluated on Zebra-CoT. FullMod generates full images; IncMod predicts visual updates."
    ));

    section.appendChild(buildTableBlock(
      "Text-Only vs. DeltaV Visual-Update Training",
      benchmarkAblation,
      "Same initialization, dataset, and 30K training steps."
    ));

    return section;
  }

  function installMethodSection() {
    if (document.querySelector("#vimo-method-section")) return true;
    const start = findHeadingBlock("Method");
    const end = findHeadingBlock("Emerging Properties");
    return replaceRange(start, end, buildMethodSection());
  }

  function installEmergingSection() {
    if (document.querySelector("#vimo-emerging-section")) return true;
    const start = findHeadingBlock("Emerging Properties");
    const end = document.querySelector("#benchmark");
    return replaceRange(start, end, buildEmergingSection());
  }

  function installPaperSections() {
    const methodReady = installMethodSection();
    const emergingReady = installEmergingSection();
    return methodReady && emergingReady;
  }

  function start() {
    if (installPaperSections()) return;

    let attempts = 0;
    const timer = window.setInterval(() => {
      attempts += 1;
      if (installPaperSections() || attempts > 100) {
        window.clearInterval(timer);
      }
    }, 100);

    const root = document.querySelector("#root");
    if (root) {
      const observer = new MutationObserver(() => installPaperSections());
      observer.observe(root, { childList: true, subtree: true });
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start, { once: true });
  } else {
    start();
  }
})();
