(function () {
  const reasoningColumns = [
    { key: "model", label: "Model" },
    { key: "params", label: "#Param" },
    { key: "z2d", label: "2D", group: "Zebra-CoT" },
    { key: "z3d", label: "3D", group: "Zebra-CoT" },
    { key: "zscience", label: "Science", group: "Zebra-CoT" },
    { key: "zstrategy", label: "Strategy", group: "Zebra-CoT" },
    { key: "zoverall", label: "Overall", group: "Zebra-CoT" },
    { key: "sstrategy", label: "Strategy Planning", group: "StructCoT" },
    { key: "sspatial", label: "Spatial Planning", group: "StructCoT" },
    { key: "slogic", label: "Logic", group: "StructCoT" },
    { key: "smath", label: "Math", group: "StructCoT" },
    { key: "sscience", label: "Science", group: "StructCoT" },
    { key: "svisual", label: "Visual Search", group: "StructCoT" },
    { key: "sjigsaw", label: "Jigsaw Restoration", group: "StructCoT" },
    { key: "soverall", label: "Overall", group: "StructCoT" }
  ];

  const reasoningRows = [
    ["GPT-5.2", "-", "67.6", "19.3", "73.3", "54.4", "53.7", "43.1", "33.8", "42.1", "76.3", "50.4", "87.0", "57.1", "55.7"],
    ["Gemini-3.1 Pro", "-", "68.7", "19.0", "83.3", "60.4", "57.9", "71.6", "28.2", "50.2", "78.3", "55.0", "79.4", "65.3", "61.1"],
    ["Gemini 3.0 Flash", "-", "66.5", "19.4", "78.4", "54.5", "54.7", "55.0", "33.3", "44.8", "74.8", "48.4", "83.6", "64.9", "57.8"],
    ["Qwen3-VL", "2B", "44.3", "13.2", "30.3", "9.2", "24.3", "3.4", "31.4", "4.6", "41.4", "29.4", "80.8", "39.3", "32.9"],
    ["Qwen3-VL", "8B", "50.7", "16.9", "56.0", "22.7", "36.6", "21.6", "25.4", "13.1", "59.3", "39.3", "83.8", "46.5", "41.3"],
    ["InternVL3.5", "8B", "29.7", "11.4", "48.9", "19.8", "27.5", "6.9", "36.3", "17.5", "36.1", "32.0", "75.8", "41.0", "35.1"],
    ["Qwen2.5-VL", "72B", "43.2", "17.3", "50.1", "25.8", "34.1", "14.8", "34.4", "31.4", "48.0", "36.5", "84.9", "47.0", "42.4"],
    ["Chameleon", "7B", "13.3", "3.0", "5.2", "9.9", "7.9", "5.6", "12.5", "4.1", "9.1", "13.1", "23.5", "14.4", "11.8", "group-start"],
    ["Anole", "7B", "10.8", "2.8", "4.8", "8.5", "6.7", "5.4", "0.1", "3.8", "8.9", "12.8", "16.8", "11.4", "9.9"],
    ["Janus-pro", "7B", "31.7", "7.7", "11.5", "18.0", "17.2", "4.3", "24.4", "13.4", "16.6", "12.0", "74.6", "33.9", "25.6"],
    ["OmniGen2", "7B", "26.5", "1.3", "9.6", "9.7", "11.8", "0.6", "25.3", "1.5", "8.4", "10.1", "78.1", "28.5", "21.8"],
    ["EMU3.5", "34B", "10.1", "3.6", "8.6", "11.8", "8.5", "2.8", "29.1", "4.6", "19.3", "15.6", "21.1", "18.8", "15.9"],
    ["ThinkMorph", "7B", "43.0", "11.6", "31.4", "22.9", "27.2", "21.4", "19.5", "26.4", "43.4", "26.0", "84.1", "49.9", "38.7", "group-start"],
    ["DeltaV", "2B", "78.9", "20.0", "41.1", "38.3", "44.6", "16.4", "53.0", "66.0", "30.1", "45.6", "84.3", "62.6", "51.1", "vimo"]
  ];

  const understandingColumns = [
    { key: "model", label: "Model" },
    { key: "params", label: "#Param" },
    { key: "vstar", label: "VStar" },
    { key: "emma", label: "EMMA" },
    { key: "m3cot", label: "M3CoT" },
    { key: "mmep", label: "MME-P" },
    { key: "mmbench", label: "MMBench" },
    { key: "mathvista", label: "MathVista" },
    { key: "mmvp", label: "MMVP" },
    { key: "visulogic", label: "VisuLogic" }
  ];

  const understandingRows = [
    ["Chameleon", "7B", "32.5", "8.6", "16.1", "530", "6.0", "21.7", "4.7", "4.5"],
    ["Anole", "7B", "34.0", "6.6", "15.8", "508", "6.2", "22.5", "6.7", "3.7"],
    ["Janus-pro", "1B", "43.5", "18.9", "45.9", "1398", "60.2", "37.6", "39.3", "25.0"],
    ["Janus-pro", "7B", "39.3", "21.5", "49.1", "1509", "66.7", "42.7", "34.7", "17.5"],
    ["OmniGen2", "7B", "41.4", "14.7", "50.3", "1588", "76.1", "60.2", "35.3", "0.1"],
    ["EMU3.5", "34B", "-", "-", "-", "791", "13.7", "28.3", "16.7", "11.4"],
    ["Qwen3-VL", "2B", "71.7", "22.2", "53.0", "1482", "77.1", "61.1", "45.0", "11.5"],
    ["Qwen3-VL", "8B", "83.7", "30.6", "61.2", "1729", "85.2", "77.6", "59.3", "22.5"],
    ["InternVL3.5", "2B", "68.1", "12.7", "51.3", "1552", "78.2", "60.8", "48.7", "26.0"],
    ["InternVL3.5", "8B", "69.1", "16.6", "59.9", "1688", "82.7", "74.1", "57.3", "29.7"],
    ["ThinkMorph", "7B", "64.4", "22.4", "48.8", "1478", "78.2", "67.8", "8.6", "6.5", "group-start"],
    ["DeltaV", "2B", "76.4", "26.4", "54.5", "1555", "82.3", "69.3", "51.3", "23.5", "vimo"]
  ];

  const reasoningBest = new Set([
    "DeltaV|2B|z2d",
    "DeltaV|2B|z3d",
    "Qwen3-VL|8B|zscience",
    "DeltaV|2B|zstrategy",
    "DeltaV|2B|zoverall",
    "Qwen3-VL|8B|sstrategy",
    "DeltaV|2B|sspatial",
    "DeltaV|2B|slogic",
    "Qwen3-VL|8B|smath",
    "DeltaV|2B|sscience",
    "Qwen2.5-VL|72B|svisual",
    "DeltaV|2B|sjigsaw",
    "DeltaV|2B|soverall"
  ]);

  const understandingBest = new Set();

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function renderCell(row, columns, index, bestCells, tag) {
    const cell = el(tag || "td", "", row[index]);
    const key = columns[index].key;
    const model = row[0];

    if (index === 0 || index === 1) {
      cell.classList.add("vimo-sticky-col");
    }
    if (index === 0) {
      cell.classList.add("vimo-model-cell");
    }
    if (bestCells.has(model + "|" + row[1] + "|" + key)) {
      const span = el("span", "font-semibold", row[index]);
      cell.textContent = "";
      cell.appendChild(span);
    }
    return cell;
  }

  function buildGroupedHeader(table, columns) {
    const thead = el("thead");
    const groupRow = el("tr", "vimo-table-group-row");

    const modelHead = el("th", "vimo-sticky-col", "Model");
    modelHead.scope = "col";
    modelHead.rowSpan = 2;
    groupRow.appendChild(modelHead);

    const paramHead = el("th", "vimo-sticky-col", "#Param");
    paramHead.scope = "col";
    paramHead.rowSpan = 2;
    groupRow.appendChild(paramHead);

    const zebraHead = el("th", "", "Zebra-CoT");
    zebraHead.scope = "colgroup";
    zebraHead.colSpan = 5;
    groupRow.appendChild(zebraHead);

    const structHead = el("th", "", "StructCoT");
    structHead.scope = "colgroup";
    structHead.colSpan = 8;
    groupRow.appendChild(structHead);

    const subRow = el("tr", "vimo-table-sub-row");
    columns.slice(2).forEach((column) => {
      const th = el("th", "", column.label);
      th.scope = "col";
      subRow.appendChild(th);
    });

    thead.appendChild(groupRow);
    thead.appendChild(subRow);
    table.appendChild(thead);
  }

  function buildFlatHeader(table, columns) {
    const thead = el("thead");
    const row = el("tr");
    columns.forEach((column, index) => {
      const th = el("th", index < 2 ? "vimo-sticky-col" : "", column.label);
      th.scope = "col";
      row.appendChild(th);
    });
    thead.appendChild(row);
    table.appendChild(thead);
  }

  function buildTable(options) {
    const table = el("table", "vimo-benchmark-table !mt-0");
    if (options.grouped) {
      buildGroupedHeader(table, options.columns);
    } else {
      buildFlatHeader(table, options.columns);
    }

    const tbody = el("tbody");
    options.rows.forEach((row) => {
      const tr = el("tr");
      const marker = row[options.columns.length];
      if (marker === "group-start") tr.classList.add("vimo-group-start");
      if (marker === "vimo") tr.classList.add("vimo-row-highlight");

      options.columns.forEach((_, index) => {
        tr.appendChild(renderCell(row, options.columns, index, options.bestCells));
      });
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    return table;
  }

  function buildTableBlock(title, table, note) {
    const block = el("div", "mb-12");
    const heading = el("h3", "text-xl md:text-2xl font-medium mb-6 text-center text-gray-800", title);
    const wrapper = el("div", "mt-6");
    const scroll = el("div", "overflow-x-auto pb-4 scrollbar-thin");
    const container = el("div", "table-container !mt-0 vimo-table-container");

    container.appendChild(table);
    scroll.appendChild(container);
    wrapper.appendChild(scroll);
    block.appendChild(heading);
    block.appendChild(wrapper);

    if (note) {
      block.appendChild(el("div", "text-xs text-gray-500 text-center mt-1 vimo-table-note", note));
    }

    return block;
  }

  function buildBenchmarkContent() {
    const container = el("div", "container mx-auto");
    const title = el("div", "text-center mb-10");
    title.appendChild(el("h2", "text-2xl md:text-3xl font-medium mb-4 text-gray-900", "Benchmark"));
    container.appendChild(title);

    container.appendChild(buildTableBlock(
      "Multimodal Reasoning",
      buildTable({
        columns: reasoningColumns,
        rows: reasoningRows,
        grouped: true,
        bestCells: reasoningBest
      }),
      "Note: StructCoT excludes image editing data because editing examples are not included in its evaluated tasks."
    ));

    container.appendChild(buildTableBlock(
      "Multimodal Understanding",
      buildTable({
        columns: understandingColumns,
        rows: understandingRows,
        grouped: false,
        bestCells: understandingBest
      })
    ));

    return container;
  }

  function installBenchmarkTables() {
    const benchmark = document.querySelector("#benchmark");
    if (!benchmark || benchmark.dataset.vimoBenchmark === "true") return false;
    benchmark.textContent = "";
    benchmark.dataset.vimoBenchmark = "true";
    benchmark.appendChild(buildBenchmarkContent());
    return true;
  }

  function start() {
    if (installBenchmarkTables()) return;

    let attempts = 0;
    const timer = window.setInterval(() => {
      attempts += 1;
      if (installBenchmarkTables() || attempts > 100) {
        window.clearInterval(timer);
      }
    }, 100);

    const root = document.querySelector("#root");
    if (root) {
      const observer = new MutationObserver(() => installBenchmarkTables());
      observer.observe(root, { childList: true, subtree: true });
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start, { once: true });
  } else {
    start();
  }
})();
