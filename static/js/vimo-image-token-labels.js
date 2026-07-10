(function () {
  const tokenCountsByImage = {
    "/images/Jigsaw_Restoration_Jigsaw_Puzzle_struct_sample_006009_out_0.png": 49,
    "/images/Jigsaw_Restoration_Jigsaw_Puzzle_struct_sample_006009_out_1.png": 100,
    "/images/Jigsaw_Restoration_Jigsaw_Puzzle_struct_sample_006009_out_2.png": 100,
    "/images/Jigsaw_Restoration_Jigsaw_Puzzle_struct_sample_006009_out_3.png": 81,
    "/images/Logic_Ciphers_zebra_sample_000640_out_0.png": 49,
    "/images/Logic_Ciphers_zebra_sample_000640_out_1.png": 49,
    "/images/Logic_Ciphers_zebra_sample_000640_out_2.png": 49,
    "/images/Logic_Ciphers_zebra_sample_000640_out_3.png": 49,
    "/images/Logic_Ciphers_zebra_sample_000640_out_4.png": 49,
    "/images/Logic_Ciphers_zebra_sample_000640_out_5.png": 49,
    "/images/Logic_Sudoku_struct_sample_001708_out_0.png": 9,
    "/images/Logic_Sudoku_struct_sample_001708_out_1.png": 9,
    "/images/Logic_light_up_struct_sample_001007_out_0.png": 9,
    "/images/Logic_light_up_struct_sample_001007_out_1.png": 9,
    "/images/Logic_light_up_struct_sample_001007_out_2.png": 9,
    "/images/Logic_light_up_struct_sample_001007_out_3.png": 9,
    "/images/Logic_light_up_struct_sample_001007_out_4.png": 9,
    "/images/Logic_light_up_struct_sample_001007_out_5.png": 9,
    "/images/Logic_rectangle_tiling_struct_sample_001002_out_0.png": 49,
    "/images/Logic_rectangle_tiling_struct_sample_001002_out_1.png": 49,
    "/images/Logic_rectangle_tiling_struct_sample_001002_out_2.png": 81,
    "/images/Logic_rectangle_tiling_struct_sample_001002_out_3.png": 49,
    "/images/Logic_rectangle_tiling_struct_sample_001002_out_4.png": 49,
    "/images/Math_Plane_Geometry_struct_sample_003046_out_0.png": 49,
    "/images/Science_Medicine_struct_sample_004121_out_0.png": 49,
    "/images/Spatial_Planning_Maze_zebra_sample_000616_out_0.png": 9,
    "/images/Strategy_Planning_Chess_zebra_sample_000107_out_0.png": 9,
    "/images/Strategy_Planning_Chess_zebra_sample_000107_out_1.png": 9,
    "/images/Strategy_Planning_Chess_zebra_sample_000107_out_2.png": 9,
    "/images/Strategy_Planning_Chess_zebra_sample_000107_out_3.png": 9,
    "/images/Visual_Search_General_VQA_struct_sample_005076_out_0.png": 81,
    "/images/Visual_Search_Visual_Search_zebra_sample_001969_out_0.png": 49
  };

  function getImagePath(image) {
    const source = image.currentSrc || image.getAttribute("src");
    if (!source) return "";

    try {
      return new URL(source, window.location.href).pathname;
    } catch (_error) {
      return source;
    }
  }

  function addTokenLabel(image) {
    const tokenCount = tokenCountsByImage[getImagePath(image)];
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

  function start() {
    labelOutputImages(document);

    const playground = document.querySelector("#playground");
    if (playground) {
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

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start, { once: true });
  } else {
    start();
  }
})();
