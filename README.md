# DeltaV Project Website

This repository contains the static project website for **DeltaV: Thinking with Visual State Updates in Unified Large Multimodal Models**.

The site is already built. No Node.js install or build step is required for normal use.

## Contents

- `index.html`: main entry page
- `404.html`: GitHub Pages fallback for client-side routes
- `static/`: bundled JavaScript, CSS, videos, PDFs, and figures
- `external/`: mirrored external assets used by the static site
- `images/` and `input/`: example images shown by the demo/gallery
- `samples_index.json`: public sample metadata with internal source paths removed

## Local Preview

From this folder, run a local static server:

```bash
python3 -m http.server 8000
```

Then open:

```text
http://localhost:8000
```

## GitHub Pages

1. Create a GitHub repository and upload the files in this folder.
2. In the repository settings, enable **Pages**.
3. Choose **Deploy from a branch** and select the branch root, usually `main` and `/`.

The included path adapter supports deployment both at a domain root and under a GitHub Pages project path such as `https://USER.github.io/REPO/`.

## License

The website source in this repository is released under the MIT License. Check the licenses and usage terms for any model weights, datasets, papers, or third-party assets before redistributing them separately.

