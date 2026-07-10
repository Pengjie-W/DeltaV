(function () {
  const ASSET_PREFIXES = ["/static/", "/images/", "/input/", "/external/"];
  const ASSET_FILES = ["/samples_index.json"];

  function isGitHubPagesProject() {
    return window.location.hostname.endsWith("github.io");
  }

  function normalizeBase(base) {
    if (!base) return "/";
    return base.endsWith("/") ? base : `${base}/`;
  }

  function getBasePath() {
    if (window.__DELTAV_BASE_PATH__) {
      return normalizeBase(window.__DELTAV_BASE_PATH__);
    }

    if (window.location.protocol === "file:") {
      const path = window.location.pathname;
      return normalizeBase(path.slice(0, path.lastIndexOf("/") + 1));
    }

    const parts = window.location.pathname.split("/").filter(Boolean);
    if (isGitHubPagesProject() && parts.length > 0) {
      return `/${parts[0]}/`;
    }

    return "/";
  }

  const basePath = getBasePath();
  window.__DELTAV_BASE_PATH__ = basePath;
  window.__DELTAV_ROUTER_BASENAME__ =
    basePath === "/" ? "/" : basePath.replace(/\/$/, "");

  function isProjectAssetPath(pathname) {
    return (
      ASSET_PREFIXES.some((prefix) => pathname.startsWith(prefix)) ||
      ASSET_FILES.includes(pathname)
    );
  }

  function rewriteUrl(value) {
    if (typeof value !== "string" || value.length === 0) return value;

    try {
      const parsed = new URL(value, window.location.href);
      if (
        parsed.origin === window.location.origin &&
        isProjectAssetPath(parsed.pathname) &&
        basePath !== "/"
      ) {
        parsed.pathname = `${basePath.replace(/\/$/, "")}${parsed.pathname}`;
        return parsed.href;
      }
    } catch (_error) {
      // Fall through to root-relative handling below.
    }

    if (isProjectAssetPath(value)) {
      return `${basePath}${value.slice(1)}`;
    }

    return value;
  }

  window.__deltavAsset = rewriteUrl;

  const redirect = new URLSearchParams(window.location.search).get("redirect");
  if (redirect && redirect.startsWith("/")) {
    const target = `${window.__DELTAV_ROUTER_BASENAME__}${redirect}`;
    window.history.replaceState(null, "", target);
  }

  const originalFetch = window.fetch;
  if (originalFetch) {
    window.fetch = function (input, init) {
      if (typeof input === "string") {
        return originalFetch.call(this, rewriteUrl(input), init);
      }

      if (input && input.url) {
        const rewritten = rewriteUrl(input.url);
        if (rewritten !== input.url) {
          return originalFetch.call(this, rewritten, init);
        }
      }

      return originalFetch.call(this, input, init);
    };
  }

  const originalOpen = XMLHttpRequest.prototype.open;
  XMLHttpRequest.prototype.open = function (method, url) {
    const args = Array.prototype.slice.call(arguments);
    args[1] = rewriteUrl(url);
    return originalOpen.apply(this, args);
  };

  const originalSetAttribute = Element.prototype.setAttribute;
  Element.prototype.setAttribute = function (name, value) {
    const attr = String(name).toLowerCase();
    if (attr === "src" || attr === "href" || attr === "poster") {
      return originalSetAttribute.call(this, name, rewriteUrl(value));
    }
    return originalSetAttribute.call(this, name, value);
  };

  function patchUrlProperty(proto, property) {
    const descriptor = Object.getOwnPropertyDescriptor(proto, property);
    if (!descriptor || !descriptor.set || !descriptor.get) return;

    Object.defineProperty(proto, property, {
      configurable: true,
      enumerable: descriptor.enumerable,
      get: descriptor.get,
      set(value) {
        descriptor.set.call(this, rewriteUrl(value));
      }
    });
  }

  [
    HTMLImageElement.prototype,
    HTMLScriptElement.prototype,
    HTMLSourceElement.prototype,
    HTMLVideoElement.prototype,
    HTMLAudioElement.prototype
  ].forEach((proto) => patchUrlProperty(proto, "src"));

  patchUrlProperty(HTMLLinkElement.prototype, "href");
})();
