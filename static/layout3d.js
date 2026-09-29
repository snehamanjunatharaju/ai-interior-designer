/* ==========================================================================
   Rotatable 3D view (three.js), built from the same layout JSON as the 2D plan.

   Why three.js and not a framework: the scene is a dozen boxes. A framework
   would add a build step and a bundle for no benefit, and the geometry here
   is already validated by the layout engine - the browser only has to draw it.

   Loaded from a CDN with a graceful fallback. If three.js is unavailable the
   panel hides itself rather than showing a broken canvas, because an empty
   grey box is worse than no box.
   ========================================================================== */

(function (global) {
  "use strict";

  const CDN = "https://unpkg.com/three@0.140.2/build/three.min.js";

  // Must match the 2D view so the two representations agree.
  const CATEGORY_COLOR = {
    bed: 0xcfe0f7, desk: 0xd9f0e0, chair: 0xe6e0f5, wardrobe: 0xf0e0d4,
    storage: 0xf0e0d4, tv_unit: 0xd6dde6, seating: 0xf7e0e6, table: 0xf3e8d0,
    soft: 0xefeaf6, decor: 0xfdf2d9, lighting: 0xfff6cf, paint: 0xeeeeee,
    kitchen: 0xdfe8ee,
  };
  const DEFAULT_COLOR = 0xe5e7eb;
  const EDGE = 0x6b7280;

  // A ceiling of 3 m keeps a 2.1 m wardrobe from looking like a monolith.
  const MAX_H = 300;


  let loadPromise = null;

  /**
   * Resolve three.js.
   *
   * A vendored copy is loaded synchronously in index.html, so in normal use
   * this resolves immediately. The dynamic path is kept only as a fallback
   * for a page that embeds layout3d.js without the vendor script; the earlier
   * version injected it asynchronously and raced its own render call, which
   * is why the canvas sometimes never appeared.
   */
  function loadThree() {
    if (global.THREE) return Promise.resolve(global.THREE);
    if (loadPromise) return loadPromise;

    loadPromise = new Promise((resolve, reject) => {
      let settled = false;
      const settle = (fn, arg) => {
        if (settled) return;
        settled = true;
        fn(arg);
      };
      const tryUrl = (url, onFail) => {
        const s = document.createElement("script");
        s.src = url;
        s.async = false;
        s.onload = () => {
          if (global.THREE) settle(resolve, global.THREE);
          else onFail();
        };
        s.onerror = onFail;
        document.head.appendChild(s);
      };
      tryUrl("/static/vendor/three.min.js", () => {
        tryUrl(CDN, () => {
          loadPromise = null;
          settle(reject, new Error("could not load three.js"));
        });
      });
    });

    return loadPromise;
  }

  /**
   * Build the scene.
   *
   * @param {HTMLElement} container
   * @param {object} layout - /api/v1/layout payload
   */
  function render(container, layout) {
    if (!container) return Promise.resolve(null);
    container.innerHTML = "";
    if (!layout || !layout.room || !layout.room.width_cm) {
      container.innerHTML = '<p class="muted small">No layout to show in 3D.</p>';
      return Promise.resolve(null);
    }

    return loadThree().then((THREE) => {
      container.innerHTML = "";
      const room = layout.room;
      const W = room.width_cm / 100; // metres
      const D = room.depth_cm / 100;
      const H = Math.min((room.ceiling_cm || 270) / 100, MAX_H / 100);

      const canvas = document.createElement("canvas");
      canvas.className = "plan3d-canvas";
      container.appendChild(canvas);

      const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
      renderer.setPixelRatio(Math.min(global.devicePixelRatio || 1, 2));
      const width = container.clientWidth || 600;
      const height = container.clientHeight || 380;
      renderer.setSize(width, height, false);

      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0xf8fafc);

      const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
      // Look at the room from above one corner, which shows both walls.
      camera.position.set(W * 1.3, H * 2.6, D * 1.5);
      camera.lookAt(W / 2, 0, D / 2);

      // --- lighting ----------------------------------------------------
      scene.add(new THREE.AmbientLight(0xffffff, 0.72));
      const sun = new THREE.DirectionalLight(0xffffff, 0.55);
      sun.position.set(W, H * 3, D);
      scene.add(sun);
      if (room.window_direction) {
        const w = new THREE.DirectionalLight(0xfff4e0, 0.5);
        // Put the light where the window is, so the render is directional.
        if (room.window_direction.indexOf("east") >= 0) w.position.set(W * 3, H, D / 2);
        else if (room.window_direction.indexOf("west") >= 0) w.position.set(-W * 3, H, D / 2);
        else if (room.window_direction.indexOf("south") >= 0) w.position.set(W / 2, H, D * 3);
        else w.position.set(W / 2, H, -D * 3);
        scene.add(w);
      }

      // --- floor and walls ---------------------------------------------
      const floor = new THREE.Mesh(
        new THREE.PlaneGeometry(W, D),
        new THREE.MeshStandardMaterial({ color: 0xfbfbfc, roughness: 0.9 })
      );
      floor.rotation.x = -Math.PI / 2;
      scene.add(floor);

      // Two back walls only - a full box hides the contents.
      const wallMat = new THREE.MeshStandardMaterial({
        color: 0xf1f5f9, roughness: 1, side: THREE.DoubleSide,
      });
      const mkWall = (w, h, x, y, z, ry) => {
        const mesh = new THREE.Mesh(new THREE.PlaneGeometry(w, h), wallMat);
        mesh.position.set(x, y, z);
        mesh.rotation.y = ry;
        scene.add(mesh);
      };
      const wallH = H * 0.65;
      mkWall(W, wallH, W / 2, wallH / 2, 0, 0);            // north
      mkWall(D, wallH, 0, wallH / 2, D / 2, Math.PI / 2);  // west

      // --- window opening ------------------------------------------------
      if (room.window_direction) {
        const glass = new THREE.Mesh(
          new THREE.PlaneGeometry(Math.min(1.2, W * 0.4), Math.min(1.1, wallH * 0.55)),
          new THREE.MeshBasicMaterial({ color: 0xbfdbfe, side: THREE.DoubleSide })
        );
        const y = wallH * 0.55;
        if (room.window_direction.indexOf("north") >= 0) glass.position.set(W / 2, y, 0.02);
        else if (room.window_direction.indexOf("west") >= 0) {
          glass.position.set(0.02, y, D / 2);
          glass.rotation.y = Math.PI / 2;
        } else if (room.window_direction.indexOf("east") >= 0) {
          glass.position.set(W - 0.02, y, D / 2);
          glass.rotation.y = Math.PI / 2;
        } else glass.position.set(W / 2, y, D - 0.02);
        scene.add(glass);
      }

      // --- furniture ------------------------------------------------------
      (layout.items || []).forEach((it) => {
        const w = Math.max(it.width / 100, 0.05);
        const d = Math.max(it.depth / 100, 0.05);
        const h = Math.max(Math.min(it.height || 40, MAX_H) / 100, 0.05);
        const geo = new THREE.BoxGeometry(w, h, d);
        const mat = new THREE.MeshStandardMaterial({
          color: CATEGORY_COLOR[it.category] || DEFAULT_COLOR,
          roughness: 0.72,
        });
        const mesh = new THREE.Mesh(geo, mat);
        // The layout is top-down in cm; three.js is y-up in metres.
        mesh.position.set(
          (it.x + it.width / 2) / 100,
          h / 2,
          (it.y + it.depth / 2) / 100
        );
        mesh.userData = { name: it.name, category: it.category };
        scene.add(mesh);

        const edges = new THREE.LineSegments(
          new THREE.EdgesGeometry(geo),
          new THREE.LineBasicMaterial({ color: EDGE })
        );
        edges.position.copy(mesh.position);
        scene.add(edges);
      });

      // --- orbit controls -------------------------------------------------
      // A minimal hand-rolled orbit, because importing OrbitControls from the
      // CDN adds a second network dependency for very little gain.
      let azimuth = 0.72, polar = 0.92, dist = Math.max(W, D) * 2.1;
      let dragging = false, lastX = 0, lastY = 0;
      const target = new THREE.Vector3(W / 2, 0, D / 2);

      canvas.addEventListener("pointerdown", (e) => {
        dragging = true; lastX = e.clientX; lastY = e.clientY;
        canvas.setPointerCapture(e.pointerId);
      });
      canvas.addEventListener("pointermove", (e) => {
        if (!dragging) return;
        azimuth -= (e.clientX - lastX) * 0.008;
        polar = Math.max(0.12, Math.min(1.5, polar - (e.clientY - lastY) * 0.008));
        lastX = e.clientX; lastY = e.clientY;
        draw();
      });
      const stop = (e) => {
        dragging = false;
        if (e && e.pointerId !== undefined && canvas.hasPointerCapture(e.pointerId)) {
          canvas.releasePointerCapture(e.pointerId);
        }
      };
      canvas.addEventListener("pointerup", stop);
      canvas.addEventListener("pointercancel", stop);
      canvas.addEventListener("wheel", (e) => {
        e.preventDefault();
        dist = Math.max(Math.max(W, D) * 0.8, Math.min(
          Math.max(W, D) * 5, dist * (1 + Math.sign(e.deltaY) * 0.1)
        ));
        draw();
      }, { passive: false });

      function draw() {
        camera.position.set(
          target.x + dist * Math.sin(polar) * Math.sin(azimuth),
          target.y + dist * Math.cos(polar),
          target.z + dist * Math.sin(polar) * Math.cos(azimuth)
        );
        camera.lookAt(target);
        renderer.render(scene, camera);
      }

      // --- resize ----------------------------------------------------------
      const onResize = () => {
        const w = container.clientWidth || 600;
        const h = container.clientHeight || 380;
        renderer.setSize(w, h, false);
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        draw();
      };
      if (global.ResizeObserver) {
        new ResizeObserver(onResize).observe(container);
      } else {
        global.addEventListener("resize", onResize);
      }

      draw();

      const hint = document.createElement("p");
      hint.className = "muted small plan3d-hint";
      hint.textContent = "Drag to rotate, scroll to zoom.";
      container.appendChild(hint);

      return { itemCount: (layout.items || []).length };
    }).catch((err) => {
        // Offline or CDN blocked: say so explicitly. An empty panel reads as
        // a broken feature rather than a missing network.
        container.innerHTML =
            '<p class="muted small">3D view needs three.js, which could not be ' +
            'loaded (no network?). The 2D floor plan shows the same measurements.</p>';
        container.dataset.threeError = "unavailable";
        return null;
    });
  }

  global.Layout3D = { render: render };
})(window);
