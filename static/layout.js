/* ==========================================================================
   Interactive 2D layout view.

   Renders the layout JSON from /api/v1/layout onto an SVG, scaled to fit.
   Deliberately dependency-free: no canvas library, no framework. The
   geometry is a handful of rects, and inline SVG is inspectable, printable,
   and accessible.

   The colour language is consistent with the rest of the app:
     - filled block  = a placed item
     - hatched area  = the door swing that must stay clear
     - blue band     = a window
     - dashed line   = the 60 cm circulation threshold
   ========================================================================== */

(function (global) {
  "use strict";

  const SVG_NS = "http://www.w3.org/2000/svg";

  // Per-category fills. Chosen to stay legible with dark text on top and to
  // read as furniture rather than UI chrome.
  const CATEGORY_STYLE = {
    bed: { fill: "#cfe0f7", stroke: "#5b86c4" },
    desk: { fill: "#d9f0e0", stroke: "#4f9d76" },
    chair: { fill: "#e6e0f5", stroke: "#8471c4" },
    wardrobe: { fill: "#f0e0d4", stroke: "#b98a63" },
    storage: { fill: "#f0e0d4", stroke: "#b98a63" },
    tv_unit: { fill: "#d6dde6", stroke: "#6c7c91" },
    seating: { fill: "#f7e0e6", stroke: "#c4788f" },
    table: { fill: "#f3e8d0", stroke: "#b39460" },
    soft: { fill: "#efeaf6", stroke: "#9a8fc0" },
    decor: { fill: "#fdf2d9", stroke: "#c4a24a" },
    lighting: { fill: "#fff6cf", stroke: "#c2a94e" },
    paint: { fill: "#eeeeee", stroke: "#999999" },
    kitchen: { fill: "#dfe8ee", stroke: "#7791a3" },
  };

  const DEFAULT_STYLE = { fill: "#e5e7eb", stroke: "#6b7280" };

  function styleFor(category) {
    return CATEGORY_STYLE[category] || DEFAULT_STYLE;
  }

  function el(name, attrs, text) {
    const node = document.createElementNS(SVG_NS, name);
    for (const k in attrs) {
      if (attrs[k] !== null && attrs[k] !== undefined) {
        node.setAttribute(k, String(attrs[k]));
      }
    }
    if (text !== undefined) node.textContent = text;
    return node;
  }

  /**
   * Render a layout into a container element.
   *
   * @param {HTMLElement} container
   * @param {object} layout  - the /api/v1/layout payload
   * @param {object} [opts]  - { scale, showClearance }
   */
  function render(container, layout, opts) {
    opts = opts || {};
    container.innerHTML = "";
    if (!container) return;
    if (!layout || !layout.room) {
      container.appendChild(
        Object.assign(document.createElement("p"), {
          className: "muted small",
          textContent: "No layout to show.",
        })
      );
      return;
    }

    const room = layout.room;
    const W = room.width_cm;
    const D = room.depth_cm;
    if (!W || !D) {
      container.appendChild(
        Object.assign(document.createElement("p"), {
          className: "muted small",
          textContent: "Room dimensions are unknown, so a plan cannot be drawn.",
        })
      );
      return;
    }

    // Padding in cm, for the dimension labels outside the walls.
    const PAD = Math.max(28, Math.min(W, D) * 0.09);
    const vbW = W + PAD * 2;
    const vbD = D + PAD * 2;
    // Small text for the inline "window" tag on the 30 cm wall band.
    const TINY_TEXT = 11;

    const svg = el("svg", {
      viewBox: `0 0 ${vbW} ${vbD}`,
      class: "plan-svg",
      role: "img",
      "aria-label":
        `Floor plan of a ${Math.round(W / 30.48)} by ${Math.round(
          D / 30.48
        )} foot room with ${(layout.items || []).length} items placed`,
    });
    svg.style.width = "100%";
    svg.style.height = "auto";
    svg.style.maxHeight = opts.maxHeight || "460px";
    svg.style.display = "block";

    // Everything below is in room coordinates, offset by the padding.
    const ox = PAD;
    const oy = PAD;

    // --- defs: hatch for the door swing -------------------------------
    const defs = el("defs", {});
    const pattern = el("pattern", {
      id: "doorHatch",
      width: 8,
      height: 8,
      patternUnits: "userSpaceOnUse",
      patternTransform: "rotate(45)",
    });
    pattern.appendChild(
      el("rect", { width: 8, height: 8, fill: "rgba(220,38,38,0.06)" })
    );
    pattern.appendChild(
      el("line", {
        x1: 0, y1: 0, x2: 0, y2: 8,
        stroke: "rgba(220,38,38,0.35)", "stroke-width": 2,
      })
    );
    defs.appendChild(pattern);
    svg.appendChild(defs);

    // --- floor ---------------------------------------------------------
    svg.appendChild(
      el("rect", {
        x: ox, y: oy, width: W, height: D,
        fill: "#fbfbfc", stroke: "#374151", "stroke-width": 3,
        rx: 2,
      })
    );

    // --- window band ----------------------------------------------------
    if (room.window_direction) {
      const bandDepth = 30;
      const bandW = Math.min(140, W * 0.4);
      let bx = ox, by = oy, bw = bandW, bh = bandDepth;
      switch (room.window_direction) {
        case "north": by = oy; break;
        case "south": by = oy + D - bandDepth; break;
        case "west": bx = ox; bw = bandDepth; bh = bandW; by = oy; break;
        case "east": bx = ox + W - bandDepth; bw = bandDepth; bh = bandW; by = oy; break;
        default: bx = null;
      }
      if (bx !== null) {
        svg.appendChild(
          el("rect", {
            x: bx, y: by, width: bw, height: bh,
            fill: "#bfdbfe", stroke: "#3b82f6", "stroke-width": 1.5,
          })
        );
        // A side-wall band is only 30 cm across, so a horizontal label is
        // clipped to a single letter. Rotate it to run along the band
        // instead, and only draw it when the band is long enough.
        const vertical = room.window_direction === "east" || room.window_direction === "west";
        const along = vertical ? bh : bw;
        if (along >= 60) {
          const cx = bx + bw / 2;
          const cy = by + bh / 2;
          svg.appendChild(
            el(
              "text",
              {
                x: cx, y: cy,
                "text-anchor": "middle",
                "dominant-baseline": "central",
                "font-size": TINY_TEXT, "font-weight": 600,
                fill: "#1e3a8a",
                transform: vertical ? `rotate(-90 ${cx} ${cy})` : null,
              },
              "window"
            )
          );
        }
      }
    }

    // --- door swing -----------------------------------------------------
    if (room.door_direction) {
      const s = 90;
      let dx = ox, dy = oy;
      if (room.door_direction === "south") dy = oy + D - s;
      if (room.door_direction === "east") dx = ox + W - s;
      if (room.door_direction.indexOf("south") >= 0) dy = oy + D - s;
      if (room.door_direction.indexOf("west") >= 0) dx = ox;
      if (room.door_direction.indexOf("east") >= 0) dx = ox + W - s;
      if (room.door_direction.indexOf("north") >= 0) dy = oy;

      svg.appendChild(
        el("rect", {
          x: dx, y: dy, width: s, height: s,
          fill: "url(#doorHatch)", stroke: "rgba(220,38,38,0.5)",
          "stroke-width": 1.5, "stroke-dasharray": "4 3",
        })
      );
      // The door leaf itself, drawn as a line on the wall.
      let lx1 = dx, ly1 = dy, lx2 = dx, ly2 = dy;
      if (room.door_direction.indexOf("north") >= 0) { lx1 = dx; ly1 = dy; lx2 = dx + 90; ly2 = dy; }
      if (room.door_direction.indexOf("south") >= 0) { lx1 = dx; ly1 = dy + s; lx2 = dx + 90; ly2 = dy + s; }
      if (room.door_direction.indexOf("west") >= 0) { lx1 = dx; ly1 = dy; lx2 = dx; ly2 = dy + 90; }
      if (room.door_direction.indexOf("east") >= 0) { lx1 = dx + s; ly1 = dy; lx2 = dx + s; ly2 = dy + 90; }
      svg.appendChild(
        el("line", {
          x1: lx1, y1: ly1, x2: lx2, y2: ly2,
          stroke: "#b91c1c", "stroke-width": 5, "stroke-linecap": "round",
        })
      );
    }

    // --- placed items ---------------------------------------------------
    const items = layout.items || [];
    items.forEach(function (it, i) {
      const st = styleFor(it.category);
      const g = el("g", {
        class: "plan-item",
        tabindex: "0",
        role: "listitem",
        "aria-label": `${it.name}, ${Math.round(it.width)} by ${Math.round(
          it.depth
        )} centimetres`,
      });

      g.appendChild(
        el("rect", {
          x: ox + it.x, y: oy + it.y,
          width: it.width, height: it.depth,
          fill: st.fill, stroke: st.stroke, "stroke-width": 2,
          rx: 3,
        })
      );

      // Label only if the footprint can actually hold the text. The viewBox
      // is in centimetres, so font size is expressed in cm too and must be
      // derived from the box it has to fit inside, not from a fixed pixel
      // value - that mismatch was overflowing the rectangles.
      const LABEL_SIZE = 11;          // ~11 cm tall glyphs in the viewBox
      const charW = LABEL_SIZE * 0.58;
      const maxChars = Math.floor(it.width / charW);
      if (maxChars >= 4 && it.depth >= LABEL_SIZE * 1.4) {
        const short =
          it.name.length > maxChars ? it.name.slice(0, maxChars - 1) + "…" : it.name;
        g.appendChild(
          el(
            "text",
            {
              x: ox + it.x + it.width / 2,
              y: oy + it.y + it.depth / 2 + LABEL_SIZE * 0.36,
              "text-anchor": "middle",
              "font-size": LABEL_SIZE,
              fill: "#1f2937", "font-weight": 600,
              // A light halo keeps the label readable over the fill.
              "paint-order": "stroke",
              stroke: "rgba(255,255,255,0.75)", "stroke-width": 2.5,
              "pointer-events": "none",
            },
            short
          )
        );
      }

      const title = el("title", {}, `${it.name}\n${Math.round(it.width)} × ${Math.round(
        it.depth
      )} cm (${it.width > it.depth ? "wide" : "deep"})${
        it.height ? `\n${Math.round(it.height)} cm tall` : ""
      }`);
      g.appendChild(title);
      svg.appendChild(g);
    });

    // --- dimension labels ------------------------------------------------
    // Also in centimetre units, scaled to the room so the text stays a
    // readable fraction of the drawing at any zoom level.
    const DIM_SIZE = Math.max(10, Math.min(16, Math.min(W, D) * 0.045));
    const dimStyle = {
      fill: "#6b7280", "font-size": DIM_SIZE, "font-weight": 600,
    };
    svg.appendChild(
      el(
        "text",
        Object.assign(
          { x: ox + W / 2, y: oy + D + PAD * 0.7, "text-anchor": "middle" },
          dimStyle
        ),
        `${Math.round(W / 30.48 * 10) / 10} ft`
      )
    );
    svg.appendChild(
      el(
        "text",
        Object.assign(
          {
            x: ox - PAD * 0.35, y: oy + D / 2,
            "text-anchor": "middle",
            transform: `rotate(-90 ${ox - PAD * 0.35} ${oy + D / 2})`,
          },
          dimStyle
        ),
        `${Math.round(D / 30.48 * 10) / 10} ft`
      )
    );

    // --- clearance ring around the longest walking gap --------------------
    if (opts.showClearance !== false && items.length) {
      // A single, honest annotation: the gap between the two nearest items
      // is the tightest point in the room, which is what a designer checks.
      let tightest = null;
      for (let a = 0; a < items.length; a++) {
        for (let b = a + 1; b < items.length; b++) {
          const A = items[a], B = items[b];
          const dx = Math.max(
            B.x - (A.x + A.width), A.x - (B.x + B.width), 0);
          const dy = Math.max(
            B.y - (A.y + A.depth), A.y - (B.y + B.depth), 0);
          const gap = dx === 0 ? dy : dy === 0 ? dx : Math.hypot(dx, dy);
          if (gap > 0 && (!tightest || gap < tightest.gap)) {
            tightest = { gap: gap, a: A, b: B };
          }
        }
      }
      if (tightest && tightest.gap < 90) {
        const A = tightest.a, B = tightest.b;
        const mx = ox + (A.x + A.width / 2 + (B.x + B.width / 2)) / 2;
        const my = oy + (A.y + A.depth / 2 + (B.y + B.depth / 2)) / 2;
        svg.appendChild(
          el(
            "text",
            {
              x: mx, y: my, "text-anchor": "middle",
              "font-size": TINY_TEXT, "font-weight": 700,
              fill: tightest.gap < 60 ? "#b91c1c" : "#a16207",
              "paint-order": "stroke", stroke: "#fff", "stroke-width": 4,
            },
            `${Math.round(tightest.gap)} cm`
          )
        );
      }
    }

    container.appendChild(svg);
  }

  /** A compact legend so the colours mean something. */
  function legend(categories) {
    const wrap = document.createElement("div");
    wrap.className = "plan-legend";
    (categories || []).forEach(function (c) {
      const st = styleFor(c);
      const chip = document.createElement("span");
      chip.className = "legend-chip";
      const dot = document.createElement("i");
      dot.style.background = st.fill;
      dot.style.borderColor = st.stroke;
      chip.appendChild(dot);
      chip.appendChild(document.createTextNode(c.replace(/_/g, " ")));
      wrap.appendChild(chip);
    });
    return wrap;
  }

  global.LayoutView = { render: render, legend: legend, styleFor: styleFor };
})(window);
