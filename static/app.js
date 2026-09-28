/**
 * InteriorAI frontend.
 *
 * Consumes the SSE endpoint so the user sees each agent execute in real time
 * rather than an opaque spinner. Falls back to the buffered POST endpoint if
 * EventSource is unavailable.
 */
(function () {
    "use strict";

    const $ = (id) => document.getElementById(id);
    const form = $("designForm");
    const btn = $("generateBtn");

    // ---------------------------------------------------------------- utils

    function inr(value) {
        const n = Number(value) || 0;
        if (n >= 10000000) return "₹" + (n / 10000000).toFixed(2) + " Cr";
        if (n >= 100000) return "₹" + (n / 100000).toFixed(2) + " L";
        if (n >= 1000) return "₹" + (n / 1000).toFixed(1) + "K";
        return "₹" + n;
    }

    function esc(str) {
        const div = document.createElement("div");
        div.textContent = str == null ? "" : String(str);
        return div.innerHTML;
    }

    function collectChips(containerId) {
        return Array.from(
            $(containerId).querySelectorAll("input[type=checkbox]:checked")
        ).map((el) => el.value);
    }

    function buildPayload() {
        const payload = {
            room_type: $("room_type").value,
            style: $("style").value,
            budget: $("budget").value,
            color_scheme: $("color_scheme").value,
            num_options: parseInt($("num_options").value, 10),
            vastu_compliance: $("vastu_compliance").value,
            client_preferences: $("client_preferences").value.trim(),
            furniture: collectChips("furnitureChips"),
            specific_requirements: collectChips("requirementChips"),
        };

        const dims = $("room_dimensions").value.trim();
        if (dims) payload.room_dimensions = dims;

        const occ = $("occupants").value.trim();
        if (occ) {
            payload.occupants = occ.split(",").map((s) => s.trim()).filter(Boolean);
        }

        const lifestyle = $("lifestyle_notes").value.trim();
        if (lifestyle) payload.lifestyle_notes = lifestyle;

        const amount = parseInt($("budget_amount_inr").value, 10);
        if (!isNaN(amount) && amount > 0) payload.budget_amount_inr = amount;

        return payload;
    }

    function setLoading(isLoading) {
        btn.disabled = isLoading;
        btn.querySelector(".spinner").hidden = !isLoading;
        btn.querySelector(".btn-text").textContent = isLoading
            ? "Generating…"
            : "Generate Designs";
    }

    // ------------------------------------------------------- pipeline panel

    let pipelineSteps = {};

    function resetPipeline() {
        pipelineSteps = {};
        $("pipeline").innerHTML = "";
        $("pipelinePanel").hidden = false;
        $("pipelineTimer").textContent = "";
        $("results").innerHTML = "";
        $("resultsPanel").hidden = true;
        $("evaluation").hidden = true;
        $("sources").innerHTML = "";
        $("sourcesPanel").hidden = true;
    }

    function addPipelineStep(step) {
        const key = step.agent + "::" + Object.keys(pipelineSteps).length;
        pipelineSteps[key] = step;

        const row = document.createElement("div");
        row.className = "step step-" + (step.status || "ok");
        row.dataset.key = key;
        row.innerHTML = `
            <div class="step-icon">${iconFor(step.agent, step.status)}</div>
            <div class="step-body">
                <div class="step-agent">${esc(step.agent)}</div>
                <div class="step-msg">${esc(step.message || "")}</div>
            </div>
            <div class="step-time">${step.duration_ms ? Math.round(step.duration_ms) + "ms" : ""}</div>
        `;
        $("pipeline").appendChild(row);
        row.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }

    function iconFor(agent, status) {
        if (status === "warning") return "⚠";
        switch (agent) {
            case "retriever": return "🔍";
            case "cache": return "⚡";
            case "planner": return "🗺️";
            case "architect": return "📐";
            case "budget": return "💰";
            case "critic": return "🎯";
            case "pipeline": return "🔄";
            default: return "•";
        }
    }

    // ------------------------------------------------------------ rendering

    function renderEvaluation(evaluation) {
        if (!evaluation || !evaluation.scores) return;
        const box = $("evaluation");
        const rows = Object.entries(evaluation.scores)
            .map(([k, v]) => {
                const pct = Math.max(0, Math.min(100, (Number(v) / 10) * 100));
                return `
                    <div class="score-row">
                        <span class="score-label">${esc(k.replace(/_/g, " "))}</span>
                        <div class="score-bar"><div class="score-fill" style="width:${pct}%"></div></div>
                        <span class="score-val">${Number(v).toFixed(1)}</span>
                    </div>`;
            })
            .join("");

        box.innerHTML = `
            <h3>Critic Evaluation <span class="overall">${Number(evaluation.overall || 0).toFixed(1)}/10</span></h3>
            ${rows}
            ${evaluation.issues && evaluation.issues.length
                ? `<div class="issues"><strong>Issues addressed:</strong><ul>${
                    evaluation.issues.map((i) => `<li>${esc(i)}</li>`).join("")
                }</ul></div>`
                : ""}
        `;
        box.hidden = false;
    }

    function renderSources(sources) {
        if (!sources || !sources.length) return;
        $("sources").innerHTML = sources
            .map(
                (s) => `
                <div class="source">
                    <div class="source-head">
                        <span class="source-name">${esc(s.source || "knowledge")}</span>
                        <span class="source-score">${Number(s.score).toFixed(3)}</span>
                    </div>
                    <p>${esc(s.content)}</p>
                </div>`
            )
            .join("");
        $("sourcesPanel").hidden = false;
    }

    function renderOptions(options) {
        $("results").innerHTML = options.map(renderOption).join("");
        $("resultsPanel").hidden = false;
    }

    function renderOption(option, index) {
        const furniture = (option.furniture || [])
            .map(
                (f) => `
                <tr>
                    <td>${esc(f.name)}${f.quantity > 1 ? ` <span class="qty">×${f.quantity}</span>` : ""}</td>
                    <td>${esc((f.brand_suggestions || []).slice(0, 2).join(", ") || "—")}</td>
                    <td class="num">${inr(f.estimated_price_inr)}</td>
                </tr>`
            )
            .join("");

        const breakdown = (option.budget_breakdown || [])
            .map(
                (b) => `
                <div class="bd-row">
                    <span>${esc(b.category)}</span>
                    <div class="bd-bar"><div class="bd-fill" style="width:${Math.min(
                        100,
                        Number(b.percentage) || 0
                    )}%"></div></div>
                    <span class="num">${inr(b.amount_inr)}</span>
                </div>`
            )
            .join("");

        const palette = option.color_palette;
        const swatches = palette
            ? [palette.primary, palette.secondary, palette.accent]
                .filter(Boolean)
                .map((c) => `<span class="swatch" title="${esc(c)}">${esc(c)}</span>`)
                .join("")
            : "";

        const hexes = palette && palette.hex_codes && palette.hex_codes.length
            ? `<div class="hexes">${palette.hex_codes
                  .map((h) => `<span class="hex" style="background:${esc(h)}" title="${esc(h)}"></span>`)
                  .join("")}</div>`
            : "";

        const list = (arr, cls) =>
            (arr || []).length
                ? `<ul class="${cls}">${arr.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>`
                : "";

        return `
        <article class="option">
            <div class="option-head">
                <h3>Option ${index + 1}: ${esc(option.title)}</h3>
                <div class="option-price">
                    <span class="price">${inr(option.estimated_cost_inr)}</span>
                    <span class="price-label">estimated total</span>
                </div>
            </div>

            <p class="concept">${esc(option.concept)}</p>

            ${swatches ? `<div class="palette">${swatches}${hexes}</div>` : ""}

            <div class="option-grid">
                <div class="opt-col">
                    <h4>Layout</h4>
                    <p>${esc(option.layout_description)}</p>
                    ${list(option.zones, "zone-list")}
                </div>
                <div class="opt-col">
                    <h4>Lighting</h4>
                    <p>${esc(option.lighting_plan)}</p>
                    <h4>Storage</h4>
                    <p>${esc(option.storage_strategy)}</p>
                </div>
            </div>

            ${furniture
                ? `<h4>Shopping List</h4>
                   <table class="furniture">
                     <thead><tr><th>Item</th><th>Brand</th><th class="num">Price</th></tr></thead>
                     <tbody>${furniture}</tbody>
                   </table>`
                : ""}

            ${breakdown
                ? `<h4>Budget Breakdown</h4><div class="breakdown">${breakdown}</div>`
                : ""}

            ${option.decor_suggestions && option.decor_suggestions.length
                ? `<h4>Decor</h4>${list(option.decor_suggestions, "decor-list")}`
                : ""}

            ${option.vastu_notes && option.vastu_notes.length
                ? `<h4>Vastu Notes</h4>${list(option.vastu_notes, "vastu-list")}`
                : ""}

            <div class="option-foot">
                ${list(option.pros, "pros")}
                ${list(option.cons, "cons")}
            </div>
        </article>`;
    }

    function renderMeta(data) {
        const parts = [
            `${data.options.length} option${data.options.length === 1 ? "" : "s"}`,
            `${(data.latency_ms / 1000).toFixed(1)}s`,
        ];
        if (data.cached) parts.push("⚡ cached");
        if (data.model_used) parts.push(data.model_used);
        $("resultsMeta").innerHTML = parts.map((p) => `<span class="meta-pill">${esc(p)}</span>`).join("");
    }

    // ------------------------------------------------------------- transport

    async function submitStreaming(payload) {
        const started = Date.now();
        const timer = setInterval(() => {
            $("pipelineTimer").textContent = `elapsed ${((Date.now() - started) / 1000).toFixed(0)}s`;
        }, 1000);

        try {
            const response = await fetch("/api/v1/generate/stream", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });

            if (!response.ok) {
                throw new Error("HTTP " + response.status);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = "";

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const frames = buffer.split("\n\n");
                buffer = frames.pop() || "";

                for (const frame of frames) {
                    const eventType = (frame.match(/^event:\s*(.+)$/m) || [])[1];
                    const dataLine = (frame.match(/^data:\s*(.+)$/m) || [])[1];
                    if (!eventType || !dataLine) continue;

                    if (eventType === "error") {
                        const err = JSON.parse(dataLine);
                        throw new Error(err.error || "Generation failed");
                    }
                    if (eventType === "step") {
                        addPipelineStep(JSON.parse(dataLine));
                    }
                    if (eventType === "complete") {
                        const data = JSON.parse(dataLine);
                        renderMeta(data);
                        renderOptions(data.options || []);
                        renderEvaluation(data.evaluation);
                        renderSources(data.context_sources);
                    }
                }
            }
        } finally {
            clearInterval(timer);
        }
    }

    async function submitBuffered(payload) {
        const url =
            "/api/v1/generate?use_cache=" + ($("use_cache").checked ? "true" : "false");
        addPipelineStep({ agent: "pipeline", status: "info", message: "Requesting designs…" });

        const response = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });

        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || "HTTP " + response.status);
        }

        const data = await response.json();
        renderMeta(data);
        renderOptions(data.options || []);
        renderEvaluation(data.evaluation);
        renderSources(data.context_sources);
    }

    // ---------------------------------------------------------------- events

    form.addEventListener("submit", async (e) => {
        e.preventDefault();

        const payload = buildPayload();
        if (!payload.client_preferences) {
            alert("Please describe the client's needs in the brief.");
            return;
        }

        resetPipeline();
        setLoading(true);

        try {
            if (typeof fetch === "undefined" || !window.ReadableStream) {
                await submitBuffered(payload);
            } else {
                await submitStreaming(payload);
            }
        } catch (err) {
            const row = document.createElement("div");
            row.className = "step step-warning";
            row.innerHTML = `
                <div class="step-icon">✕</div>
                <div class="step-body">
                    <div class="step-agent">Error</div>
                    <div class="step-msg">${esc(err.message)}</div>
                </div>`;
            $("pipeline").appendChild(row);
        } finally {
            setLoading(false);
        }
    });

    // ------------------------------------------------------------- health UI

    async function refreshHealth() {
        try {
            const response = await fetch("/api/v1/health");
            const data = await response.json();

            const up = Object.entries(data.providers || {})
                .filter(([, p]) => p.reachable)
                .map(([name, p]) => name + " · " + p.model);

            const badge = $("providerBadge");
            badge.textContent = up.length ? up.join(" | ") : "no provider available";
            badge.className = "badge " + (up.length ? "badge-ok" : "badge-warn");

            const hb = $("healthBadge");
            hb.textContent = data.status;
            hb.className = "badge " + (data.status === "ok" ? "badge-ok" : "badge-warn");
        } catch (e) {
            $("providerBadge").textContent = "offline";
            $("providerBadge").className = "badge badge-warn";
        }
    }

    refreshHealth();
    setInterval(refreshHealth, 30000);
})();
