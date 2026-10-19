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

    function displayLabel(value) {
        return String(value || "")
            .replace(/_/g, " ")
            .replace(/\b[a-z]/g, (letter) => letter.toUpperCase())
            .replace(/\bTv\b/g, "TV");
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

        // Directions feed both the layout engine (door swing, window band)
        // and the Vastu rules. Sent as explicit fields so they are
        // authoritative rather than inferred from prose.
        const door = $("door_direction") && $("door_direction").value;
        if (door) payload.door_direction = door;
        const win = $("window_direction") && $("window_direction").value;
        if (win) payload.window_direction = win;

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
        const vb = $("validation"); if (vb) { vb.hidden = true; vb.innerHTML = ""; }
        const bpb = $("budgetPlan"); if (bpb) { bpb.hidden = true; bpb.innerHTML = ""; }
        // Hiding, not clearing: emptying the panel destroys the child nodes
        // the renderers write into, and the next render then has nowhere to
        // put its output.
        const vp = $("vastuPanel"); if (vp) { vp.hidden = true; }
        const rp = $("refinePanel"); if (rp) { rp.hidden = true; }
        const p3 = $("plan3d"); if (p3) { p3.innerHTML = ""; delete p3.dataset.drawn; }
        // A new brief invalidates everything derived from the last one, so
        // refine and PDF can never act on a stale design.
        currentDesign = null;
        currentLayout = null;
        currentVastu = null;
        currentCost = null;
        currentBudgetPlan = null;
        lastValidation = null;
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

    async function renderLayout(payload, data) {
        // Once a design exists, the plan shipped with the first option is the
        // authority: it is computed from the same priced line items the
        // shopping list shows, with the door and window bearings from the
        // form. Before that, /layout draws a provisional plan from the brief
        // while the model is still thinking.
        const panel = $("layoutPanel");
        if (!panel) return;
        const shipped = data && data.options && data.options[0]
            && data.options[0].layout;
        try {
            let plan;
            if (shipped && shipped.items) {
                plan = shipped;
            } else {
                const res = await fetch("/api/v1/layout", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload),
                });
                if (!res.ok) {
                    const err = await res.json().catch(() => ({}));
                    panel.hidden = true;
                    $("layoutMeta").textContent = err.detail || "No plan available.";
                    return;
                }
                plan = await res.json();
            }
            panel.hidden = false;
            currentLayout = plan;
            wireTabs();
            LayoutView.render($("plan"), plan);

            const cats = [...new Set((plan.items || []).map((i) => i.category))];
            $("planLegend").innerHTML = "";
            $("planLegend").appendChild(LayoutView.legend(cats));

            const s = plan.score || {};
            const room = plan.room || {};
            const blocking = (data && (
                ((data.validation && data.validation.violations) || [])
                    .some((v) => v.severity === "error")
                || ((data.options || []).some((option) =>
                    ((option.validation && option.validation.violations) || [])
                        .some((v) => v.severity === "error")
                ))
            ));
            // The exact dimensions the client entered, with the area and the
            // bearings actually used, so the plan is traceable to the form.
            const dimChip = room.dimension_text
                ? `<span class="chip">${esc(room.dimension_text)} · ${room.area_sqft || "?"} sq ft</span>`
                : "";
            const bearingChip = room.door_direction || room.window_direction
                ? `<span class="chip">door ${esc(room.door_direction || "—")} · window ${esc(room.window_direction || "—")}</span>`
                : "";
            $("layoutMeta").innerHTML = dimChip + bearingChip +
                `<span class="chip ${s.errors ? "bad" : "good"}">${
                    s.errors ? `${s.errors} conflict${s.errors > 1 ? "s" : ""}` : "No conflicts"
                }</span>` +
                (s.unplaced ? ` <span class="chip warn">${s.unplaced} unplaced</span>` : "") +
                (blocking ? "" : ` <span class="chip">score ${s.overall}</span>`);

            // Approximate zone sizes, so "how big is the study corner?" has an
            // answer taken from the plan itself.
            if (plan.zones && plan.zones.length) {
                $("planLegend").innerHTML += `
                    <div class="zone-sizes">
                      <strong>Zone sizes (approx.)</strong>
                      <ul>${plan.zones.map((z) =>
                        `<li>${esc(displayLabel(z.label))}: ${z.approx_width_ft} × ${z.approx_depth_ft} ft (${z.approx_area_sqft} sq ft)</li>`
                      ).join("")}</ul>
                    </div>`;
            }

            const issues = (plan.issues || []).concat(
                (plan.unplaced || []).map((u) => ({
                    code: "unplaced", severity: "error",
                    message: `${u.name} could not be placed - ${u.reason}.`,
                }))
            );
            $("planIssues").innerHTML = issues.length
                ? issues
                      .map(
                          (i) =>
                              `<div class="plan-issue ${i.severity}"><span class="mark">${
                                  i.severity === "error" ? "✕" : "!"
                              }</span> ${esc(i.message)}</div>`
                      )
                      .join("")
                : "";

            // Vastu scores the measured layout, so it has to wait for this
            // promise to resolve. Calling it alongside the request meant
            // currentLayout was still null and the panel silently stayed
            // hidden. When the design shipped its own rule results, those are
            // used - they are the same assessment the validator blocked on.
            const shippedVastu = data && data.options && data.options[0]
                && data.options[0].vastu_results;
            if (shippedVastu && shippedVastu.length) {
                const first = data.options[0];
                showVastu({
                    score: first.vastu_score != null ? first.vastu_score : 0,
                    grade: "",
                    passed: shippedVastu.filter((r) => r.passed).length,
                    total: shippedVastu.length,
                    results: shippedVastu,
                    notes: first.vastu_notes || [],
                });
            } else if (typeof renderVastu === "function") {
                renderVastu(lastPayload || buildPayload());
            }
            // If the client is already looking at the 3D tab, draw it now
            // that the layout exists.
            if (typeof global.__draw3d === "function") global.__draw3d();
        } catch (e) {
            panel.hidden = true;
        }
    }

    // Keeps the last design so refine and PDF act on exactly what is on screen.
    let currentDesign = null;
    let currentLayout = null;
    let currentVastu = null;
    let lastPayload = null;
    let lastValidation = null;
    let currentBudgetPlan = null;
    let currentCost = null;

    // ------------------------------------------------------- Vastu panel

    async function renderVastu(payload) {
        const panel = $("vastuPanel");
        if (!panel || !currentLayout) { if (panel) panel.hidden = true; return; }
        try {
            const res = await fetch("/api/v1/vastu", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    items: currentLayout.items,
                    room_type: payload.room_type,
                    door_direction: payload.door_direction,
                    window_direction: payload.window_direction,
                    compliance: payload.vastu_compliance,
                }),
            });
            if (!res.ok) { panel.hidden = true; return; }
            showVastu(await res.json());
        } catch (e) {
            panel.hidden = true;
        }
    }

    function showVastu(v) {
        const panel = $("vastuPanel");
        if (!panel) return;
        {
            currentVastu = v;
            panel.hidden = false;

            const strict = lastPayload && lastPayload.vastu_compliance === "strict";
            // Under Strict, any failed strong rule means the score must not
            // read as a pass - the same rule the brief check applies.
            const failedStrong = (v.results || []).some(
                (r) => !r.passed && r.severity === "error"
            );
            const ok = v.score >= 75 && !(strict && failedStrong);
            $("vastuMeta").innerHTML =
                `<span class="chip ${ok ? "good" : v.score >= 60 ? "warn" : "bad"}">` +
                `${v.score} / 100</span>` +
                (v.grade ? ` <span class="chip">${esc(v.grade)}</span>` : "") +
                ` <span class="chip">${v.passed} of ${v.total} rules</span>` +
                (strict && failedStrong
                    ? ` <span class="chip bad">Strict: rule failed</span>`
                    : "");
            $("vastuBody").dataset.ready = "1";

            const rows = (v.results || []).map((r) => {
                const cls = r.passed ? "pass" : r.severity === "error" ? "error" : "warning";
                const mark = r.passed ? "✓" : r.severity === "error" ? "✕" : "!";
                return `<div class="vastu-rule ${cls}">
                    <span class="mark">${mark}</span>
                    <div>
                        <strong>${esc(r.title)}</strong>
                        <p>${esc(r.explanation)}</p>
                        ${r.remedy ? `<p class="remedy">Fix: ${esc(r.remedy)}</p>` : ""}
                    </div>
                </div>`;
            }).join("");

            const notes = (v.notes || []).map((n) => `<p class="muted small">${esc(n)}</p>`).join("");
            const missing = (v.missing_inputs || []).length
                ? `<p class="muted small">Add ${esc(v.missing_inputs.join(" and ").replace(/_/g, " "))} for a fuller score.</p>`
                : "";
            $("vastuBody").innerHTML = rows + notes + missing;
        }
    }

    // ---------------------------------------------------------- refinement

    function renderRefine() {
        const box = $("refinePanel");
        if (!box) return;
        box.hidden = false;
        const names = (currentDesign && currentDesign.furniture || [])
            .map((item) => item.name).filter(Boolean);
        const suggestions = names.slice(0, 3).flatMap((name) => [
            `Make ${name} bigger`,
            `Remove ${name}`,
        ]);
        $("refineSuggestions").innerHTML = suggestions
            .map((s) => `<button type="button" class="chip suggest">${esc(s)}</button>`)
            .join("");
        $("refineSuggestions").onclick = (e) => {
            if (e.target.classList.contains("suggest")) {
                $("refineInput").value = e.target.textContent;
                $("refineBtn").click();
            }
        };
    }

    // Build the same shape `renderOption` expects from a refine response, so
    // the edit step and the option card share one price calculation server-side.
    function _optionFromRefine(design, data) {
        const cost = data.cost || {};
        return {
            title: design.title,
            concept: design.concept,
            layout_description: design.layout_description,
            zones: design.zones,
            lighting_plan: design.lighting_plan,
            storage_strategy: design.storage_strategy,
            decor_suggestions: design.decor_suggestions,
            vastu_notes: design.vastu_notes,
            trade_offs: design.trade_offs,
            pros: design.pros,
            cons: design.cons,
            color_palette: design.color_palette,
            estimated_cost_inr: data.total_inr != null
                ? data.total_inr : cost.estimated_cost_inr,
            shopping_list: (cost.line_items || []).map((li) => ({
                sku: li.sku,
                name: li.name,
                brand: li.brand,
                quantity: li.quantity,
                unit_price_inr: li.unit_price_inr,
                line_total_inr: li.line_total_inr,
            })),
            validation: data.validation || null,
            layout: design.layout || null,
        };
    }

    async function runRefine() {
        const input = $("refineInput");
        const text = input.value.trim();
        const out = $("refineResult");
        if (!text || !currentDesign) { if (out) out.innerHTML = ""; return; }
        const btn = $("refineBtn");
        if (btn) btn.disabled = true;
        if (out) out.innerHTML = '<p class="muted small">Applying…</p>';
        try {
            const res = await fetch("/api/v1/refine", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    design: currentDesign,
                    request: text,
                    budget_inr: lastPayload ? lastPayload.budget_amount_inr : null,
                }),
            });
            const data = await res.json();
            // A request that names something absent ("make the desk bigger" in
            // a kitchen) is refused explicitly and changes nothing - it used to
            // silently edit an unrelated item and report "Add chair only".
            if (!data.understood) {
                if (out) out.innerHTML =
                    `<p class="refine-warn">${esc(data.message || "Could not tell what to change.")}</p>`;
                return;
            }
            currentDesign = data.design;
            // The edited design is repriced into the option shape before it is
            // redrawn, so the card shows the same quantity/unit/line rows and
            // the same total the edit step reports.
            const repriced = data.design && data.cost
                ? _optionFromRefine(data.design, data)
                : data.design;
            renderOptions([repriced]);
            if (data.cost && data.cost.line_items) {
                renderBudgetPlan(data.budget_plan);
            }
            if (out) {
                const changed = (data.changed_fields || []).join(", ") || "nothing";
                out.innerHTML =
                    `<p class="refine-ok">Updated: <strong>${esc(changed)}</strong></p>` +
                    `<p class="muted small">${esc(data.message)}</p>` +
                    (data.total_inr != null
                        ? `<p class="muted small">New total ${inr(data.total_inr)}</p>`
                        : "");
            }
            input.value = "";
        } catch (e) {
            if (out) out.innerHTML = '<p class="muted small">Could not apply that change.</p>';
        } finally {
            if (btn) btn.disabled = false;
        }
    }

    // ---------------------------------------------------------------- PDF

    async function downloadPdf() {
        if (!currentDesign) return;
        const btn = $("pdfBtn");
        if (btn) btn.disabled = true;
        try {
            const res = await fetch("/api/v1/proposal.pdf", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    design: currentDesign,
                    cost: currentCost,
                    layout: currentLayout,
                    validation: lastValidation,
                    vastu: currentVastu,
                    budget_plan: currentBudgetPlan,
                    client_name: "Client",
                    room_type: lastPayload ? lastPayload.room_type : "",
                }),
            });
            if (!res.ok) throw new Error("pdf failed");
            const blob = await res.blob();
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = "interiorai-proposal.pdf";
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        } catch (e) {
            if (btn) btn.title = "Could not build the PDF.";
        } finally {
            if (btn) btn.disabled = false;
        }
    }

    // Runs once a design has rendered, whichever transport delivered it.
    function _afterResult(data, payload) {
        lastValidation = data.validation || null;
        currentBudgetPlan = data.budget_plan || null;
        currentCost = data.cost_breakdown || null;
        const first = (data.options || [])[0];
        currentDesign = first ? _optionAsDesign(first, data) : null;
        renderRefine();
        // Vastu is kicked off from renderLayout, once the geometry it scores
        // actually exists.
    }

    // The refine and PDF endpoints work on the raw design dict, so the option
    // is flattened into the shape those endpoints expect.
    function _optionAsDesign(option, data) {
        const cost = data.cost_breakdown || {};
        const bySku = {};
        (cost.line_items || []).forEach((li) => { bySku[li.sku] = li; });
        return {
            title: option.title,
            concept: option.concept,
            layout_description: option.layout_description,
            zones: option.zones,
            lighting_plan: option.lighting_plan,
            storage_strategy: option.storage_strategy,
            decor_suggestions: option.decor_suggestions,
            vastu_notes: option.vastu_notes,
            pros: option.pros,
            cons: option.cons,
            color_palette: option.color_palette,
            furniture: (option.furniture || []).map((f) => {
                const line = bySku[f.sku] || {};
                return {
                    name: f.name,
                    category: f.category,
                    sku: f.sku,
                    quantity: f.quantity,
                    brand_suggestions: f.brand_suggestions || [line.brand].filter(Boolean),
                };
            }),
        };
    }

    // ------------------------------------------------------- 2D/3D tab bar

    function wireTabs() {
        const bar = document.querySelector(".tabbar");
        if (!bar || bar.dataset.wired) return;
        bar.dataset.wired = "1";
        const draw3d = () => {
            const host = $("plan3d");
            if (!host || host.dataset.drawn) return;
            if (!currentLayout) return;      // nothing to draw yet
            if (!global.Layout3D) return;
            host.dataset.drawn = "1";
            Layout3D.render(host, currentLayout);
        };
        bar.addEventListener("click", (e) => {
            const btn = e.target.closest(".tab");
            if (!btn) return;
            bar.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
            btn.classList.add("active");
            const is3d = btn.dataset.view === "3d";
            $("plan").hidden = is3d;
            $("plan3d").hidden = !is3d;
            if (is3d) draw3d();
        });
        // Exposed so the layout fetch can draw the 3D view immediately when a
        // client lands on the 3D tab, without waiting for another click.
        global.__draw3d = draw3d;
    }

    async function renderValidation(validation) {
        const box = $("validation");
        if (!box) return;
        if (!validation || !validation.checks_run) { box.hidden = true; return; }

        const v = validation.violations || [];
        const errors = v.filter((x) => x.severity === "error");
        const warnings = v.filter((x) => x.severity === "warning");
        const pass = errors.length === 0;

        const items = v
            .map(
                (x) => `<li class="vcheck ${x.severity}">
            <span class="mark">${x.severity === "error" ? "✕" : "!"}</span>
            <span>${esc(x.message)}</span>
        </li>`
            )
            .join("");

        box.hidden = false;
        box.innerHTML = `
            <h3>Brief check
                <span class="verdict ${pass ? "pass" : "fail"}">${
            pass ? `Pass · ${validation.checks_passed}/${validation.checks_run}` : "Needs work"
        }</span>
            </h3>
            <p class="muted">Code verifies these before you see the result.</p>
            <ul class="checks">${items || "<li class='vcheck pass'><span class='mark'>✓</span><span>All checks passed</span></li>"}</ul>
            ${
                errors.length
                    ? `<p class="muted small">${errors.length} blocking issue(s), ${
                          warnings.length
                      } warning(s).</p>`
                    : ""
            }`;
    }

    function renderBudgetPlan(plan) {
        const box = $("budgetPlan");
        if (!box) return;
        if (!plan || !plan.phase1) { box.hidden = true; return; }

        const line = (i) =>
            `<tr><td>${esc(i.name)}</td><td class="num">${inr(i.line_total_inr)}</td></tr>`;
        const phase = (p) => `
            <h4>${esc(p.label)}</h4>
            <table class="phase-table">${(p.line_items || []).map(line).join("")}</table>
            <div class="phase-total"><span>Total</span><span class="num">${inr(p.total_inr)}</span></div>`;

        box.hidden = false;
        box.innerHTML = `
            <h3>Budget check</h3>
            <p class="muted">Every price comes from the product catalog and is added up by code.</p>
            <p class="headline">${esc(plan.headline || "")}</p>
            ${phase(plan.phase1)}
            ${
                plan.phase1 && plan.phase1.remaining_inr
                    ? `<p class="muted small">${inr(plan.phase1.remaining_inr)} left of the budget</p>`
                    : ""
            }
            ${plan.phase2 && plan.phase2.line_items && plan.phase2.line_items.length ? phase(plan.phase2) : ""}`;
    }

    function renderEvaluation(evaluation, validation, modelUsed, options) {
        if (!evaluation || !evaluation.scores) return;
        const box = $("evaluation");
        // A high score next to blocking issues is the contradiction the client
        // sees first. When the code check has blocking issues, the overall is
        // shown as "withheld" rather than as a good-looking number.
        const validatorViolations = [
            ...((validation && validation.violations) || []),
            ...((options || []).flatMap((option) =>
                (option.validation && option.validation.violations) || []
            )),
        ];
        const blocking = validatorViolations
            .filter((v) => v.severity === "error");
        const rows = blocking.length ? "" : Object.entries(evaluation.scores)
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

        const overall = blocking.length
            ? `<span class="overall withheld" title="Blocking brief-check issues remain">score withheld</span>`
            : `<span class="overall">${Number(evaluation.overall || 0).toFixed(1)}/10</span>`;

        // "Remaining issues", not "Issues addressed". The critic's list is
        // what is still wrong, and labelling it as fixed was exactly backwards.
        const validatorIssues = blocking
            .filter((v) => v.severity === "error")
            .map((v) => v.message)
            .filter((message, index, all) => all.indexOf(message) === index);
        const remaining = validatorIssues.length
            ? validatorIssues
            : (evaluation.remaining_issues || evaluation.issues || []);
        const criticLabel = modelUsed === "Rule-based mode"
            ? "Critic Evaluation · heuristic"
            : "Critic Evaluation";

        box.innerHTML = `
            <h3>${criticLabel} ${overall}</h3>
            ${blocking.length
                ? `<p class="muted small">${blocking.length} blocking brief-check issue(s) remain, so the score is not shown as a pass.</p>`
                : ""}
            ${rows}
            ${remaining.length
                ? `<div class="issues"><strong>Remaining issues:</strong><ul>${
                    remaining.map((i) => `<li>${esc(i)}</li>`).join("")
                }</ul></div>`
                : `<div class="issues"><strong>Remaining issues:</strong> none</div>`}
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
                        <span class="source-score">${
                            (s.metadata && s.metadata.dense_similarity != null)
                                ? `similarity ${Number(s.metadata.dense_similarity).toFixed(3)}`
                                : `rank score ${Number(s.score).toFixed(3)}`
                        }</span>
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
        // One row per shopping-list line: quantity, unit price, and the line
        // total the option total is summed from. Prices come from the server's
        // `shopping_list`, which is computed from the catalog - the same rows
        // the total uses. Building rows from the model's fields is what made
        // every item show ₹0 next to a real total.
        const rows = option.shopping_list && option.shopping_list.length
            ? option.shopping_list
            : (option.furniture || []).map((f) => ({
                name: f.name,
                brand: (f.brand_suggestions || [])[0] || "—",
                quantity: f.quantity || 1,
                unit_price_inr: f.unit_price_inr || 0,
                line_total_inr: f.unit_price_inr
                    ? f.unit_price_inr * (f.quantity || 1)
                    : (f.line_total_inr || 0),
            }));
        const furniture = rows
            .map(
                (f) => `
                <tr>
                    <td>${esc(f.name)}</td>
                    <td>${esc(f.brand || "—")}</td>
                    <td class="num">${f.quantity}</td>
                    <td class="num">${inr(f.unit_price_inr)}</td>
                    <td class="num">${inr(f.line_total_inr)}</td>
                </tr>`
            )
            .join("");
        // The option total is the sum of the rows above it, computed once.
        const shoppingTotal = rows.reduce(
            (sum, f) => sum + Number(f.line_total_inr || 0), 0
        );
        const optionValidation = option.validation || null;
        const optionBlocking = ((optionValidation && optionValidation.violations) || [])
            .filter((v) => v.severity === "error");

        const breakdown = (option.budget_breakdown || [])
            .map(
                (b) => `
                <div class="bd-row">
                    <span>${esc(displayLabel(b.category))}</span>
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

        const dims = option.layout && option.layout.room;
        const zones = option.layout && option.layout.zones;
        const suggestions = (option.budget_suggestions || []).map(
            (item) => `${item.name} (${inr(item.price_inr)})`
        );
        const remainingBudget = option.remaining_budget_inr;

        return `
        <article class="option">
            <div class="option-head">
                <h3>Option ${index + 1}: ${esc(option.title)}</h3>
                <div class="option-price ${optionBlocking.length ? "fail" : ""}">
                    <span class="price">${inr(shoppingTotal || option.estimated_cost_inr)}</span>
                    <span class="price-label">${optionBlocking.length
                        ? `estimated total · ${optionBlocking.length} blocking issue(s)`
                        : "estimated total, summed from the list below"}</span>
                </div>
            </div>
            ${remainingBudget != null
                ? `<p class="muted small">${inr(remainingBudget)} left of the budget${
                    suggestions.length
                        ? ` · could buy ${suggestions.map(esc).join(", ")}`
                        : ""
                }</p>`
                : ""}

            <p class="concept">${esc(option.concept)}</p>

            ${swatches ? `<div class="palette">${swatches}${hexes}</div>` : ""}

            <div class="option-grid">
                <div class="opt-col">
                    <h4>Layout${dims ? ` <span class="hint">${esc(dims.dimension_text || "")}</span>` : ""}</h4>
                    <p>${esc(option.layout_description)}</p>
                    ${list(option.zones, "zone-list")}
                    ${zones && zones.length
                        ? `<h4 class="sub">Zone sizes <span class="hint">approximate</span></h4>
                           <ul class="zone-list">${zones.map((z) =>
                               `<li>${esc(displayLabel(z.label))}: about ${z.approx_width_ft} × ${z.approx_depth_ft} ft (${z.approx_area_sqft} sq ft)</li>`
                           ).join("")}</ul>`
                        : ""}
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
                     <thead><tr><th>Item</th><th>Brand</th><th class="num">Qty</th><th class="num">Unit price</th><th class="num">Line total</th></tr></thead>
                     <tbody>${furniture}</tbody>
                     <tfoot><tr><td colspan="4">Total</td><td class="num">${inr(shoppingTotal)}</td></tr></tfoot>
                   </table>`
                : ""}

            ${breakdown
                ? `<h4>Budget Breakdown</h4><div class="breakdown">${breakdown}</div>`
                : ""}

            ${option.decor_suggestions && option.decor_suggestions.length
                ? `<h4>Decor</h4>${list(option.decor_suggestions, "decor-list")}`
                : ""}

            ${option.trade_offs && option.trade_offs.length
                ? `<h4>Trade-offs</h4>${list(option.trade_offs, "cons")}`
                : (option.cons && option.cons.length ? `<h4>Trade-offs</h4>${list(option.cons, "cons")}` : "")}

            ${option.vastu_notes && option.vastu_notes.length
                ? `<h4>Vastu Notes <span class="hint">generated from the rule check${option.vastu_score != null ? ` · scored ${option.vastu_score}/100` : ""}</span></h4>${list(option.vastu_notes, "vastu-list")}`
                : ""}

            <div class="option-foot">
                ${list(option.pros, "pros")}
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
        if (data.options.length < (lastPayload && lastPayload.num_options || 1)) {
            const reason = (data.rejected_options || []).join("; ");
            parts.push(
                `Only ${data.options.length} distinct option(s) returned` +
                (reason ? `: ${reason}` : "")
            );
        }
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
                        renderValidation(data.validation);
                        renderBudgetPlan(data.budget_plan);
                        renderOptions(data.options || []);
                        renderEvaluation(data.evaluation, data.validation, data.model_used, data.options);
                        renderSources(data.context_sources);
                        renderLayout(payload, data);
                        _afterResult(data, payload);
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
        renderValidation(data.validation);
        renderBudgetPlan(data.budget_plan);
        renderOptions(data.options || []);
        renderEvaluation(data.evaluation, data.validation, data.model_used, data.options);
        renderSources(data.context_sources);
        renderLayout(payload, data);
        _afterResult(data, payload);   // floor plan is independent of the LLM run
    }

    // ---------------------------------------------------------------- events

    // Refinement and PDF controls. Bound once; both no-op until a design
    // exists, which is checked inside each handler.
    const refineBtn = $("refineBtn");
    if (refineBtn) {
        refineBtn.addEventListener("click", runRefine);
    }
    const refineInput = $("refineInput");
    if (refineInput) {
        refineInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter") { e.preventDefault(); runRefine(); }
        });
    }
    const pdfBtn = $("pdfBtn");
    if (pdfBtn) {
        pdfBtn.addEventListener("click", downloadPdf);
    }

    // Exposed for automated UI checks. The renderers are otherwise closure-
    // scoped, which makes the completion path impossible to drive without
    // waiting on a live LLM round trip. This is a test seam, not an API.
    window.__interiorai = {
        renderOptions: renderOptions,
        renderValidation: renderValidation,
        renderBudgetPlan: renderBudgetPlan,
        afterResult: _afterResult,
        buildPayload: buildPayload,
        state: () => ({
            currentDesign, currentLayout, currentVastu,
            currentCost, currentBudgetPlan, lastValidation,
        }),
    };

    form.addEventListener("submit", async (e) => {
        e.preventDefault();

        const payload = buildPayload();
        lastPayload = payload;
        if (!payload.client_preferences) {
            alert("Please describe the client's needs in the brief.");
            return;
        }

        resetPipeline();
        setLoading(true);
        // Draw the plan straight away. It is computed in code with no model
        // call, so the user sees geometry while the design is still thinking.
        renderLayout(payload);

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
            const configuredProviders = Object.keys(data.providers || {}).length;
            badge.textContent = up.length
                ? up.join(" | ")
                : configuredProviders
                    ? "AI unavailable · catalog fallback"
                    : "catalog mode · no AI key";
            badge.title = up.length
                ? "AI-generated design prose is available."
                : "Catalog-based designs, prices, layout, Vastu checks, and PDFs work without an AI provider.";
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
