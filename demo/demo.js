(function () {
    "use strict";

    let D = null;
    const ITEM = {};
    const USER = {};
    const BEH = {
        view: { label: "Xem", cls: "b-view", color: "var(--view)" },
        cart: { label: "Thêm giỏ", cls: "b-cart", color: "var(--cart)" },
        purchase: { label: "Mua", cls: "b-purchase", color: "var(--purchase)" },
    };

    const state = { train: null, eval: null }; // active step index per tab (null = chưa mở)
    let activeUser = null;
    let activeItem = null;

    const $ = (sel) => document.querySelector(sel);
    const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
    const th = (label, title, cls = "") => `<th${cls ? ` class="${cls}"` : ""} title="${esc(title)}">${label}</th>`;
    const itemLabel = (id) => { const it = ITEM[id]; return it ? `${it.label} · ${esc(it.category)}/${esc(it.brand)}` : `#product${id}`; };
    const behBadge = (b) => `<span class="badge ${BEH[b].cls}" title="${BEH[b].label}: loại hành vi của người dùng với sản phẩm.">${BEH[b].label}</span>`;
    const fmt = (v, digits = 3) => Number(v || 0).toLocaleString("vi-VN", {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits,
    });
    const signed = (v, digits = 3) => `${v >= 0 ? "+" : ""}${fmt(v, digits)}`;
    const sum = (arr) => arr.reduce((a, b) => a + Number(b || 0), 0);
    const dot = (a, b) => a.reduce((acc, v, i) => acc + v * b[i], 0);
    const norm = (arr) => Math.sqrt(sum(arr.map((v) => v * v)));
    const pct = (v) => `${(100 * Number(v || 0)).toFixed(1)}%`;

    function heat(v, maxAbs) {
        const t = Math.max(-1, Math.min(1, v / (maxAbs || 1)));
        return t >= 0 ? `rgba(255,180,84,${0.18 + 0.82 * t})` : `rgba(110,168,254,${0.18 + 0.82 * -t})`;
    }
    function vecStrip(vec) {
        const m = Math.max(...vec.map(Math.abs), 1e-6);
        return `<span class="vec" title="[${vec.map((x) => x.toFixed(2)).join(", ")}]">` +
            vec.map((v) => `<i style="background:${heat(v, m)}"></i>`).join("") + "</span>";
    }
    function vecCells(vec, color) {
        return `<div class="dot-calc">${vec.map((v, i) => `<div class="dot-cell"><small>d${i + 1}</small><div class="p" style="color:${color || "var(--txt)"}">${signed(v, 2)}</div></div>`).join("")}</div>`;
    }
    function vectorTable(title, vec, color, limit = vec.length) {
        return `<div class="num-panel">
            <div class="num-title">${title}<span>‖v‖=${fmt(norm(vec), 3)}</span></div>
            <div class="vec-table">${vec.slice(0, limit).map((v, i) =>
                `<div class="vec-num"><small>d${i + 1}</small><b style="color:${color || "var(--txt)"}">${signed(v, 3)}</b></div>`).join("")}</div>
        </div>`;
    }
    function twoVectorTable(title, aLabel, a, bLabel, b, productLabel = null) {
        const products = productLabel ? a.map((v, i) => v * b[i]) : null;
        const subtitle = products ? `${aLabel} · ${bLabel} = ${fmt(sum(products), 3)}` : `${aLabel} → ${bLabel}`;
        return `<div class="num-panel wide-panel">
            <div class="num-title">${title}<span>${subtitle}</span></div>
            <div class="scroll"><table class="compact-table"><thead><tr>${th("Chiều", "Chỉ số chiều trong vector embedding.")}${th(esc(aLabel), "Giá trị vector thứ nhất ở chiều này.", "num")}${th(esc(bLabel), "Giá trị vector thứ hai ở chiều này.", "num")}${products ? th(esc(productLabel), "Tích theo từng chiều, dùng để cộng thành dot product.", "num") : ""}</tr></thead><tbody>${
                a.map((v, i) => `<tr><td class="code">d${i + 1}</td><td class="num">${signed(v, 3)}</td><td class="num">${signed(b[i], 3)}</td>${products ? `<td class="num"><b>${signed(products[i], 3)}</b></td>` : ""}</tr>`).join("")
            }</tbody></table></div>
        </div>`;
    }
    function statLine(items) {
        return `<div class="num-grid">${items.map((it) => `<div class="num-chip" title="${esc(it.title || `${it.label}: ${it.value}`)}"><span>${it.label}</span><b>${it.value}</b></div>`).join("")}</div>`;
    }
    function bar(label, value, max, color) {
        const w = Math.max(2, (value / (max || 1)) * 100);
        return `<div class="bar-row" title="${esc(label)}: giá trị tương đối được biểu diễn bằng độ dài thanh."><span>${esc(label)}</span>
            <div class="bar-track"><div class="bar-fill" style="width:${w}%;background:${color}"></div></div>
            <strong>${typeof value === "number" ? value.toLocaleString("vi-VN") : value}</strong></div>`;
    }
    const statCard = (v, label, sub) =>
        `<div class="stat"><b>${v.toLocaleString("vi-VN")}</b><span>${label}</span><small>${sub}</small></div>`;

    function anonymizeDemoData() {
        const categoryMap = new Map((D.vocab.categories || []).map((name, idx) => [name, `#category${idx + 1}`]));
        const brandMap = new Map((D.vocab.brands || []).map((name, idx) => [name, `#brand${idx + 1}`]));
        D.vocab.categories = (D.vocab.categories || []).map((_name, idx) => `#category${idx + 1}`);
        D.vocab.brands = (D.vocab.brands || []).map((_name, idx) => `#brand${idx + 1}`);
        (D.vocab.items || []).forEach((item) => {
            item.label = `#product${Number(item.id) + 1}`;
            item.category = categoryMap.get(item.category) || "#category?";
            item.brand = brandMap.get(item.brand) || "#brand?";
        });
    }

    // ===================== controls + pipeline track + stage =====================
    let playTimer = null;

    function controlsHtml(tab, steps) {
        const idx = state[tab] ?? 0;
        const s = steps[idx];
        const on = playTimer && playTimer.tab === tab;
        return `<div class="controls">
            <button class="play-btn ${on ? "on" : ""}" data-play="${tab}" title="Tự chạy từng bước">${on ? "⏸" : "▶"}</button>
            <button class="ctrl-mini" data-nav="${tab}:-1" ${idx === 0 ? "disabled" : ""}>◀</button>
            <button class="ctrl-mini" data-nav="${tab}:1" ${idx === steps.length - 1 ? "disabled" : ""}>▶</button>
            <input class="scrub" type="range" min="0" max="${steps.length - 1}" value="${idx}" data-scrub="${tab}">
            <span class="ctrl-label">${on ? "Đang tự chạy · " : ""}Bước <b>${idx + 1}</b>/${steps.length} · ${esc(s.phase || "")}</span>
        </div>`;
    }

    // ----- big-box architecture diagram (tongquan.png style) -----
    // Each "box" is a major module; its `chips` are the clickable sub-layers
    // mapping to step indices (like Tokenizer/Text-Encoder inside the big
    // "Text Representation Generator" box of Diffusion Explainer).
    function archBlueprint(tab) {
        if (tab === "train") {
            return [
                { box: 1, ico: "🗂️", title: "Dữ liệu", sub: "REES46 đã xử lý", tone: "blue",
                  chips: [[0, "Log thô"], [1, "Thống kê"], [2, "Vocab"], [3, "Tách thời gian"]] },
                { arrow: "(user, item, hành vi, t)" },
                { box: 1, ico: "🕸️", title: "Đồ thị không đồng nhất", sub: "4 loại nút · 10 quan hệ", tone: "blue",
                  chips: [[4, "Dựng đồ thị"], [5, "Lấy mẫu 2-hop"]] },
                { arrow: "subgraph + Δt" },
                { box: 1, ico: "🧠", title: "BPATMP Encoder", sub: `× ${D.meta.n_layers} lớp truyền tin`, tone: "green", wide: 1,
                  chips: [[6, "1·Behavior-Aware W"], [7, "2·Temporal Attention"], [8, "3·Aggregation"], [9, "4·Intent Codebook"]] },
                { arrow: "h: vector mỗi nút" },
                { box: 1, ico: "📉", title: "Loss đa nhiệm", sub: "BPR · MBCL · Funnel · Mono", tone: "pink",
                  chips: [[10, "Hàm mất mát"]] },
                { arrow: "θ* sau hội tụ" },
                { box: 1, ico: "✨", title: "Embedding", sub: "user & sản phẩm", tone: "amber",
                  chips: [[11, "Embedding kết quả"]] },
            ];
        }
        return [
            { box: 1, ico: "👤", title: "Người dùng", sub: "lịch sử tương tác", tone: "blue",
              chips: [[0, "Lịch sử"]] },
            { arrow: "G_history(u)" },
            { box: 1, ico: "✨", title: "Embedding", sub: "user và item ứng viên", tone: "amber", wide: 1,
              chips: [[1, "Embedding user"], [2, "Embedding item"]] },
            { arrow: "h_u, h_i" },
            { box: 1, ico: "🧮", title: "Tính điểm & Xếp hạng", sub: "score = h_u · h_i", tone: "green", wide: 1,
              chips: [[3, "Dot product"]] },
        ];
    }

    function pipelineHtml(tab, steps) {
        const active = state[tab] ?? 0;
        const bp = archBlueprint(tab);
        let prevMax = -1;
        let html = "";
        bp.forEach((it, idx) => {
            if (it.arrow !== undefined) {
                const next = bp[idx + 1];
                const nextActive = next && next.chips && next.chips.some((c) => c[0] === active);
                const done = active > prevMax;
                html += `<div class="arch-arrow ${done ? "done" : ""} ${nextActive ? "flow" : ""}">
                    <span class="aa-lbl">${esc(it.arrow)}</span><span class="aa-line"></span></div>`;
                return;
            }
            const stepsIn = it.chips.map((c) => c[0]);
            const boxActive = stepsIn.includes(active);
            const passed = Math.max(...stepsIn) < active;
            prevMax = Math.max(prevMax, ...stepsIn);
            html += `<div class="arch-box tone-${it.tone} ${it.wide ? "wide" : ""} ${boxActive ? "active" : ""} ${passed ? "passed" : ""}" data-step="${tab}:${stepsIn[0]}" title="${esc(it.title)}">
                <div class="ab-head"><span class="ab-ico">${it.ico}</span><div class="ab-tt"><b>${esc(it.title)}</b><small>${esc(it.sub)}</small></div></div>
                <div class="ab-chips">${it.chips.map((c) => `<button class="ab-chip ${c[0] === active ? "on" : ""}" data-step="${tab}:${c[0]}">${esc(c[1])}</button>`).join("")}</div>
            </div>`;
        });
        const loop = tab === "train"
            ? `<div class="arch-loop"><span>↺ <b>Gradient</b> từ Loss lan ngược, cập nhật tham số của <b>mọi lớp</b> — lặp lại ${D.meta.train_iters} vòng</span></div>`
            : "";
        return `<div class="pipeline arch"><div class="arch-row">${html}</div>${loop}</div>`;
    }

    function stageHtml(tab, steps) {
        const idx = state[tab];
        if (idx === null) {
            return `<div class="stage-card"><div class="stage-empty">
                <div class="big">👆</div>
                <p><b>Chọn bước</b><br>Xem ý tưởng chính và minh họa trực quan của từng module</p>
            </div></div>`;
        }
        const s = steps[idx];
        const nav = `<div class="deck-nav">
            <button class="navbtn" data-nav="${tab}:-1" ${idx === 0 ? "disabled" : ""}>◀ Bước trước</button>
            <span class="deck-counter">Bước <b>${idx + 1}</b> / ${steps.length} · ${esc(s.label)}</span>
            <button class="navbtn primary" data-nav="${tab}:1" ${idx === steps.length - 1 ? "disabled" : ""}>Bước sau ▶</button>
        </div>`;
        return `<div class="stage-card stage-in" id="stage-${tab}">
            <div class="slide-head"><span class="slide-num">${s.ico}</span>
                <div><p class="eyebrow">${esc(s.phase || "")} · Bước ${idx + 1}/${steps.length}</p><h2>${esc(s.title)}</h2></div></div>
            ${s.desc ? `<p class="step-desc">${s.desc}</p>` : ""}
            <div class="slide-body">${s.body()}</div>
            ${nav}
        </div>`;
        // (controls bar above the pipeline already provides Play / scrub / prev-next)
    }

    // ===================== TRAIN steps =====================
    function trainSteps() {
        const m = D.meta, T = D.training, L = T.layers;
        const A = "A · Chuẩn bị dữ liệu", B = "B · Các lớp của mô hình", C = "C · Huấn luyện & Kết quả";
        return [
            {
                ico: "📜", phase: A, label: "Log thô", out: "(user, item, hành vi, t)",
                title: "Dữ liệu thô",
                desc: "Mỗi sự kiện (log) ghi lại việc một người dùng tương tác với một sản phẩm tại một thời điểm cụ thể.",
                from: "<span class='code' title='Dữ liệu sự kiện thô từ hệ thống REES46 (mock), gồm các hành vi view/cart/purchase.'>mock events</span>.",
                why: "<span class='code' title='Mỗi sự kiện được chuẩn hóa thành bộ 4: user_id (ai), item_id (sản phẩm nào), behavior (hành vi gì), timestamp (lúc nào).'>event → (user_id, item_id, behavior, timestamp)</span>.",
                outFull: "<b class='outp' title='Bộ 4 chuẩn hóa: u = người dùng, i = sản phẩm, β = loại hành vi (view/cart/purchase), t = thời điểm.'>(u, i, β, t)</b>.",
                body: () => `<div class="scroll"><table><thead><tr>${th("Người dùng", "ID người dùng phát sinh sự kiện.")}${th("Sản phẩm", "Sản phẩm được tương tác trong log.")}${th("Hành vi", "Loại tương tác: xem, thêm giỏ hoặc mua.")}${th("Thời điểm (UTC)", "Thời điểm sự kiện xảy ra, dùng cho temporal split và attention.")}</tr></thead><tbody>${
                    T.raw_sample.map((e) => `<tr><td class="code">${e.global_user}</td>
                    <td>${itemLabel(e.item)}</td><td>${behBadge(e.behavior)}</td><td class="code">${esc(e.ts_str)}</td></tr>`).join("")
                }</tbody></table></div>`,
            },
            {
                ico: "📊", phase: A, label: "Thống kê hành vi", out: "phễu view→cart→buy",
                title: "Làm sạch & thống kê hành vi",
                desc: "Đếm số lượng từng loại hành vi để hiểu phân bố dữ liệu và tỷ lệ chuyển đổi (phễu view → cart → purchase).",
                from: "<span class='code' title='Danh sách tất cả sự kiện đã chuẩn hóa từ bước trước.'>(u, i, β, t)</span>.",
                why: "<span class='code' title='Đếm số sự kiện theo từng loại hành vi β (view, cart, purchase).'>count(β)=|{e: behavior(e)=β}|</span>.",
                outFull: "<b class='outp' title='Số lượng sự kiện view, cart, purchase — cho thấy phễu chuyển đổi.'>count(view)</b>, <b class='outp'>count(cart)</b>, <b class='outp'>count(purchase)</b>.",
                body: () => {
                    const bc = T.clean_stats.behavior_counts, mx = Math.max(bc.view, bc.cart, bc.purchase);
                    return `<div class="grid cols-4">
                        ${statCard(m.counts.users, "người dùng", `thật: ${m.real_counts.users.toLocaleString("vi-VN")}`)}
                        ${statCard(m.counts.products, "sản phẩm", `thật: ${m.real_counts.products.toLocaleString("vi-VN")}`)}
                        ${statCard(m.counts.categories, "danh mục", `thật: ${m.real_counts.categories}`)}
                        ${statCard(m.counts.brands, "thương hiệu", `thật: ${m.real_counts.brands.toLocaleString("vi-VN")}`)}</div>
                        ${statLine([{ label: "events", value: m.counts.events }, { label: "d", value: m.embed_dim }, { label: "layers", value: m.n_layers }, { label: "train iters", value: m.train_iters }])}` +
                        bar("Xem", bc.view, mx, "var(--view)") + bar("Thêm giỏ", bc.cart, mx, "var(--cart)") + bar("Mua", bc.purchase, mx, "var(--purchase)") +
                        statLine([
                            { label: "view/cart", value: fmt(bc.view / (bc.cart || 1), 2) },
                            { label: "cart/purchase", value: fmt(bc.cart / (bc.purchase || 1), 2) },
                            { label: "view/purchase", value: fmt(bc.view / (bc.purchase || 1), 2) },
                        ]);
                },
            },
            {
                ico: "🔢", phase: A, label: "Ánh xạ vocab", out: "id cục bộ 0..N",
                title: "Ánh xạ từ điển (vocab)",
                desc: "Chuyển ID thật (có thể không liên tục) sang ID cục bộ liên tục 0..N để dùng làm chỉ mục bảng embedding.",
                from: "<span class='code' title='Tập hợp tất cả ID thật (real_id) duy nhất, theo từng loại nút: user, item, category, brand.'>unique(real_id)</span> theo từng loại nút.",
                why: "<span class='code' title='Sắp xếp các real_id, rồi gán chỉ mục 0, 1, 2... làm local_id. Đảm bảo mỗi nút có chỉ mục liên tục để tra bảng embedding.'>local_id = index(sorted(unique(real_id)))</span>.",
                outFull: "<b class='outp' title='Bảng ánh xạ 2 chiều: từ ID thật sang ID cục bộ và ngược lại.'>real_id → local_id</b>.",
                body: () => `<div class="scroll"><table><thead><tr>${th("idx thật", "ID gốc trong dữ liệu sau xử lý.")}${th("", "Ánh xạ từ ID thật sang ID cục bộ.")}${th("idx cục bộ", "ID liên tục dùng để tra embedding.")}${th("Danh mục", "Danh mục của sản phẩm.")}${th("Thương hiệu", "Thương hiệu của sản phẩm.")}</tr></thead><tbody>${
                    D.vocab.items.slice(0, 10).map((it) => `<tr><td class="code">${it.global_idx}</td><td>→</td><td class="code">${it.id}</td><td>${esc(it.category)}</td><td>${esc(it.brand)}</td></tr>`).join("")
                }</tbody></table></div>`,
            },
            {
                ico: "✂️", phase: A, label: "Tách thời gian", out: "train + nhãn",
                title: "Tách theo thời gian (temporal split)",
                desc: "Chia lịch sử mỗi user theo mốc thời gian: phần trước mốc dùng để huấn luyện, phần sau mốc là nhãn đúng (ground truth).",
                from: "<span class='code' title='Tất cả sự kiện của user u, đã sắp theo thời gian tăng dần.'>events_u sorted by t</span>.",
                why: "<span class='code' title='train_u: tập sự kiện có thời gian ≤ mốc cắt (dùng để huấn luyện).'>train_u={e:t_e≤cut_u}</span><br><span class='code' title='gt_u: tập sản phẩm mà user thực sự mua sau mốc cắt (dùng để đánh giá).'>gt_u={i:purchase(i,t>cut_u)}</span>.",
                outFull: "<b class='outp' title='train_u = dữ liệu huấn luyện, gt_u = nhãn đúng để đánh giá, mask_u = sản phẩm đã mua (loại khỏi ứng viên).'>train_u</b>, <b class='outp'>gt_u</b>, <b class='outp'>mask_u</b>.",
                body: () => {
                    const us = Object.keys(T.user_history).sort((a, b) => a - b);
                    const cards = us.map((u) => {
                        const hist = (T.user_history[u] || []).slice(-6);
                        const cut = T.split.cutoffs[u], gt = T.split.ground_truth[u] || [];
                        const tl = hist.map((e) => `<div class="tl-ev">${behBadge(e.behavior)}<small>#product${e.item}</small></div>`).join('<span class="tl-arrow">→</span>');
                        return `<div class="step" style="margin:0;background:var(--bg-soft)"><b>Người dùng #${u}</b>
                            <div class="timeline" style="margin:8px 0">${tl || '<span class="note">(trống)</span>'}
                            <span class="tl-cut">✂ ${esc(cut.ts_str)}</span>
                            ${gt.map((i) => `<div class="tl-ev" style="border-color:var(--purchase)">🎯<small>#product${i}</small></div>`).join("")}</div>
                            <div class="formula mini" style="margin-top:8px">train: t ≤ cut · gt: purchase(t &gt; cut) = [${gt.map((i) => "#product" + i).join(", ")}]</div></div>`;
                    }).join("");
                    return `<div class="grid cols-2">${cards}</div>`;
                },
            },
            {
                ico: "🕸️", phase: B, label: "Dựng đồ thị", out: "10 quan hệ",
                title: "Dựng đồ thị không đồng nhất",
                desc: "Xây dựng đồ thị có 4 loại nút (user, item, category, brand) và 10 loại quan hệ (3 hành vi × 2 chiều + 4 cấu trúc).",
                from: "<span class='code' title='Dữ liệu huấn luyện của mọi user, kết hợp với thông tin metadata (danh mục, thương hiệu) của từng sản phẩm.'>train_u</span> + metadata item/category/brand.",
                why: "<span class='code' title='Mỗi cạnh gồm: nút nguồn, loại quan hệ, nút đích, và thời gian. Ví dụ: user—view→item, item—belongs_to→category.'>edge=(src, relation, dst, t)</span><br><span class='code' title='10 loại quan hệ: view, cart, purchase (× 2 chiều thuận/nghịch) + belongs_to, contains, producedBy, brands.'>relation ∈ 10 loại</span>.",
                outFull: "<b class='outp' title='Tập cạnh theo từng loại quan hệ, tạo thành đồ thị không đồng nhất (heterogeneous graph).'>E_relation</b> cho 10 quan hệ.",
                body: () => `<div class="schema">${
                    T.graph.edges.map((e) => {
                        const isS = ["belongs_to", "contains", "producedBy", "brands"].includes(e.name);
                        const col = isS ? "var(--struct)" : (BEH[e.name.replace("rev_", "")] ? BEH[e.name.replace("rev_", "")].color : "var(--accent)");
                        return `<div class="rel-pill" title="Số cạnh và chiều truyền tin của quan hệ ${esc(e.name)} trong đồ thị dị thể."><b style="color:${col}">${e.count}</b><span>${esc(e.src_type)} <span style="color:${col}">—${esc(e.name)}→</span> ${esc(e.dst_type)}</span></div>`;
                    }).join("")
                }</div>`,
            },
            {
                ico: "🎯", phase: B, label: "Lấy mẫu láng giềng", out: "subgraph nhỏ",
                title: "Lấy mẫu láng giềng (neighbor sampling)",
                from: "<span class='code' title='G: Toàn bộ đồ thị không đồng nhất.\nseed user: Người dùng mục tiêu đang được tính toán.'>G, seed user</span>.",
                why: "<span class='code' title='Từ user ban đầu, lấy ngẫu nhiên tối đa B₁ sản phẩm có tương tác.'>N₁=sample(adj(seed), B₁)</span><br><span class='code' title='Từ các sản phẩm N₁, lấy tiếp tối đa B₂ láng giềng (các user khác, danh mục, thương hiệu...) để mở rộng.'>N₂=sample(adj(N₁), B₂)</span>.",
                outFull: "<b class='outp' title='Đồ thị con nhỏ gọn chỉ chứa các nút xung quanh user, giúp giảm khối lượng tính toán thay vì dùng toàn bộ đồ thị.'>subgraph = {seed, N₁, N₂}</b>.",
                body: () => {
                    const s = T.sampler;
                    const hop = Object.keys(s.hop1).map((b) => `<div class="bar-row" style="grid-template-columns:90px 1fr"><span>${behBadge(b)}</span><span class="code">${s.hop1[b].length ? s.hop1[b].map((p) => "#product" + p).join(", ") : "—"}</span></div>`).join("");
                    return `<p class="note" style="margin-bottom:12px"><b>Mô tả:</b> Thay vì tính toán trên toàn bộ đồ thị khổng lồ, ta chỉ trích xuất một vùng lân cận (subgraph) xung quanh người dùng (seed) qua 2 bước nhảy (2-hop). Di chuột vào các công thức/phép tính ở trên để xem giải thích chi tiết.</p>
                        <div class="formula" title='Budget B₁ là số lượng nút tối đa lấy ở hop 1, B₂ là số lượng nút tối đa lấy ở hop 2'>seed = user #${s.seed_user} · B₁=${s.hop1_budget} · B₂=${s.hop2_budget}</div>
                        <div style="margin: 12px 0 8px 0;"><b>Hop 1: Sản phẩm user trực tiếp tương tác</b> <small style="color:var(--txt-dim)">(được lấy mẫu theo từng loại hành vi)</small></div>
                        ${hop}
                        <div style="margin: 16px 0 8px 0;"><b>Hop 2: Lân cận mở rộng từ Hop 1</b> <small style="color:var(--txt-dim)">(sản phẩm liên quan, danh mục, thương hiệu...)</small></div>
                        <div class="schema"><div class="rel-pill" title="Các sản phẩm được mở rộng ở hop 2 từ vùng lân cận của user.">📦 ${s.hop2_products.length} sản phẩm</div>
                        <div class="rel-pill" style="border-color:var(--struct)" title="Các category liên quan được lấy qua cạnh cấu trúc product-category.">🏷️ ${s.hop2_categories.map((c) => D.vocab.categories[c]).join(", ")}</div>
                        <div class="rel-pill" style="border-color:var(--struct)" title="Các brand liên quan được lấy qua cạnh cấu trúc product-brand.">™️ ${s.hop2_brands.map((b) => D.vocab.brands[b]).join(", ")}</div></div>`;
                },
            },
            {
                ico: "🧩", phase: B, label: "1️⃣ Behavior-Aware Weight", out: "thông điệp m",
                title: "Lớp 1 · BehaviorAwareWeight (biến đổi thông điệp)",
                desc: "Biến đổi embedding nguồn thành thông điệp m, với ma trận trọng số khác nhau tùy loại quan hệ ρ và hành vi β.",
                from: "<span class='code' title='h_src: vector embedding của nút nguồn gửi tin. ρ: loại quan hệ (ví dụ view, cart). β: loại hành vi.'>h_src</span>, quan hệ <span class='code'>ρ</span>, hành vi <span class='code'>β</span>.",
                why: "<span class='code' title='Ma trận Wρβ được phân rã thành: Wρ (chung cho quan hệ) + hiệu chỉnh low-rank Aρ·diag(zβ)·Bρᵀ theo hành vi β. Giúp tiết kiệm tham số.'>Wρβ = Wρ + Aρ·diag(zβ)·Bρᵀ</span><br><span class='code' title='Nhân ma trận Wρβ với h_src để tạo thông điệp m, chứa thông tin ngữ cảnh của nút nguồn.'>m = Wρβ·h_src</span>.",
                outFull: "<b class='outp' title='Thông điệp m (d chiều) trên mỗi cạnh — đóng vai trò lời nhắn mà nút nguồn gửi cho nút đích.'>m ∈ ℝ<sup>d</sup></b> trên từng cạnh.",
                body: () => behaviorWeightBlock(L),
            },
            {
                ico: "⏱️", phase: B, label: "2️⃣ Temporal Attention", out: "α, gate",
                title: "Lớp 2 · Temporal Attention (chú ý theo thời gian)",
                desc: "Tính trọng số chú ý α và cổng gate cho mỗi cạnh, có tính đến khoảng cách thời gian Δt — sự kiện gần hơn được ưu tiên hơn.",
                from: "<span class='code' title='m_e: thông điệp trên cạnh e. Δt_e: khoảng cách thời gian từ sự kiện e đến hiện tại. Q·K: tích vô hướng giữa query và key.'>m_e</span>, <span class='code'>Δt_e</span>, <span class='code'>Q·K</span>.",
                why: "<span class='code' title='logit gồm 4 thành phần: (1) QK/√d: độ tương quan nội dung, (2) bρ: bias theo loại quan hệ, (3) u·Φ(Δt): bias thời gian học được, (4) -λβ·log(...): suy giảm thời gian theo hành vi.'>logit_e = QK/√d + bρ + u·Φ(Δt) − λβ·log(1+Δt/τ)</span><br><span class='code' title='α = softmax(logit): chuẩn hóa thành trọng số chú ý. gate = σ(...): cổng 0→1 kiểm soát mức độ đóng góp thực tế.'>α = softmax(logit)</span>, <span class='code'>gate = σ(...)</span>.",
                outFull: "<b class='outp' title='α_e: trọng số chú ý (tổng = 1). gate_e: cổng (0..1). Tích α_e·gate_e quyết định mức đóng góp thực sự của mỗi cạnh.'>α_e</b>, <b class='outp'>gate_e</b>, <b class='outp'>α_e·gate_e</b>.",
                body: () => attentionBlock(T.attention),
            },
            {
                ico: "🧮", phase: B, label: "3️⃣ Aggregation", out: "h' mỗi nút",
                title: "Lớp 3 · Behavior-Normalized Aggregation (gộp tin)",
                desc: "Gộp tất cả thông điệp đến theo từng loại hành vi, chuẩn hóa, rồi cộng với vector ban đầu h⁰ (residual connection).",
                from: "<span class='code' title='Tích (α_e·gate_e)·m_e: thông điệp đã được cân bằng theo trọng số chú ý và cổng. Bucket β: nhóm cạnh theo loại hành vi.'>(α_e·gate_e)·m_e</span>, bucket <span class='code'>β</span>.",
                why: "<span class='code' title='aggβ: gộp tất cả thông điệp trong cùng nhóm hành vi β bằng tổng có trọng số.'>aggβ = Σe∈β (αe·gatee)·me</span><br><span class='code' title='Chuẩn hóa (LayerNorm) từng bucket, nhân trọng số wβ, cộng tất cả lại với h⁰, rồi qua hàm kích hoạt ELU.'>h' = ELU(Σβ wβ·LayerNorm(aggβ)+h⁰)</span>.",
                outFull: "<b class='outp' title='Vector h-prime mới chứa thông tin tổng hợp từ tất cả láng giềng, được cập nhật qua truyền tin.'>h' = ELU(Σ wᵦ·LayerNorm(aggᵦ) + h⁰)</b> cho mỗi nút.",
                body: () => aggBlock(T.attention),
            },
            {
                ico: "💡", phase: B, label: "4️⃣ Intent Codebook", out: "h'' (residual)",
                title: "Lớp 4 · Intent Codebook (mã ý định dùng chung)",
                desc: "Bổ sung thông tin ý định mua sắm bằng cách kết hợp h' với E mã ý định dùng chung (codebook), qua attention mềm.",
                from: "<span class='code' title='h-prime: vector nút sau aggregation. C: bảng codebook gồm E vector ý định, mỗi vector đại diện cho một xu hướng mua sắm.'>h'</span>, codebook <span class='code'>C ∈ ℝ<sup>E×d</sup></span>.",
                why: "<span class='code' title='Tính trọng số attention giữa h-prime và từng mã ý định Ce trong codebook.'>a = softmax(h'·Cᵀ/√d)</span><br><span class='code' title='Cộng residual: h-prime + tổ hợp tuyến tính các mã ý định theo trọng số a. Giúp bổ sung thông tin ý định toàn cục.'>h'' = h' + Σe ae·Ce</span>.",
                outFull: "<b class='outp' title='Vector embedding cuối cùng sau khi đã kết hợp cả thông tin láng giềng lẫn ý định dùng chung.'>h'' = h' + Σ aₑ·Cₑ</b> (chú ý mềm trên E mã ý định).",
                body: () => intentBlock(L),
            },
            {
                ico: "📉", phase: C, label: "Hàm mất mát", out: "gradient, loss↓",
                title: "Hàm mất mát đa nhiệm & hội tụ",
                desc: "Kết hợp 4 hàm loss: BPR (xếp hạng), MBCL (contrastive), Funnel (thứ tự hành vi), Mono (đơn điệu) để tối ưu mô hình.",
                from: "<span class='code' title='h_u: embedding user. h_i+: embedding sản phẩm tương tác (positive). h_i−: embedding sản phẩm không tương tác (negative).'>h_u</span>, <span class='code'>h_i+</span>, <span class='code'>h_i−</span>, nhãn temporal split.",
                why: "<span class='code' title='Loss tổng = BPR (ranking) + λcl×MBCL (contrastive learning đa hành vi) + λconv×Funnel (thứ tự purchase ≥ cart ≥ view) + λmono×Mono (đơn điệu decay).'>L = L_BPR + λcl·L_MBCL + λconv·L_funnel + λmono·L_mono</span>.",
                outFull: "<b class='outp' title='L_total: giá trị loss tổng. ∇θ: gradient lan ngược để cập nhật tất cả tham số mô hình.'>L_total</b>, <b class='outp'>∇θ</b>, cập nhật tham số.",
                body: () => lossBlock(T),
            },
            {
                ico: "✨", phase: C, label: "Embedding kết quả", out: "vector mỗi nút",
                title: "Kết quả: embedding đã học",
                desc: "Sau huấn luyện, mỗi user và item đều có vector embedding. Hai sản phẩm giống nhau sẽ có vector gần nhau trong không gian d chiều.",
                from: `<span class='code' title='θ: tất cả tham số của mô hình (embedding tables, ma trận W, codebook...) sau khi huấn luyện xong.'>θ</span> sau ${D.meta.train_iters} iter.`,
                why: "<span class='code' title='Điểm tương thích giữa user u và item i = tích vô hướng của 2 vector embedding tương ứng.'>score(u,i)=h_u·h_i</span>.",
                outFull: "<b class='outp' title='h_user và h_item: vector embedding cuối cùng, dùng để tính điểm và xếp hạng sản phẩm gợi ý.'>h_user</b>, <b class='outp'>h_item</b> dùng cho ranking.",
                body: () => `<div class="scroll"><table><thead><tr>${th("Sản phẩm", "Sản phẩm đã có embedding sau huấn luyện.")}${th("Danh mục", "Danh mục giúp diễn giải ngữ cảnh sản phẩm.")}${th(`Embedding (d=${D.meta.embed_dim})`, "Dải màu biểu diễn vector embedding; rê vào vector để xem giá trị.")}</tr></thead><tbody>${
                    D.vocab.items.slice(0, 8).map((it) => `<tr><td>${it.label}</td><td>${esc(it.category)}</td><td>${vecStrip(it.vec)}</td></tr>`).join("")
                }</tbody></table></div>
                <div class="grid cols-2" style="margin-top:12px">
                    ${D.vocab.items.slice(0, 4).map((it) => vectorTable(`${it.label} · ${esc(it.category)}`, it.vec, "var(--cart)")).join("")}
                </div>
                <div class="legend"><span><i class="swatch" style="background:var(--view)"></i> âm</span><span><i class="swatch" style="background:var(--cart)"></i> dương</span></div>`,
            },
        ];
    }

    function embeddingBlock(L) {
        const e = L.input_embedding;
        return `<div class="formula">h⁰<sub>user</sub> = E<sub>user</sub>[${L.focus_user}] &nbsp;&nbsp; h⁰<sub>item</sub> = E<sub>item</sub>[${L.focus_product}] &nbsp;&nbsp; d=${L.embed_dim}</div>
        ${statLine([
            { label: "chiều vector", value: `d=${L.embed_dim}` },
            { label: "user đang xét", value: `#${L.focus_user}` },
            { label: "sản phẩm đang xét", value: `#${L.focus_product}` },
            { label: "cos(h_user,h_item)", value: fmt(dot(e.user_vec, e.product_vec) / ((norm(e.user_vec) * norm(e.product_vec)) || 1), 3) },
        ])}
        <div class="veclane"><span class="vlab">Người dùng #${L.focus_user} → h⁰</span>${vecStrip(e.user_vec)}</div>
        ${vecCells(e.user_vec, "var(--view)")}
        <div class="veclane" style="margin-top:14px"><span class="vlab">Sản phẩm #${L.focus_product} → h⁰</span>${vecStrip(e.product_vec)}</div>
        ${vecCells(e.product_vec, "var(--cart)")}
        <div class="grid cols-2" style="margin-top:12px">
            ${vectorTable(`h⁰ user #${L.focus_user}`, e.user_vec, "var(--view)")}
            ${vectorTable(`h⁰ sản phẩm #${L.focus_product}`, e.product_vec, "var(--cart)")}
        </div>`;
    }

    function behaviorWeightBlock(L) {
        const b = L.behavior_aware, mx = Math.max(...Object.values(b.behaviors).map((x) => x.w_norm));
        const zHeader = Array.from({ length: L.rank }, (_, i) => th(`z${i + 1}`, "Một chiều trong vector hành vi zβ, dùng để điều chỉnh ma trận theo từng hành vi.", "num")).join("");
        const zRows = ["view", "cart", "purchase"].map((k) =>
            `<tr><td>${behBadge(k)}</td>${b.behaviors[k].z_beta.map((v) => `<td class="num">${fmt(v, 3)}</td>`).join("")}<td class="num"><b>${fmt(b.behaviors[k].w_norm, 3)}</b></td></tr>`).join("");
        const rows = ["view", "cart", "purchase"].map((k) =>
            `<div class="veclane"><span class="vlab" title="Vector zβ đại diện cho phần điều chỉnh riêng của hành vi này trong Behavior-Aware Weight.">${behBadge(k)} z<sub>β</sub></span>${vecStrip(b.behaviors[k].z_beta)}
            <span class="note" style="margin:0">‖W<sub>ρ,β</sub>‖ = <b>${b.behaviors[k].w_norm}</b></span></div>`).join("");
        return `<div class="formula">W<sub>ρ,β</sub> = W<sub>ρ</sub> + A<sub>ρ</sub> · diag(<b class="hl-purchase">z<sub>β</sub></b>) · B<sub>ρ</sub><sup>T</sup><br>
            m = W<sub>ρ,β</sub> · h<sub>src</sub> &nbsp;&nbsp; r=${L.rank}</div>
        ${statLine([
            { label: "quan hệ ví dụ", value: esc(b.relation) },
            { label: "‖Wρ‖ gốc", value: fmt(b.w_base_norm, 3) },
            { label: "‖h nguồn‖", value: fmt(norm(b.example_src_vec), 3) },
            { label: "‖m mua‖", value: fmt(norm(b.example_msg_vec), 3) },
        ])}
        ${rows}
        <div class="scroll" style="margin-top:10px"><table class="compact-table"><thead><tr>${th("Hành vi", "Loại hành vi gốc của thông điệp: view, cart hoặc purchase.")}${zHeader}${th("‖Wρ,β‖", "Độ lớn của ma trận biến đổi sau khi kết hợp quan hệ ρ và hành vi β.", "num")}</tr></thead><tbody>${zRows}</tbody></table></div>
        ${bar("‖W‖ Xem", b.behaviors.view.w_norm, mx, "var(--view)")}${bar("‖W‖ Giỏ", b.behaviors.cart.w_norm, mx, "var(--cart)")}${bar("‖W‖ Mua", b.behaviors.purchase.w_norm, mx, "var(--purchase)")}
        <div class="veclane"><span class="vlab" title="Vector của sản phẩm nguồn trước khi biến đổi thành thông điệp.">h (sản phẩm #${L.focus_product})</span>${vecStrip(b.example_src_vec)}</div>
        <div class="veclane"><span class="vlab" title="Thông điệp sau khi đi qua ma trận phụ thuộc quan hệ và hành vi.">m = W<sub>ρ,mua</sub>·h</span>${vecStrip(b.example_msg_vec)}</div>`;
    }

    function attentionBlock(a) {
        const rows = a.edges.slice(0, 14).map((e) => `<tr><td>${behBadge(e.behavior)}</td><td class="code">#product${e.product}</td>
            <td class="num">${e.delta_days}</td><td class="num">${e.qk}</td><td class="num">${e.time_bias}</td>
            <td class="num" style="color:var(--cart)">−${e.decay}</td><td class="num">${e.logit}</td>
            <td class="num"><b style="color:var(--accent)">${e.alpha}</b></td><td class="num">${e.gate}</td><td class="num">${fmt(e.alpha * e.gate, 3)}</td></tr>`).join("");
        const lam = a.lambda_per_behavior;
        const top = a.edges.slice().sort((x, y) => (y.alpha * y.gate) - (x.alpha * x.gate)).slice(0, 5);
        const sample = top[0] || a.edges[0];
        const totalAlpha = sum(a.edges.map((e) => e.alpha));
        const topRows = top.map((e) => `<tr><td>${behBadge(e.behavior)}</td><td class="code">#product${e.product}</td><td class="num">${fmt(e.alpha, 3)}</td><td class="num">${fmt(e.gate, 3)}</td><td class="num"><b>${fmt(e.alpha * e.gate, 3)}</b></td></tr>`).join("");
        return `<div class="formula">logit = <span class="hl-view">Q·K/√d</span> + b<sub>ρ</sub> + <span class="hl-cart">u·Φ(Δt)</span> − <span style="color:var(--cart)">λ<sub>β</sub>·log(1+Δt/τ)</span> &nbsp;⟶&nbsp; α = softmax(logit) &nbsp;·&nbsp; gate = σ(c + r·Φ(Δt) − μ·log(1+Δt/τ))</div>
        ${statLine([
            { label: "số cạnh vào user", value: a.edges.length },
            { label: "Σα sau softmax", value: fmt(totalAlpha, 3) },
            { label: "τ thời gian", value: fmt(a.tau, 1) },
            { label: "top α·gate", value: fmt(sample.alpha * sample.gate, 3) },
        ])}
        <div class="num-panel">
            <div class="num-title">Tách số cho cạnh mạnh nhất<span>${BEH[sample.behavior].label} · #product${sample.product}</span></div>
            <div class="formula mini">logit = ${signed(sample.qk, 3)} + ${signed(sample.b_rho || 0, 3)} + ${signed(sample.time_bias, 3)} − ${fmt(sample.decay, 3)} = <b>${signed(sample.logit, 3)}</b><br>
            hệ số gửi tin = α × gate = ${fmt(sample.alpha, 3)} × ${fmt(sample.gate, 3)} = <b>${fmt(sample.alpha * sample.gate, 3)}</b></div>
        </div>
        <div class="scroll" style="margin-top:12px"><table><thead><tr>
            ${th("Hành vi", "Loại tương tác tạo ra cạnh: view, cart hoặc purchase.")}
            ${th("Nguồn", "Sản phẩm hoặc nút nguồn đang gửi thông điệp tới user.")}
            ${th("Δt (ngày)", "Khoảng cách từ thời điểm xảy ra hành vi đến thời điểm dự đoán.", "num")}
            ${th("Q·K", "Mức tương đồng nội dung giữa query của nút đích và key của nút nguồn.", "num")}
            ${th("time bias", "Độ lệch học được từ mã hóa thời gian Fourier.", "num")}
            ${th("decay", "Mức phạt do tương tác đã cũ; phụ thuộc vào loại hành vi.", "num")}
            ${th("logit", "Điểm chú ý trước softmax, kết hợp nội dung, quan hệ và thời gian.", "num")}
            ${th("α", "Trọng số attention sau softmax; các cạnh cạnh tranh với nhau.", "num")}
            ${th("gate", "Cổng giá trị từ 0 đến 1, kiểm soát lượng thông tin được truyền qua.", "num")}
            ${th("α·gate", "Mức đóng góp thực tế của cạnh sau khi nhân attention với gate.", "num")}
        </tr></thead><tbody>${rows}</tbody></table></div>
        <div class="scroll" style="margin-top:12px"><table class="compact-table"><thead><tr><th colspan="5" title="Các cạnh có α·gate cao nhất, tức ảnh hưởng mạnh nhất trong bước attention.">Top cạnh đóng góp lớn nhất</th></tr><tr>${th("Hành vi", "Loại hành vi của cạnh.")}${th("Nguồn", "Nút nguồn gửi thông điệp.")}${th("α", "Trọng số attention.", "num")}${th("gate", "Cổng giá trị.", "num")}${th("α·gate", "Đóng góp cuối cùng của cạnh.", "num")}</tr></thead><tbody>${topRows}</tbody></table></div>
        <div style="margin-top:14px">
            ${bar("Xem", lam.view, Math.max(lam.view, lam.cart, lam.purchase), "var(--view)")}${bar("Giỏ", lam.cart, Math.max(lam.view, lam.cart, lam.purchase), "var(--cart)")}${bar("Mua", lam.purchase, Math.max(lam.view, lam.cart, lam.purchase), "var(--purchase)")}
        </div>`;
    }

    function aggBlock(a) {
        const w = a.behavior_bucket_weights;
        const buckets = ["view", "cart", "purchase"].map((b) => {
            const es = a.edges.filter((e) => e.behavior === b);
            return { key: b, count: es.length, alpha: sum(es.map((e) => e.alpha)), gated: sum(es.map((e) => e.alpha * e.gate)), weight: w[b] };
        });
        buckets.push({ key: "struct", count: 0, alpha: 0, gated: 0, weight: w.struct });
        const bucketRows = buckets.map((b) => {
            const label = b.key === "struct" ? '<span class="badge b-struct">Cấu trúc</span>' : behBadge(b.key);
            return `<tr><td>${label}</td><td class="num">${b.count}</td><td class="num">${fmt(b.alpha, 3)}</td><td class="num">${fmt(b.gated, 3)}</td><td class="num">${fmt(b.weight, 3)}</td><td class="num"><b>${fmt(b.gated * b.weight, 3)}</b></td></tr>`;
        }).join("");
        return `<div class="formula">aggᵦ = Σ<sub>e∈β</sub> (α<sub>e</sub>·gate<sub>e</sub>)·m<sub>e</sub><br>
        h' = <b>ELU</b>(Σ<sub>β</sub> wᵦ·<b>LayerNorm</b>(aggᵦ) + h⁰)</div>
        ${bar("Xem", w.view, 1, "var(--view)")}${bar("Giỏ", w.cart, 1, "var(--cart)")}${bar("Mua", w.purchase, 1, "var(--purchase)")}${bar("Cấu trúc", w.struct, 1, "var(--struct)")}
        <div class="scroll" style="margin-top:12px"><table class="compact-table"><thead><tr>${th("Bucket", "Nhóm thông điệp theo hành vi hoặc cấu trúc.")}${th("số cạnh", "Số cạnh thuộc bucket này.", "num")}${th("Σα", "Tổng trọng số attention trong bucket.", "num")}${th("Σα·gate", "Tổng đóng góp sau khi nhân attention với gate.", "num")}${th("wᵦ", "Trọng số học được để trộn bucket hành vi β.", "num")}${th("wᵦ·Σα·gate", "Mức ảnh hưởng cuối cùng của bucket sau khi trộn.", "num")}</tr></thead><tbody>${bucketRows}</tbody></table></div>
        ${vectorTable("h' user sau aggregation, trước intent", D.training.layers.intent.pre_vec, "var(--purchase)")}`;
    }

    function intentBlock(L) {
        const t = L.intent, mx = Math.max(...t.attn, 1e-6);
        const bars = t.attn.map((a, i) => bar("Ý định " + (i + 1), a, mx, "var(--struct)")).join("");
        const intentRows = t.attn.map((a, i) => `<tr><td class="code">C${i + 1}</td><td class="num">${fmt(a, 3)}</td><td class="num">${pct(a)}</td></tr>`).join("");
        return `<div class="formula">a = softmax(h' · C<sup>T</sup> / √d)<br>
        h'' = h' + Σ<sub>e=1..${L.n_intents}</sub> aₑ · Cₑ</div>
        ${bars}
        <div class="scroll" style="margin-top:10px"><table class="compact-table"><thead><tr>${th("Mã ý định", "Một vector nguyên mẫu trong intent codebook.")}${th("aₑ", "Trọng số attention của embedding lên mã ý định này.", "num")}${th("tỷ trọng", "Tỷ lệ đóng góp của mã ý định trong tổ hợp cuối.", "num")}</tr></thead><tbody>${intentRows}</tbody></table></div>
        <div class="veclane" style="margin-top:8px"><span class="vlab" title="Embedding trước khi bổ sung ý định từ codebook.">h' (trước)</span>${vecStrip(t.pre_vec)}</div>
        <div class="veclane"><span class="vlab" title="Phần ý định được trích từ codebook và cộng bù vào embedding.">+ Σ aₑ·Cₑ (residual)</span>${vecStrip(t.residual_vec)}</div>
        <div class="veclane"><span class="vlab" title="Embedding cuối sau khi đã bổ sung ý định.">h'' (sau) → embedding cuối</span>${vecStrip(t.post_vec)}</div>
        ${vectorTable("h'' user cuối cùng", t.post_vec, "var(--purchase)")}`;
    }

    function lossBlock(T) {
        const c = T.curve, lw = T.loss_weights, maxL = Math.max(...c.map((p) => p.total));
        const W = 560, H = 150, pad = 24;
        const xs = (i) => pad + (i / (c.length - 1)) * (W - 2 * pad);
        const ys = (v) => H - pad - (v / (maxL || 1)) * (H - 2 * pad);
        const lossValue = (p, k) => {
            if (k === "mbcl") return lw.lambda_cl * p.cl;
            if (k === "funnel") return lw.lambda_conv * p.conv;
            if (k === "mono") return lw.lambda_mono * p.mono;
            return p[k];
        };
        const line = (k) => "M" + c.map((p, i) => `${xs(i).toFixed(1)},${ys(lossValue(p, k)).toFixed(1)}`).join(" L");
        const legend = [
            { label: "tổng loss", color: "var(--accent)", y: 16 },
            { label: "BPR", color: "var(--purchase)", y: 30 },
            { label: "MBCL", color: "var(--view)", y: 44 },
            { label: "Funnel", color: "var(--cart)", y: 58 },
            { label: "Mono", color: "var(--struct)", y: 72 },
        ].map((l) => `<text x="${W - pad}" y="${l.y}" fill="${l.color}" font-size="11" text-anchor="end">${l.label}</text>`).join("");
        const pickIters = [0, 20, 100, 300, 500, 699];
        const picked = pickIters.map((it) => c.find((p) => p.iter === it)).filter(Boolean);
        const lossRows = picked.map((p) => {
            const weighted = p.bpr + lw.lambda_cl * p.cl + lw.lambda_conv * p.conv + lw.lambda_mono * p.mono;
            return `<tr><td class="num">${p.iter}</td><td class="num"><b>${fmt(p.total, 4)}</b></td><td class="num">${fmt(p.bpr, 4)}</td><td class="num">${fmt(lw.lambda_cl * p.cl, 4)}</td><td class="num">${fmt(lw.lambda_conv * p.conv, 4)}</td><td class="num">${fmt(lw.lambda_mono * p.mono, 4)}</td><td class="num">${fmt(weighted, 4)}</td></tr>`;
        }).join("");
        return `<div class="formula">L = L<sub>BPR</sub> + ${lw.lambda_cl}·L<sub>MBCL</sub> + ${lw.lambda_conv}·L<sub>Funnel</sub> + ${lw.lambda_mono}·L<sub>Mono</sub><br>
        L<sub>BPR</sub> = −log σ(s⁺ − s⁻), &nbsp; s(u,i)=h<sub>u</sub>·h<sub>i</sub></div>
        <svg viewBox="0 0 ${W} ${H}" style="width:100%;height:auto;background:var(--bg-ink);border:1px solid var(--line);border-radius:10px;margin-top:10px">
            <path d="${line("total")}" fill="none" stroke="var(--accent)" stroke-width="2"/>
            <path d="${line("bpr")}" fill="none" stroke="var(--purchase)" stroke-width="1.6" stroke-dasharray="4 3"/>
            <path d="${line("mbcl")}" fill="none" stroke="var(--view)" stroke-width="1.5" stroke-dasharray="2 4"/>
            <path d="${line("funnel")}" fill="none" stroke="var(--cart)" stroke-width="1.5" stroke-dasharray="6 3"/>
            <path d="${line("mono")}" fill="none" stroke="var(--struct)" stroke-width="1.5" stroke-dasharray="1 3"/>
            ${legend}</svg>
        <div class="scroll" style="margin-top:10px"><table class="compact-table"><thead><tr>${th("iter", "Vòng lặp huấn luyện.", "num")}${th("total", "Tổng loss tại iter này.", "num")}${th("BPR", "Loss xếp hạng positive cao hơn negative.", "num")}${th("0.15·MBCL", "Thành phần contrastive learning đa hành vi sau khi nhân trọng số.", "num")}${th("0.10·Funnel", "Ràng buộc thứ tự view-cart-purchase sau khi nhân trọng số.", "num")}${th("0.05·Mono", "Ràng buộc suy giảm thời gian theo hành vi sau khi nhân trọng số.", "num")}${th("cộng thành phần", "Tổng các thành phần loss chính trước regularization.", "num")}</tr></thead><tbody>${lossRows}</tbody></table></div>
        <div class="grid cols-2" style="margin-top:10px">
            <div class="formula">w<sub>BPR</sub>: view=${lw.bpr_task_weights.view}, cart=${lw.bpr_task_weights.cart}, purchase=${lw.bpr_task_weights.purchase}</div>
            <div class="formula">L<sub>Funnel</sub>: s<sub>purchase</sub> ≥ s<sub>cart</sub> ≥ s<sub>view</sub><br>L<sub>Mono</sub>: λ<sub>view</sub> ≥ λ<sub>cart</sub> ≥ λ<sub>purchase</sub></div></div>`;
    }

    function renderTrain() {
        const steps = trainSteps();
        $("#tab-train").innerHTML =
            `<p class="pipe-intro"><b>Luồng ý tưởng chính.</b> Chọn module/layer để xem vai trò và minh họa trực quan.</p>` +
            controlsHtml("train", steps) + pipelineHtml("train", steps) + stageHtml("train", steps);
    }

    // ===================== EVAL steps =====================
    function evalSteps() {
        const ev = D.evaluation;
        const p = ev.per_user.find((x) => x.user === activeUser);
        const masked = new Set(p.masked_items);
        const hist = D.training.user_history[p.user] || [];
        if (activeItem === null || !p.ranking.some((r) => r.item === activeItem))
            activeItem = p.ranking[0].item;
        const histCounts = hist.reduce((acc, e) => {
            acc[e.behavior] = (acc[e.behavior] || 0) + 1;
            return acc;
        }, { view: 0, cart: 0, purchase: 0 });
        const candidateItems = D.vocab.items.filter((it) => !masked.has(it.id));

        return [
            {
                ico: "👤", phase: "Suy luận", label: "Lịch sử tương tác", out: "history graph",
                title: `Đầu vào: lịch sử người dùng #${p.user}`,
                desc: "Thu thập toàn bộ lịch sử tương tác của user trong tập train, gồm các sự kiện view/cart/purchase theo thời gian.",
                from: "<span class='code' title='Lịch sử tương tác của user u trong tập huấn luyện.'>history_train[u]</span>.",
                why: "<span class='code' title='Đồ thị lịch sử: tập hợp các bộ (user, hành vi, item, khoảng cách thời gian).'>G_history(u) = {(u, β, i, Δt)}</span><br><span class='code' title='Danh sách sản phẩm đã mua trong train — sẽ bị loại khỏi danh sách gợi ý.'>mask_u = purchased_train[u]</span>.",
                outFull: "<span class='code' title='G_history(u): đồ thị lịch sử dùng để tính embedding. mask_u: danh sách sản phẩm cần loại trừ.'>G_history(u)</span>, <span class='code'>mask_u</span>.",
                body: () => `<div class="timeline">${
                    hist.slice(-14).map((e) => `<div class="tl-ev">${behBadge(e.behavior)}<small>${ITEM[e.item] ? esc(ITEM[e.item].category) : ""}</small><small>#product${e.item}</small></div>`).join('<span class="tl-arrow">→</span>') || '<span class="note">(trống)</span>'
                }</div>${statLine([
                    { label: "history_len", value: hist.length },
                    { label: "view", value: histCounts.view },
                    { label: "cart", value: histCounts.cart },
                    { label: "purchase", value: histCounts.purchase },
                    { label: "masked_items", value: masked.size },
                ])}
                <div class="scroll"><table class="compact-table"><thead><tr>${th("t", "Thời điểm tương tác trong lịch sử train.")}${th("β", "Loại hành vi: view, cart hoặc purchase.")}${th("item", "Sản phẩm user từng tương tác.")}${th("category", "Danh mục của sản phẩm.")}${th("brand", "Thương hiệu của sản phẩm.")}</tr></thead><tbody>${
                    hist.slice(-10).map((e) => `<tr><td class="code">${esc(e.ts_str)}</td><td>${behBadge(e.behavior)}</td><td class="code">#product${e.item}</td><td>${ITEM[e.item] ? esc(ITEM[e.item].category) : ""}</td><td>${ITEM[e.item] ? esc(ITEM[e.item].brand) : ""}</td></tr>`).join("")
                }</tbody></table></div>`,
            },
            {
                ico: "✨", phase: "Suy luận", label: "Embedding user", out: "h_u",
                title: `Embedding người dùng #${p.user}`,
                desc: "Lấy vector embedding đã học của user sau khi mô hình BPATMP xử lý toàn bộ lịch sử tương tác.",
                from: "<span class='code' title='Đồ thị lịch sử tương tác, được đưa qua mô hình BPATMP đã huấn luyện xong.'>G_history(u)</span> sau khi train BPATMP.",
                why: "<span class='code' title='Cho đồ thị lịch sử qua mô hình BPATMP, lấy ra vector tại vị trí nút user u.'>h_u = BPATMP(G_history(u))[u]</span><br><span class='code' title='Vector h_u có d chiều, đại diện cho sở thích tổng hợp của user.'>h_u ∈ ℝ^d</span>.",
                outFull: "<b class='outp' title='Vector embedding cuối cùng của user, sẽ được dùng để tính dot product với embedding từng sản phẩm.'>h_u</b> dùng để so với từng sản phẩm.",
                body: () => `<div class="formula">h<sub>u</sub> = H<sub>user</sub>[${p.user}] · d=${p.user_vec.length}</div>
                    ${statLine([
                        { label: "‖h_u‖", value: fmt(norm(p.user_vec), 3) },
                        { label: "history_len", value: hist.length },
                        { label: "view/cart/purchase", value: `${histCounts.view}/${histCounts.cart}/${histCounts.purchase}` },
                    ])}
                    ${vectorTable(`h_user #${p.user}`, p.user_vec, "var(--view)")}`,
            },
            {
                ico: "📦", phase: "Suy luận", label: "Embedding item", out: "h_i ứng viên",
                title: "Embedding sản phẩm ứng viên",
                desc: "Loại bỏ sản phẩm đã mua (mask), lấy embedding của tất cả sản phẩm ứng viên còn lại để so sánh với user.",
                from: "<span class='code' title='all_items: toàn bộ sản phẩm. mask_u: sản phẩm user đã mua — cần loại ra để không gợi ý lại.'>all_items</span> và <span class='code'>mask_u</span>.",
                why: "<span class='code' title='C_u = tập ứng viên = tất cả sản phẩm trừ những cái đã mua.'>C_u = all_items \\ mask_u</span><br><span class='code' title='h_i: tra bảng embedding sản phẩm để lấy vector tương ứng.'>h_i = H_item[i]</span>.",
                outFull: "<b class='outp' title='Vector embedding h_i cho từng sản phẩm ứng viên, sẽ được nhân dot product với h_u.'>h_i</b> cho từng sản phẩm ứng viên.",
                body: () => {
                    const it = ITEM[activeItem];
                    const sampleItems = [it, ...candidateItems.filter((cand) => cand.id !== activeItem).slice(0, 9)];
                    const rows = sampleItems.map((cand) =>
                        `<tr class="${cand.id === activeItem ? "active-row" : ""}"><td class="code">#product${cand.id}</td><td>${esc(cand.category)}</td><td>${esc(cand.brand)}</td><td>${vecStrip(cand.vec)}</td></tr>`).join("");
                    return `${statLine([
                        { label: "all_items", value: D.meta.counts.products },
                        { label: "masked_items", value: masked.size },
                        { label: "candidate_items", value: candidateItems.length },
                        { label: "item đang xem", value: `#product${activeItem}` },
                    ])}
                    <div class="scroll"><table class="compact-table"><thead><tr>${th("item", "Sản phẩm ứng viên sau khi loại các item đã mua.")}${th("category", "Danh mục của sản phẩm ứng viên.")}${th("brand", "Thương hiệu của sản phẩm ứng viên.")}${th("embedding", "Vector embedding của sản phẩm; rê vào dải màu để xem giá trị.")}</tr></thead><tbody>${rows}</tbody></table></div>
                    ${vectorTable(`h_item #product${activeItem} · ${esc(it.category)}/${esc(it.brand)}`, it.vec, "var(--cart)")}`;
                },
            },
            {
                ico: "🧮", phase: "Suy luận", label: "Tính điểm & xếp hạng", out: "top gợi ý",
                title: "Tính điểm = emb(user) · emb(sản phẩm) → xếp hạng",
                desc: "Tính dot product giữa vector user và vector từng sản phẩm ứng viên, sắp xếp theo điểm giảm dần để gợi ý.",
                from: "<span class='code' title='h_u: embedding user. h_i: embedding từng sản phẩm ứng viên (sau khi loại mask).'>h_u</span>, <span class='code'>h_i</span> với mọi item ứng viên.",
                why: "<span class='code' title='score_i = tích vô hướng (dot product) giữa user và item = tổng tích từng chiều. Điểm càng cao ⇒ user càng có khả năng thích item đó.'>score_i = Σ_k h_u[k] · h_i[k]</span><br><span class='code' title='Sắp xếp tất cả sản phẩm theo score giảm dần, lấy K sản phẩm đầu tiên.'>topK = argsort(score_i)</span>.",
                outFull: "<b class='outp' title='score_i: điểm tương thích. rank_i: thứ hạng. Top gợi ý: danh sách K sản phẩm có điểm cao nhất.'>score_i</b>, <b class='outp'>rank_i</b>, top gợi ý.",
                body: () => {
                    const it = ITEM[activeItem], uv = p.user_vec, iv = it.vec;
                    let dot = 0;
                    const cells = uv.map((u, k) => { const pr = u * iv[k]; dot += pr; return `<div class="dot-cell"><small>d${k + 1}</small><div class="u">${u.toFixed(2)}</div><div class="i">${iv[k].toFixed(2)}</div><div class="p">${pr >= 0 ? "+" : ""}${pr.toFixed(2)}</div></div>`; }).join("");
                    const ro = p.ranking.find((r) => r.item === activeItem);
                    const recs = p.ranking.slice(0, 10).map((r) => {
                        return `<div class="rec-row ${r.item === activeItem ? "active" : ""}" data-item="${r.item}"><span class="rec-rank">#${r.rank}</span><span>${itemLabel(r.item)}</span><span class="rec-score">${r.score.toFixed(3)}</span></div>`;
                    }).join("");
                    return `<div class="score-2col">
                        <div><div class="formula">topK = argsort(score)<br>score = h<sub>u</sub> · h<sub>i</sub></div>${recs}</div>
                        <div><div class="formula">score(user #${p.user}, #product${activeItem}) = Σ<sub>k=1..${uv.length}</sub> user<sub>k</sub> × item<sub>k</sub></div>
                        <div class="dot-calc">${cells}</div>
                        <p style="margin-top:10px">Điểm = <b style="color:var(--purchase);font-size:18px">${dot.toFixed(3)}</b> ⇒ hạng <b>#${ro ? ro.rank : "?"}</b>/${p.ranking.length}.</p>
                        ${twoVectorTable(`Bảng số đầy đủ: user #${p.user} · #product${activeItem}`, "user_i", uv, "item_i", iv, "user_i×item_i")}
                        </div></div>`;
                },
            },
        ];
    }

    function renderEval() {
        const ev = D.evaluation;
        if (activeUser === null) activeUser = ev.per_user[0].user;
        const steps = evalSteps();
        const pick = ev.per_user.map((p) => {
            const h = (D.training.user_history[p.user] || []).length;
            return `<button class="user-btn ${p.user === activeUser ? "active" : ""}" data-user="${p.user}"><b>Người dùng #${p.user}</b><small>${h} sự kiện lịch sử</small></button>`;
        }).join("");
        $("#tab-eval").innerHTML =
            `<p class="pipe-intro"><b>Luồng suy luận.</b> Chọn user và bước để xem minh họa trực quan.</p>
            <div class="userbar"><h3>Chọn người dùng để phân tích</h3><div class="user-pick">${pick}</div></div>` +
            controlsHtml("eval", steps) + pipelineHtml("eval", steps) + stageHtml("eval", steps);
    }

    // ===================== wiring =====================
    const stepsOf = (tab) => (tab === "train" ? trainSteps() : evalSteps());
    const rerender = (tab) => (tab === "train" ? renderTrain() : renderEval());

    function setStep(tab, idx) {
        const n = stepsOf(tab).length;
        state[tab] = Math.max(0, Math.min(n - 1, idx));
    }
    function stopPlay() {
        if (playTimer) { clearInterval(playTimer.id); const t = playTimer.tab; playTimer = null; return t; }
        return null;
    }
    function startPlay(tab) {
        stopPlay();
        const n = stepsOf(tab).length;
        playTimer = { tab, id: setInterval(() => {
            const next = (state[tab] ?? 0) + 1;
            if (next >= n) { stopPlay(); setStep(tab, n - 1); rerender(tab); return; }
            setStep(tab, next); rerender(tab);
        }, 3200) };
        rerender(tab);
    }

    document.addEventListener("click", (e) => {
        const tabBtn = e.target.closest(".tab");
        if (tabBtn) {
            stopPlay();
            document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === tabBtn));
            document.querySelectorAll(".tab-panel").forEach((p) => p.classList.toggle("active", p.id === "tab-" + tabBtn.dataset.tab));
            return;
        }
        const play = e.target.closest("[data-play]");
        if (play) {
            const tab = play.dataset.play;
            if (playTimer && playTimer.tab === tab) { stopPlay(); rerender(tab); }
            else startPlay(tab);
            return;
        }
        const node = e.target.closest("[data-step]");
        if (node) {
            stopPlay();
            const [tab, i] = node.dataset.step.split(":");
            state[tab] = Number(i);
            rerender(tab);
            const card = document.getElementById("stage-" + tab);
            if (card) card.scrollIntoView({ behavior: "smooth", block: "start" });
            const chip = document.querySelector(`.ab-chip[data-step="${tab}:${i}"]`);
            if (chip) { const p = chip.closest(".pipeline"); if (p) p.scrollLeft = Math.max(0, chip.offsetLeft - p.clientWidth / 2 + chip.clientWidth / 2); }
            return;
        }
        const nav = e.target.closest("[data-nav]");
        if (nav) {
            stopPlay();
            const [tab, d] = nav.dataset.nav.split(":");
            setStep(tab, (state[tab] ?? 0) + Number(d));
            rerender(tab);
            const card = document.getElementById("stage-" + tab);
            if (card) card.scrollIntoView({ behavior: "smooth", block: "start" });
            const chip = document.querySelector(`.ab-chip[data-step="${tab}:${state[tab]}"]`);
            if (chip) { const p = chip.closest(".pipeline"); if (p) p.scrollLeft = Math.max(0, chip.offsetLeft - p.clientWidth / 2 + chip.clientWidth / 2); }
            return;
        }
        const ub = e.target.closest(".user-btn");
        if (ub) { activeUser = Number(ub.dataset.user); activeItem = null; rerender("eval"); return; }
        const rr = e.target.closest(".rec-row");
        if (rr) { activeItem = Number(rr.dataset.item); rerender("eval"); return; }
    });

    document.addEventListener("input", (e) => {
        const sc = e.target.closest("[data-scrub]");
        if (!sc) return;
        const tab = sc.dataset.scrub;
        stopPlay();
        setStep(tab, Number(sc.value));
        rerender(tab);
    });

    document.addEventListener("keydown", (e) => {
        if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
        const tab = $("#tab-train").classList.contains("active") ? "train" : "eval";
        stopPlay();
        setStep(tab, (state[tab] ?? 0) + (e.key === "ArrowRight" ? 1 : -1));
        rerender(tab);
    });

    fetch("demo_data.json").then((r) => r.json()).then((data) => {
        D = data;
        anonymizeDemoData();
        D.vocab.items.forEach((it) => (ITEM[it.id] = it));
        D.vocab.users.forEach((u) => (USER[u.id] = u));
        state.train = 0; state.eval = 0;
        $("#loading").style.display = "none";
        renderTrain();
        renderEval();
    }).catch((err) => { $("#loading").textContent = "Lỗi nạp demo_data.json: " + err.message; });
}());
