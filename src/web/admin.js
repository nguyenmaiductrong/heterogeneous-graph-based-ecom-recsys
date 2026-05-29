(function () {
    "use strict";

    const Store = window.DemoStore;
    const Backend = window.DemoBackend;
    const $ = (id) => document.getElementById(id);
    const IMAGE_TYPES = ["headset", "keyboard", "monitor", "mouse", "speaker", "chair", "watch", "tablet", "camera"];
    const IMAGE_LABELS = {
        headset: "Tai nghe",
        keyboard: "Bàn phím",
        monitor: "Màn hình",
        mouse: "Chuột",
        speaker: "Loa",
        chair: "Ghế",
        watch: "Đồng hồ",
        tablet: "Máy tính bảng",
        camera: "Camera",
    };
    let results = [];
    let previousRanks = {};
    let selectedResultId = null;
    let selectedProductHistoryId = null;
    let selectedUserId = null;
    let selectedSourceSessionId = null;
    let simulatorVisible = false;
    const CONFIG_LIMITS = {
        weight: { min: 0, max: 6 },
        recencyDecay: { min: 0, max: 0.2 },
        categoryWeight: { min: 0, max: 4 },
        brandWeight: { min: 0, max: 4 },
        popularityWeight: { min: 0, max: 2 },
        topK: { min: 1, max: 50 },
        refShiftDays: { min: 0, max: 365 },
    };

    function clampNumber(value, limits, fallback) {
        const number = Number(value);
        if (!Number.isFinite(number)) {
            return fallback;
        }
        return Math.min(limits.max, Math.max(limits.min, number));
    }

    function behaviorLabel(behavior) {
        return Store.BEHAVIORS[behavior]?.label || behavior;
    }

    function allEvents(state) {
        return sessionsForAdmin(state).flatMap((session) => session.events);
    }

    function sessionsForAdmin(state) {
        if (!state.draftSession) {
            return state.eventSessions;
        }
        return [{ ...state.draftSession, live: true }, ...state.eventSessions];
    }

    function visibleSessions(state) {
        const query = $("session-search").value.trim().toLowerCase();
        const userId = $("session-user-filter").value;
        const createdDate = $("session-date-filter").value;
        const behavior = $("session-behavior-filter").value;
        return sessionsForAdmin(state).filter((session) => {
            const labelMatch = session.label.toLowerCase().includes(query);
            const userMatch = !userId || session.userId === userId;
            const dateMatch = !createdDate || session.createdAt.slice(0, 10) === createdDate;
            const behaviorMatch = !behavior || session.events.some((event) => event.behavior === behavior);
            return labelMatch && userMatch && dateMatch && behaviorMatch;
        });
    }

    function currentConfig() {
        return {
            weights: {
                view: clampNumber($("weight-view").value, CONFIG_LIMITS.weight, 1),
                cart: clampNumber($("weight-cart").value, CONFIG_LIMITS.weight, 3),
                purchase: clampNumber($("weight-purchase").value, CONFIG_LIMITS.weight, 5),
            },
            recencyDecay: clampNumber($("recency-decay").value, CONFIG_LIMITS.recencyDecay, 0.055),
            categoryWeight: clampNumber($("category-weight").value, CONFIG_LIMITS.categoryWeight, 1.9),
            brandWeight: clampNumber($("brand-weight").value, CONFIG_LIMITS.brandWeight, 1.35),
            popularityWeight: clampNumber($("popularity-weight").value, CONFIG_LIMITS.popularityWeight, 0.85),
            topK: Math.round(clampNumber($("top-k").value, CONFIG_LIMITS.topK, 30)),
            maskPurchased: $("mask-purchased").checked,
            refDate: $("ref-date").value,
            refShiftDays: Math.round(clampNumber($("ref-shift-days").value, CONFIG_LIMITS.refShiftDays, 0)),
        };
    }

    function localDateValue(value) {
        const date = new Date(value);
        const localTime = date.getTime() - date.getTimezoneOffset() * 60000;
        return new Date(localTime).toISOString().slice(0, 10);
    }

    function renderReferenceShiftLabel() {
        const days = Number($("ref-shift-days").value || 0);
        $("ref-shift-label").textContent = days ? `${days} ngày sau mốc` : "Đúng mốc ngày";
    }

    function referenceTime(events, config) {
        const newestTime = events.length
            ? Math.max(...events.map((event) => new Date(event.timestamp).getTime()))
            : Date.now();
        const dateTime = config.refDate ? new Date(`${config.refDate}T23:59:59`).getTime() : NaN;
        const baseTime = Number.isFinite(dateTime) ? dateTime : Math.max(Date.now(), newestTime);
        return baseTime + config.refShiftDays * 86400000;
    }

    function setConfig(preset) {
        $("weight-view").value = clampNumber(preset.weights.view, CONFIG_LIMITS.weight, 1);
        $("weight-cart").value = clampNumber(preset.weights.cart, CONFIG_LIMITS.weight, 3);
        $("weight-purchase").value = clampNumber(preset.weights.purchase, CONFIG_LIMITS.weight, 5);
        $("recency-decay").value = clampNumber(preset.recencyDecay, CONFIG_LIMITS.recencyDecay, 0.055);
        $("category-weight").value = clampNumber(preset.categoryWeight, CONFIG_LIMITS.categoryWeight, 1.9);
        $("brand-weight").value = clampNumber(preset.brandWeight, CONFIG_LIMITS.brandWeight, 1.35);
        $("popularity-weight").value = clampNumber(preset.popularityWeight, CONFIG_LIMITS.popularityWeight, 0.85);
        $("top-k").value = Math.round(clampNumber(preset.topK, CONFIG_LIMITS.topK, 30));
        $("mask-purchased").checked = Boolean(preset.maskPurchased);
        if (preset.refDate) {
            $("ref-date").value = preset.refDate;
        }
        $("ref-shift-days").value = Math.round(clampNumber(preset.refShiftDays ?? 0, CONFIG_LIMITS.refShiftDays, 0));
        renderReferenceShiftLabel();
    }

    function sourceEvents(state) {
        const chosenSession = sessionsForAdmin(state).find((session) => session.id === selectedSourceSessionId);
        if (chosenSession) {
            return { label: chosenSession.label, events: chosenSession.events, session: chosenSession };
        }
        const userId = selectedUserId;
        const events = sessionsForAdmin(state)
            .filter((session) => session.userId === userId)
            .flatMap((session) => session.events)
            .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))
            .slice(0, 24);
        return { label: "Lịch sử người dùng", events, session: null };
    }

    function simulate(state, config) {
        const products = Store.productMap(state);
        const rawSource = sourceEvents(state);
        const refTime = referenceTime(rawSource.events, config);
        const source = {
            ...rawSource,
            allEvents: rawSource.events,
            events: rawSource.events.filter((event) => new Date(event.timestamp).getTime() <= refTime),
            refTime,
        };
        const hidden = state.products.filter((product) => !product.active).length;
        const purchased = new Set(source.events
            .filter((event) => event.behavior === "purchase")
            .map((event) => event.productId));
        const warnings = [];
        const futureEvents = rawSource.events.length - source.events.length;
        if (!source.events.length) {
            warnings.push("Nguồn chọn chưa có sự kiện. Kết quả lúc này chủ yếu dựa vào độ phổ biến.");
        }
        if (futureEvents) {
            warnings.push(`${futureEvents} sự kiện sau mốc ngày chạy chưa được đưa vào mô phỏng.`);
        }
        if (hidden) {
            warnings.push(`${hidden} sản phẩm đang ẩn sẽ không được xếp hạng.`);
        }
        if (config.maskPurchased && purchased.size) {
            warnings.push(`${purchased.size} sản phẩm đã mua bị loại khỏi danh sách gợi ý.`);
        }

        const ranked = state.products
            .filter((product) => product.active && (!config.maskPurchased || !purchased.has(product.id)))
            .map((product) => {
                const contribution = {
                    behavior: 0,
                    recency: 0,
                    category: 0,
                    brand: 0,
                    popularity: (Number(product.popularityScore) / 20) * config.popularityWeight,
                };
                const traces = [];
                source.events.forEach((event) => {
                    const origin = products[event.productId];
                    if (!origin) {
                        return;
                    }
                    const ageHours = Math.max(0, (refTime - new Date(event.timestamp).getTime()) / 3600000);
                    const decay = Math.exp(-config.recencyDecay * ageHours);
                    const base = (config.weights[event.behavior] || 0) * decay;
                    const sameProduct = event.productId === product.id;
                    const sameCategory = origin.category === product.category;
                    const sameBrand = origin.brand === product.brand;
                    if (!sameProduct && !sameCategory && !sameBrand) {
                        return;
                    }
                    if (sameProduct) {
                        contribution.behavior += base * 0.9;
                    }
                    contribution.recency += base * (sameProduct ? 0.18 : 0.08);
                    if (sameCategory) {
                        contribution.category += decay * config.categoryWeight;
                    }
                    if (sameBrand) {
                        contribution.brand += decay * config.brandWeight;
                    }
                    traces.push({
                        event,
                        origin,
                        ageHours,
                        relation: [
                            sameProduct ? "cùng sản phẩm" : "",
                            sameCategory ? "cùng danh mục" : "",
                            sameBrand ? "cùng thương hiệu" : "",
                        ].filter(Boolean).join(", "),
                    });
                });
                const score = Object.values(contribution).reduce((sum, value) => sum + value, 0);
                return {
                    product,
                    score,
                    contribution,
                    traces: traces.slice(0, 6),
                    noRelation: source.events.length > 0 && !traces.length,
                };
            })
            .sort((a, b) => b.score - a.score)
            .slice(0, config.topK);

        return { ranked, warnings, source };
    }

    function renderOptions(state) {
        const chosenSessionUser = $("session-user-filter").value;
        const chosenProductCategory = $("admin-product-filter").value;
        const chosenProductBrand = $("admin-brand-filter").value;
        $("preset-select").innerHTML = state.simulatorPresets.map((preset) => `
            <option value="${Store.escapeHtml(preset.id)}">${Store.escapeHtml(preset.name)}</option>
        `).join("");
        $("preset-select").value = state.activePresetId;
        const userOptions = state.users.map((user) => `
            <option value="${Store.escapeHtml(user.id)}">${Store.escapeHtml(user.name)}${user.active ? "" : " (ẩn)"}</option>
        `).join("");
        $("session-user-filter").innerHTML = `<option value="">Mọi người dùng</option>${userOptions}`;
        const sessions = sessionsForAdmin(state);
        $("product-image").innerHTML = IMAGE_TYPES
            .map((type) => `<option value="${type}">${IMAGE_LABELS[type]}</option>`).join("");
        const categories = [...new Set(state.products.map((product) => product.category))].sort();
        const brands = [...new Set(state.products.map((product) => product.brand))].sort();
        $("admin-product-filter").innerHTML = `<option value="">Mọi danh mục</option>${categories
            .map((value) => `<option>${Store.escapeHtml(value)}</option>`).join("")}`;
        $("admin-brand-filter").innerHTML = `<option value="">Mọi thương hiệu</option>${brands
            .map((value) => `<option>${Store.escapeHtml(value)}</option>`).join("")}`;
        const recentEventUserId = sessions
            .flatMap((session) => session.events.map((event) => ({ ...event, userId: session.userId })))
            .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))[0]?.userId;
        const fallbackUser = state.users.find((user) => user.id === selectedUserId)
            || state.users.find((user) => user.id === state.signedInUserId)
            || state.users.find((user) => user.id === recentEventUserId)
            || state.users.find((user) => user.active)
            || state.users[0];
        selectedUserId = fallbackUser?.id || "";
        if (!sessions.some((session) => session.id === selectedSourceSessionId)) {
            selectedSourceSessionId = null;
        }
        $("session-user-filter").value = chosenSessionUser;
        $("admin-product-filter").value = chosenProductCategory;
        $("admin-brand-filter").value = chosenProductBrand;
        if (!$("ref-date").value) {
            $("ref-date").value = localDateValue(Date.now());
        }
        renderReferenceShiftLabel();
    }

    function productInteractionHistory(state, productId) {
        const users = Store.userMap(state);
        return allEvents(state)
            .filter((event) => event.productId === productId)
            .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))
            .slice(0, 8)
            .map((event) => `
                <li><b>${Store.escapeHtml(behaviorLabel(event.behavior))}</b> · ${Store.escapeHtml(users[event.userId]?.name || event.userId)} · ${Store.formatTime(event.timestamp)}</li>
            `).join("");
    }

    function renderProducts(state) {
        const query = $("admin-product-search").value.trim().toLowerCase();
        const category = $("admin-product-filter").value;
        const brand = $("admin-brand-filter").value;
        const maxPrice = Number($("admin-price-filter").value || 0);
        const products = state.products.filter((product) => {
            const queryMatch = `${product.name} ${product.brand}`.toLowerCase().includes(query);
            const categoryMatch = !category || product.category === category;
            const brandMatch = !brand || product.brand === brand;
            const priceMatch = !maxPrice || Number(product.price) <= maxPrice;
            return queryMatch && categoryMatch && brandMatch && priceMatch;
        });
        $("product-table").innerHTML = products.map((product) => `
            <article class="admin-row ${product.active ? "" : "inactive"}">
                <span class="mini-art art-${Store.escapeHtml(product.image)}"></span>
                <div>
                    <strong>${Store.escapeHtml(product.name)}</strong>
                    <small>${Store.escapeHtml(product.brand)} · ${Store.escapeHtml(product.category)} · ${Store.formatMoney(product.price)}</small>
                </div>
                <b>${Number(product.popularityScore)}</b>
                <div class="button-group row-actions">
                    <button data-edit-product="${Store.escapeHtml(product.id)}">Sửa</button>
                    <button data-history-product="${Store.escapeHtml(product.id)}">Lịch sử</button>
                    <button data-toggle-product="${Store.escapeHtml(product.id)}">${product.active ? "Ẩn" : "Hiện"}</button>
                </div>
            </article>
        `).join("") || `<div class="empty-state">Không có sản phẩm khớp bộ lọc.</div>`;
        if (selectedProductHistoryId) {
            const product = state.products.find((item) => item.id === selectedProductHistoryId);
            $("product-history").innerHTML = product ? `
                <h3>Lịch sử ${Store.escapeHtml(product.name)}</h3>
                <ul>${productInteractionHistory(state, product.id) || "<li>Chưa có sự kiện.</li>"}</ul>
            ` : "";
        }
    }

    function userEvents(state, userId) {
        const products = Store.productMap(state);
        const sessions = sessionsForAdmin(state).filter((session) => session.userId === userId);
        return sessions.flatMap((session) => session.events.map((event) => ({
            ...event,
            label: session.label,
            product: products[event.productId],
        })))
            .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
    }

    function hashNumber(value) {
        return [...String(value)].reduce((hash, char) => {
            return Math.imul(hash ^ char.charCodeAt(0), 16777619) >>> 0;
        }, 2166136261);
    }

    function intentScores(state, userId) {
        const scores = Array.from({ length: 64 }, (_, index) => ({
            id: index + 1,
            score: 0.18 + (hashNumber(`${userId}:${index}`) % 16) / 100,
        }));
        const behaviorBoost = { view: 0.75, cart: 1.35, purchase: 2.1 };
        userEvents(state, userId).forEach((event, index) => {
            const product = event.product;
            if (!product) {
                return;
            }
            const freshness = 1 / (1 + index * 0.16);
            const key = `${product.category}:${product.brand}`;
            const primary = hashNumber(key) % scores.length;
            const neighbor = (primary + hashNumber(product.id) % 11 + 1) % scores.length;
            const boost = (behaviorBoost[event.behavior] || 0.4) * freshness;
            scores[primary].score += boost;
            scores[neighbor].score += boost * 0.34;
        });
        return scores.sort((a, b) => b.score - a.score).slice(0, 8);
    }

    function renderSelectedUser(state) {
        const user = state.users.find((item) => item.id === selectedUserId)
            || state.users.find((item) => item.active)
            || state.users[0];
        if (!user) {
            $("selected-user-card").innerHTML = `<div class="empty-state">Chưa có người dùng demo.</div>`;
            $("selected-user-log").innerHTML = "";
            $("user-intent-chart").innerHTML = "";
            return;
        }
        selectedUserId = user.id;
        const events = userEvents(state, user.id);
        const countByBehavior = Object.fromEntries(["view", "cart", "purchase"].map((behavior) => {
            return [behavior, events.filter((event) => event.behavior === behavior).length];
        }));
        const eventCounts = Object.entries(countByBehavior).map(([behavior, value]) => {
            return [behaviorLabel(behavior), value];
        });
        $("selected-user-card").innerHTML = `
            <div>
                <p class="eyebrow">Người dùng đang phân tích</p>
                <h3>${Store.escapeHtml(user.name)}</h3>
                <p>${Store.escapeHtml(user.id)}</p>
                <small>${Store.escapeHtml(user.notes || "Chưa có ghi chú.")}</small>
            </div>
            <div class="pill-row user-signal-pills">
                ${eventCounts.map(([label, value]) => `<span class="pill">${Store.escapeHtml(label)} <b>${value}</b></span>`).join("")}
            </div>
        `;
        $("selected-user-log").innerHTML = events.map((event) => `
            <article class="log-row behavior-${Store.escapeHtml(event.behavior)}">
                <b>${Store.escapeHtml(behaviorLabel(event.behavior))}</b>
                <span>${Store.escapeHtml(event.product?.name || event.productId)}</span>
                <small>${Store.escapeHtml(event.label)} · ${Store.formatTime(event.timestamp)}</small>
            </article>
        `).join("") || `<div class="empty-state">User này chưa có log hành vi để đọc.</div>`;
        const intents = intentScores(state, user.id);
        const maxIntent = Math.max(...intents.map((intent) => intent.score), 0.01);
        $("intent-note").textContent = "Top 8 trên 64 mã ý định. Đây là biểu đồ demo từ log user; attention thật cần chạy model.";
        $("user-intent-chart").innerHTML = intents.map((intent) => `
            <div class="intent-row">
                <span>I${String(intent.id).padStart(2, "0")}</span>
                <i><b style="width:${Math.max(7, intent.score / maxIntent * 100).toFixed(1)}%"></b></i>
                <strong>${intent.score.toFixed(2)}</strong>
            </div>
        `).join("");
    }

    function renderUsers(state) {
        const query = $("admin-user-search").value.trim().toLowerCase();
        $("user-table").innerHTML = state.users.filter((user) => {
            return `${user.name} ${user.id} ${user.notes || ""}`.toLowerCase().includes(query);
        }).map((user) => {
            const events = userEvents(state, user.id);
            const counts = Object.fromEntries(["view", "cart", "purchase"].map((behavior) => {
                return [behavior, events.filter((event) => event.behavior === behavior).length];
            }));
            return `
            <article class="user-pick-row ${user.active ? "" : "inactive"} ${user.id === selectedUserId ? "selected" : ""}" data-select-user="${Store.escapeHtml(user.id)}">
                <button class="user-pick-main" data-select-user="${Store.escapeHtml(user.id)}">
                    <span class="user-mark">${Store.escapeHtml(user.name.charAt(0))}</span>
                    <span>
                        <strong>${Store.escapeHtml(user.name)}</strong>
                        <small>${Store.escapeHtml(user.id)}${user.id === selectedUserId ? " · đang xem" : ""}</small>
                    </span>
                    <span class="user-behavior-counts">
                        <span class="behavior-count view"><b>${counts.view}</b><small>Xem</small></span>
                        <span class="behavior-count cart"><b>${counts.cart}</b><small>Giỏ</small></span>
                        <span class="behavior-count purchase"><b>${counts.purchase}</b><small>Mua</small></span>
                    </span>
                </button>
                <div class="button-group row-actions">
                    <button data-select-user="${Store.escapeHtml(user.id)}">Chọn</button>
                    <button data-edit-user="${Store.escapeHtml(user.id)}">Sửa</button>
                </div>
            </article>
        `;
        }).join("") || `<div class="empty-state">Không có người dùng khớp bộ lọc.</div>`;
    }

    function renderModelRunLog(simulation, config) {
        const state = Store.read();
        const products = Store.productMap(state);
        const users = Store.userMap(state);
        const sourceUser = simulation.source.events[0]?.userId || simulation.source.session?.userId || selectedUserId;
        const eventRows = simulation.source.events.slice(0, 10).map((event) => `
            <li>
                <b>${Store.escapeHtml(behaviorLabel(event.behavior))}</b>
                ${Store.escapeHtml(products[event.productId]?.name || event.productId)}
                <time>${Store.formatTime(event.timestamp)}</time>
            </li>
        `).join("");
        const resultRows = simulation.ranked.map((result, index) => `
            <li>#${index + 1} ${Store.escapeHtml(result.product.name)} <b>${result.score.toFixed(2)}</b></li>
        `).join("");
        $("model-run-log").innerHTML = `
            <article>
                <span>Người dùng</span>
                <strong>${Store.escapeHtml(users[sourceUser]?.name || sourceUser || "Chưa chọn")}</strong>
                <small>${Store.escapeHtml(simulation.source.label)} · ${simulation.source.events.length} sự kiện đầu vào</small>
            </article>
            <article>
                <span>Cấu hình đang chạy</span>
                <p>Xem ${config.weights.view} · Giỏ ${config.weights.cart} · Mua ${config.weights.purchase}</p>
                <small>Mốc ${Store.formatTime(simulation.source.refTime)} · suy giảm ${config.recencyDecay} · top ${config.topK}${config.maskPurchased ? " · ẩn sản phẩm đã mua" : ""}</small>
            </article>
            <article>
                <span>Sự kiện được đọc</span>
                <ol>${eventRows || "<li>Chưa có sự kiện; dùng độ phổ biến làm tín hiệu nền.</li>"}</ol>
            </article>
            <article>
                <span>Kết quả xếp hạng</span>
                <ol>${resultRows || "<li>Không có ứng viên.</li>"}</ol>
            </article>
        `;
    }

    function renderModelRunLog(simulation, config) {
        const state = Store.read();
        const products = Store.productMap(state);
        const users = Store.userMap(state);
        const sourceUser = simulation.source.events[0]?.userId || simulation.source.session?.userId || selectedUserId;
        const pipelineRows = (simulation.source.pipeline || []).map((step, index) => `
            <li>
                <b>${index + 1}. ${Store.escapeHtml(step.name)}</b>
                <small>${Store.escapeHtml(step.detail)}</small>
            </li>
        `).join("");
        const eventRows = simulation.source.events.slice(0, 10).map((event) => `
            <li>
                <b>${Store.escapeHtml(behaviorLabel(event.behavior))}</b>
                ${Store.escapeHtml(products[event.productId]?.name || event.productId)}
                <time>${Store.formatTime(event.timestamp)}</time>
            </li>
        `).join("");
        const resultRows = simulation.ranked.map((result, index) => `
            <li>
                #${index + 1} ${Store.escapeHtml(result.product.name)}
                <b>${result.score.toFixed(2)}</b>
                <small>model ${Number(result.contribution.model || 0).toFixed(2)} · category ${Number(result.contribution.category || 0).toFixed(2)} · brand ${Number(result.contribution.brand || 0).toFixed(2)} · popularity ${Number(result.contribution.popularity || 0).toFixed(2)}</small>
            </li>
        `).join("");
        $("model-run-log").innerHTML = `
            <article>
                <span>Người dùng</span>
                <strong>${Store.escapeHtml(users[sourceUser]?.name || sourceUser || "Chưa chọn")}</strong>
                <small>${Store.escapeHtml(simulation.source.querySource || simulation.source.label)} · ${simulation.source.events.length} sự kiện đầu vào</small>
            </article>
            <article>
                <span>Checkpoint</span>
                <p>${Store.escapeHtml(simulation.source.checkpoint || "Chưa có checkpoint")}</p>
                <small>epoch ${Store.escapeHtml(simulation.source.checkpointEpoch ?? "?")} · mốc ${Store.formatTime(simulation.source.refTime)}</small>
            </article>
            <article>
                <span>Các bước tính toán</span>
                <ol>${pipelineRows || "<li>Backend chưa trả trace pipeline.</li>"}</ol>
            </article>
            <article>
                <span>Cấu hình đang chạy</span>
                <p>Xem ${config.weights.view} · Giỏ ${config.weights.cart} · Mua ${config.weights.purchase}</p>
                <small>recency ${config.recencyDecay} · category ${config.categoryWeight} · brand ${config.brandWeight} · popularity ${config.popularityWeight} · top ${config.topK}${config.maskPurchased ? " · ẩn sản phẩm đã mua" : ""}</small>
            </article>
            <article>
                <span>Sự kiện được đọc</span>
                <ol>${eventRows || "<li>Chưa có sự kiện; chỉ dùng user embedding hoặc popularity fallback.</li>"}</ol>
            </article>
            <article>
                <span>Kết quả xếp hạng</span>
                <ol>${resultRows || "<li>Không có ứng viên.</li>"}</ol>
            </article>
        `;
    }

    function renderRecommendations(simulation, config) {
        results = simulation.ranked;
        $("recommendation-panel").hidden = !simulatorVisible;
        $("explanation-panel").hidden = !simulatorVisible;
        $("model-log-panel").hidden = !simulatorVisible;
        $("warning-list").innerHTML = simulation.warnings.map((warning) => `<p>${Store.escapeHtml(warning)}</p>`).join("");
        $("simulation-note").textContent = `Nguồn: ${simulation.source.label} · ${simulation.source.events.length} sự kiện trước ${Store.formatTime(simulation.source.refTime)}`;
        if (!selectedResultId || !results.some((result) => result.product.id === selectedResultId)) {
            selectedResultId = results[0]?.product.id || null;
        }
        $("recommendation-list").innerHTML = results.map((result, index) => {
            const oldRank = previousRanks[result.product.id];
            const shift = oldRank ? oldRank - (index + 1) : 0;
            const delta = oldRank ? (shift === 0 ? "giữ hạng" : shift > 0 ? `tăng ${shift}` : `giảm ${Math.abs(shift)}`) : "mới";
            return `
                <button class="recommendation-row ${result.product.id === selectedResultId ? "selected" : ""}" data-result="${Store.escapeHtml(result.product.id)}">
                    <span>#${index + 1}</span>
                    <span class="mini-art art-${Store.escapeHtml(result.product.image)}"></span>
                    <strong>${Store.escapeHtml(result.product.name)}</strong>
                    <b>${result.score.toFixed(2)}</b>
                    <small>${Store.escapeHtml(delta)}</small>
                </button>
            `;
        }).join("") || `<div class="empty-state">Không có ứng viên để xếp hạng.</div>`;
        previousRanks = Object.fromEntries(results.map((result, index) => [result.product.id, index + 1]));
        renderExplanation();
        renderModelRunLog(simulation, config);
    }

    function renderExplanation() {
        const result = results.find((item) => item.product.id === selectedResultId);
        if (!result) {
            $("explanation-detail").innerHTML = `<div class="empty-state">Chạy mô phỏng để xem giải thích.</div>`;
            return;
        }
        const maxPart = Math.max(...Object.values(result.contribution), 0.01);
        const labels = {
            model: "Điểm checkpoint",
            behavior: "Hành vi trực tiếp",
            recency: "Độ mới thời gian",
            category: "Cùng danh mục",
            brand: "Cùng thương hiệu",
            popularity: "Độ phổ biến",
        };
        $("explanation-detail").innerHTML = `
            <div class="detail-head compact-result">
                <span class="product-art art-${Store.escapeHtml(result.product.image)}"></span>
                <div>
                    <h3>${Store.escapeHtml(result.product.name)}</h3>
                    <p>${Store.escapeHtml(result.product.category)} · ${Store.escapeHtml(result.product.brand)}</p>
                </div>
            </div>
            <div class="score-bars">
                ${Object.entries(result.contribution).map(([key, value]) => `
                    <div class="score-line">
                        <span>${labels[key]}</span>
                        <i><b style="width:${Math.max(4, value / maxPart * 100).toFixed(1)}%"></b></i>
                        <strong>${value.toFixed(2)}</strong>
                    </div>
                `).join("")}
            </div>
            <h4>Sự kiện ảnh hưởng</h4>
            <ul class="trace-list">
                ${result.traces.map((trace) => `
                    <li>
                        <b>${Store.escapeHtml(behaviorLabel(trace.event.behavior))}</b>
                        ${Store.escapeHtml(trace.origin.name)}
                        <small>${Store.escapeHtml(trace.relation)} · ${trace.ageHours.toFixed(1)} giờ</small>
                    </li>
                `).join("") || "<li>Không có liên hệ danh mục hoặc thương hiệu; điểm chủ yếu đến từ độ phổ biến.</li>"}
            </ul>
            ${result.noRelation ? `<p class="warning-inline">Không tìm thấy tín hiệu danh mục hoặc thương hiệu phù hợp trong nguồn sự kiện.</p>` : ""}
        `;
    }

    async function runSimulation(event) {
        event?.preventDefault();
        const config = currentConfig();
        if (Backend && selectedUserId) {
            try {
                const simulation = await Backend.recommend({
                    userId: selectedUserId,
                    topK: config.topK,
                    maskPurchased: config.maskPurchased,
                    weights: config.weights,
                    recencyDecay: config.recencyDecay,
                    categoryWeight: config.categoryWeight,
                    brandWeight: config.brandWeight,
                    popularityWeight: config.popularityWeight,
                    refTime: new Date(referenceTime(sourceEvents(Store.read()).events, config)).toISOString(),
                });
                renderRecommendations(simulation, config);
                return;
            } catch (error) {
                $("simulation-note").textContent = "Backend khong san sang, dang dung mo phong local.";
            }
        }
        renderRecommendations(simulate(Store.read(), config), config);
    }

    function showSimulator() {
        simulatorVisible = true;
        $("simulator-panel").hidden = false;
        $("model-log-panel").hidden = true;
        selectedSourceSessionId = null;
        runSimulation();
        $("simulator-panel").scrollIntoView({ behavior: "smooth", block: "start" });
    }

    function fillProductForm(product) {
        $("product-id").value = product?.id || "";
        $("product-name").value = product?.name || "";
        $("product-brand").value = product?.brand || "";
        $("product-category").value = product?.category || "";
        $("product-price").value = product?.price || "";
        $("product-popularity").value = product?.popularityScore ?? 50;
        $("product-image").value = product?.image || "headset";
        $("product-active").checked = product ? Boolean(product.active) : true;
        $("product-dialog-title").textContent = product ? "Sửa sản phẩm" : "Tạo sản phẩm mới";
    }

    function openProductDialog(product) {
        fillProductForm(product);
        $("product-editor-dialog").showModal();
    }

    function setProductImportNote(message, tone) {
        $("product-import-note").textContent = message;
        $("product-import-note").dataset.tone = tone || "";
    }

    function importedProductsFromJson(text) {
        const parsed = JSON.parse(text);
        const rows = Array.isArray(parsed) ? parsed : parsed.products;
        if (!Array.isArray(rows)) {
            throw new Error("File JSON cần là mảng product hoặc object có trường products.");
        }
        return rows.map((row, index) => {
            const name = String(row.name || "").trim();
            const brand = String(row.brand || "").trim();
            const category = String(row.category || "").trim();
            if (!name || !brand || !category) {
                throw new Error(`Product dòng ${index + 1} thiếu tên, thương hiệu hoặc danh mục.`);
            }
            const image = IMAGE_TYPES.includes(row.image) ? row.image : "headset";
            const price = Number(row.price);
            const popularity = Number(row.popularityScore);
            return {
                id: String(row.id || Store.uid("product")),
                name,
                brand,
                category,
                price: Number.isFinite(price) ? Math.max(0, price) : 0,
                popularityScore: Number.isFinite(popularity)
                    ? Math.min(100, Math.max(0, popularity))
                    : 50,
                image,
                active: row.active !== false,
            };
        });
    }

    function fillUserForm(user) {
        $("admin-user-id").value = user?.id || "";
        $("admin-user-name").value = user?.name || "";
        $("admin-user-notes").value = user?.notes || "";
        $("user-dialog-title").textContent = user ? "Sửa người dùng" : "Thêm người dùng";
    }

    function openUserDialog(user) {
        fillUserForm(user);
        $("user-editor-dialog").showModal();
    }

    function renderAll() {
        const state = Store.read();
        renderOptions(state);
        renderProducts(state);
        renderUsers(state);
        renderSelectedUser(state);
        if (!$("weight-view").value) {
            setConfig(state.simulatorPresets.find((preset) => preset.id === state.activePresetId) || Store.defaultPreset());
        }
        $("simulator-panel").hidden = !simulatorVisible;
        $("recommendation-panel").hidden = !simulatorVisible || !results.length;
        $("explanation-panel").hidden = !simulatorVisible || !results.length;
        $("model-log-panel").hidden = !simulatorVisible || !results.length;
        if (simulatorVisible && !results.length) {
            runSimulation();
        }
    }

    $("simulator-form").addEventListener("submit", runSimulation);
    $("load-preset").addEventListener("click", () => {
        const state = Store.read();
        const preset = state.simulatorPresets.find((item) => item.id === $("preset-select").value);
        if (preset) {
            Store.update((draft) => {
                draft.activePresetId = preset.id;
            });
            setConfig(preset);
            runSimulation();
        }
    });
    $("reset-config").addEventListener("click", () => {
        setConfig(Store.defaultPreset());
        runSimulation();
    });
    $("save-preset").addEventListener("click", () => {
        $("preset-save-note").textContent = "";
        $("preset-name").value = "";
        $("preset-dialog").showModal();
    });
    $("preset-form").addEventListener("submit", (event) => {
        event.preventDefault();
        const name = $("preset-name").value.trim();
        if (!name) {
            $("preset-save-note").textContent = "Nhập tên cấu hình trước khi lưu.";
            return;
        }
        const config = currentConfig();
        const state = Store.update((draft) => {
            const preset = { id: Store.uid("preset"), name, ...config };
            draft.simulatorPresets.unshift(preset);
            draft.activePresetId = preset.id;
        });
        $("preset-name").value = "";
        $("preset-dialog").close();
        $("simulation-note").textContent = `Đã lưu cấu hình: ${name}`;
        renderOptions(state);
    });
    $("ref-date").addEventListener("input", runSimulation);
    $("ref-shift-days").addEventListener("input", () => {
        renderReferenceShiftLabel();
        runSimulation();
    });

    $("product-form").addEventListener("submit", (event) => {
        event.preventDefault();
        const id = $("product-id").value || Store.uid("product");
        Store.update((state) => {
            const product = state.products.find((item) => item.id === id);
            const payload = {
                id,
                name: $("product-name").value.trim(),
                brand: $("product-brand").value.trim(),
                category: $("product-category").value.trim(),
                price: Number($("product-price").value),
                popularityScore: Number($("product-popularity").value),
                image: $("product-image").value,
                active: $("product-active").checked,
            };
            if (product) {
                Object.assign(product, payload);
            } else {
                state.products.unshift(payload);
            }
        });
        $("product-editor-dialog").close();
        fillProductForm();
        renderAll();
        runSimulation();
    });
    $("user-admin-form").addEventListener("submit", (event) => {
        event.preventDefault();
        const id = $("admin-user-id").value || Store.uid("user");
        selectedUserId = id;
        Store.update((state) => {
            const user = state.users.find((item) => item.id === id);
            const payload = {
                id,
                name: $("admin-user-name").value.trim(),
                notes: $("admin-user-notes").value.trim(),
                active: user ? Boolean(user.active) : true,
            };
            if (user) {
                Object.assign(user, payload);
            } else {
                state.users.unshift(payload);
            }
        });
        const savedUser = Store.read().users.find((item) => item.id === id);
        if (Backend && savedUser) {
            Backend.createUser(savedUser).catch(() => {});
        }
        $("user-editor-dialog").close();
        fillUserForm();
        renderAll();
        if (simulatorVisible) {
            runSimulation();
        }
    });

    document.addEventListener("click", (event) => {
        const target = event.target;
        const resultButton = target.closest("[data-result]");
        const userSelectButton = target.closest("[data-select-user]");
        const state = Store.read();
        if (resultButton) {
            selectedResultId = resultButton.dataset.result;
            document.querySelectorAll("[data-result]").forEach((row) => {
                row.classList.toggle("selected", row.dataset.result === selectedResultId);
            });
            renderExplanation();
        }
        if (target.dataset.editProduct) {
            openProductDialog(state.products.find((product) => product.id === target.dataset.editProduct));
        }
        if (target.dataset.historyProduct) {
            selectedProductHistoryId = target.dataset.historyProduct;
            renderProducts(state);
        }
        if (target.dataset.toggleProduct) {
            Store.update((draft) => {
                const product = draft.products.find((item) => item.id === target.dataset.toggleProduct);
                product.active = !product.active;
            });
            renderAll();
            runSimulation();
        }
        if (target.dataset.editUser) {
            openUserDialog(state.users.find((user) => user.id === target.dataset.editUser));
        }
        if (userSelectButton && !target.dataset.editUser) {
            selectedUserId = userSelectButton.dataset.selectUser;
            selectedSourceSessionId = null;
            renderUsers(state);
            renderSelectedUser(state);
            if (simulatorVisible) {
                runSimulation();
            }
        }
    });

    ["admin-product-search", "admin-product-filter", "admin-brand-filter", "admin-price-filter"].forEach((id) => {
        $(id).addEventListener("input", () => renderProducts(Store.read()));
    });
    $("admin-user-search").addEventListener("input", () => renderUsers(Store.read()));
    $("new-product").addEventListener("click", () => openProductDialog());
    $("close-product-dialog").addEventListener("click", () => $("product-editor-dialog").close());
    $("cancel-product-dialog").addEventListener("click", () => $("product-editor-dialog").close());
    $("import-products").addEventListener("click", () => $("import-products-file").click());
    $("import-products-file").addEventListener("change", async (event) => {
        const file = event.target.files[0];
        if (!file) {
            return;
        }
        try {
            const imported = importedProductsFromJson(await file.text());
            Store.update((state) => {
                imported.reverse().forEach((payload) => {
                    const product = state.products.find((item) => item.id === payload.id);
                    if (product) {
                        Object.assign(product, payload);
                    } else {
                        state.products.unshift(payload);
                    }
                });
            });
            setProductImportNote(`Đã import ${imported.length} sản phẩm từ ${file.name}.`, "success");
            renderAll();
            if (simulatorVisible) {
                runSimulation();
            }
        } catch (error) {
            setProductImportNote(error.message || "Không đọc được file sản phẩm.", "error");
        }
        event.target.value = "";
    });
    $("new-user-admin").addEventListener("click", () => openUserDialog());
    $("close-user-dialog").addEventListener("click", () => $("user-editor-dialog").close());
    $("cancel-user-dialog").addEventListener("click", () => $("user-editor-dialog").close());
    $("close-preset-dialog").addEventListener("click", () => $("preset-dialog").close());
    $("cancel-preset-dialog").addEventListener("click", () => $("preset-dialog").close());
    $("show-simulator").addEventListener("click", showSimulator);
    $("restore-demo").addEventListener("click", () => {
        Store.restoreSeed();
        results = [];
        previousRanks = {};
        selectedUserId = null;
        simulatorVisible = false;
        fillProductForm();
        fillUserForm();
        renderAll();
    });
    $("export-sessions-json").addEventListener("click", () => {
        const sessions = visibleSessions(Store.read());
        Store.download("bp-atmp-phien-test.json", JSON.stringify(sessions, null, 2), "application/json");
    });
    $("export-sessions-csv").addEventListener("click", () => {
        const events = visibleSessions(Store.read()).flatMap((session) => session.events);
        Store.download("bp-atmp-phien-test.csv", Store.toCsv(events), "text/csv;charset=utf-8");
    });

    fillProductForm();
    fillUserForm();
    renderAll();
    if (Backend) {
        Backend.hydrateStore(Store).then((result) => {
            if (result.ok) {
                results = [];
                previousRanks = {};
                selectedUserId = null;
                renderAll();
            }
        });
    }
    window.addEventListener("storage", () => {
        results = [];
        previousRanks = {};
        renderAll();
    });
}());
