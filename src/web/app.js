(function () {
    "use strict";

    const Store = window.DemoStore;
    const Backend = window.DemoBackend;
    const $ = (id) => document.getElementById(id);
    let selectedProductId = null;
    let toastTimer = null;

    function signedInUser(state) {
        return state.users.find((user) => user.id === state.signedInUserId && user.active);
    }

    function activeUsers(state) {
        return state.users.filter((user) => user.active);
    }

    function behaviorLabel(behavior) {
        return Store.BEHAVIORS[behavior]?.label || behavior;
    }

    function filterProducts(state) {
        const query = $("product-search").value.trim().toLowerCase();
        const category = $("category-filter").value;
        const brand = $("brand-filter").value;
        const maxPrice = Number($("price-filter").value || 0);
        return state.products.filter((product) => {
            const nameMatch = product.name.toLowerCase().includes(query);
            const categoryMatch = !category || product.category === category;
            const brandMatch = !brand || product.brand === brand;
            const priceMatch = !maxPrice || Number(product.price) <= maxPrice;
            return product.active && nameMatch && categoryMatch && brandMatch && priceMatch;
        });
    }

    function renderSelectors(state) {
        const chosenCategory = $("category-filter").value;
        const chosenBrand = $("brand-filter").value;
        $("user-select").innerHTML = activeUsers(state)
            .map((user) => `<option value="${Store.escapeHtml(user.id)}">${Store.escapeHtml(user.name)}</option>`)
            .join("");
        const categories = [...new Set(state.products.map((product) => product.category))].sort();
        const brands = [...new Set(state.products.map((product) => product.brand))].sort();
        $("category-filter").innerHTML = `<option value="">Mọi danh mục</option>${categories
            .map((value) => `<option>${Store.escapeHtml(value)}</option>`).join("")}`;
        $("brand-filter").innerHTML = `<option value="">Mọi thương hiệu</option>${brands
            .map((value) => `<option>${Store.escapeHtml(value)}</option>`).join("")}`;
        $("category-filter").value = chosenCategory;
        $("brand-filter").value = chosenBrand;
    }

    function renderLoginState(state) {
        const user = signedInUser(state);
        $("login-panel").hidden = Boolean(user);
        $("catalog-panel").hidden = !user;
        $("product-panel").hidden = !user;
        $("logout-user").hidden = !user;
        $("shopper-title").textContent = user ? `Xin chào, ${user.name}` : "Chào mừng bạn";
        $("shopper-user-note").textContent = user
            ? "Hành vi mua sắm sẽ được ghi vào hệ thống."
            : "Đăng nhập để bắt đầu phiên mua sắm.";
    }

    function renderProducts(state) {
        const products = filterProducts(state);
        if (!selectedProductId || !products.some((product) => product.id === selectedProductId)) {
            selectedProductId = products[0]?.id || null;
        }
        $("product-count").textContent = `${products.length} sản phẩm`;
        $("product-grid").innerHTML = products.length ? products.map((product) => `
            <article class="product-card ${product.id === selectedProductId ? "selected" : ""}">
                <button class="product-select" data-product-select="${Store.escapeHtml(product.id)}">
                    <span class="product-art art-${Store.escapeHtml(product.image)}" aria-hidden="true"></span>
                    <span class="product-copy">
                        <strong>${Store.escapeHtml(product.name)}</strong>
                        <small>${Store.escapeHtml(product.category)} · ${Store.escapeHtml(product.brand)}</small>
                        <b>${Store.formatMoney(product.price)}</b>
                    </span>
                </button>
                <div class="behavior-row shopper-actions">
                    <button class="icon-command cart" data-event="cart" data-product="${Store.escapeHtml(product.id)}">Thêm giỏ</button>
                    <button class="icon-command purchase" data-event="purchase" data-product="${Store.escapeHtml(product.id)}">Mua nhanh</button>
                </div>
            </article>
        `).join("") : `<div class="empty-state">Không có sản phẩm khớp bộ lọc.</div>`;
        renderDetail(state);
    }

    function renderDetail(state) {
        const product = state.products.find((item) => item.id === selectedProductId);
        if (!product) {
            $("product-detail").innerHTML = `<div class="empty-state">Chọn sản phẩm để xem chi tiết.</div>`;
            return;
        }
        $("product-detail").innerHTML = `
            <div class="detail-head">
                <span class="product-art art-${Store.escapeHtml(product.image)}" aria-hidden="true"></span>
                <div>
                    <p class="eyebrow">${Store.escapeHtml(product.category)}</p>
                    <h2>${Store.escapeHtml(product.name)}</h2>
                    <p>${Store.escapeHtml(product.brand)} · ${Store.formatMoney(product.price)}</p>
                </div>
            </div>
            <div class="detail-actions">
                <button class="primary cart" data-event="cart" data-product="${Store.escapeHtml(product.id)}">Thêm vào giỏ</button>
                <button class="primary purchase" data-event="purchase" data-product="${Store.escapeHtml(product.id)}">Mua nhanh</button>
            </div>
        `;
    }

    function rerender() {
        const state = Store.read();
        renderSelectors(state);
        renderLoginState(state);
        if (signedInUser(state)) {
            Store.ensureDraft(state.signedInUserId);
            renderProducts(Store.read());
        }
    }

    function showToast(message, tone) {
        const toast = $("shopper-toast");
        toast.textContent = message;
        toast.dataset.tone = tone || "info";
        toast.hidden = false;
        window.clearTimeout(toastTimer);
        toastTimer = window.setTimeout(() => {
            toast.hidden = true;
        }, 2200);
    }

    function addEvent(productId, behavior) {
        const state = Store.read();
        if (!signedInUser(state)) {
            return;
        }
        const updated = Store.addDraftEvent(productId, behavior);
        const event = updated.draftSession?.events?.[0];
        if (Backend && event) {
            Backend.createEvent(event).catch(() => {});
        }
        if (behavior === "cart") {
            showToast("Đã thêm sản phẩm vào giỏ.", "cart");
        }
        if (behavior === "purchase") {
            showToast("Đã mua sản phẩm.", "purchase");
        }
        rerender();
    }

    function selectProduct(productId) {
        selectedProductId = productId;
        addEvent(productId, "view");
    }

    function createUser(event) {
        event.preventDefault();
        const name = $("new-user-name").value.trim();
        if (!name) {
            return;
        }
        const state = Store.update((draft) => {
            draft.users.unshift({
                id: Store.uid("user"),
                name,
                notes: "Tạo từ cửa hàng demo.",
                active: true,
            });
        });
        if (Backend) {
            Backend.createUser(state.users[0]).catch(() => {});
        }
        event.target.reset();
        Store.startUserSession(state.users[0].id);
        rerender();
    }

    document.addEventListener("click", (event) => {
        const eventButton = event.target.closest("[data-event]");
        const selectButton = event.target.closest("[data-product-select]");
        if (eventButton) {
            addEvent(eventButton.dataset.product, eventButton.dataset.event);
        }
        if (selectButton) {
            selectProduct(selectButton.dataset.productSelect);
        }
    });

    $("login-form").addEventListener("submit", (event) => {
        event.preventDefault();
        Store.startUserSession($("user-select").value);
        rerender();
    });
    $("user-form").addEventListener("submit", createUser);
    $("logout-user").addEventListener("click", () => {
        Store.closeUserSession();
        rerender();
    });
    ["product-search", "category-filter", "brand-filter", "price-filter"].forEach((id) => {
        $(id).addEventListener("input", () => renderProducts(Store.read()));
    });
    window.addEventListener("storage", rerender);
    rerender();
    if (Backend) {
        Backend.hydrateStore(Store).then((result) => {
            if (result.ok) {
                selectedProductId = null;
                rerender();
            }
        });
    }
}());
