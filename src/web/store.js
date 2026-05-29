(function () {
    "use strict";

    const STORAGE_KEY = "bpatmp-demo-state-v2";
    const BEHAVIORS = {
        view: { label: "Xem", color: "view" },
        cart: { label: "Thêm vào giỏ", color: "cart" },
        purchase: { label: "Mua", color: "purchase" },
    };

    const now = Date.now();
    const hoursAgo = (hours) => new Date(now - hours * 60 * 60 * 1000).toISOString();

    const seed = {
        products: [
            {
                id: "p-sony-headset",
                name: "Tai nghe chống ồn Sony",
                brand: "Sony",
                category: "Âm thanh",
                price: 8490000,
                popularityScore: 88,
                image: "headset",
                active: true,
            },
            {
                id: "p-keyboard-k2",
                name: "Bàn phím cơ K2",
                brand: "Keychron",
                category: "Phụ kiện máy tính",
                price: 2390000,
                popularityScore: 71,
                image: "keyboard",
                active: true,
            },
            {
                id: "p-monitor-27",
                name: "Màn hình 27 inch",
                brand: "LG",
                category: "Thiết bị làm việc",
                price: 6290000,
                popularityScore: 67,
                image: "monitor",
                active: true,
            },
            {
                id: "p-mouse-mx",
                name: "Chuột không dây MX",
                brand: "Logitech",
                category: "Phụ kiện máy tính",
                price: 2190000,
                popularityScore: 83,
                image: "mouse",
                active: true,
            },
            {
                id: "p-speaker-mini",
                name: "Loa để bàn Mini",
                brand: "Sony",
                category: "Âm thanh",
                price: 1590000,
                popularityScore: 59,
                image: "speaker",
                active: true,
            },
            {
                id: "p-chair-flex",
                name: "Ghế làm việc Flex",
                brand: "Ergo",
                category: "Thiết bị làm việc",
                price: 4190000,
                popularityScore: 62,
                image: "chair",
                active: true,
            },
            {
                id: "p-watch-fit",
                name: "Đồng hồ theo dõi vận động",
                brand: "Fitbit",
                category: "Thiết bị đeo",
                price: 3590000,
                popularityScore: 75,
                image: "watch",
                active: true,
            },
            {
                id: "p-tablet-air",
                name: "Máy tính bảng Air",
                brand: "Apple",
                category: "Thiết bị di động",
                price: 14990000,
                popularityScore: 91,
                image: "tablet",
                active: true,
            },
            {
                id: "p-camera-web",
                name: "Camera họp trực tuyến",
                brand: "Logitech",
                category: "Thiết bị làm việc",
                price: 1890000,
                popularityScore: 54,
                image: "camera",
                active: false,
            },
        ],
        users: [
            {
                id: "u-203063",
                name: "Người dùng 203063",
                notes: "Mẫu hành vi có nhiều lượt xem và mua mới.",
                active: true,
            },
            {
                id: "u-office",
                name: "Khách làm việc tại nhà",
                notes: "Thường chuyển từ phụ kiện sang màn hình và ghế.",
                active: true,
            },
            {
                id: "u-audio",
                name: "Khách âm thanh",
                notes: "Hay quay lại nhóm tai nghe và loa.",
                active: true,
            },
        ],
        eventSessions: [
            {
                id: "s-seed-audio",
                label: "Chuỗi xem tai nghe",
                userId: "u-audio",
                createdAt: hoursAgo(18),
                events: [
                    {
                        id: "e-seed-1",
                        userId: "u-audio",
                        productId: "p-sony-headset",
                        behavior: "view",
                        timestamp: hoursAgo(18),
                        sessionId: "s-seed-audio",
                    },
                    {
                        id: "e-seed-2",
                        userId: "u-audio",
                        productId: "p-speaker-mini",
                        behavior: "cart",
                        timestamp: hoursAgo(16),
                        sessionId: "s-seed-audio",
                    },
                ],
            },
            {
                id: "s-seed-office",
                label: "Chuyển đổi bàn phím",
                userId: "u-office",
                createdAt: hoursAgo(5),
                events: [
                    {
                        id: "e-seed-3",
                        userId: "u-office",
                        productId: "p-keyboard-k2",
                        behavior: "view",
                        timestamp: hoursAgo(5),
                        sessionId: "s-seed-office",
                    },
                    {
                        id: "e-seed-4",
                        userId: "u-office",
                        productId: "p-keyboard-k2",
                        behavior: "cart",
                        timestamp: hoursAgo(4),
                        sessionId: "s-seed-office",
                    },
                    {
                        id: "e-seed-5",
                        userId: "u-office",
                        productId: "p-keyboard-k2",
                        behavior: "purchase",
                        timestamp: hoursAgo(3),
                        sessionId: "s-seed-office",
                    },
                ],
            },
        ],
        simulatorPresets: [
            {
                id: "preset-default",
                name: "Cân bằng",
                weights: { view: 1, cart: 3, purchase: 5 },
                recencyDecay: 0.055,
                categoryWeight: 1.9,
                brandWeight: 1.35,
                popularityWeight: 0.85,
                topK: 30,
                maskPurchased: true,
            },
            {
                id: "preset-category",
                name: "Ưu tiên danh mục",
                weights: { view: 1.2, cart: 2.6, purchase: 4.2 },
                recencyDecay: 0.04,
                categoryWeight: 3.1,
                brandWeight: 0.9,
                popularityWeight: 0.45,
                topK: 30,
                maskPurchased: true,
            },
        ],
        activePresetId: "preset-default",
        signedInUserId: null,
        draftSession: null,
    };

    function clone(value) {
        return JSON.parse(JSON.stringify(value));
    }

    function uid(prefix) {
        if (window.crypto && window.crypto.randomUUID) {
            return `${prefix}-${window.crypto.randomUUID()}`;
        }
        return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    }

    function read() {
        try {
            const stored = window.localStorage.getItem(STORAGE_KEY);
            return stored ? JSON.parse(stored) : clone(seed);
        } catch (error) {
            return clone(seed);
        }
    }

    function write(state) {
        window.localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
        return state;
    }

    function update(mutator) {
        const state = read();
        mutator(state);
        return write(state);
    }

    function createSession(userId, label) {
        return {
            id: uid("session"),
            label: label || "Phiên truy cập mới",
            userId,
            createdAt: new Date().toISOString(),
            events: [],
        };
    }

    function archiveDraft(state) {
        if (state.draftSession && state.draftSession.events.length) {
            state.eventSessions.unshift(clone(state.draftSession));
        }
    }

    function startUserSession(userId) {
        return update((state) => {
            archiveDraft(state);
            const user = state.users.find((item) => item.id === userId);
            state.signedInUserId = userId;
            state.draftSession = createSession(
                userId,
                `Phiên của ${user?.name || "người dùng"}`,
            );
        });
    }

    function closeUserSession() {
        return update((state) => {
            archiveDraft(state);
            state.signedInUserId = null;
            state.draftSession = null;
        });
    }

    function ensureDraft(userId) {
        const state = read();
        if (!state.draftSession || (userId && state.draftSession.userId !== userId)) {
            archiveDraft(state);
            state.draftSession = createSession(userId || state.users[0].id, "Phiên đang tạo");
            write(state);
        }
        return state.draftSession;
    }

    function addDraftEvent(productId, behavior) {
        return update((state) => {
            if (!state.draftSession) {
                state.draftSession = createSession(state.users[0].id, "Phiên đang tạo");
            }
            state.draftSession.events.unshift({
                id: uid("event"),
                userId: state.draftSession.userId,
                productId,
                behavior,
                timestamp: new Date().toISOString(),
                sessionId: state.draftSession.id,
            });
        });
    }

    function saveDraft(label) {
        return update((state) => {
            if (!state.draftSession || !state.draftSession.events.length) {
                return;
            }
            state.draftSession.label = label || state.draftSession.label;
            state.eventSessions.unshift(clone(state.draftSession));
            state.draftSession = createSession(state.draftSession.userId, "Phiên đang tạo");
        });
    }

    function resetDraft(userId) {
        return update((state) => {
            state.draftSession = createSession(userId || state.users[0].id, "Phiên đang tạo");
        });
    }

    function formatMoney(value) {
        return new Intl.NumberFormat("vi-VN").format(Number(value || 0)) + " đ";
    }

    function formatTime(value) {
        return new Intl.DateTimeFormat("vi-VN", {
            dateStyle: "short",
            timeStyle: "short",
        }).format(new Date(value));
    }

    function escapeHtml(value) {
        return String(value ?? "")
            .replaceAll("&", "&amp;")
            .replaceAll("<", "&lt;")
            .replaceAll(">", "&gt;")
            .replaceAll('"', "&quot;");
    }

    function toCsv(events) {
        const header = "user,product,behavior,timestamp,session";
        const rows = events.map((event) => [
            event.userId,
            event.productId,
            event.behavior,
            event.timestamp,
            event.sessionId,
        ].map((value) => `"${String(value).replaceAll('"', '""')}"`).join(","));
        return [header, ...rows].join("\n");
    }

    function download(filename, content, type) {
        const blob = new Blob([content], { type });
        const link = document.createElement("a");
        link.href = URL.createObjectURL(blob);
        link.download = filename;
        link.click();
        setTimeout(() => URL.revokeObjectURL(link.href), 0);
    }

    function productMap(state) {
        return Object.fromEntries(state.products.map((product) => [product.id, product]));
    }

    function userMap(state) {
        return Object.fromEntries(state.users.map((user) => [user.id, user]));
    }

    function defaultPreset() {
        return clone(seed.simulatorPresets[0]);
    }

    function restoreSeed() {
        return write(clone(seed));
    }

    window.DemoStore = {
        BEHAVIORS,
        addDraftEvent,
        closeUserSession,
        createSession,
        defaultPreset,
        download,
        ensureDraft,
        escapeHtml,
        formatMoney,
        formatTime,
        productMap,
        read,
        resetDraft,
        restoreSeed,
        saveDraft,
        startUserSession,
        toCsv,
        uid,
        update,
        userMap,
        write,
    };
}());
