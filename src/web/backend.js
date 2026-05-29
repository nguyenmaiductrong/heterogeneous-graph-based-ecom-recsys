(function () {
    "use strict";

    const API_BASE = "";

    async function request(path, options) {
        const response = await fetch(`${API_BASE}${path}`, {
            headers: { "Content-Type": "application/json", ...(options?.headers || {}) },
            ...options,
        });
        if (!response.ok) {
            throw new Error(`${response.status} ${response.statusText}`);
        }
        return response.json();
    }

    async function health() {
        return request("/api/health");
    }

    async function bootstrap() {
        return request("/api/bootstrap");
    }

    async function createUser(user) {
        return request("/api/users", {
            method: "POST",
            body: JSON.stringify(user),
        });
    }

    async function createEvent(event) {
        return request("/api/events", {
            method: "POST",
            body: JSON.stringify(event),
        });
    }

    async function recommend(payload) {
        return request("/api/recommendations", {
            method: "POST",
            body: JSON.stringify(payload),
        });
    }

    async function hydrateStore(Store) {
        try {
            const existing = Store.read();
            const state = await bootstrap();
            const presets = new Map();
            [...(state.simulatorPresets || []), ...(existing.simulatorPresets || [])].forEach((preset) => {
                presets.set(preset.id, preset);
            });
            state.simulatorPresets = [...presets.values()];
            if (existing.activePresetId && state.simulatorPresets.some((preset) => preset.id === existing.activePresetId)) {
                state.activePresetId = existing.activePresetId;
            }
            Store.write(state);
            return { ok: true, state };
        } catch (error) {
            return { ok: false, error };
        }
    }

    window.DemoBackend = {
        health,
        bootstrap,
        createUser,
        createEvent,
        recommend,
        hydrateStore,
    };
}());
