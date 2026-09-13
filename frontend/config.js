// Where the static frontend finds the API.
//
// Empty = same origin (the page is served by the FastAPI backend itself, which
// also serves this file). On Vercel, set VERITAS_API_URL in the project and
// build-config.mjs overwrites this file at build time.
window.VERITAS_API = window.VERITAS_API || "";
