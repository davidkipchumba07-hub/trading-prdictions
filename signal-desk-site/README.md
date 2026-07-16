# Signal Desk — build & deploy guide

A full-stack site: a Flask API that runs the technical-analysis / backtest
engine, and a static frontend that calls it. This is a step-by-step guide to
take it from these files to a live public website.

```
site/
  backend/
    app.py              <- Flask API
    signal_engine.py    <- indicators, scoring, backtest engine
    requirements.txt
    Dockerfile
  frontend/
    index.html           <- the whole frontend, one file
  README.md               <- you are here
```

---

## Step 1 — Run it locally first

Always confirm it works on your machine before deploying anything.

```bash
cd backend
pip install -r requirements.txt --break-system-packages
python app.py
# -> running on http://localhost:5000
```

In a second terminal:

```bash
cd frontend
python3 -m http.server 8080
# -> open http://localhost:8080 in your browser
```

The frontend defaults to `http://localhost:5000` for the API. You should see
live indicator values, a signal, and a backtest chart using demo (synthetic)
data. Switch the dropdown to "Live" and type a symbol like `EURUSD=X` to pull
real data via yfinance.

If nothing loads, open the browser console (F12) — the status line under the
indicator panel will also show the actual error.

---

## Step 2 — Put the backend somewhere public

You need a host that runs Python continuously (not a static host — the API
does real computation per request). Pick one:

### Option A: Render.com (easiest, free tier available)
1. Push the `backend/` folder to a GitHub repo.
2. On Render: **New > Web Service**, connect the repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn --bind 0.0.0.0:$PORT app:app`
5. Deploy. Render gives you a URL like `https://signal-desk-api.onrender.com`.

### Option B: Railway.app (similarly simple)
1. Push to GitHub, `New Project > Deploy from repo` on Railway.
2. Railway auto-detects the `Dockerfile` and builds it.
3. It assigns a public URL automatically.

### Option C: Your own VPS (DigitalOcean, Hetzner, etc.) — more control
```bash
# On the server, with Docker installed:
git clone <your-repo>
cd site/backend
docker build -t signal-desk-api .
docker run -d -p 5000:5000 --restart unless-stopped signal-desk-api
```
Then put nginx in front of it for HTTPS (see Step 4).

Whichever you choose, **test the deployed API directly** before touching the
frontend:
```bash
curl https://your-backend-url/api/health
curl "https://your-backend-url/api/signal?demo=true"
```

---

## Step 3 — Point the frontend at your live backend

Open `frontend/index.html` and find this line near the top of the `<script>`
block:

```js
const API_BASE = window.SIGNAL_API_BASE || "http://localhost:5000";
```

Add one line right before it to hardcode your deployed backend URL:

```js
window.SIGNAL_API_BASE = "https://signal-desk-api.onrender.com";
```

(Alternatively, set `window.SIGNAL_API_BASE` in a small inline `<script>`
tag before this file loads — useful if you want different URLs for staging
vs. production without editing this file each time.)

---

## Step 4 — Put the frontend somewhere public

The frontend is one static HTML file, so any static host works:

### Option A: Netlify / Vercel (drag-and-drop, free)
Drag the `frontend/` folder onto Netlify's deploy page, or connect the repo.
Done — you get a URL immediately, and can attach a custom domain in settings.

### Option B: GitHub Pages
1. Push `frontend/index.html` to a repo (rename to keep it at the repo root,
   or use a `docs/` folder if that's your Pages source).
2. Repo Settings > Pages > enable, pick the branch/folder.

### Option C: Same VPS as the backend, via nginx
```nginx
server {
    listen 443 ssl;
    server_name yourdomain.com;

    ssl_certificate     /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;

    location / {
        root /var/www/signal-desk/frontend;
        index index.html;
    }

    location /api/ {
        proxy_pass http://localhost:5000/api/;
        proxy_set_header Host $host;
    }
}
```
Get the SSL cert with `certbot --nginx -d yourdomain.com` (free, via Let's
Encrypt). This setup also lets the frontend call `/api/...` on the same
domain, so you can leave `API_BASE` as an empty string instead of a full URL.

---

## Step 5 — Buy and connect a domain (optional but recommended)

1. Buy a domain (Namecheap, Google Domains, Porkbun — any registrar).
2. Point its DNS at your host:
   - Netlify/Vercel/Render: add a `CNAME` record to the URL they give you,
     following their "custom domain" instructions.
   - Your own VPS: add an `A` record pointing to the server's IP address.
3. Wait for DNS to propagate (minutes to a few hours), then enable HTTPS —
   Netlify/Vercel/Render do this automatically; on a VPS use certbot as above.

---

## Step 6 — Keep it running and know its limits

- **Free tiers sleep.** Render's free web services spin down after
  inactivity and take ~30s to wake on the next request — fine for a demo,
  not for something you're charging for. Upgrade to a paid tier for
  always-on behavior.
- **Rate limits.** yfinance is a free, unofficial data source and can rate-limit
  or block if you call it too often. For a real product, budget for a paid
  market-data API (e.g. Twelve Data, Polygon.io, Alpha Vantage) once you have
  real users.
- **No trade execution.** This site only analyzes and displays — it never
  places trades. If you ever add real order execution, that's a materially
  different (and more heavily regulated) product; treat it as a separate
  project with its own legal review.
- **Compliance.** Binary options are banned or restricted for retail traders
  in several jurisdictions. If you monetize this or target specific
  countries, check local financial-promotion rules before publishing —
  a "not financial advice" disclaimer (already on the page) helps, but does
  not substitute for actual legal review if you start taking payment for
  signals.

---

## Extending it later

- **Add a database** (Postgres via Render/Railway/Supabase) to log every
  signal generated, so you can show a real historical track record instead
  of only live/demo runs.
- **Add the Telegram/Discord bot next** — it can call the same
  `/api/signal` endpoint on a timer and push alerts, reusing this backend
  with no changes.
- **Add auth** (e.g. Auth0, Clerk, or a simple email+password with Flask-Login)
  if you want to gate the tool behind a signup or paywall.
