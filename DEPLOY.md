# Deploying paraligilo

Single-server VPS deployment behind nginx + Let's Encrypt. Written for
`masawayh` (the author's VPS) but the pattern is portable.

Two deployment shapes covered:
- **Sub-path under an existing site** (e.g. `https://sypian.ski/paraligilo/`)
- **Dedicated subdomain** (e.g. `https://paraligilo.example.com/`)

Pick one. The systemd unit is the same; only the nginx config and the
`PARALIGILO_ROOT_PATH` env var differ.

## Prerequisites

- A VPS (Linux, x86_64 or ARM64)
- A DNS A/AAAA record pointing to it (e.g. `paraligilo.example.com`)
- `nginx` installed and running
- `certbot` with the nginx plugin
- [uv](https://github.com/astral-sh/uv) installed for the service user

## 1. Pull the code

```bash
sudo useradd -m -s /bin/bash paraligilo
sudo -iu paraligilo
git clone https://github.com/sypianski/paraligilo.git ~/app
cd ~/app
# uv installs Python + deps on first run — pre-warm:
./web/app.py --help 2>&1 | tail -1 || true
```

## 2. systemd service

`/etc/systemd/system/paraligilo.service`:

```ini
[Unit]
Description=paraligilo web (EPUB bitext alignment)
After=network.target

[Service]
Type=simple
User=paraligilo
WorkingDirectory=/home/paraligilo/app
Environment=PORT=8088
# Sub-path mount: set this to the URL prefix nginx routes here (with leading
# slash, no trailing slash). Leave empty for subdomain deployment.
Environment=PARALIGILO_ROOT_PATH=/paraligilo
Environment=PARALIGILO_JOBS=/var/lib/paraligilo/jobs
Environment=PARALIGILO_MAX_CONCURRENT=2
Environment=PARALIGILO_MAX_MB=30
Environment=PARALIGILO_TTL_HOURS=24
# Rate limits (per IP). Tighten if abused.
Environment=PARALIGILO_RL_ALIGN=10/hour
Environment=PARALIGILO_RL_STATUS=120/minute
Environment=PARALIGILO_RL_DOWNLOAD=60/hour

# Uncomment to offer a server-key fallback ("free trial" checkbox).
# This makes YOU pay for those users' OpenAI calls — set a monthly cap
# in https://platform.openai.com/account/limits before enabling.
# EnvironmentFile=/etc/paraligilo.env   # contains OPENAI_API_KEY=...

ExecStart=/home/paraligilo/.local/bin/uv run --script /home/paraligilo/app/web/app.py
Restart=on-failure
RestartSec=5

# Sandboxing
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths=/var/lib/paraligilo
ProtectKernelTunables=true
ProtectKernelModules=true
ProtectControlGroups=true
LockPersonality=true
MemoryDenyWriteExecute=false   # uv's interpreter cache needs it

[Install]
WantedBy=multi-user.target
```

Create the data directory and enable:

```bash
sudo mkdir -p /var/lib/paraligilo/jobs
sudo chown paraligilo:paraligilo /var/lib/paraligilo /var/lib/paraligilo/jobs
sudo systemctl daemon-reload
sudo systemctl enable --now paraligilo
sudo systemctl status paraligilo --no-pager
```

## 3. nginx reverse proxy

### Option A — sub-path under an existing site (e.g. sypian.ski/paraligilo)

Add to your existing `server { listen 443 ssl ... }` block for `sypian.ski`:

```nginx
# Allow larger uploads on this path specifically.
location /paraligilo/ {
    # Strip the prefix when forwarding: backend sees the request at /.
    # The trailing slash on proxy_pass is what makes the rewrite happen.
    proxy_pass http://127.0.0.1:8088/;

    proxy_http_version 1.1;
    proxy_set_header Host              $host;
    proxy_set_header X-Real-IP         $remote_addr;
    proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_set_header X-Forwarded-Host  $host;
    # Tell the app what URL prefix it lives behind (matches PARALIGILO_ROOT_PATH).
    proxy_set_header X-Forwarded-Prefix /paraligilo;

    # Big enough for 2 × 30MB EPUBs plus form overhead.
    client_max_body_size 80M;
    client_body_timeout 120s;
    send_timeout 120s;
    proxy_read_timeout 600s;
}
```

Reload nginx — no new certificate needed since you're under an existing
HTTPS host:

```bash
sudo nginx -t && sudo systemctl reload nginx
curl -I https://sypian.ski/paraligilo/
```

In the systemd unit, keep `PARALIGILO_ROOT_PATH=/paraligilo` so the app
generates absolute URLs like `/paraligilo/align` rather than `/align`.

### Option B — dedicated subdomain (e.g. paraligilo.example.com)

`/etc/nginx/sites-available/paraligilo.example.com`:

```nginx
server {
    listen 80;
    listen [::]:80;
    server_name paraligilo.example.com;

    # Allow ACME challenges; everything else redirects to HTTPS.
    location /.well-known/acme-challenge/ { root /var/www/html; }
    location / { return 301 https://$host$request_uri; }
}

server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name paraligilo.example.com;

    # ssl_certificate / ssl_certificate_key — filled in by certbot below.

    # Big enough for 2 × 30MB EPUBs plus form overhead.
    client_max_body_size 80M;
    client_body_timeout 120s;
    send_timeout 120s;

    location / {
        proxy_pass http://127.0.0.1:8088;
        proxy_http_version 1.1;
        proxy_set_header Host              $host;
        proxy_set_header X-Real-IP         $remote_addr;
        proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        # Status polling: 10-minute reads tolerated.
        proxy_read_timeout 600s;
    }
}
```

Enable + obtain certificate:

```bash
sudo ln -s /etc/nginx/sites-available/paraligilo.example.com \
          /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx

sudo certbot --nginx -d paraligilo.example.com \
     --redirect --email you@example.com --agree-tos --no-eff-email
```

certbot rewrites the `server { listen 443 ... }` block in place to fill in
`ssl_certificate` paths and adds an HTTP→HTTPS redirect. Verify with:

```bash
curl -I https://paraligilo.example.com/
```

For Option B, clear `PARALIGILO_ROOT_PATH=` in the systemd unit (empty
string) — the app should generate root-relative URLs like `/align`.

## 4. Sanity checks

```bash
# Health: home page loads.
curl -fI https://paraligilo.example.com/

# Submit a deterministic alignment from the CLI.
JOB=$(curl -sS -D - -o /dev/null -X POST https://paraligilo.example.com/align \
  -F "orig=@orig.epub" -F "trans=@trans.epub" -F "mode=deterministic" \
  | awk '/^[Ll]ocation:/ {print $2}' | tr -d '\r')
echo "Job URL: $JOB"

# Poll until done.
while :; do
  curl -sS "https://paraligilo.example.com${JOB}/status" | tee /dev/null
  echo
  sleep 2
done
```

## 5. Operational notes

### Server-key fallback (paid trial)

If you set `OPENAI_API_KEY` (via `EnvironmentFile=/etc/paraligilo.env`), the
form shows a checkbox letting users run AI mode with **your** key. Strongly
recommended before enabling:

1. Set a **monthly budget cap** at
   https://platform.openai.com/account/limits — e.g. `$20/month`. This is
   your only hard guarantee against runaway cost.
2. Drop `PARALIGILO_RL_ALIGN` to `3/hour` per IP — server-key abusers can
   still rotate IPs, but raise the noise floor.
3. Audit `journalctl -u paraligilo` weekly for unusual traffic patterns.

If you don't set the env var, the checkbox doesn't appear and only BYOK
users can use AI mode. **You pay zero** in that case.

### Storage growth

Jobs are pruned after `PARALIGILO_TTL_HOURS` (default 24h). With 100
jobs/day at ~100MB each (deterministic) or ~300MB (AI), expect a working
set around 10–30GB. Watch `df -h /var/lib/paraligilo`.

If disk fills, the upload handler will fail mid-stream. Defend with
`/var/lib/paraligilo` on a dedicated volume or set
`PARALIGILO_TTL_HOURS=6`.

### Logs

```bash
journalctl -u paraligilo -f          # live tail
journalctl -u paraligilo --since '1 hour ago'
```

The app logs to stderr (request lines + tracebacks). Per-job pipeline
output lands in `/var/lib/paraligilo/jobs/<id>/run.log` — visible to the
user via the download endpoint, useful for self-diagnosis.

### Backups

Nothing in paraligilo is worth backing up — every job artifact is
regenerable from the source EPUBs. Don't bother.

### Updating

```bash
sudo -iu paraligilo
cd ~/app
git pull
sudo systemctl restart paraligilo
```

If a stage script's CLI signature changed, `git pull` plus a restart picks
it up — uv re-resolves dependencies on demand. No build step.

### Tuning rate limits

All four limits read from env vars at process start. To change without
editing the unit file:

```bash
sudo systemctl edit paraligilo
# add an [Service] override:
#   Environment=PARALIGILO_RL_ALIGN=20/hour
sudo systemctl restart paraligilo
```

Values use [`limits` library syntax](https://limits.readthedocs.io/en/stable/quickstart.html#rate-limit-string-notation):
`<count>/<period>` where period is one of `second`, `minute`, `hour`, `day`.

### Behind multiple proxies

If nginx sits behind another reverse proxy (Cloudflare, etc.), the
`X-Forwarded-For` header is a comma-separated chain. Our `_client_ip()`
picks the **leftmost** entry, which is the real client *if* you trust your
proxy chain not to allow spoofing. Cloudflare strips spoofed XFF by
default; bare nginx in front of paraligilo with no upstream proxy also
fine. If you have an untrusted edge, replace `_client_ip` with a more
careful parser.

## 6. Decommissioning

```bash
sudo systemctl disable --now paraligilo
sudo rm /etc/systemd/system/paraligilo.service
sudo rm -rf /var/lib/paraligilo
sudo userdel -r paraligilo
sudo rm /etc/nginx/sites-enabled/paraligilo.example.com
sudo rm /etc/nginx/sites-available/paraligilo.example.com
sudo nginx -t && sudo systemctl reload nginx
sudo certbot delete --cert-name paraligilo.example.com
```
