#!/usr/bin/env bash
#
# Provisions the Music Recommendation System on a headless Ubuntu 26.04 host:
# uv + .venv + dependencies, nginx serving the static pages and proxying /api
# to the FastAPI service, plus a systemd unit for server.py.
#
# Usage: sudo ./install.sh
#
set -euo pipefail

APP_NAME="music-recommender"
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEB_ROOT="/var/www/${APP_NAME}/html"
SERVICE_FILE="/etc/systemd/system/${APP_NAME}.service"
NGINX_SITE="/etc/nginx/sites-available/${APP_NAME}"
UV_BIN="/usr/local/bin/uv"

SWAP_FILE="${SWAP_FILE:-/swapfile}"
SWAP_SIZE_MB="${SWAP_SIZE_MB:-1024}"

PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
API_HOST="${API_HOST:-127.0.0.1}"
#API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8000}"
# The service (and therefore the .venv) runs as the human who invoked sudo, so
# the checkout stays writable for them.
SERVICE_USER="${SERVICE_USER:-${SUDO_USER:-root}}"

HTML_FILES=(
    index.html
    find_artists_by_tag.html
    find_similar_artists.html
    find_similar_users.html
    recommend_artists_content.html
    recommend_artists_cf.html
)

log() { printf '\n\033[1;32m==>\033[0m %s\n' "$*"; }
die() { printf '\n\033[1;31mError:\033[0m %s\n' "$*" >&2; exit 1; }

as_service_user() {
    if [[ "$SERVICE_USER" == "root" ]]; then
        env "$@"
    else
        sudo -u "$SERVICE_USER" -H env "$@"
    fi
}

# --- sanity checks -----------------------------------------------------------

[[ $EUID -eq 0 ]] || die "run with sudo: sudo $0"
id -u "$SERVICE_USER" >/dev/null 2>&1 || die "user '$SERVICE_USER' does not exist"

for f in server.py requirements.txt "${HTML_FILES[@]}"; do
    [[ -f "$APP_DIR/$f" ]] || die "missing $f in $APP_DIR"
done

# --- swap --------------------------------------------------------------------

if [[ -n "$(swapon --show=NAME --noheadings)" ]]; then
    log "Swap already enabled, leaving it as is"
    swapon --show
else
    log "Enabling ${SWAP_SIZE_MB}MB of swap at $SWAP_FILE"

    avail_mb="$(df -Pm "$(dirname "$SWAP_FILE")" | awk 'NR==2 {print $4}')"
    (( avail_mb > SWAP_SIZE_MB + 512 )) \
        || die "only ${avail_mb}MB free on $(dirname "$SWAP_FILE"), need $((SWAP_SIZE_MB + 512))MB"

    # Nothing is swapped on, so a file left behind by an earlier failed run is
    # unused and safe to replace.
    rm -f "$SWAP_FILE"
    fallocate -l "${SWAP_SIZE_MB}M" "$SWAP_FILE" 2>/dev/null \
        || dd if=/dev/zero of="$SWAP_FILE" bs=1M count="$SWAP_SIZE_MB" status=none
    chmod 600 "$SWAP_FILE"
    mkswap "$SWAP_FILE" >/dev/null
    swapon "$SWAP_FILE"

    grep -qs "^${SWAP_FILE}[[:space:]]" /etc/fstab \
        || printf '%s none swap sw 0 0\n' "$SWAP_FILE" >> /etc/fstab

    swapon --show
fi

# --- system packages ---------------------------------------------------------

log "Installing system packages"
export DEBIAN_FRONTEND=noninteractive
# With a kernel upgrade pending, needrestart's post-install dialog blocks
# forever on a host with no TTY, so suspend it and keep existing configs.
export NEEDRESTART_MODE=a NEEDRESTART_SUSPEND=1
# unattended-upgrades often holds the dpkg lock on a freshly booted host, so
# wait for it rather than aborting the whole install.
APT_OPTS=(-o DPkg::Lock::Timeout=600)
apt-get update -qq "${APT_OPTS[@]}"
apt-get install -y -qq --no-install-recommends \
    "${APT_OPTS[@]}" -o Dpkg::Options::=--force-confold \
    ca-certificates curl nginx git

# --- clone repo ---
git clone https://github.com/Areso/music-recommendation-system.git

# --- uv ----------------------------------------------------------------------

if [[ -x "$UV_BIN" ]]; then
    log "uv already present: $("$UV_BIN" --version)"
else
    log "Installing uv into /usr/local/bin"
    curl -fsSL https://astral.sh/uv/install.sh \
        | env UV_INSTALL_DIR=/usr/local/bin sh
    [[ -x "$UV_BIN" ]] || die "uv installation failed"
fi

# --- virtualenv + dependencies -----------------------------------------------

log "Creating .venv (Python ${PYTHON_VERSION}) and installing requirements"
chown "$SERVICE_USER" "$APP_DIR"
# --clear so the run succeeds whether or not a previous .venv survived the
# deployment's directory cleanup; uv rebuilds it from its cache in seconds.
as_service_user "$UV_BIN" venv --clear --python "$PYTHON_VERSION" "$APP_DIR/.venv"
as_service_user "$UV_BIN" pip install --python "$APP_DIR/.venv/bin/python" \
    -r "$APP_DIR/requirements.txt"

# server.py starts uvicorn itself, but the unit calls uvicorn directly so that
# --host/--port stay configurable from here.
[[ -x "$APP_DIR/.venv/bin/uvicorn" ]] || die "uvicorn missing from the venv"

# --- static files ------------------------------------------------------------

log "Publishing HTML to $WEB_ROOT"
install -d -o www-data -g www-data "$WEB_ROOT"
for f in "${HTML_FILES[@]}"; do
    # The pages ship with an absolute API URL for local development; behind
    # nginx they must call the same-origin /api path instead.
    sed "s#http://127\.0\.0\.1:8000/api#/api#g" "$APP_DIR/$f" > "$WEB_ROOT/$f"
    chown www-data:www-data "$WEB_ROOT/$f"
    chmod 0644 "$WEB_ROOT/$f"
done

# --- systemd service ---------------------------------------------------------

log "Registering systemd unit $SERVICE_FILE"
cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=Music Recommendation System API (FastAPI)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${SERVICE_USER}
# server.py resolves its artifacts (tfidf_matrix.npz, clean/, biblioteca/)
# relative to the current directory.
WorkingDirectory=${APP_DIR}
ExecStart=${APP_DIR}/.venv/bin/uvicorn server:app --host ${API_HOST} --port ${API_PORT}
Restart=on-failure
RestartSec=5
# Loading the TF-IDF matrix and vectorizer takes a while on a small box.
TimeoutStartSec=300
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=full
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable "$APP_NAME"
systemctl restart "$APP_NAME"

# --- nginx -------------------------------------------------------------------

log "Configuring nginx"
cat > "$NGINX_SITE" <<EOF
server {
    listen 80 default_server;
    listen [::]:80 default_server;
    server_name _;

    root ${WEB_ROOT};
    index index.html;

    location / {
        try_files \$uri \$uri/ =404;
    }

    location /api/ {
        proxy_pass http://${API_HOST}:${API_PORT};
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        # Cosine similarity over the full matrix can take a few seconds.
        proxy_read_timeout 120s;
    }
}
EOF

ln -sfn "$NGINX_SITE" "/etc/nginx/sites-enabled/${APP_NAME}"
rm -f /etc/nginx/sites-enabled/default
nginx -t
systemctl enable nginx
systemctl reload nginx

if command -v ufw >/dev/null 2>&1 && ufw status | grep -q "^Status: active"; then
    log "Opening HTTP in ufw"
    ufw allow 'Nginx HTTP'
fi

# --- smoke test --------------------------------------------------------------

log "Waiting for the API to finish loading its artifacts"
for _ in $(seq 1 60); do
    if curl -fsS "http://127.0.0.1/api/autocomplete?q=rock&limit=1" >/dev/null 2>&1; then
        break
    fi
    systemctl is-active --quiet "$APP_NAME" \
        || die "$APP_NAME failed to start; check: journalctl -u $APP_NAME -n 50"
    sleep 5
done

if curl -fsS "http://127.0.0.1/api/autocomplete?q=rock&limit=1" >/dev/null 2>&1; then
    log "Done. Site is live on http://<server-ip>/"
else
    die "API did not answer through nginx; check: journalctl -u $APP_NAME -n 50"
fi

cat <<EOF

  Service:    systemctl status ${APP_NAME}
  Logs:       journalctl -u ${APP_NAME} -f
  HTML root:  ${WEB_ROOT}   (re-run this script after editing the pages)
  API:        http://${API_HOST}:${API_PORT} (localhost only, proxied at /api)

EOF
