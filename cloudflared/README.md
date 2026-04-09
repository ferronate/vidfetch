# Cloudflare Tunnel

This folder contains the local tunnel scaffolding for keeping the backend and data on this Windows machine while exposing the API through a public HTTPS hostname.

## Files

- `config.example.yml` is the template you copy to `config.yml` after you create your tunnel.
- `install-tunnel.ps1` checks that `cloudflared` is available and prepares the local config file.
- `run-tunnel.ps1` starts the tunnel using `config.yml`.

## Setup

1. Install `cloudflared`.
2. Authenticate with Cloudflare.
3. Create a named tunnel for this machine.
4. Copy `config.example.yml` to `config.yml` and fill in:
   - the tunnel UUID
   - the credentials file path
   - the public hostname you want to route to `http://localhost:8000`
5. Run `run-tunnel.ps1` while the backend is already running.

## Notes

- Keep `data/`, `index_store/`, and `models/` on this machine.
- Set `VIDFETCH_API_URL` for Streamlit if the backend is not on `http://localhost:8000`.
- Update `VF_CORS_ORIGINS` to include your Streamlit origin(s) and any additional allowed origins.
