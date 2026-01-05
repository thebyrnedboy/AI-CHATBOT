Pre-Deploy / Post-Deploy Smoke Checklist

Pre-deploy (local)
- Run tests: `pytest -q`
- Start app locally and confirm it boots without tracebacks.
- Log in as a normal user and verify:
  - Dashboard loads.
  - Setup page shows embed snippet.
  - Helpdesk chatbot loads.
- Confirm protected routes:
  - `/admin/helpdesk` redirects or 403 for non-admin.
  - Admin account can access `/admin/helpdesk`.

Post-deploy (production)
- Open Railway logs and confirm startup has no tracebacks.
- Auth:
  - Log in and out successfully.
  - Dashboard loads and shows API key pill.
- Admin gating:
  - Non-admin cannot access `/admin/helpdesk`.
  - Admin can access `/admin/helpdesk` and see tickets.
- Widget:
  - Visit a page with the real embed snippet.
  - `/embed.js` loads with 200 (no Mixed Content errors).
  - Launcher opens panel; UI renders; send button visible.
- Helpdesk:
  - User can open a ticket.
  - Admin can reply.
  - User sees reply in `/helpdesk` or `/helpdesk/<ticket_id>`.
- Email (if SMTP/Brevo configured):
  - Ticket creation sends admin email.
  - Admin reply sends user email.
  - If email fails, app continues without crashing.
- Stripe (safe check only):
  - If webhook is configured, confirm `/stripe/webhook` responds 200 to a test event in Stripe test mode.

Notes
- If any step fails, capture the error and the Railway log snippet.
