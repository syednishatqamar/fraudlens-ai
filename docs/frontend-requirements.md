# Frontend and dashboard requirements

Requirement from the project owner: the frontend and analyst dashboard must look and behave like an
enterprise-grade, professional product, not a demo page. This file is the checklist we build and review against.
Status: planned; nothing built yet.

## Users and core workflow
- Analyst: works the queue of REVIEW/BLOCK cases, reads the evidence, and APPROVES, REJECTS (with a different action and a comment)
  or ESCALATES (with a comment). Decisions are final and audited.
- Senior analyst: handles ESCALATED cases.
- Manager (later): sees volume, agreement rate, overrides, and workload.

## Product quality bar
- Consistent design system: one component library, one colour/spacing/typography scale, no one-off styling.
- Dense, scannable data tables: sorting, filters (band, status, amount, date), pagination, sticky headers, risk shown by band and probability.
- Case view built for decisions: the model's recommendation and why (reasons with direction), the cost explanation, card history,
  merchant record, comparable cases, the analyst note and its source (LLM or template), and the three actions in one place.
- Clear states everywhere: loading, empty, error and retry, success confirmation, and confirmation for irreversible decisions.
- Keyboard-first for analysts (next case, approve, open filters) with visible shortcut hints.
- Accessibility: WCAG 2.1 AA contrast, full keyboard use, labelled controls, visible focus, not colour alone for risk.
- Responsive from tablet to wide desktop; works in light and dark themes.
- Trust and safety cues: masked card numbers only, "placeholder costs" and "synthetic data" labels, who decided what and when.
- Monitoring views (later): queue size, decisions per day, agreement and override rate, calibration check.

## Engineering bar
- TypeScript, typed API client generated from or checked against the FastAPI schema.
- Component and integration tests, plus a few end-to-end tests of the decision flow; CI runs lint, type check and tests.
- No secrets in the frontend; API key or login handled server side; CORS configured per environment.
- Containerised build served by a web server, wired into Docker Compose with the API and database.

## Candidate stack (to confirm)
React + TypeScript + Vite, a component library (for example Ant Design, MUI or shadcn/ui with Tailwind), TanStack Table and Query,
Recharts. Final choice depends on where the project is run and demoed, and whether Node tooling runs on the owner's machine.

## Open questions
- Authentication for the demo: API key only, or a simple login with roles?
- Where it will be demoed: local, GitHub Codespaces, or a free hosting tier.
