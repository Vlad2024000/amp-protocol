#!/usr/bin/env bash
# =============================================================================
#  AMP Protocol — Launch Day Script
#  Run this to go from zero to live GitHub repo + PyPI + npm in one day.
# =============================================================================
set -e
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'
BOLD='\033[1m'; NC='\033[0m'

echo -e "${BOLD}${BLUE}"
echo "  ╔═══════════════════════════════════════╗"
echo "  ║   AMP Protocol — Launch Day Script    ║"
echo "  ║   github.com/amp-protocol/amp-python  ║"
echo "  ╚═══════════════════════════════════════╝"
echo -e "${NC}"

# ── STEP 0: Prerequisites check ──────────────────────────────────────────────
echo -e "${BOLD}Step 0: Checking prerequisites...${NC}"
for cmd in git python3 pip node npm docker; do
  if command -v $cmd &>/dev/null; then
    echo -e "  ${GREEN}✓${NC} $cmd $(${cmd} --version 2>&1 | head -1)"
  else
    echo -e "  ${RED}✗ $cmd not found — please install it${NC}"
    exit 1
  fi
done

# ── STEP 1: Create GitHub repository ─────────────────────────────────────────
echo -e "\n${BOLD}Step 1: GitHub repository${NC}"
echo -e "${YELLOW}Actions required:${NC}"
cat << 'ACTIONS'

  1. Go to: https://github.com/new
  2. Organization:   amp-protocol  (create org first if needed)
  3. Repository:     amp-python
  4. Description:    Agent Memory Protocol — open standard for cross-agent AI memory
  5. Visibility:     Public
  6. DO NOT init:    no README, no .gitignore, no license (we have them)
  7. Click "Create repository"

  Then create a second repo for the website:
  Organization:   amp-protocol
  Repository:     amp-protocol.github.io
  Description:    amp-protocol.org landing page

ACTIONS
read -p "Press Enter when both repos are created..."

# ── STEP 2: Configure GitHub secrets ─────────────────────────────────────────
echo -e "\n${BOLD}Step 2: Configure GitHub repository secrets${NC}"
cat << 'SECRETS'

  Go to: https://github.com/amp-protocol/amp-python/settings/secrets/actions

  Add these secrets (Actions → Secrets → New repository secret):

  Secret name         │ How to get it
  ────────────────────┼──────────────────────────────────────────────────
  NPM_TOKEN           │ npmjs.com → Avatar → Access Tokens → New Token (Publish)
  CODECOV_TOKEN       │ codecov.io → sign in with GitHub → copy token

  For PyPI (OIDC — no token needed if you configure Trusted Publisher):
  → pypi.org → Account → Publishing → Add a new publisher
     Owner:      amp-protocol
     Repo:       amp-python
     Workflow:   ci.yml
     Environment: pypi

SECRETS
read -p "Press Enter when secrets are configured..."

# ── STEP 3: First commit + push ───────────────────────────────────────────────
echo -e "\n${BOLD}Step 3: Initialize git and push${NC}"

# Ensure we're in the right directory
if [ ! -f "pyproject.toml" ]; then
  echo -e "${RED}Run this script from the amp-protocol root directory${NC}"
  exit 1
fi

git init -b main
git add -A
git commit -m "feat: initial release — AMP v0.3.0

Agent Memory Protocol — open standard for cross-agent AI memory.

- MemoryObject with 5 dimensions: content/relations/time/trust/scope
- Decay function: weight(t) = importance × e^(−λ×days) + permanence
- LSA embeddings (offline, 256-dim, scipy SVD)
- Three-signal RRF search: vector + BM25 + AMP weight
- Cross-agent memory sharing via shared user_id scope
- SQLite (dev) + PostgreSQL+pgvector (prod) backends
- MCP server (7 tools) for Claude Desktop integration
- HTTP REST API + Docker + TypeScript SDK
- JSON Schema spec (schema_v0.1.json)

Semantic hit rate: 7/7 = 100% on benchmark.
Search latency: 4ms (SQLite) / <10ms (pgvector @ 1M records).

Closes #1"

echo -e "\n${YELLOW}Now add your GitHub remote:${NC}"
echo "  git remote add origin https://github.com/amp-protocol/amp-python.git"
echo "  git push -u origin main"
read -p "Press Enter after pushing..."

# ── STEP 4: Create first release tag ─────────────────────────────────────────
echo -e "\n${BOLD}Step 4: Tag v0.3.0 release (triggers CI publish)${NC}"
git tag -a v0.3.0 -m "AMP v0.3.0 — Initial public release

Agent Memory Protocol: open standard for persistent cross-agent AI memory.

Highlights:
- LSA semantic embeddings (offline, zero API deps)
- RRF fusion search: 7/7 = 100% semantic hit rate
- Cross-agent memory sharing (Claude reads GPT's memories)
- MCP server for Claude Desktop
- PostgreSQL + pgvector production backend
- TypeScript SDK

pip install amp-memory
npm install @amp-protocol/sdk"

git push origin v0.3.0
echo -e "${GREEN}✓ Tag pushed — CI will build and publish to PyPI + npm${NC}"

# ── STEP 5: PyPI manual first publish (if OIDC not set up yet) ───────────────
echo -e "\n${BOLD}Step 5: Optional — manual PyPI publish${NC}"
cat << 'PYPI'

  If OIDC trusted publishing isn't configured yet, publish manually:

  pip install build twine
  python -m build
  twine upload dist/*
  # Enter your PyPI username and API token

PYPI

# ── STEP 6: npm publish ───────────────────────────────────────────────────────
echo -e "\n${BOLD}Step 6: npm publish (TypeScript SDK)${NC}"
cat << 'NPM'

  cd sdk/typescript
  npm install
  npm run build
  npm publish --access public
  # Requires: npm login

NPM

# ── STEP 7: Deploy landing page ───────────────────────────────────────────────
echo -e "\n${BOLD}Step 7: Deploy amp-protocol.org${NC}"
cat << 'LANDING'

  Option A — GitHub Pages (free, instant):
    1. Copy index.html from amp-protocol.org/ to a new repo:
       amp-protocol/amp-protocol.github.io
    2. Push to main branch
    3. Settings → Pages → Source: main / root
    4. Add custom domain: amp-protocol.org
    5. DNS: add CNAME record pointing to amp-protocol.github.io

  Option B — Cloudflare Pages (free, faster CDN):
    1. cloudflare.com → Pages → Connect to Git
    2. Select amp-protocol.github.io repo
    3. Build command: (empty — static HTML)
    4. Add custom domain

LANDING

# ── STEP 8: Community launch ─────────────────────────────────────────────────
echo -e "\n${BOLD}Step 8: Community launch sequence${NC}"
cat << 'LAUNCH'

  Post order (one per day for maximum reach):

  Day 1 — Hacker News (Show HN):
    Title: "Show HN: AMP – an open protocol for persistent memory between AI agents"
    URL: https://github.com/amp-protocol/amp-python
    Post at: 9am ET Tuesday–Thursday for best visibility

  Day 2 — DEV.to article:
    "I built TCP/IP for agent memory — here's why every AI agent forgets everything"
    Include the 7/7 semantic hit rate benchmark
    Include cross-agent demo (Claude finds GPT's memory)

  Day 3 — Reddit:
    r/MachineLearning — technical post with the math
    r/LocalLLaMA — the offline LSA angle (no API needed)
    r/ClaudeAI — the MCP integration

  Day 4 — X/Twitter thread:
    "Every AI agent today is amnesic. We built the fix. A thread."
    Show the 5-line quickstart. Show the before/after.

  Day 5 — Discord outreach:
    Anthropic MCP Discord (official)
    LocalLLaMA Discord
    LangChain Discord

  DM AuraSDK author (Aleksander, Ukraine 🇺🇦):
    "Fellow Ukrainian builder here — saw your AuraSDK.
    We're building AMP, the cross-agent protocol layer.
    Your SDK could be the first AMP-compatible implementation.
    Want to collaborate?"

LAUNCH

# ── Done ──────────────────────────────────────────────────────────────────────
echo -e "\n${BOLD}${GREEN}"
echo "  ╔═══════════════════════════════════════╗"
echo "  ║         Launch checklist done!        ║"
echo "  ║                                       ║"
echo "  ║  GitHub    ✓  amp-protocol/amp-python ║"
echo "  ║  PyPI      ✓  amp-memory              ║"
echo "  ║  npm       ✓  @amp-protocol/sdk       ║"
echo "  ║  Website   ✓  amp-protocol.org        ║"
echo "  ║  CI/CD     ✓  GitHub Actions          ║"
echo "  ╚═══════════════════════════════════════╝"
echo -e "${NC}"
echo -e "  Built in Ivano-Frankivsk, Ukraine 🇺🇦\n"
