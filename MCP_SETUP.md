# Running with Claude Desktop (no API key, no cost)

The MCP server exposes the same four tools to Claude Desktop. The Claude app
supplies the model, so there is no API key and nothing to pay for.

## 1. Config file

macOS:  `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "peptide": {
      "command": "/ABSOLUTE/PATH/TO/peptide-agent/.venv/bin/python",
      "args": ["-m", "peptide_agent.mcp_server"],
      "cwd": "/ABSOLUTE/PATH/TO/peptide-agent",
      "env": {
        "PYTHONPATH": "/ABSOLUTE/PATH/TO/peptide-agent",
        "PEPTIDE_DATASET": "/ABSOLUTE/PATH/TO/peptide-agent/data/peptides.csv"
      }
    }
  }
}
```

Replace all four paths. `pwd` in the project folder prints the right value.
Relative paths and `~` do not work here — Claude Desktop does not expand them.

`PYTHONPATH` is required, not optional. Running `python -m peptide_agent...`
from a terminal works because Python puts the current directory on the import
path; launched by Claude Desktop it does not, and the server dies with
`ModuleNotFoundError: No module named 'peptide_agent'`. Setting `cwd` alone
does not fix this.

## 2. Restart

Quit Claude Desktop completely (Cmd+Q, not just closing the window) and
reopen it. The config is read only at startup.

## 3. Check it worked

Ask: *"Which peptides in my dataset are strongly amphipathic?"*

It should call `dataset_summary`, then `query_dataset`.

## Troubleshooting

Verify the server starts on its own first:

```bash
.venv/bin/python -c "from peptide_agent import mcp_server; print('ok')"
```

If that fails, the config will fail too. Common causes:

| Symptom | Cause |
|---|---|
| No tools appear | Wrong path in the config, or Claude Desktop not fully quit |
| `No module named peptide_agent` | `PYTHONPATH` missing from `env` (most common) |
| `No module named mcp` | Used system Python instead of `.venv/bin/python` |
| Tools appear, dataset questions fail | `PEPTIDE_DATASET` path wrong, or the CSV was never fetched |

## What this mode does and does not give you

Gives you: a working demo, all four tools, no cost.

Does not give you: the eval suite. `evals/run_evals.py` calls the agent loop
programmatically and needs an API key. In MCP mode Claude Desktop *is* the
loop, so `agent.py` is unused.

Because the client supplies its own system prompt, `agent.SYSTEM_PROMPT`
never reaches the model here. The equivalent rules live in
`mcp_server.INSTRUCTIONS` and in each tool's docstring. Change a rule in one
place and change it in the other.
