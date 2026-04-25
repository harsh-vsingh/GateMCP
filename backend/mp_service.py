import json
from pathlib import Path
from langchain_mcp_adapters.client import MultiServerMCPClient
import httpx
from pydantic import BaseModel, Field
from typing import Optional, List
import shutil 

BASE_DIR = Path(__file__).parent.parent.resolve()
MCP_CONFIG_DIR = BASE_DIR / "agent_workspace" / "mcp_servers"
MCP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

class MCPServerSchema(BaseModel):
    transport: str
    command: Optional[str] = None
    args: Optional[List[str]] = None
    url: Optional[str] = None

    @classmethod
    def validate_config(cls, config: dict):
        transport = config.get("transport")

        if transport not in {"stdio", "sse", "http", "streamable_http", "streamable-http"}:
            raise ValueError("Unsupported transport")

        # Normalize
        transport = transport.replace("-", "_")

        # Stdio rules
        if transport == "stdio":
            cmd = config.get("command")
            if not isinstance(cmd, str) or not cmd.strip():
                raise ValueError("Stdio requires valid 'command'")

            if "args" in config:
                args = config["args"]
                if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
                    raise ValueError("'args' must be list of strings")

            if "url" in config:
                raise ValueError("Stdio cannot have 'url'")

        # Remote rules
        else:
            url = config.get("url")
            if not isinstance(url, str) or not url.strip():
                raise ValueError(f"{transport} requires valid 'url'")

            if not url.startswith(("http://", "https://")):
                raise ValueError("URL must start with http/https")

        return True

class MCPService:
    def __init__(self):
        self.current_client = None

    def save_server_config(self, name: str, config: dict):
        """Saves a single MCP configuration to a JSON file."""
        file_path = MCP_CONFIG_DIR / f"{name}.json"
        with open(file_path, 'w') as f:
            json.dump(config, f, indent=4)

    def delete_server_config(self, name: str):
        """Removes a server configuration file."""
        file_path = MCP_CONFIG_DIR / f"{name}.json"
        if file_path.exists():
            file_path.unlink()

    def list_servers(self) -> dict:
        """Reads all JSON files in the directory and returns a unified config dict."""
        servers = {}
        for file in MCP_CONFIG_DIR.glob("*.json"):
            try:
                with open(file, 'r') as f:
                    servers[file.stem] = json.load(f)
            except Exception:
                continue
        
        # Ensure the built-in tls toolset is always present
        if "tls" not in servers:
            default_tls = {
                "transport": "stdio",
                "command": "/usr/bin/uv",
                "args": ["run", "fastmcp", "run", str(BASE_DIR / "backend" / "tools.py")]
            }
            self.save_server_config("tls", default_tls)
            servers["tls"] = default_tls
            
        return servers

    async def refresh_client(self, config_dict: dict):
        """Kills existing subprocesses and initializes a fresh client."""
        if self.current_client:
            try:
                await self.current_client.close()
            except Exception:
                pass
        
        self.current_client = MultiServerMCPClient(config_dict)
        return self.current_client
    
    async def validate_server(self, config_dict: dict):
        """Pre-save validation to prevent graph crashes."""
        # 1. Schema Check 
        srv = MCPServerSchema(**config_dict)
        
        # 2. Connectivity Check
        if srv.transport == "stdio":
            if not shutil.which(srv.command):
                raise ValueError(f"System error: '{srv.command}' is not an executable command.")
        elif srv.transport == "sse":
            async with httpx.AsyncClient() as client:
                # Quick ping to ensure the cloud MCP is alive
                resp = await client.get(str(srv.url), timeout=5.0)
                resp.raise_for_status()
        return True