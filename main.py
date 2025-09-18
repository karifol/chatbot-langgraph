import os
import uvicorn
import json
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import tool
import dotenv

dotenv.load_dotenv()

@tool
def generate_chart(labels: list=[], datasets: list=[], chart_type: str="line") -> dict:
    """
    Chart.js 用のグラフデータを生成するツール
    LLMへの説明：これを使えばチャット上でグラフが勝手に描画されるので、LLMが自ら描く必要はないよ
    Args:
        data: {"labels": [...], "datasets": [...]}
        chart_type: "bar", "line", "pie" など
    Returns:
        Chart.js の構造に従った dict
    """
    # とりあえずダミーデータを返す
    # return {
    #     "type": "chart",
    #     "chartType": "line",
    #     "data": {
    #         "labels": ["January", "February", "March", "April", "May"],
    #         "datasets": [
    #             {
    #                 "label": "Sample Dataset",
    #                 "data": [65, 59, 80, 81, 56],
    #                 "fill": False,
    #                 "borderColor": "rgb(255, 192, 192)",
    #                 "lineTension": 0.1
    #             }
    #         ]
    #     },
    #     "options": {
    #         "responsive": True,
    #         "plugins": {
    #             "legend": {"position": "top"}
    #         }
    #     }
    # }
    return {
        "type": "chart",
        "chartType": chart_type,
        "data": {
            "labels": labels,
            "datasets": datasets
        },
        "options": {
            "responsive": True,
            "plugins": {
                "legend": {"position": "top"}
            }
        }
    }

# -------------------------
# OpenAI LLM
# -------------------------
# --- LLM ---
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    streaming=True,
)

# --- MCP Client ---
# ValueError: Unsupported transport: see. Must be one of: 'stdio', 'sse', 'websocket', 'streamable_http'
server_connections = {
    "aws": {
        "url": "https://knowledge-mcp.global.api.aws",
        "transport": "streamable_http",
    },
    "WxTech1kmMeshPinpointWeatherForecast": {
        "url": "https://wxtech.weathernews.com/api/beta/ss1wx/mcp/",
        "headers": {"X-API-KEY": os.getenv("WX_API_KEY")},
        "transport": "streamable_http",
    },
    "WxTech5kmMeshGlobalWeatherForecast": {
        "url": "https://wxtech.weathernews.com/api/beta/global/wx/mcp/",
        "headers": {"X-API-KEY": os.getenv("WX_API_KEY")},
        "transport": "streamable_http",
    }
}

# -------------------------
# Lazy Init で agent を構築
# -------------------------

_agent = None

async def get_agent():
    global _agent
    if _agent is None:
        mcp_client = MultiServerMCPClient(server_connections)
        tools = await mcp_client.get_tools()  # ← await OK
        # ローカルツールを追加
        tools.append(generate_chart)
        print(f"✅ MCPツールをロードしました: {[t.name for t in tools]}")
        _agent = create_react_agent(llm, tools)
    return _agent

# --- FastAPI アプリ ---
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- エンドポイント ---
@app.post("/chat")
async def chat_endpoint(request: Request):
    agent = await get_agent()   # ← await に変更
    body = await request.json()
    messages = body.get("messages", [])

    async def event_stream():
        async for event in agent.astream_events(
            {"messages": messages},
            version="v1",
        ):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                delta = event["data"]["chunk"].content
                if delta:
                    yield f"data: {json.dumps({'type': 'token', 'content': delta}, ensure_ascii=False)}\n\n"

            elif kind == "on_tool_start":
                yield f"data: {json.dumps({'type': 'tool_start', 'tool_name': event['name'], 'tool_input': event['data']['input'], 'tool_id': event['run_id']}, ensure_ascii=False)}\n\n"

            elif kind == "on_tool_end":
                tool_output = event["data"]["output"]
                try:
                    if hasattr(tool_output, "content"):
                        tool_output = json.loads(tool_output.content)
                except Exception:
                    pass

                if isinstance(tool_output, dict) and tool_output.get("type") == "chart":
                    # グラフデータの場合
                    yield f"data: {json.dumps({'type': 'chart', 'tool_name': event['name'], 'tool_response': tool_output, 'tool_id': event['run_id'], 'chart': tool_output}, ensure_ascii=False)}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'tool_end', 'tool_name': event['name'], 'tool_response': tool_output, 'tool_id': event['run_id']}, ensure_ascii=False)}\n\n"
    return StreamingResponse(event_stream(), media_type="text/event-stream")

if __name__ == "__main__":
    # uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "80")))
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
