"""
Lark webhook notification for experiment progress.
"""

import json
import requests

LARK_WEBHOOK = "https://open.larksuite.com/open-apis/bot/v2/hook/98fa7f97-652a-43fb-ae34-aafbd5ad62b4"


def send(title: str, content: str):
    """Send a notification to Lark."""
    payload = {
        "msg_type": "interactive",
        "card": {
            "header": {
                "title": {"tag": "plain_text", "content": title},
                "template": "blue",
            },
            "elements": [
                {
                    "tag": "markdown",
                    "content": content,
                }
            ],
        },
    }
    try:
        requests.post(LARK_WEBHOOK, json=payload, timeout=5)
    except Exception:
        pass


def experiment_start(experiment: str, model: str, samples: int):
    send(
        f"🚀 实验开始: {experiment}",
        f"**模型**: {model}\n**样本数**: {samples}\n**状态**: 运行中...",
    )


def experiment_progress(experiment: str, current: int, total: int, pass_rate: float):
    bar = "█" * int(pass_rate * 10) + "░" * (10 - int(pass_rate * 10))
    send(
        f"📊 {experiment} 进度: {current}/{total}",
        f"**通过率**: {bar} {pass_rate*100:.1f}%\n**进度**: {current}/{total}",
    )


def experiment_done(experiment: str, results: dict):
    send(
        f"✅ 实验完成: {experiment}",
        f"**JSON通过率**: {results.get('json_pass_rate', 0)*100:.1f}%\n"
        f"**API通过率**: {results.get('api_pass_rate', 0)*100:.1f}%\n"
        f"**全部通过**: {results.get('all_pass_rate', 0)*100:.1f}%\n"
        f"**样本数**: {results.get('total', 0)}",
    )


def experiment_error(experiment: str, error: str):
    send(
        f"❌ 实验失败: {experiment}",
        f"**错误**: {error[:500]}",
    )


if __name__ == "__main__":
    send("🧪 Compiler-Reward Agent RL", "通知系统已连接！实验进展将自动推送到此群。")
