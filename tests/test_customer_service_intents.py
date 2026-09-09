"""Exercise the notebook's chatbot without model downloads or interactive input."""

import ast
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import unittest


NOTEBOOK = Path(__file__).resolve().parents[1] / "agent-demo" / "agent.ipynb"


def load_bot_class():
    required = {
        "MLIntentRecognizer", "TextPreprocessor", "SlotExtractor",
        "HybridIntentRecognizer", "IntentRecognitionEngine", "SmartCustomerServiceBot",
    }
    namespace = {}
    for cell in json.loads(NOTEBOOK.read_text())["cells"]:
        source = "".join(cell.get("source", []))
        if not any(f"class {name}:" in source for name in required):
            continue
        tree = ast.parse(source)
        tree.body = [
            node for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom))
            or isinstance(node, ast.ClassDef) and node.name in required
        ]
        exec(compile(tree, str(NOTEBOOK), "exec"), namespace)
    return namespace["SmartCustomerServiceBot"]


class CustomerServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bot_class = load_bot_class()

    def setUp(self):
        with redirect_stdout(io.StringIO()):
            self.bot = self.bot_class()
            self.bot.train()

    def test_common_intents(self):
        cases = {
            "明天北京天气怎么样": "查询天气",
            "北京现在多少度": "查询天气",
            "我想订票去上海": "订票",
            "帮我买张去广州的票": "订票",
            "查询我的订单": "查询订单",
            "我的快递什么时候到": "查询订单",
            "我要退款": "退款",
            "这个订单申请退货": "退款",
            "转人工客服": "人工客服",
            "你好！": "问候",
        }
        for text, intent in cases.items():
            with self.subTest(text=text):
                result = self.bot.engine.process(text)
                self.assertNotEqual(result["status"], "uncertain")
                self.assertEqual(result["intent"], intent)

    def test_unknown_negation_and_multiple_intents(self):
        for text in ("写一首诗", "量子纠缠", "不要订票", "不要退款", "查天气然后订票", ""):
            with self.subTest(text=text):
                self.assertEqual(self.bot.engine.process(text)["status"], "uncertain")

    def test_booking_followups(self):
        self.assertIn("去哪", self.bot.chat("帮我订票"))
        self.assertIn("什么时候", self.bot.chat("上海"))
        response = self.bot.chat("明天")
        self.assertIn("目的地：上海", response)
        self.assertIsNone(self.bot.current_intent)
        self.assertEqual(self.bot.current_slots, {})

    def test_order_followup(self):
        self.assertIn("订单号", self.bot.chat("查询我的订单"))
        self.assertIn("运输中", self.bot.chat("ord123456"))

    def test_weather_followup(self):
        self.assertIn("城市", self.bot.chat("天气怎么样"))
        self.assertIn("北京", self.bot.chat("北京"))
        self.assertIsNone(self.bot.current_intent)

    def test_topic_switch_clears_pending_slots(self):
        self.bot.chat("我想订票去上海")
        self.assertIn("订单号", self.bot.chat("查订单"))
        self.assertNotIn("to", self.bot.current_slots)
        self.assertIn("运输中", self.bot.chat("ORD123456"))

    def test_greeting_then_out_of_scope(self):
        self.assertIn("您好", self.bot.chat("你好"))
        self.assertIn("没太理解", self.bot.chat("写一首诗"))


if __name__ == "__main__":
    unittest.main()
