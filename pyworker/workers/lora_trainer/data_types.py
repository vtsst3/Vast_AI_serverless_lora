import dataclasses
import inspect
from typing import Dict, Any, Optional

from lib.data_types import ApiPayload, JsonDataException

@dataclasses.dataclass
class InputData(ApiPayload):
    """
    LoRA学習ジョブの入力データを定義するクラス。
    """
    user_id: str
    job_id: str
    image_r2_key: str
    webhook_url: Optional[str] = None

    @classmethod
    def for_test(cls) -> "InputData":
        """テスト用のダミーデータを生成します。"""
        return cls(
            user_id="test_user",
            job_id="test_job_12345",
            image_r2_key="test/image.jpg",
            webhook_url="https://example.com/webhook"
        )

    def generate_payload_json(self) -> Dict[str, Any]:
        """
        このPyWorkerはモデルAPIを直接呼び出さないため、このメソッドは実質的に不要です。
        しかし、抽象クラスの要件を満たすために実装します。
        """
        return dataclasses.asdict(self)

    def count_workload(self) -> float:
        """
        LoRA学習ジョブのワークロードを定義します。
        ここでは単純に1ジョブあたり100の負荷と仮定します。
        Vast.aiはこの値を見てスケーリングの判断材料にします。
        """
        return 100.0

    @classmethod
    def from_json_msg(cls, json_msg: Dict[str, Any]) -> "InputData":
        """
        クライアントから受け取ったJSONペイロードをInputDataオブジェクトに変換します。
        """
        errors = {}
        # 必須パラメータのチェック
        for param in inspect.signature(cls).parameters:
            # オプショナルではない、かつ、デフォルト値がないパラメータが入力に含まれているかチェック
            if param not in json_msg and cls.__dataclass_fields__[param].default is dataclasses.MISSING:
                 errors[param] = "missing parameter"
        
        if errors:
            raise JsonDataException(errors)
            
        # 知らないパラメータは無視して、知っているパラメータだけを使ってインスタンス化する
        known_params = {k: v for k, v in json_msg.items() if k in inspect.signature(cls).parameters}
        return cls(**known_params)
