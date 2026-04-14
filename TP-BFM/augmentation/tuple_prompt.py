

from typing import List, Dict, Optional


class TuplePromptBuilder:
    """
    元组提示 (Tuple Prompting) 构建器

    元组提示定义为 S = (T, L, V)，其中:
        T: 文本类型 (Text Type)
        L: 标签类型 (Label Type)
        V: 标签词具体化器 (Label-token Verbalizer)

    通过 "意图类别标识 + 类内真实示例" 的双重约束，
    引导大语言模型生成符合目标语义的合成样本
    """

    def __init__(
        self,
        text_type: str = "用户描述",
        label_type: str = "意图类型",
        language: str = "en",
    ):
       
        self.text_type = text_type
        self.label_type = label_type
        self.language = language

    def build_prompt(
        self,
        class_name: str,
        examples: List[str],
        num_to_generate: int = 10,
    ) -> str:

        if self.language == "zh":
            return self._build_chinese_prompt(class_name, examples, num_to_generate)
        else:
            return self._build_english_prompt(class_name, examples, num_to_generate)

    def _build_english_prompt(
        self, class_name: str, examples: List[str], num_to_generate: int
    ) -> str:
       
        # 系统指令
        system_instruction = (
            "You are a data augmentation assistant. "
            "Your task is to simulate real user intents and generate diverse intent sentences."
        )

        # 示例部分
        example_lines = []
        for example in examples:
            example_lines.append(
                f"User description: {example}  [Intent type: {class_name}]"
            )

        examples_text = "\n".join(example_lines)

        # 生成指令
        generation_instruction = (
            f"\nPlease generate {num_to_generate} sentences that match the intent label "
            f"'{class_name}', one per line. Do not add numbering or extra explanation."
        )

        prompt = f"{system_instruction}\n\n{examples_text}\n{generation_instruction}"
        return prompt

   
    def build_soft_label_prompt(self, text: str, label_list: List[str]) -> str:
       
        labels_str = ", ".join(label_list)

        if self.language == "en":
            prompt = (
                f"Given the following user utterance, classify it into one of these intent categories: "
                f"[{labels_str}]\n\n"
                f"User utterance: \"{text}\"\n"
                f"Intent category:"
            )
        else:
            prompt = (
                f"给定以下用户语句，将其分类到以下意图类别之一: "
                f"[{labels_str}]\n\n"
                f"用户语句: \"{text}\"\n"
                f"意图类别:"
            )

        return prompt