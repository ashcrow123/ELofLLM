from pydantic import BaseModel, model_validator
from typing import List, Optional, Self, Literal,Dict,Any
import json
_OPTIONS = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
def deal_json_format(text):
    text = text.replace("**EXPECTED FORMAT:**", "")
    text = text.replace('json', "")
    text = text.replace('`', '')
    text = text.strip()
    return text
class speaker_retrieval_response(BaseModel):
    player_id: int | str
    max_length: int
    target_object: str
    object_num: int
    analysis:Optional[str]=None
    num_list: Optional[List[int]]=None

    @model_validator(mode="after")
    def _check(self) -> Self:
        for i, v in enumerate(self.num_list):
            if v < 0 or v >= self.object_num:
                raise ValueError(
                    f"num_list[{i}]={v} must be in [0, {self.object_num})"
                )
        return self
    def __str__(self):
        header = f"{'id':<14} {'max_length':<12} {'target_object':<20} {'object_num':<12}"
        sep = "-" * 58
        analysis_line = f"  analysis : {self.analysis}"
        num_list_line = f"  num_list : {self.num_list}"
        return (
            "\n"
            f"speaker_retrieval_response:\n"
            f"{header}\n"
            f"{sep}\n"
            f"  {self.player_id!s:<12} {self.max_length!s:<12} {self.target_object!s:<20} {self.object_num!s:<12}\n"
            f"{sep}\n"
            f"{analysis_line}\n"
            f"{num_list_line}\n"
            f"{sep}\n"
        )

class speaker_generate_response(BaseModel):
    player_id: int | str
    max_length: int
    target_object: str
    letter_list:List[str]
    analysis:str
    word:str
    @model_validator(mode="after")
    def _check(self) -> Self:
        word_split=self.word.split("-")
        if len(word_split)>self.max_length or len(word_split)==0:
            raise ValueError(
                f"word length {len(word_split)} must be <= max_length {self.max_length} and >=0"
            )
        for letter in word_split:
            if letter not in self.letter_list:
                raise ValueError(
                    f"letter '{letter}' in word must be in letter_list {self.letter_list}"
                )
        return self
    def __str__(self):
        header = f"{'id':<14} {'max_length':<12} {'target_object':<20} {'letter_list':<30}"
        sep = "-" * 80
        analysis_line = f"  analysis : {self.analysis}"
        word_line = f"  word     : {self.word}"
        return (
            "\n"
            f"speaker_generate_response:\n"
            f"{header}\n"
            f"{sep}\n"
            f"  {self.player_id!s:<12} {self.max_length!s:<12} {self.target_object!s:<20} {str(self.letter_list)!s:<30}\n"
            f"{sep}\n"
            f"{analysis_line}\n"
            f"{word_line}\n"
            f"{sep}\n"
        )

class listener_retrieval_response(BaseModel):
    player_id: int | str
    target_word: str
    vocab: List[str]
    analysis:str
    num_list: List[int]
    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.num_list:
            for num in self.num_list:
                if num < 0 or num > len(self.vocab):
                    raise ValueError(
                        f"num_list contains invalid index {num} for vocab of length {len(self.vocab)}"
                    )
        return self

    @property
    def word_list(self)->List[str]:
        return [self.vocab[num-1] for num in self.num_list]

    def __str__(self):
        header = f"{'id':<14} {'target_word':<20} {'vocab_size':<12} {'num_list':<30}"
        sep = "-" * 80
        analysis_line = f"  analysis : {self.analysis}"
        num_list_line = f"  num_list : {self.num_list}"
        return (
            "\n"
            f"listener_retrieval_response:\n"
            f"{header}\n"
            f"{sep}\n"
            f"  {self.player_id!s:<12} {self.target_word!s:<20} {len(self.vocab)!s:<12} {str(self.num_list)!s:<30}\n"
            f"{sep}\n"
            f"{analysis_line}\n"
            f"{num_list_line}\n"
            f"{sep}\n"
        )

class listener_select_response(BaseModel):
    player_id: int | str
    target_object: str
    word:str
    choices:Dict[str, Any]
    analysis:str
    option:Literal["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.option in _OPTIONS[:len(self.choices)]:
            return self
        else:
            raise ValueError(
                f"option '{self.option}' must be one of {_OPTIONS[:len(self.choices)]} for choices of length {len(self.choices)}"
            )

    def __str__(self):
        header = f"{'id':<14} {'target_object':<20} {'word':<20} {'option':<8} {'choices':<30}"
        sep = "-" * 96
        analysis_line = f"  analysis : {self.analysis}"
        choice_line = f"  choice   : {self.option} -> {self.choices.get(self.option, 'N/A')}"
        return (
            "\n"
            f"listener_select_response:\n"
            f"{header}\n"
            f"{sep}\n"
            f"  {self.player_id!s:<12} {self.target_object!s:<20} {self.word!s:<20} {self.option!s:<8} {str(list(self.choices.keys()))!s:<30}\n"
            f"{sep}\n"
            f"{analysis_line}\n"
            f"{choice_line}\n"
            f"{sep}\n"
        )


class select_feature_response(BaseModel):
    data: List[str]

    @model_validator(mode="after")
    def _check(self) -> Self:
        all_categories = [
            'encyclopaedic',
            'function',
            'smell',
            'sound',
            'tactile',
            'taste',
            'taxonomic',
            'visual_colour',
            'visual_form_and_surface',
            'visual_motion',
        ]
        if set(self.data).issubset(all_categories):
            return self
        raise ValueError(
            f"Categories {set(self.data) - set(all_categories)} not in allowed categories"
        )
