from pandas import DataFrame # type: ignore

class Notebook:
    def __init__(self) -> None:
        self.data = []

    def write(self, input_data: DataFrame, short_description: str):
        self.data.append({"Short Description": short_description, "Content":input_data})
        return f"The information has been recorded in Notebook, and its index is {len(self.data)-1}."