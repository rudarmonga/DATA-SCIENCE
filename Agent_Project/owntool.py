from langchain.tools import tool

@tool
def get_greeting(name : str ) -> str :
    """Generate greeting message for user"""
    return f"Hello {name}, Welcome to AI world."

result = get_greeting.run({"name" : "Rudar"})
print(result)
print(get_greeting.name)
print(get_greeting.description)
print(get_greeting.args)