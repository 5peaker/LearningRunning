"""Console script for learningrunning."""
import learningrunning

import typer
from rich.console import Console

app = typer.Typer()
console = Console()


@app.command()
def main():
    """Console script for learningrunning."""
    console.print("Replace this message by putting your code into "
               "learningrunning.cli.main")
    console.print("See Typer documentation at https://typer.tiangolo.com/")
    


if __name__ == "__main__":
    app()
