import sys
from textwrap import dedent

try:
    from nox_poetry import session
except ImportError:
    message = f"""\
    Nox failed to import the 'nox-poetry' package.

    Please install it using the following command:

    {sys.executable} -m pip install nox-poetry"""
    raise SystemExit(dedent(message)) from None

LOCATIONS = ("src/", "tests/", "noxfile.py")


@session
def lint(session):
    session.install("flake8")
    session.install("flake8-bugbear")
    session.install("flake8-isort")
    session.install("black")

    args = session.posargs or LOCATIONS
    session.run("flake8", *args)
    session.run("black", "--check", "--diff", *args)


@session(python=["3.8", "3.9", "3.10", "3.11", "3.12"])
def tests(session):
    session.install(".", "--extra-index-url", "https://download.pytorch.org/whl/cpu")
    session.install("pytest")
    session.install("pytest-mock")
    session.run("pytest", *session.posargs)
