import nox

LOCATIONS = ("src/", "tests/", "noxfile.py", "setup.py")


@nox.session
def lint(session):
    session.install("flake8")
    session.install("flake8-bugbear")
    session.install("flake8-isort")
    session.install("black==24.3.0")

    args = session.posargs or LOCATIONS
    session.run("flake8", *args)
    session.run("black", "--check", "--diff", *args)


@nox.session(python=["3.8", "3.9", "3.10", "3.11", "3.12"])
def tests(session):
    session.install(
        "torch==2.2.1",
        "torchvision",
        "--index-url",
        "https://download.pytorch.org/whl/cpu",
    )
    session.install(".")
    session.install("pytest")
    session.install("pytest-mock")
    session.run("pytest", *session.posargs)
