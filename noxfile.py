import nox

LOCATIONS = ('src/', 'noxfile.py', 'setup.py')


@nox.session
def lint(session):
    session.install('flake8')
    session.install('flake8-bugbear')
    session.install('flake8-isort')

    args = session.posargs or LOCATIONS
    session.run('flake8', *args)
