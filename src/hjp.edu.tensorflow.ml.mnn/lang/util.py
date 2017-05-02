from progress.bar import Bar

class ProgressBar(Bar):
    message = 'loading'
    fill = '#'
    suffix = '%(percent).1f%% | ETA: %(eta)ds'
