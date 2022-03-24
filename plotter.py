import matplotlib.pyplot as p
from IPython import display

p.ion()

def plot(score, mean):
    display.clear_output(wait=True)
    display.display(p.gcf())
    p.clf()
    p.title('AI Playing Game....')
    p.xlabel('Games')
    p.ylabel('Score')
    p.plot(score)
    p.plot(mean)
    p.ylim(ymin = 0)
    p.text(len(score) - 1, score[-1], str(score[-1]))
    p.text(len(mean) - 1, mean[-1], str(mean[-1]))
