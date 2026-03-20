import editdistance

def cer(p,g):

    return editdistance.eval(p,g)/len(g)

def wer(p,g):

    pw = p.split()
    gw = g.split()

    return editdistance.eval(pw,gw)/len(gw)