##Importing the required libraries
import bz2
import glob
import json
import nltk
import pycorenlp as nlp 
from collections import Counter
import sys
import csv
import re

from pycorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('http://localhost:9000')

##Changing directory to a folder with all reddit files that should be included
cd "C:/Users/m_pis/Dropbox/Documents/SoDA 502/project/reddit data"


##Finding files and adding it to a list of all Reddit Comments
files=glob.glob("*.bz2")

subreddits="nfl"

partsOfSpeech=("NN","PRP","CC","CD","DT","EX","FW","IN","JJ","JJR","JJS","LS","MD","NNS","NNP","NNPS","PDT","POS","PRP$","RB","RBR","RBS","RP","SYM","TO","UH","VB","VBD","VBG","VBN","VBP","VBZ","WDT","WP","WP$","WRB")

phrases=('(S, ADJP)','(S, ADVP)','(S, CONJP)','(S, FRAG)','(S, INTJ)','(S, LST)', '(S, NAC)', '(S, NP)','(S, NX)','(S, PP)','(S, PRN)', '(S, PRT)', '(S, QP)', '(S, RRC)', '(S, UCP)', '(S, VP)','(S, WHADJP)', '(S, WHNP)', '(S, X)','(TRUE, ADJP)','(TRUE, ADVP)','(TRUE, CONJP)','(TRUE, FRAG)','(TRUE, INTJ)', '(TRUE, LST)', '(TRUE, NAC)', '(TRUE, NP)','(TRUE, NX)','(TRUE, PP)','(TRUE, PRN)', '(TRUE, PRT)', '(TRUE, QP)','(TRUE, RRC)', '(TRUE, UCP)', '(TRUE, VP)','(TRUE, WHADJP)','(TRUE, WHNP)','(TRUE, X)','(SINV, ADJP)','(SINV, ADVP)','(SINV, CONJP)','(SINV, FRAG)', '(SINV, INTJ)','(SINV, LST)', '(SINV, NAC)', '(SINV, NP)','(SINV, NX)','(SINV, PP)','(SINV, PRN)', '(SINV, PRT)', '(SINV, QP)','(SINV, RRC)', '(SINV, UCP)', '(SINV, VP)', 
 '(SINV, WHADJP)','(SINV, WHNP)','(SINV, X)','(SQ, ADJP)','(SQ, ADVP)','(SQ, CONJP)',
 '(SQ, FRAG)','(SQ, INTJ)','(SQ, LST)','(SQ, NAC)','(SQ, NP)', '(SQ, NX)', '(SQ, PP)', '(SQ, PRN)','(SQ, PRT)','(SQ, QP)', '(SQ, RRC)','(SQ, UCP)', '(SQ, VP)', '(SQ, WHADJP)','(SQ, WHNP)','(SQ, X)','(ADJP, NN)','(ADJP, PRP)', '(ADJP, CC)','(ADJP, CD)','(ADJP, DT)','(ADJP, EX)','(ADJP, FW)','(ADJP, IN)','(ADJP, JJ)','(ADJP, JJR)', '(ADJP, JJS)', '(ADJP, LS)','(ADJP, MD)','(ADJP, NNS)', '(ADJP, NNP)', '(ADJP, NNPS)','(ADJP, PDT)', '(ADJP, POS)', '(ADJP, PRP$)','(ADJP, RB)','(ADJP, RBR)', '(ADJP, RBS)', '(ADJP, RP)','(ADJP, SYM)', '(ADJP, TO)','(ADJP, UH)','(ADJP, VB)','(ADJP, VBD)', '(ADJP, VBG)', '(ADJP, VBN)', '(ADJP, VBP)', '(ADJP, VBZ)', '(ADJP, WDT)', '(ADJP, WP)','(ADJP, WP$)', '(ADJP, WRB)', '(ADVP, NN)','(ADVP, PRP)','(ADVP, CC)','(ADVP, CD)','(ADVP, DT)','(ADVP, EX)','(ADVP, FW)','(ADVP, IN)','(ADVP, JJ)','(ADVP, JJR)', '(ADVP, JJS)', '(ADVP, LS)','(ADVP, MD)','(ADVP, NNS)', '(ADVP, NNP)', '(ADVP, NNPS)','(ADVP, PDT)', '(ADVP, POS)', '(ADVP, PRP$)','(ADVP, RB)','(ADVP, RBR)', '(ADVP, RBS)', '(ADVP, RP)','(ADVP, SYM)', '(ADVP, TO)','(ADVP, UH)', '(ADVP, VB)','(ADVP, VBD)', '(ADVP, VBG)', '(ADVP, VBN)', '(ADVP, VBP)', '(ADVP, VBZ)','(ADVP, WDT)', '(ADVP, WP)','(ADVP, WP$)', '(ADVP, WRB)', '(CONJP, NN)', '(CONJP, PRP)','(CONJP, CC)', '(CONJP, CD)', '(CONJP, DT)', '(CONJP, EX)', '(CONJP, FW)', '(CONJP, IN)','(CONJP, JJ)', '(CONJP, JJR)','(CONJP, JJS)','(CONJP, LS)', '(CONJP, MD)', '(CONJP, NNS)', '(CONJP, NNP)','(CONJP, NNPS)','(CONJP, PDT)','(CONJP, POS)','(CONJP, PRP$)','(CONJP, RB)', '(CONJP, RBR)','(CONJP, RBS)','(CONJP, RP)', '(CONJP, SYM)','(CONJP, TO)', '(CONJP, UH)', '(CONJP, VB)', '(CONJP, VBD)','(CONJP, VBG)','(CONJP, VBN)','(CONJP, VBP)','(CONJP, VBZ)', '(CONJP, WDT)','(CONJP, WP)', '(CONJP, WP$)','(CONJP, WRB)','(FRAG, NN)','(FRAG, PRP)', '(FRAG, CC)','(FRAG, CD)','(FRAG, DT)','(FRAG, EX)','(FRAG, FW)','(FRAG, IN)','(FRAG, JJ)','(FRAG, JJR)', '(FRAG, JJS)', '(FRAG, LS)','(FRAG, MD)','(FRAG, NNS)', '(FRAG, NNP)', '(FRAG, NNPS)','(FRAG, PDT)', '(FRAG, POS)', '(FRAG, PRP$)','(FRAG, RB)', '(FRAG, RBR)', '(FRAG, RBS)', '(FRAG, RP)','(FRAG, SYM)', '(FRAG, TO)','(FRAG, UH)', '(FRAG, VB)','(FRAG, VBD)', '(FRAG, VBG)', '(FRAG, VBN)', '(FRAG, VBP)', '(FRAG, VBZ)', '(FRAG, WDT)', '(FRAG, WP)','(FRAG, WP$)', '(FRAG, WRB)', '(INTJ, NN)','(INTJ, PRP)', '(INTJ, CC)','(INTJ, CD)','(INTJ, DT)','(INTJ, EX)','(INTJ, FW)','(INTJ, IN)','(INTJ, JJ)','(INTJ, JJR)', '(INTJ, JJS)', '(INTJ, LS)','(INTJ, MD)','(INTJ, NNS)', '(INTJ, NNP)', '(INTJ, NNPS)','(INTJ, PDT)', '(INTJ, POS)', '(INTJ, PRP$)','(INTJ, RB)','(INTJ, RBR)', '(INTJ, RBS)', '(INTJ, RP)','(INTJ, SYM)', '(INTJ, TO)','(INTJ, UH)','(INTJ, VB)','(INTJ, VBD)', '(INTJ, VBG)', '(INTJ, VBN)', '(INTJ, VBP)', '(INTJ, VBZ)', '(INTJ, WDT)', '(INTJ, WP)','(INTJ, WP$)', '(INTJ, WRB)', '(LST, NN)','(LST, PRP)', '(LST, CC)','(LST, CD)','(LST, DT)','(LST, EX)','(LST, FW)','(LST, IN)', '(LST, JJ)','(LST, JJR)','(LST, JJS)','(LST, LS)','(LST, MD)','(LST, NNS)', 
 '(LST, NNP)','(LST, NNPS)', '(LST, PDT)','(LST, POS)','(LST, PRP$)', '(LST, RB)','(LST, RBR)','(LST, RBS)','(LST, RP)','(LST, SYM)','(LST, TO)','(LST, UH)','(LST, VB)','(LST, VBD)','(LST, VBG)','(LST, VBN)','(LST, VBP)','(LST, VBZ)', '(LST, WDT)','(LST, WP)','(LST, WP$)','(LST, WRB)','(NAC, NN)','(NAC, PRP)', '(NAC, CC)','(NAC, CD)','(NAC, DT)','(NAC, EX)','(NAC, FW)','(NAC, IN)', '(NAC, JJ)','(NAC, JJR)','(NAC, JJS)','(NAC, LS)','(NAC, MD)','(NAC, NNS)','(NAC, NNP)','(NAC, NNPS)', '(NAC, PDT)','(NAC, POS)','(NAC, PRP$)', '(NAC, RB)', '(NAC, RBR)','(NAC, RBS)','(NAC, RP)','(NAC, SYM)','(NAC, TO)','(NAC, UH)','(NAC, VB)','(NAC, VBD)','(NAC, VBG)','(NAC, VBN)','(NAC, VBP)','(NAC, VBZ)', '(NAC, WDT)','(NAC, WP)','(NAC, WP$)','(NAC, WRB)','(NP, NN)', '(NP, PRP)', '(NP, CC)', '(NP, CD)', '(NP, DT)', '(NP, EX)', '(NP, FW)', '(NP, IN)', '(NP, JJ)', '(NP, JJR)','(NP, JJS)','(NP, LS)', '(NP, MD)', '(NP, NNS)', '(NP, NNP)','(NP, NNPS)','(NP, PDT)','(NP, POS)','(NP, PRP$)','(NP, RB)', '(NP, RBR)','(NP, RBS)','(NP, RP)', '(NP, SYM)','(NP, TO)', '(NP, UH)', '(NP, VB)', '(NP, VBD)','(NP, VBG)','(NP, VBN)','(NP, VBP)','(NP, VBZ)', '(NP, WDT)','(NP, WP)', '(NP, WP$)','(NP, WRB)','(NX, NN)', '(NX, PRP)', '(NX, CC)', '(NX, CD)', '(NX, DT)', '(NX, EX)', '(NX, FW)', '(NX, IN)', '(NX, JJ)', '(NX, JJR)','(NX, JJS)','(NX, LS)', '(NX, MD)', '(NX, NNS)', '(NX, NNP)','(NX, NNPS)','(NX, PDT)','(NX, POS)','(NX, PRP$)','(NX, RB)', '(NX, RBR)','(NX, RBS)','(NX, RP)', '(NX, SYM)','(NX, TO)', '(NX, UH)', '(NX, VB)', '(NX, VBD)','(NX, VBG)','(NX, VBN)','(NX, VBP)','(NX, VBZ)', '(NX, WDT)','(NX, WP)', '(NX, WP$)','(NX, WRB)','(PP, NN)', '(PP, PRP)', '(PP, CC)', '(PP, CD)', '(PP, DT)', '(PP, EX)', '(PP, FW)', '(PP, IN)', '(PP, JJ)', '(PP, JJR)','(PP, JJS)','(PP, LS)', '(PP, MD)', '(PP, NNS)',
 '(PP, NNP)','(PP, NNPS)','(PP, PDT)','(PP, POS)','(PP, PRP$)','(PP, RB)', '(PP, RBR)','(PP, RBS)','(PP, RP)', '(PP, SYM)','(PP, TO)', '(PP, UH)', '(PP, VB)', '(PP, VBD)','(PP, VBG)','(PP, VBN)','(PP, VBP)','(PP, VBZ)', '(PP, WDT)','(PP, WP)', '(PP, WP$)','(PP, WRB)','(PRN, NN)','(PRN, PRP)','(PRN, CC)','(PRN, CD)','(PRN, DT)','(PRN, EX)','(PRN, FW)','(PRN, IN)', '(PRN, JJ)','(PRN, JJR)','(PRN, JJS)','(PRN, LS)','(PRN, MD)','(PRN, NNS)', '(PRN, NNP)','(PRN, NNPS)', '(PRN, PDT)','(PRN, POS)','(PRN, PRP$)', '(PRN, RB)', '(PRN, RBR)','(PRN, RBS)','(PRN, RP)','(PRN, SYM)','(PRN, TO)','(PRN, UH)', '(PRN, VB)','(PRN, VBD)','(PRN, VBG)','(PRN, VBN)','(PRN, VBP)','(PRN, VBZ)','(PRN, WDT)','(PRN, WP)','(PRN, WP$)','(PRN, WRB)','(PRT, NN)','(PRT, PRP)', '(PRT, CC)','(PRT, CD)','(PRT, DT)','(PRT, EX)','(PRT, FW)','(PRT, IN)', '(PRT, JJ)','(PRT, JJR)','(PRT, JJS)','(PRT, LS)','(PRT, MD)','(PRT, NNS)', '(PRT, NNP)','(PRT, NNPS)', '(PRT, PDT)','(PRT, POS)','(PRT, PRP$)', '(PRT, RB)', '(PRT, RBR)','(PRT, RBS)','(PRT, RP)','(PRT, SYM)','(PRT, TO)','(PRT, UH)',
 '(PRT, VB)','(PRT, VBD)','(PRT, VBG)','(PRT, VBN)','(PRT, VBP)','(PRT, VBZ)', '(PRT, WDT)','(PRT, WP)','(PRT, WP$)','(PRT, WRB)','(QP, NN)', '(QP, PRP)', '(QP, CC)', '(QP, CD)', '(QP, DT)', '(QP, EX)', '(QP, FW)', '(QP, IN)', '(QP, JJ)', '(QP, JJR)','(QP, JJS)','(QP, LS)', '(QP, MD)', '(QP, NNS)', '(QP, NNP)','(QP, NNPS)','(QP, PDT)','(QP, POS)','(QP, PRP$)','(QP, RB)', '(QP, RBR)','(QP, RBS)','(QP, RP)', '(QP, SYM)','(QP, TO)', '(QP, UH)', '(QP, VB)', '(QP, VBD)','(QP, VBG)','(QP, VBN)','(QP, VBP)','(QP, VBZ)', '(QP, WDT)','(QP, WP)', '(QP, WP$)','(QP, WRB)','(RRC, NN)','(RRC, PRP)', '(RRC, CC)','(RRC, CD)','(RRC, DT)','(RRC, EX)','(RRC, FW)','(RRC, IN)', '(RRC, JJ)','(RRC, JJR)','(RRC, JJS)','(RRC, LS)','(RRC, MD)','(RRC, NNS)','(RRC, NNP)','(RRC, NNPS)', '(RRC, PDT)','(RRC, POS)','(RRC, PRP$)', '(RRC, RB)','(RRC, RBR)','(RRC, RBS)','(RRC, RP)','(RRC, SYM)','(RRC, TO)','(RRC, UH)', '(RRC, VB)','(RRC, VBD)','(RRC, VBG)','(RRC, VBN)','(RRC, VBP)','(RRC, VBZ)','(RRC, WDT)','(RRC, WP)','(RRC, WP$)','(RRC, WRB)','(UCP, NN)','(UCP, PRP)','(UCP, CC)','(UCP, CD)','(UCP, DT)','(UCP, EX)','(UCP, FW)','(UCP, IN)',
 '(UCP, JJ)','(UCP, JJR)','(UCP, JJS)','(UCP, LS)','(UCP, MD)','(UCP, NNS)', '(UCP, NNP)','(UCP, NNPS)', '(UCP, PDT)','(UCP, POS)','(UCP, PRP$)', '(UCP, RB)', '(UCP, RBR)','(UCP, RBS)','(UCP, RP)','(UCP, SYM)','(UCP, TO)','(UCP, UH)', '(UCP, VB)','(UCP, VBD)','(UCP, VBG)','(UCP, VBN)','(UCP, VBP)','(UCP, VBZ)', '(UCP, WDT)','(UCP, WP)','(UCP, WP$)','(UCP, WRB)','(VP, NN)', '(VP, PRP)', '(VP, CC)', '(VP, CD)', '(VP, DT)', '(VP, EX)', '(VP, FW)', '(VP, IN)', '(VP, JJ)', '(VP, JJR)','(VP, JJS)','(VP, LS)', '(VP, MD)', '(VP, NNS)', '(VP, NNP)','(VP, NNPS)','(VP, PDT)','(VP, POS)','(VP, PRP$)','(VP, RB)', '(VP, RBR)','(VP, RBS)','(VP, RP)', '(VP, SYM)','(VP, TO)', '(VP, UH)', '(VP, VB)', '(VP, VBD)','(VP, VBG)','(VP, VBN)','(VP, VBP)','(VP, VBZ)', '(VP, WDT)','(VP, WP)', '(VP, WP$)','(VP, WRB)','(WHADJP, NN)','(WHADJP, PRP)', '(WHADJP, CC)','(WHADJP, CD)','(WHADJP, DT)','(WHADJP, EX)','(WHADJP, FW)','(WHADJP, IN)', '(WHADJP, JJ)','(WHADJP, JJR)','(WHADJP, JJS)','(WHADJP, LS)','(WHADJP, MD)','(WHADJP, NNS)', '(WHADJP, NNP)','(WHADJP, NNPS)','(WHADJP, PDT)','(WHADJP, POS)','(WHADJP, PRP$)','(WHADJP, RB)', '(WHADJP, RBR)','(WHADJP, RBS)','(WHADJP, RP)','(WHADJP, SYM)','(WHADJP, TO)','(WHADJP, UH)', '(WHADJP, VB)','(WHADJP, VBD)','(WHADJP, VBG)','(WHADJP, VBN)','(WHADJP, VBP)','(WHADJP, VBZ)','(WHADJP, WDT)','(WHADJP, WP)','(WHADJP, WP$)','(WHADJP, WRB)','(WHNP, NN)','(WHNP, PRP)', '(WHNP, CC)','(WHNP, CD)','(WHNP, DT)','(WHNP, EX)','(WHNP, FW)','(WHNP, IN)','(WHNP, JJ)','(WHNP, JJR)', '(WHNP, JJS)', '(WHNP, LS)','(WHNP, MD)','(WHNP, NNS)', '(WHNP, NNP)', '(WHNP, NNPS)','(WHNP, PDT)', '(WHNP, POS)', '(WHNP, PRP$)','(WHNP, RB)', '(WHNP, RBR)', '(WHNP, RBS)', '(WHNP, RP)','(WHNP, SYM)', '(WHNP, TO)','(WHNP, UH)','(WHNP, VB)','(WHNP, VBD)', '(WHNP, VBG)', '(WHNP, VBN)', '(WHNP, VBP)', '(WHNP, VBZ)', '(WHNP, WDT)', '(WHNP, WP)','(WHNP, WP$)', '(WHNP, WRB)', '(X, NN)','(X, PRP)', '(X, CC)','(X, CD)','(X, DT)','(X, EX)','(X, FW)','(X, IN)','(X, JJ)','(X, JJR)', '(X, JJS)', '(X, LS)','(X, MD)','(X, NNS)', '(X, NNP)', '(X, NNPS)','(X, PDT)', '(X, POS)', '(X, PRP$)','(X, RB)','(X, RBR)', '(X, RBS)', '(X, RP)','(X, SYM)', '(X, TO)','(X, UH)','(X, VB)','(X, VBD)', '(X, VBG)', '(X, VBN)', '(X, VBP)', '(X, VBZ)', '(X, WDT)', '(X, WP)','(X, WP$)', '(X, WRB)', '(ADJP, ADVP)','(ADJP, CONJP)',
 '(ADJP, FRAG)','(ADJP, INTJ)','(ADJP, LST)', '(ADJP, NAC)', '(ADJP, NP)','(ADJP, NX)','(ADJP, PP)','(ADJP, PRN)', '(ADJP, PRT)', '(ADJP, QP)','(ADJP, RRC)', '(ADJP, UCP)', '(ADJP, VP)','(ADJP, WHADJP)','(ADJP, WHNP)','(ADJP, X)','(ADVP, ADJP)','(ADVP, CONJP)', '(ADVP, FRAG)','(ADVP, INTJ)','(ADVP, LST)', '(ADVP, NAC)', '(ADVP, NP)','(ADVP, NX)', '(ADVP, PP)','(ADVP, PRN)', '(ADVP, PRT)', '(ADVP, QP)','(ADVP, RRC)', '(ADVP, UCP)', '(ADVP, VP)','(ADVP, WHADJP)','(ADVP, WHNP)','(ADVP, X)','(CONJP, ADJP)','(CONJP, ADVP)', '(CONJP, FRAG)','(CONJP, INTJ)','(CONJP, LST)','(CONJP, NAC)','(CONJP, NP)', '(CONJP, NX)', '(CONJP, PP)', '(CONJP, PRN)','(CONJP, PRT)','(CONJP, QP)', '(CONJP, RRC)','(CONJP, UCP)', '(CONJP, VP)', '(CONJP, WHADJP)', '(CONJP, WHNP)','(CONJP, X)','(FRAG, ADJP)','(FRAG, ADVP)', '(FRAG, CONJP)','(FRAG, INTJ)','(FRAG, LST)', '(FRAG, NAC)', '(FRAG, NP)','(FRAG, NX)','(FRAG, PP)','(FRAG, PRN)', '(FRAG, PRT)', '(FRAG, QP)','(FRAG, RRC)', '(FRAG, UCP)', '(FRAG, VP)','(FRAG, WHADJP)','(FRAG, WHNP)','(FRAG, X)','(INTJ, ADJP)','(INTJ, ADVP)', '(INTJ, CONJP)','(INTJ, FRAG)','(INTJ, LST)', '(INTJ, NAC)', '(INTJ, NP)','(INTJ, NX)','(INTJ, PP)','(INTJ, PRN)', '(INTJ, PRT)', '(INTJ, QP)','(INTJ, RRC)', '(INTJ, UCP)','(INTJ, VP)','(INTJ, WHADJP)','(INTJ, WHNP)','(INTJ, X)','(LST, ADJP)', '(LST, ADVP)','(LST, CONJP)','(LST, FRAG)', '(LST, INTJ)', '(LST, NAC)','(LST, NP)','(LST, NX)', '(LST, PP)','(LST, PRN)','(LST, PRT)','(LST, QP)','(LST, RRC)','(LST, UCP)','(LST, VP)','(LST, WHADJP)','(LST, WHNP)', '(LST, X)', '(NAC, ADJP)', '(NAC, ADVP)', '(NAC, CONJP)','(NAC, FRAG)', '(NAC, INTJ)', '(NAC, LST)','(NAC, NP)','(NAC, NX)', '(NAC, PP)','(NAC, PRN)','(NAC, PRT)','(NAC, QP)','(NAC, RRC)','(NAC, UCP)','(NAC, VP)','(NAC, WHADJP)','(NAC, WHNP)', '(NAC, X)', '(NP, ADJP)','(NP, ADVP)','(NP, CONJP)', '(NP, FRAG)','(NP, INTJ)','(NP, LST)','(NP, NAC)','(NP, NX)', '(NP, PP)', '(NP, PRN)','(NP, PRT)','(NP, QP)', '(NP, RRC)','(NP, UCP)', '(NP, VP)', '(NP, WHADJP)','(NP, WHNP)','(NP, X)','(NX, ADJP)','(NX, ADVP)','(NX, CONJP)', '(NX, FRAG)','(NX, INTJ)','(NX, LST)','(NX, NAC)','(NX, NP)', '(NX, PP)', '(NX, PRN)','(NX, PRT)','(NX, QP)', '(NX, RRC)','(NX, UCP)', '(NX, VP)', '(NX, WHADJP)','(NX, WHNP)','(NX, X)','(PP, ADJP)','(PP, ADVP)','(PP, CONJP)', '(PP, FRAG)','(PP, INTJ)','(PP, LST)','(PP, NAC)','(PP, NP)', '(PP, NX)', '(PP, PRN)','(PP, PRT)','(PP, QP)', '(PP, RRC)','(PP, UCP)', '(PP, VP)', '(PP, WHADJP)','(PP, WHNP)','(PP, X)','(PRN, ADJP)', '(PRN, ADVP)', '(PRN, CONJP)','(PRN, FRAG)', '(PRN, INTJ)', '(PRN, LST)','(PRN, NAC)','(PRN, NP)',
 '(PRN, NX)','(PRN, PP)','(PRN, PRT)','(PRN, QP)','(PRN, RRC)','(PRN, UCP)','(PRN, VP)','(PRN, WHADJP)','(PRN, WHNP)', '(PRN, X)', '(PRT, ADJP)', '(PRT, ADVP)', '(PRT, CONJP)','(PRT, FRAG)', '(PRT, INTJ)', '(PRT, LST)','(PRT, NAC)','(PRT, NP)', '(PRT, NX)','(PRT, PP)','(PRT, PRN)','(PRT, QP)','(PRT, RRC)','(PRT, UCP)','(PRT, VP)','(PRT, WHADJP)','(PRT, WHNP)', '(PRT, X)', '(QP, ADJP)','(QP, ADVP)','(QP, CONJP)', '(QP, FRAG)','(QP, INTJ)','(QP, LST)','(QP, NAC)','(QP, NP)', '(QP, NX)', '(QP, PP)', '(QP, PRN)','(QP, PRT)','(QP, RRC)','(QP, UCP)', '(QP, VP)', '(QP, WHADJP)','(QP, WHNP)','(QP, X)','(RRC, ADJP)', '(RRC, ADVP)', '(RRC, CONJP)','(RRC, FRAG)', '(RRC, INTJ)', '(RRC, LST)','(RRC, NAC)','(RRC, NP)', '(RRC, NX)','(RRC, PP)','(RRC, PRN)','(RRC, PRT)','(RRC, QP)','(RRC, UCP)','(RRC, VP)','(RRC, WHADJP)','(RRC, WHNP)', '(RRC, X)', '(UCP, ADJP)', '(UCP, ADVP)', '(UCP, CONJP)','(UCP, FRAG)', '(UCP, INTJ)', '(UCP, LST)','(UCP, NAC)','(UCP, NP)', '(UCP, NX)','(UCP, PP)','(UCP, PRN)','(UCP, PRT)','(UCP, QP)','(UCP, RRC)','(UCP, VP)','(UCP, WHADJP)','(UCP, WHNP)', '(UCP, X)', '(VP, ADJP)','(VP, ADVP)','(VP, CONJP)', '(VP, FRAG)','(VP, INTJ)','(VP, LST)','(VP, NAC)','(VP, NP)', '(VP, NX)', '(VP, PP)', '(VP, PRN)','(VP, PRT)','(VP, QP)', '(VP, RRC)', '(VP, UCP)','(VP, WHADJP)','(VP, WHNP)','(VP, X)','(WHADJP, ADJP)','(WHADJP, ADVP)','(WHADJP, CONJP)', '(WHADJP, FRAG)','(WHADJP, INTJ)','(WHADJP, LST)','(WHADJP, NAC)','(WHADJP, NP)', '(WHADJP, NX)','(WHADJP, PP)','(WHADJP, PRN)','(WHADJP, PRT)','(WHADJP, QP)','(WHADJP, RRC)', '(WHADJP, UCP)','(WHADJP, VP)','(WHADJP, WHNP)','(WHADJP, X)', '(WHNP, ADJP)','(WHNP, ADVP)', '(WHNP, CONJP)','(WHNP, FRAG)','(WHNP, INTJ)','(WHNP, LST)', '(WHNP, NAC)', '(WHNP, NP)','(WHNP, NX)','(WHNP, PP)','(WHNP, PRN)', '(WHNP, PRT)', '(WHNP, QP)','(WHNP, RRC)', '(WHNP, UCP)', '(WHNP, VP)','(WHNP, WHADJP)','(WHNP, X)','(X, ADJP)','(X, ADVP)', '(X, CONJP)','(X, FRAG)','(X, INTJ)','(X, LST)', '(X, NAC)', '(X, NP)','(X, NX)','(X, PP)','(X, PRN)', '(X, PRT)', '(X, QP)','(X, RRC)', '(X, UCP)', '(X, VP)','(X, WHADJP)', '(X, WHNP)') ##Need to add all the phrase pairs

pos_col = list(map(lambda x: "pos_{}".format(x), partsOfSpeech))
phr_col = list(map(lambda x: "phr_{}".format(x), phrases))
other_vars=["id","author","subreddit"]

file= open("phrase_features.csv", 'w')
fwriter = csv.DictWriter(file, fieldnames = other_vars + pos_col + phr_col)
fwriter.writeheader()

            
def PhrasePairs(tmp):
    tmp=tmp.productions()
    
    lst=[]
    for objects in tmp:
        objects=str(objects)
        objects=objects.split(" -> ")
        if('"' in str(objects)):
            continue
        lst.append(objects)
    
    pairs=[]
    for items in lst:
        parent=str(items[0])
        children=str(items[1]).split(" ")
        for child in children:
            hold="("+parent +", " +child+")"
            pairs.append(hold)
    return pairs
  

with bz2.BZ2File(files[0],mode="r") as f: ##Have to read line by line
    for line in f:
        tmp = f.readline().decode('utf8')
        comment=json.loads(tmp)
        ##We don't want to include deleted authors or deleted body text.
        if comment['author'] == "[deleted]":
            continue
        if comment['body'] == "[deleted]":
            continue
        if subreddits is None:
            continue
        if comment['subreddit'].lower() not in subreddits: ##Only posts that are in our subreddit
            continue
        if len(comment['body'])<100:
            continue
        text= re.sub(r'http\S+', '', comment['body'])
        output = nlp.annotate(text, properties={
                'annotators': 'tokenize,pos,parse',
                'outputFormat': 'json',
                'timeout': 10000
                })
        if isinstance(output,str):
            continue
        pos=()
        for s in output["sentences"]:
            s=[t["pos"] for t in s["tokens"]]
            pos=pos+(tuple(s))
        PartsSpeech = ["pos_{}".format(word) for word in pos if word in partsOfSpeech]
        pos_count=Counter(PartsSpeech) 
        features = {key: value for key, value in pos_count.items()}
        
        tmp=()
        for s in output["sentences"]:
            sentParse=s["parse"].encode('ascii','ignore').decode('ascii')
            tmp=tmp+ (tuple(PhrasePairs(nltk.Tree.fromstring(sentParse))))
            
        Phrases = ["phr_{}".format(phrase) for phrase in tmp if phrase in phrases]
        phrase_count=Counter(Phrases)
        features.update(phrase_count)
        comfeat={"id":comment['id'],"author":comment['author'],"subreddit":comment['subreddit']}
        features.update(comfeat)
        fwriter.writerow(features)

