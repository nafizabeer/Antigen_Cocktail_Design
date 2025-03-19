# dictionary for Mixture project
import csv
from AMcode.antigens import cons, AgWT, AgC1, AgC2, AgC3, AgC4, AgC5, AgC6 # antigens creation

####### parameters #######
time_Steps = [2, 420, 780,900]
concseq0 = 0.14*6/2
concseq1 = 0.14*6/2
concseq2 = 0.40
flag = 0

GCDur1 = time_Steps[1]
GCDur2 = time_Steps[2]
GCDur3 = time_Steps[3]

dicAgs = {c:[AgC1,AgC2,AgC3,AgC4,AgC5,AgC6] for c in list(range(time_Steps[0],time_Steps[1]))}
dicAgs.update({c:[AgC1,AgC2,AgC3,AgC4,AgC5,AgC6] for c in list(range(time_Steps[1],time_Steps[2]))})

# dicAgs = {c:[AgC6]*6 for c in list(range(time_Steps[0],time_Steps[1]))}
# dicAgs.update({c:[AgC6]*6 for c in list(range(time_Steps[1],time_Steps[2]))})

dicconc = {c:concseq0 for c in list(range(time_Steps[0], time_Steps[1]))}
dicconc.update({c:concseq1 for c in list(range(time_Steps[1], time_Steps[2]))})
dicGCDur = {c:GCDur1 for c in list(range(time_Steps[0],time_Steps[1]+1))}
dicGCDur.update({c:GCDur2 for c in list(range(time_Steps[1]+1,time_Steps[2]+1))})


# with open('output/output_sequenceDict.csv', 'w') as output:
#     w = csv.writer(output)
#     w.writerows(dicAgs.items())

# with open('output/output_GC_duration.csv', 'w') as output:
#     w = csv.writer(output)
#     w.writerows(dicGCDur.items())
