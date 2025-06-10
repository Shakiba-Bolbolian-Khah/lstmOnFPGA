inputs = []
dspStat = []


newInput = {}
newInput['m']= 128
newInput['n'] = 128
newInput['p'] = 128
newInput['batch'] = 512
newInput['hiddenUnits'] = 64 * newInput['batch']
newInput['bitWidth'] = 16
newInput['mulPerDSP'] = 2
newInput['MVMParDeg'] = 2
newInput['cc'] = 4199918 #14913
newInput['dsp'] = 384
newInput['tag'] = 'RNN-128'
inputs += [newInput]


newInput = {}
newInput['m']= 256
newInput['n'] = 256
newInput['p'] = 256
newInput['batch'] = 256
newInput['hiddenUnits'] = 128 * newInput['batch']
newInput['bitWidth'] = 16
newInput['mulPerDSP'] = 2
newInput['MVMParDeg'] = 2
newInput['cc'] = 8937045
newInput['dsp'] = 768
newInput['tag'] = 'RNN-256'
inputs += [newInput]

newInput = {}
newInput['m']= 512
newInput['n'] = 512
newInput['p'] = 512
newInput['batch'] = 512
newInput['hiddenUnits'] = 256 * newInput['batch']
newInput['bitWidth'] = 16
newInput['mulPerDSP'] = 2
newInput['MVMParDeg'] = 1
newInput['cc'] = 69549846
newInput['dsp'] = 768
newInput['tag'] = 'RNN-512'
inputs += [newInput]

newInput = {}
newInput['m']= 1024
newInput['n'] = 1024
newInput['p'] = 1024
newInput['batch'] = 256
newInput['hiddenUnits'] = 512 * newInput['batch']
newInput['bitWidth'] = 8
newInput['mulPerDSP'] = 2
newInput['MVMParDeg'] = 1
newInput['cc'] = 136871559
newInput['dsp'] = 1518
newInput['tag'] = 'RNN-1024'
inputs += [newInput]



dspStat += [[192*2*0.2, '- 192 DSP']]
dspStat += [[384*2*0.2, '- 384 DSP']]
dspStat += [[768*2*0.2, '- 768 DSP']]


def rnn_roofline(inputs):
    ioRoof = 6.9
    totalDSP = 1518
    rooflinePoints = []
    for i in inputs:
        m = i['m']
        n = i['n']
        p = i['p']
        hiddenUnits = i['hiddenUnits']
        cc = i['cc']
        bitWidth = i['bitWidth']
        dsp = i['dsp']
        mulPerDSP = i['mulPerDSP']

        ops_h = m*(n+m) #m*( 2*n+ 2*m- 1)
        ops_y = p*m # p*( 2*m -1)
        ops = hiddenUnits *( ops_h+ ops_y)
        print("#OPS:", ops)
        print("#OPS/cycle:", ops/cc)

        f = 200 * (10**6)

        gops = ops * f / cc
        print("***Performance***\n#OPS/s:", gops, "\n#GOPS/s:", gops * 10**(-9))


        mem = ((hiddenUnits * n)+(m*m)+(n*m)+(p*m) + (p*hiddenUnits)) * bitWidth / 8

        oi = ops/ mem
        print("***Operation (Computational) Intensity***\n #OI:", oi)


        muls = hiddenUnits*( m*( m+ n+ p))
        # mulPerDSP = 1 if bitWidth > 16 else int(36/bitWidth)
        # print("Mul/DSP:", mulPerDSP)
        # print("Mul: ", muls)

        dspEff = muls / (dsp * mulPerDSP*cc)
        print("DSP efficiency:", dspEff)


        peakPerf = totalDSP * mulPerDSP * f * (10 ** (-9))
        print("(Computational Roof) Peak Performance: #GOPs/s", peakPerf)


        print("(I/O Bandwidth Roof): (GB/sec)", ioRoof)

        rooflinePoint = min(peakPerf, oi * ioRoof)
        print("***Roofline Point***\nX: ", oi, ", Y:  ", gops * 10**(-9), ",  Roofline Point:  ", rooflinePoint)

        rooflinePoints += [[round(oi,2), round(gops * 10**(-9),2), i['tag'], dsp]]
        print('-----------------------------------------------------------------------')

    return rooflinePoints

