'''
Created on 13 Feb 2015

@author: Josy
'''
from __future__ import print_function

import os
import sys

from myhdl import *


def mask( width, start):
    ''' building a simple mask '''
    return (2**width - 1) << start

def str2bits( s ):
    ''' converts a string into an integer representation to initialise an intbv with
        trouble ahead: what will we do with a binary string like e.g. "0100_0010" (0x42)
        we could check if it contains only 0, 1 or _
    '''
    # have a 'text' string to convert
    nrbits = len(s) * 8
    lval = 0
    # reverse the string
    for i in range(len(s)-1, -1, -1):
        lval = lval * 256 + ord(s[i])
    return nrbits, lval


# the pertinent RTL
def regrw( OFFSET, LENGTH, START, WIDTH, Clk, Reset, A, WD, Wr, Q, Pulse = None, BUSWIDTH = 32):
    ''' the WriteRead register '''

    if WIDTH > BUSWIDTH:
        if START != 0:
            raise ValueError(" Start should be zero for large (>'BUSWIDTH' bits) vectors")

        @always_seq( Clk.posedge, reset = Reset)
        def ccregrw():
            for i in range((WIDTH + BUSWIDTH - 1) // BUSWIDTH):
                if Wr and A == OFFSET +i:
                    if i != (WIDTH + BUSWIDTH - 1) // BUSWIDTH - 1:
                        Q.next[(i+1) * BUSWIDTH : i * BUSWIDTH] = WD
                        else:
                            # may have to assign a slice for the last iteration
                        Q.next[:(WIDTH // BUSWIDTH) * BUSWIDTH] = WD[WIDTH % BUSWIDTH:]
    else :
        @always_seq( Clk.posedge, reset = Reset)
        def ccregrw():
            if Wr and (A == OFFSET):
                Q.next = WD[WIDTH+START:START]

    if not Pulse is None:
        @always_seq( Clk.posedge, reset = Reset)
        def ccpulserw():
            Pulse.next = 0
            if Wr and (A == OFFSET + LENGTH - 1): # this always holds
                Pulse.next = 1
        return ccregrw, ccpulserw

    else:
        return ccregrw


def regac(OFFSET, START, WIDTH, Clk, Reset, A, WD, Wr, Q):
    ''' the AutoClear register '''
    @always_seq( Clk.posedge, reset = Reset)
    def ccregac():
        Q.next = 0
        if Wr and (A == OFFSET):
            Q.next = WD[WIDTH + START : START]

    return ccregac


def regroa(OFFSET, LENGTH, A, Rd, Pulse):
    ''' the ReadOnlyAck register '''
    @always_comb
    def ccregroa():
        Pulse.next = 0
        if Rd and (A == OFFSET + LENGTH - 1): # only on highest word
            Pulse.next = 1

    return ccregroa

def regrowc(OFFSET, LENGTH, A, Wr, Pulse):
    ''' the ReadOnlyWriteClear register '''
    @always_comb
    def ccregrowc():
        Pulse.next = 0
        if Wr and (A >= OFFSET) and (A < OFFSET+LENGTH): # on any word
            Pulse.next = 1

    return ccregrowc

def ccassign( D, Q):
    ''' to assign the output ports from internal signals '''
    @always_comb
    def cca():
        Q.next = D

    return cca


class ControlStatus(object):
    ''' a Class to build a Control and Status Register '''
    # mimic, to some extent, what was done in VHDL
    # constant regcontroldefinitions :  array_of_regcontroldesc
    #    := (( strfmt("YHeight" , 39)               , 32 ,  0 , 1 ,  REG , WriteRead           ,  11 ,  0 , false ) ,
    #        ( strfmt("FrameCount" , 39)            , 32 ,  1 , 1 ,  REG , ReadOnlyWriteClear  ,  32 ,  0 , false ) ,
    #        ( strfmt("AoI0Start" , 39)             , 32 ,  2 , 1 ,  REG , WriteRead           ,  11 ,  0 , false ) ,
    #        ( strfmt("AoI0End" , 39)               , 32 ,  2 , 1 ,  REG , WriteRead           ,  11 , 16 , false ) ,
    #        ( strfmt("AoI0Channel" , 39)           , 32 ,  2 , 1 ,  REG , WriteRead           ,   3 , 28 , false ) ,
    #        ( strfmt("AoI0Accumulate" , 39)        , 32 ,  2 , 1 ,  REG , WriteRead           ,   1 , 31 , false ) ,
    #        ( strfmt("AoI1Start" , 39)             , 32 ,  3 , 1 ,  REG , WriteRead           ,  11 ,  0 , false ) ,
    #        ( strfmt("AoI1End" , 39)               , 32 ,  3 , 1 ,  REG , WriteRead           ,  11 , 16 , false ) ,
    #        ( strfmt("AoI1Channel" , 39)           , 32 ,  3 , 1 ,  REG , WriteRead           ,   3 , 28 , false ) ,
    #        ( strfmt("AoI1Accumulate" , 39)        , 32 ,  3 , 1 ,  REG , WriteRead           ,   1 , 31 , false ) ,
    #        ( strfmt("XInvert" , 39)               , 32 ,  4 , 1 ,  REG , WriteRead           ,   1 ,  0 , false ) ,
    #        ( strfmt("Status" , 39)                , 32 ,  5 , 1 ,  REG , ReadOnly            ,  32 ,  0 , false ) ,
    #        ( strfmt("ResetCounters" , 39)         , 32 ,  6 , 1 ,  REG , AutoClear           ,   1 ,  3 , false ) ,
    #        ( strfmt("ReadFifo" , 39)              , 32 ,  7 , 1 ,  REG , ReadOnlyAck         ,   1 ,  3 , false ) ,
    #        ) ;


    def __init__(self, description = None, BUSWIDTH = 32):
        ''' '''
        # initially we don't do much more then starting a dictionary
        self._dict = {}
        self._isbuilt = False
        self.regs = []
        self.BUSWIDTH = BUSWIDTH
        if not description is None:
            for desc in description:
                self.addentry(desc[0], desc[1], desc[2], desc[3], desc[4], desc[5] if len(desc) > 5 else None)

    def regcount(self):
        ''' this gives the current used number of registers '''
        return len(self.regs)

    def addentry(self, portname, offset, start, mode, srcdst, pulse = None):
        '''
            add an entry in the register file
                portname: obvious
                offset: relative to 0, in words (of BUSWIDTH bits)
                start: offset in bits relative to bit 0 starting at offset
                mode: what we can do
                    WriteRead:             writing can generate an update signal, reading (back) is silent
                    ReadOnly:              silent reading of info with no sidesignals
                    ReadOnlyAck:           will also generate an update pulse to e.g. acknowledge reading from a Fifo
                    AutoClear:             the bit(s) written will only last for one clock period
                    ReadOnlyWriteClear :   any write sends a 'clr' pulse
                srcdst: in case of a ReadOnly, ReadOnlyAck or ReadOnlyWriteClear this must be the vector to be read
                        for write registers, this is the receiving Signal
                        for constants this must be a tuple( constant_value, bitwidth )
                pulse: in case an update, acknowledge or clear pulse is required
            Note that is allowed to reuse the same offset (and length) to define sub-fields with their own names
            to efficiently pack things a bit together. Caveat Scribor!

            Note that we do it little-endian over here (I never understood the benefits of big-endian ...)
        '''

        if isinstance(srcdst,list):
            lenlov    = len(srcdst)
            lenvector = len(srcdst[0])
            lenentry  = (lenvector + self.BUSWIDTH - 1) / self.BUSWIDTH
            lenregs   = lenlov * lenentry
            # see if we have to extend
            if len(self.regs) < (offset + lenregs):
                for _ in range(offset + lenregs - len(self.regs)):
                    self.regs.append( [0, portname] ) # not yet occupied

            # must check if they are free
            for i in range(lenregs):
                    if self.regs[offset + i][0] :
                        # already taken
                        raise ValueError( "Overlapping definitions {} and {}" .format(self.regs[offset+i][1], portname ))
                    else:
                        #mark all bits as taken
                        self.regs[offset + i][0] = 0xffffffff

            # if the pulse is also a list ...
            if isinstance(pulse, list):
                npulse = pulse
            else:
                npulse = []
                for _ in range(lenlov - 1):
                    npulse.append( None )
                npulse.append( pulse )

            # add in the dictionary
            for i in range( lenlov ):
                nname = '{}{}' .format(portname,i)
                if not nname in self._dict:
                    self._dict[ nname] = {  'isout'  : mode in ('WriteRead', 'AutoClear'),
                                            'offset' : offset + i,
                                            'length' : lenentry,
                                            'start'  : start,
                                            'width'  : lenvector,
                                            'mode'   : mode,
                                            'srcdst' : srcdst[i],
                                            'pulse'  : npulse[i],
                                            'isig'   : None
                                            }
                else:
                    raise ValueError( "Port {} already set in dictionary" .format( portname ))

        else:
            if isinstance(srcdst, (int, tuple)):
                # get length from second element in tuple
                lenvector = srcdst[1]
                lenregs = (lenvector + self.BUSWIDTH - 1) / self.BUSWIDTH

            elif isinstance(srcdst, intbv):
                # receiving a bitstring, already converted by MyHDL intbv()
                lenvector = len(srcdst)
                lenregs = (lenvector + self.BUSWIDTH - 1) / self.BUSWIDTH

            else:
                lenvector = len(srcdst)
                lenregs = (lenvector + self.BUSWIDTH - 1) / self.BUSWIDTH

            # see if we have to extend
            if len(self.regs) < (offset + lenregs):
                for _ in range(offset + lenregs - len(self.regs)):
                    self.regs.append( [0, portname] ) # not yet occupied
            # check / flag the used bits
            if lenvector > self.BUSWIDTH :
                for i in range(lenregs):
                    if self.regs[offset + i][0] :
                        # already taken
                        raise ValueError( "Overlapping definitions {} and {}" .format(self.regs[offset+i][1], portname ))
                    else:
                        #mark all bits as taken
                        self.regs[offset + i][0] = 0xffffffff
            else:
                m = mask( lenvector, start)
                if m & self.regs[offset][0]:
                    #overlapping definitions
                    raise ValueError( "Overlapping definitions {} and {}" .format(self.regs[offset][1], portname ))
                else :
                    # add bits for later checks
                    self.regs[offset][0] |= m

            # add description to the dictionary
            if not portname in self._dict:
                self._dict[ portname] = {'isout'  : mode in ('WriteRead', 'AutoClear'),
                                         'offset' : offset,
                                         'length' : lenregs,
                                         'start'  : start,
                                         'width'  : lenvector,
                                         'mode'   : mode,
                                         'srcdst' : srcdst,
                                         'pulse'  : pulse,
                                         'isig'   : None
                                         }
            else:
                raise ValueError( "Port {} already set in dictionary" .format( portname ))


    # these can only be called when all entries have been added
    # which is a reasonable requirement

    def build(self, Clk, Reset, A, WD, Wr, Rd, RQ):
        ''' the actual process of instantiating the registers '''

        if self._isbuilt:
            raise "Can only build register file once"
        else:
            self._isbuilt = True
            # now go through the dictionary and create the writer/readers
            rw  = [] # to collect the created registers
            ro  = [] # to collect the created registers
            roa = [] # to collect the created registers
            rwc = [] # to collect the created registers
            ac  = [] # to collect the created registers
            for key in self._dict.keys():
                ioffset = self._dict[key]['offset']
                ilength = self._dict[key]['length']
                istart  = self._dict[key]['start']
                iwidth  = self._dict[key]['width']
                ccmode  = self._dict[key]['mode']
                srcdst  = self._dict[key]['srcdst']
                if self._dict[key]['isout']:
                    # create an intermediate signal and append into the sub-dictionary held in the dictionary
                    self._dict[key].update( { 'isig' : Signal(intbv(0)[iwidth:]) } )

                if ccmode == 'WriteRead':
                    # create the register
                    rw.append( regrw( ioffset , ilength , istart, iwidth , Clk, Reset, A, WD, Wr, self._dict[key]['isig'], self._dict[key]['pulse'] ) )

                elif ccmode == 'ReadOnly':
                    # we insert the readonly object (signal, intbv, str) in the readback vector
                    if isinstance(srcdst, tuple):
                        if isinstance(srcdst[0], (int, long)):
                            # make an intbv
                            self._dict[key].update( { 'srcdst' :  intbv( srcdst[0] )[iwidth:] } )

                        elif isinstance(srcdst[0] , str) :
                            # replace it by an intbv
                            n,v = str2bits(srcdst[0])
                            if n <= iwidth:
                                self._dict[key].update( { 'srcdst' :   intbv( v )[iwidth:] } )
                            else:
                                raise ValueError( "Converted String {} having {} bits does not fit in specified {} bits" .format( srcdst, n, iwidth ))

                    elif isinstance(srcdst, intbv):
                        # well defined
                            pass

                    else:
                        # must be a Signal
                        pass

                elif ccmode == 'ReadOnlyAck':
                    # create the pulse
                    roa.append( regroa( ioffset , ilength , A, Rd, self._dict[key]['pulse']) )

                elif ccmode == 'ReadOnlyWriteClear':
                    # create the pulse
                    rwc.append( regrowc( ioffset , ilength , A, Wr, self._dict[key]['pulse']) )

                elif ccmode == 'AutoClear':
                    if ilength > 1:
                        raise ValueError( "AutoClear bits should fit inside a single word (of BUSWIDTH bits)")
                    # create the register
                    ac.append( regac( ioffset , istart, iwidth , Clk, Reset, A, WD, Wr, self._dict[key]['isig'] ) )

                else :
                    raise ValueError( 'Unsupported register mode <{}>' .format( ccmode ))

            # assemble the readback vector
            # by making a list and concatenating this into a wide vector adding zero padding bits in between non-contiguous objects
            # need to order things ...
            # extract a list of needed things, and sort it along the way
            rbl = sorted([ [val['offset'] , val['start'], val['width'], val['srcdst'], val['isig'], val['isout']] for val in self._dict.itervalues()])
            elements = []
            ccassigns = []
            index = 0
            for item in rbl:
                istart = item[0] * self.BUSWIDTH + item[1]
                if  istart > index:
                    # insert padding
                    elements.append(  intbv(0)[istart - index:] )
                    index += istart - index
                # insert the srcdst
                if item[5] :
                    elements.append( item[4])
                    # must also connect the output
                    ccassigns.append( ccassign( item[4], item[3]))
                else:
                    elements.append( item[3])

                index += item[2]

            # may have to pad the end(top) as well
            padding = index % self.BUSWIDTH
            if padding:
                # we're not on a boundary
                elements.append(  intbv(0)[self.BUSWIDTH - padding:] )

            self.dr = ConcatSignal(*reversed(elements))

        rdr = self._MMread(Clk, Reset, A, Rd, RQ, self.BUSWIDTH)

        return rw, ro, roa, rwc, ac, ccassigns, rdr

    # not - public?
    def _MMread(self, Clk, Reset, A, Rd, RQ, BUSWIDTH):
        ''' our (Avalon MM) Master reading (back) '''
        # can use a 'brute force' memory read ...
        @always_seq(Clk.posedge, reset = Reset)
        def mmrdr():
            if Rd:
                RQ.next = 0
                for i in range(len(self.regs)) :
                    if A == i :
                        RQ.next = self.dr[(i+1) * BUSWIDTH : i * BUSWIDTH]

        return mmrdr

    def genheaderfile(self):
        '''  a C-type header file is always nice to have?'''
        #TODO:
        pass





if __name__ == '__main__':
    # extracted utility functions
    def simulate( timesteps, mainclass ):
        """Runs simulation for MyHDL Class"""
        # Remove old vcd File, otherwise we get a list of renamed .vcd files lingering about
        filename = ( mainclass.__name__ + ".vcd" )
        if os.access( filename, os.F_OK ):
            os.unlink( filename )

        # Run Simulation
        tb = traceSignals( mainclass )
        sim = Simulation( tb )
        sim.run( timesteps )


    def genClk( Clk, tCK , ClkCount = None ):
        """ generate a clock and possible associate a clockcounter with it """
        while True:
            Clk.next = 1
            yield delay( int( tCK / 2 ) )
            Clk.next = 0
            if not ( ClkCount is None ):
                ClkCount.next = ClkCount + 1
            yield delay( int( tCK / 2 ) )


    def genReset( Clk, tCK, Reset ):
        """ although not strictly needed generates an asynchronous Reset with valid de-assert"""
        Reset.next = 1
        yield delay( int( tCK * 3.5 ) )
        Reset.next = 0


    def delayclks( Clk, tCK, count ):
        """ delays 'count' 'Clk's """
        for _ in range( count ):
            yield Clk.posedge
        yield delay( int( tCK / 4 ) )


    def MMwrite( Clk, tCK, A, WD, Wr, offset , data, WaitRequest = None ):
        """ write a register """
        WD.next = data
        A.next = offset
        Wr.next = 1
        if not ( WaitRequest is None ):
            yield Clk.posedge
            while WaitRequest:
                yield Clk.posedge
        else:
            yield Clk.posedge

        yield delay( int( tCK / 4 ) )
        Wr.next = 0


    def MMread( Clk, tCK, A, Rd , RQ, READ_DELAY, offset, WaitRequest = None, result = None ):
        """ read a register
             if waitrequest is used, READ_DELAY is ignored
        """
        A.next = offset
        if WaitRequest is None:
            Rd.next = 1
            yield Clk.posedge
            yield delay( int( tCK / 4 ) )
            Rd.next = 0  # must be a single pulse
            for _ in range( READ_DELAY ):
                yield Clk.posedge
            # only transfer after READ_DELAY
            if not result is None:
                result.next = RQ

            yield delay( int( tCK / 4 ) )

        else:
            # here we hold Rd
            Rd.next = 1
            while WaitRequest:
                yield Clk.posedge

            if not result is None:
                result.next = RQ

            yield delay( int( tCK / 4 ) )
            Rd.next = 0


    def flatten( lov, f ):
        WIDTH_V = len( lov[0] )
        @always_comb
        def fl():
            for i in range( len( lov ) ):
                f.next[( i + 1 ) * WIDTH_V:i * WIDTH_V] = lov[i]

        return fl


    WIDTH_A = 8
    USE_LIST = True
def tcsr(Clk, Reset, A, WD, Wr, Rd, RQ, Status, Status2, TestBit, TestVector, TestVector2, Update,
         TestVector3,  Pulse, TestAutoClearBit, LargeVector, FlattenedListOfVectors, Status3, Pulse3,
         SingleReadOnlyBit, SingleROIntbvBit):
    ''' a routine to actually build a csr
        both for Simulation and Conversion
    '''

    # mimic a 1D*1D type
    # as this is top file to be converted
    llov = [Signal(intbv(0)[24:]) for _ in range(3)]

    # we have two ways to build a registerfile
    if not USE_LIST:
        # initialise the control/status object
        csr = ControlStatus()
        # Add the definitions one by one
        # These fields can be set in random order
        # but for sanity reasons, better keep the offsets in sequence
        # The addentry function does some checking, but e.g. the optional update pulses are not checked
        # for double use, simulation will work, but the (VHDL) conversion will raise a 'multiple drivers' error
        # The 'length' and 'width' could be inferred from the Signals themselves
        # except for constants where we could use a tuple and also specify the desired width
        # unfortunately Signals have no name assigned here, so we still need the 'double' reference
        csr.addentry('TestBit'            ,  0,    0, 'WriteRead'             , TestBit)
        csr.addentry('TestAutoClearBit'   ,  0,    2, 'AutoClear'             , TestAutoClearBit)
        csr.addentry('SingleReadOnlyBit'  ,  0,    4, 'ReadOnly'              , SingleReadOnlyBit)
        csr.addentry('SingleROIntbvBit'   ,  0,    5, 'ReadOnly'              , SingleROIntbvBit)
        csr.addentry('TestVector'         ,  1,    0, 'WriteRead'             , TestVector )
        csr.addentry('TestVector2'        ,  1,   16, 'WriteRead'             , TestVector2 )
        csr.addentry('Status'             ,  2,    0, 'ReadOnly'              , Status)
        # large parameters (> BUSWIDTH bits) have to start from 0
        csr.addentry('LargeVector'        ,  3,    0, 'WriteRead'             , LargeVector )
        csr.addentry('Status2'            ,  6,    0, 'ReadOnlyAck'           , Status2, Pulse)
        csr.addentry('Status3'            ,  7,    0, 'ReadOnlyWriteClear'    , Status3, Pulse3)
        # leaving a large gap to pad up (48 bits)
        csr.addentry('TestVector3'        ,  9,    0, 'WriteRead'             , TestVector3)
        # Mimic a 1D*1D type
        # The optional update pulse is only active at the last sub-register,
        # unless it is a list of equal dimension too (Tip: specify None for unused ones)
        csr.addentry('ListOfVectors'      , 10,    0, 'WriteRead'             , llov, [None, Update, None])
        # Finally we have some 'constant' data, like e.g. Version Numbers
        csr.addentry('42'                 ,  0,   16, 'ReadOnly'              , (42,8)) # "0100_0010") # trying a constant !this is now broken!
        csr.addentry('BitString'          ,  0,   24, 'ReadOnly'              , intbv("0100_0010"))
        csr.addentry('MyHDL'              , 13,    0, 'ReadOnly'              , ('MyHDL0.9', 64)) # trying another constant
        csr.addentry('C-Cam'              , 15,    0, 'ReadOnly'              , ('Ccam', 32)) # trying another constant

    else:
        # use a list of descriptors
        description =  (('TestBit'            ,  0,    0, 'WriteRead'             , TestBit),
                        ('TestAutoClearBit'   ,  0,    2, 'AutoClear'             , TestAutoClearBit),
                        ('SingleReadOnlyBit'  ,  0,    4, 'ReadOnly'              , SingleReadOnlyBit),
                        ('SingleROIntbvBit'   ,  0,    5, 'ReadOnly'              , SingleROIntbvBit),
                        ('TestVector'         ,  1,    0, 'WriteRead'             , TestVector ),
                        ('TestVector2'        ,  1,   16, 'WriteRead'             , TestVector2 ),
                        ('Status'             ,  2,    0, 'ReadOnly'              , Status),
                        # large parameters (> 'BUSWIDTH' bits) have to start from 0
                        ('LargeVector'        ,  3,    0, 'WriteRead'             , LargeVector ),
                        ('Status2'            ,  6,    0, 'ReadOnlyAck'           , Status2, Pulse),
                        ('Status3'            ,  7,    0, 'ReadOnlyWriteClear'    , Status3, Pulse3),
                        # leaving a large gap to pad up (48 bits)
                        ('TestVector3'        ,  9,    0, 'WriteRead'             , TestVector3),
                        # Mimic a 1D*1D type
                        # The optional update pulse is only active at the last sub-register,
                        # unless it is a list of equal dimension too (Tip: specify None for unused ones)
                        ('ListOfVectors'      , 10,    0, 'WriteRead'             , llov, [None, Update, None]),
                        # Finally we have some 'constant' data, like e.g. Version Numbers
                        ('42'                 ,  0,   16, 'ReadOnly'              , (42,8)),
                        # specifying "0100_0010" directly doesn't work as it will be interpreted as a general string
                        ('BitString'          ,  0,   24, 'ReadOnly'              , intbv("0100_0010")),
                        # some general strings
                        ('MyHDL'              , 13,    0, 'ReadOnly'              , ('MyHDL0.9', 64)),
                        ('C-Cam'              , 15,    0, 'ReadOnly'              , ('Ccam', 32)) # yes, that's us!
                       )
        # and pass this list to the constructor
        csr = ControlStatus( description )

    # this will build the control and status register
    # at the moment the width of A is a chicken-or-egg issue, i.o.w. you have to check it
    control = csr.build(Clk, Reset, A, WD, Wr, Rd, RQ)
    # flatten the generated ListOfVectors in to one vector
    # this is only required here as this is a top-level module
    mimic = flatten( llov, FlattenedListOfVectors )

    return control, mimic

def tb_ControlStatus():
    Clk                     =  Signal(bool(0))
    Reset                   =  ResetSignal(0, active=1, async=True)
    A                       =  Signal(intbv(0)[WIDTH_A:])
    WD, RQ                  = [Signal(intbv(0)[32:]) for _ in range(2)]
    Wr , Rd                 = [Signal(bool(0)) for _ in range(2)]
    Status                  =  Signal(intbv(0)[32:])
    Status2                 =  Signal(intbv(0)[32:])
    Pulse                   =  Signal(bool(0))
    Status3                 =  Signal(intbv(0)[16:])
    Pulse3                  =  Signal(bool(0))
    TestAutoClearBit        =  Signal(bool(0))
    TestBit                 =  Signal(bool(0))
    TestVector              =  Signal(intbv(0)[10:])
    TestVector2             =  Signal(intbv(0)[16:])
    TestVector3             =  Signal(intbv(0)[32:])
    LargeVector             =  Signal(intbv(0)[72:])
    FlattenedListOfVectors  =  Signal(intbv(0)[3 * 24:])
    Update                  =  Signal(bool(0))
    SingleReadOnlyBit       =  Signal(bool(0))
    SingleROIntbvBit        =  Signal(intbv(0)[1:])

    dut = tcsr(Clk, Reset, A, WD, Wr, Rd, RQ, Status, Status2, TestBit, TestVector, TestVector2,  Update,
               TestVector3, Pulse, TestAutoClearBit, LargeVector, FlattenedListOfVectors, Status3, Pulse3,
               SingleReadOnlyBit, SingleROIntbvBit)
    ClkCount = Signal( intbv( 0 )[32:])
    tCK = 20

    @instance
    def clkgen():
        yield genClk(Clk, tCK, ClkCount)

    @instance
    def resetgen():
        yield genReset(Clk, tCK, Reset)

    @instance
    def stimulus():
        yield delayclks(Clk, tCK, 10)
        Status.next = 1
        Status2.next = 2
        Status3.next = 3
        for i in range(16):
            yield MMread(Clk, tCK, A, Rd, RQ, 1, i, None, None)

        # write a few things
        yield MMwrite(Clk, tCK, A, WD, Wr,  0, 1)
        yield delayclks(Clk, tCK, 2)
        yield MMwrite(Clk, tCK, A, WD, Wr,  0, 0)
        yield MMwrite(Clk, tCK, A, WD, Wr,  0, 4)
        yield MMwrite(Clk, tCK, A, WD, Wr,  1, 513 << 16 | 0x1)
        yield MMwrite(Clk, tCK, A, WD, Wr,  1, 513 << 16 | 0x2)
        yield MMwrite(Clk, tCK, A, WD, Wr,  3, 0xcccccccc)
        yield MMwrite(Clk, tCK, A, WD, Wr,  4, 0x33333333)
        yield MMwrite(Clk, tCK, A, WD, Wr,  5, 0xaa)
        yield MMwrite(Clk, tCK, A, WD, Wr,  5, 0x55)
        yield MMwrite(Clk, tCK, A, WD, Wr,  7, 0)
        yield MMwrite(Clk, tCK, A, WD, Wr,  9, 0x6d616363)
        yield MMwrite(Clk, tCK, A, WD, Wr, 10, 0x10)
        yield MMwrite(Clk, tCK, A, WD, Wr, 11, 0x20)
        yield MMwrite(Clk, tCK, A, WD, Wr ,12, 0x30)

        # read back and check
        _, myhdl09 = str2bits('MyHDL0.9')
        cd = [ (0x42 << 24) + (42 << 16), 513 << 16 | 0x2, 1 , 0xcccccccc,  0x33333333, 0x55, 2, 3, 0,  0x6d616363, 0x10, 0x20, 0x30 , myhdl09 & 0xffffffff, myhdl09 >> 32,   0x6d616343   ]
        for i in range(16):
            yield MMread(Clk, tCK, A, Rd, RQ, 1, i, None, None)
            if RQ != cd[i]:
                print( "Fail at offset {}: expected {} <> received {}" .format(i, cd[i], hex(RQ)) )

        raise StopSimulation

    return instances()


def convert():

    Clk                 = Signal(bool(0))
    Reset               = ResetSignal(0, active=1, async=True)
    A                   = Signal(intbv(0)[WIDTH_A:])
    WD, RQ              = [Signal(intbv(0)[32:]) for _ in range(2)]
    Wr , Rd             = [Signal(bool(0)) for _ in range(2)]
    Status              = Signal(intbv(0)[32:])
    Status2             = Signal(intbv(0)[32:])
    Pulse               = Signal(bool(0))
    Status3             = Signal(intbv(0)[16:])
    Pulse3              = Signal(bool(0))
    TestAutoClearBit    = Signal(bool(0))
    TestBit             = Signal(bool(0))
    TestVector          = Signal(intbv(0)[10:])
    TestVector2         = Signal(intbv(0)[16:])
    TestVector3         = Signal(intbv(0)[32:])
    LargeVector         = Signal(intbv(0)[72:])
    FlattenedListOfVectors  =  Signal(intbv(0)[3 * 24:])
    Update              = Signal(bool(0))
    SingleReadOnlyBit       =  Signal(bool(0))
    SingleROIntbvBit        =  Signal(intbv(0)[1:])

    # As this is an 'internal module' ( i.e. not exposed to Qsys)
    # we can stick to use unsigned vectors as ports
    # but eventually some of the ports will come from the top-level module (exposed to Qsys)
    # and will not be 'numeric'
        # Force std_logic_vector ports instead of unsigned ports
    toVHDL.std_logic_ports = True
    # Convert
    toVHDL(tcsr, Clk, Reset, A, WD, Wr, Rd, RQ, Status, Status2, TestBit, TestVector, TestVector2,  Update,
           TestVector3, Pulse, TestAutoClearBit, LargeVector, FlattenedListOfVectors, Status3, Pulse3,
           SingleReadOnlyBit, SingleROIntbvBit)
    toVerilog(tcsr, Clk, Reset, A, WD, Wr, Rd, RQ, Status, Status2, TestBit, TestVector, TestVector2,  Update,
              TestVector3, Pulse, TestAutoClearBit, LargeVector, FlattenedListOfVectors, Status3, Pulse3,
              SingleReadOnlyBit, SingleROIntbvBit)


    simulate(3000, tb_ControlStatus)
    convert()
    print('All done!')
    sys.exit(0)

