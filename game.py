# ----------------------------------------------------------------------------
# Copyright (c) 2016 Joshua Dempster
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
#  * Neither the name of pyglet nor the names of its
#    contributors may be used to endorse or promote products
#    derived from this software without specific prior written
#    permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# ----------------------------------------------------------------------------

'''Trains a pair of neural networks to play dice poker.'''

import numpy as np
import neural_net as nn
import random
from time import sleep
import cProfile
import copy
import os

NFACES = 5 #possible outcomes per die
NDICE = 4 #how many dice each player gets

def weighted_choice(choices, weights):
    total = sum(w for c, w in zip(choices, weights))
    r = random.uniform(0, total)
    upto = 0
    for c, w in zip(choices, weights):
       if upto + w >= r:
          return c
       upto += w
      

class Bid:
    ''' Convenience class for managing a bid. Key attributes are:
            n: the number of dice in the bid
            die: the face shown by the dice
            index: unique integer associated with all possible bids, arranged in increasing bid order
            '''
    def __init__(self, index):
        self.nfaces = NFACES
        self.ndice = NDICE
        self.max = 2*self.nfaces*self.ndice
        self.n = 0
        self.die = 0
        self.convert(index)
        self.y = np.zeros(self.max+1)
        self._call = False

    def convert(self, index):
        '''set n and die for a given index'''
        self.index = index
        if index < 0:
            self.die = 0
            self.n = 0
            self._call = False
        elif index >= self.max:
            self._call = True
        else:
            self._call = False
            self.die = index%self.nfaces
            self.n = index/self.nfaces+1

    def set_y(self):
        '''convenience method; sets an array equivalent to a neural net output for the bid'''
        self.y[self.die + self.nfaces*(self.n-1)] = 1

    def clear(self):
        self.n = 0
        self.die = 0
        self.y[:] = 0
        self._call = False

    def is_in(self, count):
        '''check whether the bid exists in a given combination of dice. Count is a set of total dice for
            each face'''
        if self.index < 0 or self.n == 0:
            return False
        elif count[self.die] >= self.n:
            return True
        else:
            return False

    def is_possible(self, count):
        '''check whether the bid is possible given NDICE known dice and NDICE unknown dice'''
        if self.n <= count[self.die]+self.ndice:
           return True
        else:
           return False
       
    def set_index(self):
        if not self._call:
            self.index = self.die + self.nfaces*(self.n-1)
        else:
            self.index = self.max

    def get_count(self):
        count = np.zeros(self.nfaces, np.int)
        if self.index >= 0:
            count[self.die] = self.n
        return count

    def set_to_max(self, count):
        '''set self to highest bid for which self.is_possible(count) returns True'''
        #returns index of max of count with highest die
        self.die = self.nfaces - 1 - count[::-1].argmax()
        self.n = int(count[self.die])
        self.set_index()
        self._call = False

    def set_to_min(self, count):
        self.die = count.argmin()
        self.n = int(count[self.die])
        self.set_index()
        self._call = False

    def call(self):
        self.convert(self.max)

    def set_values(self, ndice, face):
        self.convert((ndice-1)*self.nfaces+face)

    def __cmp__(self, bid):
        if not isinstance(bid, Bid):
            raise ValueError, '%r is not a Bid instance' %bid
        if self._call and not bid._call:
            return 1
        elif not self._call and bid._call:
            return -1
        elif self._call and bid._call:
            return 0
        
        if self.n > bid.n:
            return 1
        elif self.n < bid.n:
            return -1
        else:
            if self.die > bid.die:
                return 1
            elif self.die < bid.die:
                return -1
            else:
                return 0

    def __iadd__(self, bid):
        if isinstance(bid, Bid):
            if self.die == bid.die:
                self.n += bid.n
            else:
                if bid.n > self.n:
                    self.n = bid.n
                    self.die = bid.die
                else:
                    pass
        elif isinstance(bid, int):
            self.convert(self.index + bid)
        else:
            raise ValueError, 'Bids can only be incremented by other bids or integers'
        return self

    def __repr__(self):
        if self._call:
            return 'called'
        else:
            return 'bid %i %is' %(self.n, self.die+1)
            

class Player:
    '''Abstract Player class. Includes a `count` attribute assigned at the beginning of the game'''
    name = 'generic player'
    def __init__(self):
        self.max = 2*NFACES*NDICE
        self.bid = Bid(-1)

    def play(self, bid, verbose):
        '''bid: Bid from previous player. Verbose will print information for debugging'''
        raise NotImplementedError, "base class method must be overridden"

    def setup(self, count):
        self.bid = Bid(-1)
        self.count = count

    def __repr__(self): return self.name

class Conservative(Player):
    '''will only make bids that can be satisfied by its own dice'''
    name = 'conservative player'
    
    def play(self, bid, verbose):
        if not bid.is_possible(self.count):
           self.bid.call()
           return self.bid
        new_bid = self.bid
        new_bid.clear()
        for index in range(bid.index+1, self.max+1):
            new_bid.convert(index)
            if new_bid.is_in(self.count):
                break
        return new_bid

class Regular(Player):
    '''will go one die higher than the maximum bid it can make with its own dice'''
    name = 'regular player'
    
    def setup(self, count):
        Player.setup(self, count)
        self.count = self.count[:] + 1.0
    
    def play(self, bid, verbose):
        if not bid.is_possible(self.count):
           self.bid.call()
           return self.bid
        new_bid = self.bid
        new_bid.clear()
        count = self.count
        for index in range(bid.index+1, self.max+1):
            new_bid.convert(index)
            if new_bid.is_in(self.count):
                break
        return new_bid

class Aggressive(Regular):
    '''like Conservative, but opens with its highest possible count'''
    name = 'aggressive player'
    
    def play(self, bid, verbose):
        if not bid.is_possible(self.count):
           self.bid.call()
           return self.bid
        new_bid = self.bid
        new_bid.clear()
        new_bid.set_to_max(self.count)
        if not new_bid > bid:
            new_bid.call()
        return new_bid

class Trusting(Player):
    '''Like Conservative, but adds opponent's last bid to its dice total before deciding whether
            the bid exists'''
    name = 'trusting player'
    
    def __init__(self):
        Player.__init__(self)
        self.limit = np.array([2*NDICE]*NFACES, np.int)
    
    def play(self, bid, verbose):
        if not bid.is_possible(self.count):
           self.bid.call()
           return self.bid
        new_count = np.copy(self.count)
        new_count += bid.get_count()
        np.putmask(new_count, new_count>self.limit, self.limit)
        self.bid.set_to_max(new_count)
        if self.bid == bid:
            self.bid.call()
        return self.bid

class Chaotic(Player):
    '''Bid unconnected to dice. Used to provide untrustworthy and extreme bid examples to
        the networks'''
    name = 'chaotic  player'
    
    def play(self, bid, verbose):
        if not bid.is_possible(self.count):
           self.bid.call()
           return self.bid
        if bid.index < 0:
            self.bid.convert(random.randint(bid.index+1, self.max))
            return self.bid
        elif random.random() > .5 and bid.n > 1:
            self.bid.call()
        else:
           try: self.bid.convert(random.randint(bid.index+1, self.max+1))
           except ValueError: assert False, 'bid invalid! %s (index=%i) call=%r' %(
               bid, bid.index, bid.call)
        return self.bid


class Human(Player):
    name = 'human'
    numberdict = {
        'one': '1',
        'two': '2',
        'three': '3',
        'four': '4',
        'five': '5',
        'ix': '6',
        'even': '7',
        'eight': '8',
        'nine': '9',
        'ten': '10'}
    
    def setup(self, count):
        Player.setup(self, count)
        dice = []
        for i, j in enumerate(self.count[:]):
            for k in range(j):
                dice.append(i+1)
        print 'You rolled %r' %dice

    def play(self, bid, verbose):
        if verbose:
            print 'Human playing'
        self.turn = 0
        self.turn = 1
        if bid.index >= 0:
            print 'Your opponent %r' %bid
        self.set_bid(bid, verbose)
        print 'you %r' %self.bid
        return self.bid

    def set_bid(self, bid, verbose):
        '''Get bid from human player.
            Includes some language parsing to allow for different bid entry styles'''
        string = raw_input('Enter your bid (number of dice first) or type call:\n')
        clean = ''
        clean = clean.join(s for s in string if s.isalnum())
        clean = clean.lower()
        clean = clean.strip('s')
        if verbose:
            print 'clean: %s' %clean
        
        if clean == 'call':
            self.bid.call()
            return
        elif 'call' in clean:
            print "I'm gonna take that as a call"
            self.bid.call()
            return
            
        out = []
        while len(out) < 2:
            found = False
            try:
                out.append(int(clean[0]))
                clean = clean[1:]
                found = True
            except ValueError:
                for key in self.numberdict:
                    try:
                        if clean.index(key) == 0:
                            out.append(int(self.numberdict[clean[:len(key)]]))
                            found = True
                            if verbose: print 'out: %r' %out
                            try:
                                clean = clean[len(key):]
                                if verbose: print 'clean: %s' %clean
                            except IndexError:
                                if len(out) < 2:
                                    print 'Bid too short!'
                                    self.set_bid(bid, verbose)
                                else:
                                    clean = ''
                            break
                    except ValueError:
                        if verbose: print '%s not in clean string' %key
                if not found:
                    print 'Bad bid! Try again'
                    self.set_bid(bid, verbose)
                    
        self.bid.set_values(out[0], out[1]-1)
        if len(clean) > 0:
            print 'Gonna give it my best guess'
        if self.bid <= bid:
            print 'You bid too low!'
            if verbose: print self.bid.index, bid.index
            self.set_bid(bid, verbose)


class AI(Player, nn.NeuralNet):
    '''Neural network player. Key attributes:
        x: list of np.array. Stores player dice and bidding history. Each array corresponds to a round
            in a game, each element in the array corresponds to a face, integer values are number of
            dice with that face. The first NFACES elements are the player's dice. Each following block
            of NFACES is a bid in reverse chronological order
        y: list of np.array. Stores the output of the network each round for reinforcement learning
        turn: an integer that internally tracks which round it is and whether this player is first or second
        '''
    regularization = 0.0001
    lm = .5 # lambda value for time-difference learning
    alpha = .005 # learning rate. Lower learning rate is slower but better retains old games
    punish_factor = .4 #factor by which punishment is discounted relative to rewards.

    def __init__(self, shape, max_rounds, name='ai player'):
        self.name = name
        Player.__init__(self)
        nn.NeuralNet.__init__(self, shape)
        self.nfaces = NFACES
        self.ndice = NDICE
        self.randomize(100.0)
        self.lower_bias(.01)
        
        self.x = [np.zeros(self.nfaces+(2*max_rounds+1)*self.nfaces) for i in range(max_rounds)]
        self.y = [np.zeros(1+2*self.ndice*self.nfaces, dtype=np.float32) for i in range(max_rounds)]
        self.ytest = np.zeros(1+2*self.ndice*self.nfaces, dtype=np.float32)
        self.ylen = (self.y)[0].shape[0]-1
        assert shape[0] == (self.x)[0].shape[0] + 1, 'mismatch between input vector (%i) and input layer (%i)'%(
            (self.x)[0].shape[0], shape[0])
        assert shape[-1] == (self.y)[0].shape[0], 'mismatch between output layer and bid permutations'
        self.turn = 0

    def setup(self, count):
        for x, y in zip(self.x, self.y):
            x[:] = 0
            x[:self.nfaces] = count[:]
            y[:] = 0
        self.bid = Bid(-1)
        self.turn = 0
        self.count = count

    def get_out(self, x):
        y = self.predict(x)
        indices = np.argsort(self.predict(x))
        #uncomment the following lines to encourage exploring new lines of play
        #leave commented to improve performance vs humans
##        for index in reversed(indices):
##            if y[index] > random.random():
##                return index
        return indices[-1]

    def play(self, bid, verbose):
        if bid.index >= 0:
            if self.turn == 0:
                self.turn = 1
        if bid.index == self.ylen-1:
            #this action is mandatory, so we don't bother training the network to do it
            self.bid.call()
            return self.bid
        
        count = bid.get_count()
        #add bid information to current and all future inputs for this game:
        for i, x in enumerate(self.x[self.turn/2:]):
            start = (2*i+1)*self.nfaces
            end = start + self.nfaces
            x[start:end] = count[:]
        if verbose:
            print '%r playing turn %i' %(self.name, self.turn)
            
        nrounds = self.turn/2
        out = self.get_out(self.x[nrounds])
        if self.turn == 0:
            self.ytest[self.ylen] = 0
            self.ytest[0:self.ylen] = 1
            while out == self.ylen:
                #calling is illegal on the first turn
                if verbose: print 'wrong out: %i %r' %(out, self.predict(self.x[nrounds]))
                self.train([(self.x[nrounds], self.ytest)], alpha=.1, max_iterations=10 ,
                           verbose=False)
                out = self.get_out(self.x[nrounds])
        elif out <= bid.index:
            #next bid must be greater than the previous bid
            self.ytest[0:bid.index+1] = 0
            self.ytest[bid.index+1:] = 1
            while out <= bid.index:
                if verbose:
                    print 'wrong out: %i %r' %(out, self.predict(self.x[nrounds]))
                    print 'training with %r' %self.ytest
                self.train([(self.x[nrounds], self.ytest)], alpha=.1, max_iterations=10,
                           verbose=False)
                out = self.get_out(self.x[nrounds])

        self.y[nrounds][:] = 0
        self.y[nrounds][out] = 1
        self.bid.convert(out)
        if verbose: print '%r outputting %i' %(self, out)
        #add bid information to inputs for all future rounds
        for i, x in enumerate(self.x[self.turn/2+1:]):
            start = (2*i+2)*self.nfaces
            end = start + self.nfaces
            x[start:end] = self.bid.get_count()[:]
##        if verbose:
##          print 'x: %r' %self.x
        self.turn += 2
        return self.bid

    def reward(self, verbose=False):
        #reinforces winning lines of play; assigns temporal credit equally to all moves
        nrounds = self.turn/2
        if nrounds == 0:
            print self.turns, self.x[0], self.y[0]
        if verbose:
            print 'rewarding %r nrounds=%i' %(self, nrounds)
            print 'last input: %r' %self.x[nrounds-1]
            print 'rewarded output: %r' %self.y[:nrounds]
            print 'predict before: %r' %self.predict(self.x[nrounds-1])
        self.train(zip(self.x[:nrounds], self.y[:nrounds]), alpha=self.alpha, max_iterations=1,
               verbose=False, regularization=self.regularization)
        if verbose:
            print 'predict after: %r' %self.predict(self.x[nrounds-1])

    def punish(self, history, verbose=False):
        #encourages exploring alternate play to avoid traps; history contains bid indices
        nrounds = self.turn/2
        turn = self.turn%2
        if verbose:
            print 'punishing %r: nrounds %i, turn %i' %(self, nrounds, turn)
            print 'history: %r' %history
        for r in range(nrounds):
            self.y[r][:] = 0
            self.y[r][history[2*r+turn]+1:] = 1
            if r==0 and turn == 0:
                self.y[r][self.ylen] = 0
            self.y[r][history[2*r+turn+1]] = 0
            #assign (dis)credit according to TD(lambda)
            alpha = self.lm**(nrounds - 1 - r) * self.alpha*self.punish_factor
            if verbose:
                print 'previous out: %i' %history[2*r+turn]
                print 'prediction: %r' %self.predict(self.x[r])
                print 'wrong answer: %i' %history[2*r+turn+1]
                print 'training for input %r' %self.x[r]
                print 'desired output %r with alpha %1.2f' %(self.y[r], alpha)
            self.train([(self.x[r], self.y[r])], alpha=alpha,
                       max_iterations=1, verbose=False, regularization=self.regularization)
            if verbose:
                print 'new prediction: %r' %self.predict(self.x[r])

    def load(self, name):
        replica = nn.load_net(name)
        self.matrices = copy.deepcopy(replica.matrices)


######################################################
######################################################
class Game:
    ''' Plays dice poker. Contains 'markov', a predefined policy for training networks; two neural
        networks AI1 and AI2 that go first and second respectively; and a human player to allow
        for human-machine play through the interpreter'''
    max_rounds = 6
    
    def __init__(self):
        self.ndice = NDICE
        self.nfaces = NFACES
        
        #set up predefined policy for training:
        self.conservative = Conservative()
        self.regular = Regular()
        self.aggressive = Aggressive()
        self.trusting = Trusting()
        self.chaotic = Chaotic()
        self.human = Human()
        self.markov_players = [self.conservative, self.regular, self.aggressive, self.trusting,
                               self.chaotic]
        self.weights = [.6, .8, .9, .5, .2] #weights based on win rates for each markov player style

        #set up neural networks
        ninput = (self.max_rounds+1)*2*self.nfaces+1
        noutput = 2*self.ndice*self.nfaces+1
        self.ai1 = AI((ninput, ninput, ninput, noutput), self.max_rounds, name='AI1')
        self.ai2 = AI((ninput, ninput, ninput, noutput), self.max_rounds, name='AI2')
        self.players = self.markov_players + [self.ai1, self.ai2, self.human]

        #set up trackers for win rates
        self.winrates = {}
        self.wins = {}
        self.losses = {}
        for player in self.players:
            self.winrates[player] = 0.0
            self.wins[player] = 0
            self.losses[player] = 0

        #preallocating arrays for playing many games faster
        self.count1 = np.zeros(self.nfaces, dtype=np.int)
        self.count2 = np.copy(self.count1)
        self.total = np.copy(self.count1)

    def clear_wins(self):
        for player in self.players:
            self.wins[player] = 0
            self.losses[player] = 0

    def save(self, name):
        if not os.path.isdir(os.getcwd()+'/' +name):
            os.mkdir(name)
        self.ai1.save(name + '/ai1.dat')
        self.ai2.save(name+'/ai2.dat')

    def load(self, name):
        self.ai1.load(name+'/ai1.dat')
        self.ai2.load(name+'/ai2.dat')

    def set_up_game(self, player1, player2, verbose):
        self.count1[:] = 0
        self.count2[:] = 0
        dice1 = [random.randint(1,self.nfaces) for i in range(self.ndice)]
        for i in dice1:
            self.count1[i-1] += 1
        dice2 = [random.randint(1,self.nfaces) for i in range(self.ndice)]
        for i in dice2:
            self.count2[i-1] += 1
        self.total = self.count1+self.count2
        player1.setup(self.count1)
        player2.setup(self.count2)
        self.dice = (dice1, dice2)

        if verbose:
            print "\nplayer 1 rolled %r (counts: %r)" %(dice1, self.count1)
            print  "player 2 rolled %r (counts: %r)" %(dice2, self.count2)
            print 'dice totals: %r' %self.total
            print 'game has dice totals %r' %self.total.tolist()


    def play_ai(self, player10, player20, verbose=False, new_setup=True):
        if player10 == 'markov':
            player1 = weighted_choice(self.markov_players, self.weights)
        else:
            player1 = player10
        if player20 == 'markov':
            player2 = weighted_choice(self.markov_players, self.weights)
        else:
            player2 = player20

        if verbose:
            print 'player 1: %r, player 2: %r' %(player1, player2)
            
        if new_setup:
            self.set_up_game(player1, player2, verbose)
        history = [0 for i in range(2*self.max_rounds +1)]
        history[0] = -1
        players = [player1, player2]
        human_flag = False
        if any([isinstance(player, Human) for player in players]):
            human_flag = True
            
        counts = [self.count1, self.count2]
        prev_bid = Bid(-1)
        nrounds = 0
        loser = None
        winner = None
        iplayer = 0
        
        #game loop
        while nrounds < self.max_rounds:
            current_bid = players[iplayer].play(prev_bid, verbose)
            history[2*nrounds+iplayer + 1] = current_bid.index
            if verbose:
                print 'player %i, %r (%i)' %(iplayer+1,
                                                 current_bid, current_bid.index)
                print 'history: %r' %history
            if current_bid._call:
                if human_flag and not isinstance(players[iplayer], Human):
                    print 'Your opponent called'
                if prev_bid.is_in(self.total):
                    loser = iplayer
                else:
                    loser = (iplayer+1)%2
                break
            else:
                prev_bid = current_bid
            iplayer = (iplayer+1)%2
            if iplayer == 0:
                nrounds += 1

        if verbose:
            if loser == 0:
                print '2 won!\n'
            elif loser == 1:
                print '1 won!\n'
            else:
                print 'A tie!\n'

        try:
            winner = (loser+1)%2
        except TypeError:
            pass
        try:
            if isinstance(players[winner], Human):
                print 'The dice showed %r and %r' %self.dice
                print 'You won!\n'
            elif isinstance(players[loser], Human):
                print 'The dice showed %r and %r' %self.dice
                print 'You lost!\n'
        except TypeError:
            if human_flag:
                print "It's a tie! (You exceeded the maximum allowed rounds. How intense!)\n"
        try: players[loser].punish(history, verbose)
        except AttributeError:
            pass
        try: players[winner].reward(verbose)
        except AttributeError:
            pass
        try:
            return players[winner], players[loser]
        except TypeError:
            return None, None

    def markov_training(self, ngames):
        '''used to determine the initial weights for the different predefined policies'''
        print 'evaluating Markovian policies'
        total_games = 2*(len(self.markov_players)-1)*ngames
        for player in self.markov_players:
            for player2 in self.markov_players:
                if player != player2:
                    print 'playing %r vs %r' %(player, player2)
                    for i in range(ngames):
                        winner, loser = self.play_ai(player, player2)
                        if winner:
                            self.wins[winner] += 1
                            self.losses[loser] += 1
        for key in self.winrates:
            self.winrates[key] = self.wins[key]*1.0/(self.wins[key]+self.losses[key]+.00001)
        print 'results:'
        for key, val in self.winrates.items():
            print key, val
 
    def train_ai(self, ngames, player1, player2, verbose=False, report_rate=500):
        for player in self.players:
            self.wins[player] = 0
            self.losses[player] = 0
        for i in range(ngames):
            if player1 == 'random':
                mplayer1 = random.choice([self.ai1, 'markov'])
            else:
                mplayer1 = player1
            if player2 == 'random':
                mplayer2 = random.choice([self.ai2, 'markov'])
            else:
                mplayer2 = player2
            if mplayer1 == mplayer2 == 'markov':
                mplayer1 = self.ai1
                mplayer2 = self.ai2
            winner, loser = self.play_ai(mplayer1, mplayer2, verbose=False)
            if winner:
                self.wins[winner] += 1
                self.losses[loser] += 1
            if not i%report_rate:
                print '%i games completed' %i
                print 'sample game: '
                winner, loser = self.play_ai(mplayer1, mplayer2, verbose=verbose)
                for player in self.players:
                    self.winrates[player] = self.wins[player]*1.0/(self.wins[player]+self.losses[player]+.0001)
                print '\n win rates: %r\n\n' %self.winrates
                self.clear_wins()

    def play_human(self, verbose=False):
        print 'Starting game'
        if random.random() > .5:
            player1 = self.human
            player2 = self.ai2
            print 'You go first'
        else:
            player1 = self.ai1
            player2 = self.human
            print 'Your opponent goes first'
        winner, loser = self.play_ai(player1, player2, verbose=verbose)
        keep_playing = raw_input('Play again? (y/n)\n').lower()
        try:
            if keep_playing[0] == 'y':
                    self.play_human(verbose)
        except IndexError:
            pass

             
if __name__ == '__main__':
    game = Game()
    game.load('test')
    #uncomment these lines to train from scratch; expect ~hour for training
    #recommend uncommenting the key lines in AI.get_out() to encourage exploration
    #       if training
##    game.train_ai(1000000, 'markov', game.ai2, report_rate=50000, verbose=True)
##    game.train_ai(1000000, game.ai1, 'markov',  report_rate=50000, verbose=True)
##    game.train_ai(1000000, game.ai1, game.ai2,  report_rate=50000, verbose=True)
##    game.save('test')
    print "Welcome to dice poker!"
    print "The rules are simple: each player rolls %i %i-sided dice without"
    print "showing the other player. They take turns bidding. Each bid states"
    print "a number of dice and a face. Example: the bid 'two 3s' means the player thinks"
    print "that when all the dice in the game are combined, there will be at least"
    print "two dice showing 3. When a player doesn't believe the last player's bid, that"
    print "player calls. If the last bid exists, the calling player loses; otherwise, they win.\n"
    game.play_human()
    game.save('test') 
