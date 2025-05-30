import random
import numpy as np
from abc import ABC, abstractmethod

        
class Fragment():
    def __init__(self, type, duration, channel, packet, intensity):
        self.packet = packet
        self.duration = duration
        self.success = 0
        self.transmitted = 0
        self.type = type
        self.channel = channel
        self.timestamp = 0
        self.id = id(self)
        self.collided = []
        self.intensity = intensity

class Packet():
    def __init__(self, node_id, obw, headers, payloads, header_duration, payload_duration, fading_generator):
        self.id = id(self)
        self.node_id = node_id
        self.index_transmission = 0
        self.success = 0
        self.channels = random.choices(range(obw), k=headers+payloads)
        self.fragments = []

        #Na geração dos fragmentos já coloca uma propriedade de intensidade
        for h in range(headers):
            intensity = fading_generator.fading_function()
            self.fragments.append(Fragment('header',header_duration, self.channels[h], self.id, intensity))
        for p in range(payloads):
            intensity = fading_generator.fading_function()
            self.fragments.append(Fragment('payload',payload_duration, self.channels[p+h+1], self.id, intensity))

    def next(self):
        self.index_transmission+=1
        try:
            return self.fragments[self.index_transmission-1]
        except:
            return False
        
# Instead of grid selection, we consider one grid of obw (usually 35 for EU) channels, as it is faster to simulate and extrapolate the number.
# Later we can implement the grid selection in case of interest of studying it.
#    def new_channels(self, obw, fragments):
#        self.channels = random.sample(range(obw), fragments)


class Traffic(ABC):
    @abstractmethod
    def __init__(self, traffic_param):
        self.traffic_param = traffic_param

    @abstractmethod
    def traffic_function(self):
        pass

class Fading(ABC):
    @abstractmethod
    def __init__(self, fading_param):
        self.fading_param = fading_param
    
    @abstractmethod
    def fading_function(self):
        pass
        
class Node():
    def __init__(self, obw, headers, payloads, header_duration, payload_duration, transceiver_wait, traffic_generator, fading_generator, max_distance, transmission_power):
        self.id = id(self)
        self.transmitted = 0
        self.traffic_generator = traffic_generator
        self.fading_generator = fading_generator
        self.transceiver_wait = transceiver_wait
        # Packet info that Node has to store
        self.obw = obw
        self.headers = headers
        self.payloads = payloads
        self.header_duration = header_duration
        self.payload_duration = payload_duration
        self.packet = Packet(self.id, self.obw, self.headers, self.payloads, self.header_duration, self.payload_duration, self.fading_generator)
        self.distance = random.uniform(0, max_distance)
        self.transmission_power = transmission_power

    def next_transmission(self):
        return self.traffic_generator.traffic_function()

    def end_of_transmission(self):
        self.packet = Packet(self.id, self.obw, self.headers, self.payloads, self.header_duration, self.payload_duration, self.fading_generator)

    def transmit(self, env, bs):
        while 1:
            #time between transmissions
            yield env.timeout(self.next_transmission())
            self.transmitted += 1
            bs.add_packet(self.packet)
            next_fragment = self.packet.next()
            first_payload = 0
            while next_fragment:
                if first_payload == 0 and next_fragment.type=='payload': #account for the transceiver wait time between the last header and first payload fragment
                    first_payload=1
                    yield env.timeout(self.transceiver_wait)
                next_fragment.timestamp = env.now
                #checks if the fragment is colliding with the fragments in transmission now
                bs.check_collision(next_fragment)
                #add the fragment to the list of fragments being transmitted.
                bs.receive_packet(next_fragment)
                #wait the duration (time on air) of the fragment
                yield env.timeout(next_fragment.duration)
                #removes the fragment from the list.
                bs.finish_fragment(next_fragment, self.distance, self.transmission_power)
                #check if base can decode the packet now.
                #tries to decode if not decoded yet.
                if self.packet.success == 0:
                    bs.try_decode(self.packet,env.now)
                #select the next fragment
                next_fragment = self.packet.next()
            
            #end of transmission procedure
            self.end_of_transmission()

class Base():
    def __init__(self, obw, threshold, sensitivity):
        self.id = id(self)
        self.transmitting = {}
        for channel in range(obw):
            self.transmitting[channel] = []
        self.packets_received = {}
        self.threshold = threshold
        self.sensitivity = sensitivity

    def add_packet(self, packet):
        pass

    def add_node(self, id):
        self.packets_received[id] = 0

    def receive_packet(self, fragment):
        self.transmitting[fragment.channel].append(fragment)

    def finish_fragment(self, fragment, distance, transmission_power):
        self.transmitting[fragment.channel].remove(fragment)
        transmission_power_W = 10 ** (transmission_power / 10) / 1000
        sensitivity_W = 10 ** (self.sensitivity / 10) / 1000

        snr = (transmission_power_W * (fragment.intensity**2))/(distance**4)

        #só tem sucesso se além de não ter colisão, intensidade maior que threshold
        if len(fragment.collided) == 0 and snr > sensitivity_W:
            fragment.success = 1
        fragment.transmitted = 1
 
    def check_collision(self,fragment):
        for f in self.transmitting[fragment.channel]:
            f.collided.append(fragment)
            fragment.collided.append(f)

    def try_decode(self,packet,now):
        #na contagem, só contar quem tá acima do fading threshold
        #h_success = sum( ((len(f.collided)==0) and f.transmitted==1) if (f.type=='header') else 0 for f in packet.fragments)
        #p_success = sum( ((len(f.collided)==0) and f.transmitted==1) if (f.type=='payload') else 0 for f in packet.fragments)

        h_success = sum(1 for f in packet.fragments if f.type == 'header' and f.success == 1)
        p_success = sum(1 for f in packet.fragments if f.type == 'payload' and f.success == 1)

        success = 1 if ((h_success>0) and (p_success >= self.threshold)) else 0
        if success == 1:
            self.packets_received[packet.node_id] += 1
            packet.success = 1
            return True
        else:
            return False
