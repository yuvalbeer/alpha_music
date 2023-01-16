# %%
import mido
import numpy as np
import matplotlib.pyplot as plt 
import string 
import dissonant as D

def msg2dict(msg): # a costume function with a messege variable in it
    result = dict()
    if 'note_on' in msg:
        on_ = True
    elif 'note_off' in msg:
        on_ = False
    else:
        on_ = None
    #conditions for when note is on/off, if it is then on_ = true and if it isnt then on_ = false

    result['time'] = int(msg[msg.rfind('time'):]
    # time in the result array equals to an integer of the messege's last mentioned 'time' value
    .split(' ')[0].split('=')[1]
    # then 
    .translate(str.maketrans({a: None for a in string.punctuation})))

    if on_ is not None: # if on_ is equal to false/true
        for k in ['note', 'velocity']:
            result[k] = int(msg[msg.rfind(k):]
            .split(' ')[0].split('=')[1]
            .translate(str.maketrans({a: None for a in string.punctuation})))
    return [result, on_]

'''
The function msg2dict extracts important informationP
(note, velocity, time, on or off) from each message.
'''

def switch_note(last_state, note, velocity, on_=True):
    # piano has 88 notes, corresponding to note id 21 to 108, any note out of this range will be ignored
    result = [0] * 88 if last_state is None else last_state.copy()
    if 21 <= note <= 108:
        result[note-21] = velocity if on_ else 0
    return result

'''
The function switch_note changes the last_state 
(the state of the 88 note at the previous time step)
based on new value of note, velocity, note on or note off.
The state of each time step contains 88 values.
'''

def get_new_state(new_msg, last_state):
    #print(new_msg)
    new_dict, on_ = msg2dict(str(new_msg))
    #print(new_dict)
    new_state = switch_note(last_state, note=new_dict['note'], velocity=new_dict['velocity'], on_=on_) if on_ is not None else last_state
    #print(len(new_state))
    #print(new_state)
    return [new_state, new_dict['time']]


'''
The function track2seq takes the threshold of minimum number of messages into consideration while filtering tracks,
and combines all tracks into one numpy array.
If two sound track on the same note at the same time, it takes the larger velocity.
'''

def track2seq(track):
    # piano has 88 notes, corresponding to note id 21 to 108, any note out of the id range will be ignored
    result = []
    jump = []
    msg = str(track[0])
    last_state, last_time = get_new_state(msg, [0]*88)
    for i in range(1, len(track)):
        msg = str(track[i])
        #check if state dosnt change and there is still a message ???????? 
        new_state, new_time = get_new_state(msg, last_state)
        if new_time > 0:
            #print(new_time)
            result += [last_state]*new_time
            jump.append(len(result))
        last_state, last_time = new_state, new_time
        #debug = str(last_state) + ":"+ str(last_time) 
    return jump , result

'''
The function track2seq converts each message in a track to a list of 88 values,
and stores each list in the result list in order.
'''

def mid2arry(mid, min_msg_pct=0.1):
    tracks_len = [len(tr) for tr in mid.tracks]
    min_n_msg = max(tracks_len) * min_msg_pct
    # convert each track to nested list
    all_arys = []
    all_jumps = []
    
    for i in range(len(mid.tracks)):
        if len(mid.tracks[i]) > min_n_msg:
            #print("==========================================")
            #print(mid.tracks[i])
            jump_i , ary_i = track2seq(mid.tracks[i])
            #print("+++++++++++++++++++++++++++++++++++++++++++++++++")
            #print("==========================================")
            all_arys.append(ary_i)
            all_jumps.append(jump_i)
    # make all nested list the same length
    max_len = max([len(ary) for ary in all_arys])
    for i in range(len(all_arys)):
        if len(all_arys[i]) < max_len:
            all_arys[i] += [[0] * 88] * (max_len - len(all_arys[i]))
    all_arys  = np.array(all_arys)
    all_arys = all_arys.max(axis=0)
    # trim: remove consecutive 0s in the beginning and at the end
    sums = all_arys.sum(axis=1)
    ends = np.where(sums > 0)[0]
    all_jumps = np.array(all_jumps)
    return all_arys[min(ends): max(ends)],all_jumps


def note_number_to_name(note_number):
    """Convert a MIDI note number to its name, in the format
    ``'(note)(accidental)(octave number)'`` (e.g. ``'C#4'``).
    Parameters
    ----------
    note_number : int
        MIDI note number.  If not an int, it will be rounded.
    Returns
    -------
    note_name : str
        Name of the supplied MIDI note number.
    Notes
    -----
        Thanks to Brian McFee.
    """

    # Note names within one octave
    semis = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    # Ensure the note is an int
    note_number = int(np.round(note_number))

    # Get the semitone and the octave, and concatenate to create the name
    return semis[note_number % 12] + str(note_number//12 - 1)

def note_number_to_hz(note_number):
    """Convert a (fractional) MIDI note number to its frequency in Hz.
    Parameters
    ----------
    note_number : float
        MIDI note number, can be fractional.
    Returns
    -------
    note_frequency : float
        Frequency of the note in Hz.
    """
    # MIDI note numbers are defined as the number of semitones relative to C0
    # in a 440 Hz tuning
    return 440.0*(2.0**((note_number - 69)/12.0))


#a new fuction that gets the changes of the notes in specific times
def get_notes(result_array,jump_array): 
    dictinary_array = []
    start_time  = 0
    for element in jump_array[0]:
        len_result_array = result_array.shape[0]
        if element < len_result_array: 
            res = result_array[element]
            notes    = [i+21 for i, e in enumerate(res) if e != 0]
            velocity = [float(e) for e in res if e != 0]
            hertz = []  
            name = []  
            for note in notes:
                hertz.append(note_number_to_hz(note))
                name.append(note_number_to_name(note))
            
            if len(notes) > 1: 
                dissonance = D.dissonance(np.array(hertz),np.array(velocity), model='sethares1993')
            else:
                dissonance = 0
            
            end_time = element   
            dictinary = dict({'start': start_time, 'end': end_time, 'name': name, 'notes': notes,'velocity': velocity, 'hertz': hertz, 'dissonance': dissonance}) # i read that velocity = amplitude
            start_time = end_time
            dictinary_array.append(dictinary)
    
    return dictinary_array 



def calc_disaonant(mid): 
    result_array,jump_array = mid2arry(mid)
    res = get_notes(result_array,jump_array)
    dissonance_array = []
    time_array = []
    for element in res:
        dissonance = element.get('dissonance')
        dissonance_array.append(dissonance)
        dissonance_array.append(dissonance)
        time_array.append(element.get('start'))
        time_array.append(element.get('end'))
    return time_array,dissonance_array

for index in range(0,3):
    mid = mido.MidiFile(f'/home/beer/yuval/model_dir/Alpha_generated_files/alpha_{index}.mid', clip=True)

    #  for debuging
    #  print(mid.tracks[1])

    time_array,dissonance_array = calc_disaonant(mid)
    # print(time_array)
    # print(dissonance_array)
    result_array,jump_array = mid2arry(mid)
    mul = np.multiply(np.where(result_array>0, 1, 0), range(1, 89))
    # printing dissonance graph

    fig, ax = plt.subplots(2, 1, figsize=(8, 6), layout='constrained')

    ax[1].plot(time_array, dissonance_array, linewidth=2, markersize=12)
    ax[1].set_xlabel('time')
    ax[1].set_ylabel('dissonance')
    ax[1].set_title('Dissonance Graph')
    ax[0].plot(range(result_array.shape[0]),mul, marker='.', markersize=1, linestyle='')
    ax[0].set_xlabel('time')
    ax[0].set_ylabel('note')
    ax[0].set_title('MIDI Analysis')

    fig.savefig(f'plot_{index}.png')

# %%



