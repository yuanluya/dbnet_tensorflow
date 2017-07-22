import tensorflow as tf
from multiprocessing import Condition, Lock, Process, Manager
import random
from utils import train_ids, test_ids, get_data
import pdb

class DataLoader:
    """ Class for loading data
    Attributes:
        num_processor: an integer indicating the number of processors 
            for loading the data, normally 4 is enough
        capacity: an integer indicating the capacity of the data load
            queue, default set to 10
        batch_size: an integer indicating the batch size for each 
            extraction from the data load queue
        phase: an string indicating the phase of the data loading process,
            can only be 'train' or 'test'
    """
    def __init__(self, num_processor, batch_size, phase,
                 batch_idx_init = 0, data_ids_init = train_ids, capacity = 10):
        self.num_processor = num_processor
        self.batch_size = batch_size
        self.data_load_capacity = capacity
        self.manager = Manager()
        self.batch_lock = Lock()
        self.mutex = Lock()
        self.cv_full = Condition(self.mutex)
        self.cv_empty = Condition(self.mutex)
        self.data_load_queue = self.manager.list()
        self.cur_batch = self.manager.list([batch_idx_init])
        self.processors = []
        if phase == 'train':
            self.data_ids = self.manager.list(data_ids_init)
        elif phase == 'test':
            self.data_ids = self.manager.list(test_ids)
        else:
            raise ValueError('Could not set phase to %s' % phase)

    def __load__(self):
        while True:
            image_dicts = []
            self.batch_lock.acquire()
            image_ids = self.data_ids[self.cur_batch[0] * self.batch_size : 
                                     (self.cur_batch[0] + 1) * self.batch_size]
            self.cur_batch[0] += 1
            if (self.cur_batch[0] + 1) * self.batch_size >= len(self.data_ids):
                self.cur_batch[0] = 0
                random.shuffle(self.data_ids)
            self.batch_lock.release()
            
            self.cv_full.acquire()
            if len(self.data_load_queue) > self.data_load_capacity:
                self.cv_full.wait()
            self.data_load_queue.append(get_data(image_ids))
            self.cv_empty.notify()
            self.cv_full.release()

    def start(self):
        for _ in range(self.num_processor):
            p = Process(target = self.__load__)
            p.start()
            self.processors.append(p)

    def get_batch(self):
        self.cv_empty.acquire()
        if len(self.data_load_queue) == 0:
            self.cv_empty.wait()
        batch_data = self.data_load_queue.pop()
        self.cv_full.notify()
        self.cv_empty.release()
        return batch_data

    def get_status(self):
        self.batch_lock.acquire()
        current_cur_batch = self.cur_batch[0]
        current_data_ids = self.data_ids
        self.batch_lock.release()
        return {'batch_idx': int(current_cur_batch), 'data_ids': list(current_data_ids)}

    def stop(self):
        for p in self.processors:
            p.terminate()
