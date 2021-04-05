'''
This is the core server:
    1. Store the main model weight of DDPG
    2. Send new weights to actors' models
This object will communicate with:
    1. Worker: Param will be sent from server to each worker
    2. Learner: New learnt param will be updated 
'''
import ray 

@ray.remote
class ParameterServer(object):

    def __init__(self):
        self.params = {'actor': [], 'critic': []}
        self.update_step = 0
    
    def define_param_list(self, new_params):
        self.params = new_params
    
    # the update_params() functions receives new_params
    # from the actor. In particular, this function is called
    # in the learner process, with its own parameters as args
    def update_params(self, new_params):
        # Actor
        if len(self.params['actor']) <= 0: # is empty
            for new_param in new_params['actor']:
                self.params['actor'].append(new_param)
        else:
            for new_param, idx in zip(new_params['actor'], range(len(self.params['actor']))):
                self.params['actor'][idx] = new_param
        # Critic
        if len(self.params['critic']) <= 0: # is empty
            for new_param in new_params['critic']:
                self.params['critic'].append(new_param)
        else:
            for new_param, idx in zip(new_params['critic'], range(len(self.params['critic']))):
                self.params['critic'][idx] = new_param

        self.update_step += 1

    def return_params(self):
        return self.params

    def get_update_step(self):
        return self.update_step
    