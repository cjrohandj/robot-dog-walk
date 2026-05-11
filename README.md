This is a homework assignment for EEC289A - set up the curriculum for a robot dog to follow joint and individual commands in +-{vx,vy,yaw}

Demonstration video:

<video src = "awesome_dog.mp4" width="320" height="240" controls></video>


Some analysis:
Based on the provided rewards there seems to be a neccesary tradeoff between standing still and walking backwards. Walking backwards requires a lower posture reward and a higher tolerance for high leg movement, while standing still requires the opposite. Changing the rewards present in the reward curriculum could significantly improve perfomance. 

Longer training periods are likely required to make the dog better at walking for low speeds because while slipping the dog can maintain low speeds, and despite a relatively high slipping reward it still doesn't completely walk for those low speeds due to a higher penalty for failure. 
