"""
 === Introduction ===

   A few months ago a new rover was sent to McMurdo Station in the Antarctic. The rover is a technical marvel
   as it is equipped with the latest scientific sensors and analyzers capable of surviving the harsh climate of the
   South Pole.  The goal is for the rover to reach a series of test sites and perform scientific sampling and analysis.
   Due to the extreme conditions, the rover will be air dropped via parachute into the test area.  The good news is
   the surface is smooth and free from any type of obstacles, the bad news is the surface is entirely ice.  The
   station scientists are ready to deploy the new rover, but first we need to create and test the planning software
   that will be used on board to ensure it can complete it's goals.

   The assignment is broken up into two parts.

   Part A:
        Create a SLAM implementation to process a series of landmark measurements and movement updates.

        Hint: A planner with an unknown number of motions works well with an online version of SLAM.

    Part B:
        Here you will create the planner for the rover.  The rover does unfortunately has a series of limitations:

        - Start position
          - The rover will land somewhere within range of at least 3 or more satellites for measurements.

        - Measurements
          - Measurements will come from satellites and test sites within range of the rover's antenna horizon.
            * The format is {'landmark id':{'distance':0.0, 'bearing':0.0, 'type':'satellite'}, ...}
          - Satellites and test sites will always return a measurement if in range.

        - Movements
          - Action: 'move 1.570963 1.0'
            * The rover will turn counterclockwise 90 degrees and then move 1.0
          - stochastic due to the icy surface.
          - if max distance or steering is exceeded, the rover will not move.

        - Samples
          - Provided as list of x and y coordinates, [[0., 0.], [1., -3.5], ...]
          - Action: 'sample'
            * The rover will attempt to take a sample at the current location.
          - A rover can only take a sample once per requested site.
          - The rover must be with 0.25 distance to successfully take a sample.
            * Hint: Be sure to account for floating point limitations
          - The is a 100ms penalty if the robot is requested to sample a site not on the list or if the site has
            previously been sampled.
          - Use sys.stdout = open('stdout.txt', 'w') to directly print data if necessary.

        The rover will always execute a measurement first, followed by an action.

        The rover will have a time limit of 10 seconds to find and sample all required sites.
"""

from matrix import matrix
import math
import random

class SLAM:
    """Create a basic SLAM module.
    """
    def __init__(self, initialX = 0.0, initialY = 0.0, heading = 0.0):
        """Initialize SLAM components here.
        """
        # initialize Omega
        self.Omega = matrix()
        # Can only initialize the initial x , y positions
        self.Omega.zero(2, 2)
        self.Omega.value[0][0] = 1.0
        self.Omega.value[1][1] = 1.0

        # intialize Xi
        self.Xi = matrix()
        # Can only initialize Xi with X and Y values of 0.0 , 0.0 --> initial believe of world is at center.
        self.Xi.zero(2, 1)
        self.Xi.value[0][0] = initialX
        self.Xi.value[1][0] = initialY

        # initialize the list of collected list of landmark IDs and index in the Xi/Omega matrix. Or just a list of landmarks.
        self.knownLandmarks = []

        # keep an internal log for the heading.
        # This heading does not matter. It is not absolute... so the steering angle does not have an affect on it.
        # The first steering after initialization does not matter.
        self.heading = heading

    def check_knownLM(self, thisLM):
        # This will look in the self:
        index = []
        for i in range(len(self.knownLandmarks)):
            if self.knownLandmarks[i] == thisLM:
                index = i
        return index

    def truncate_angle(self, t):
        PI = math.pi
        return ((t + PI) % (2 * PI)) - PI

    def process_measurements(self, measurements):
        """Process a new series of measurements.

        Args:
            measurements(dict): Collection of measurements
                in the format {'landmark id':{'distance':0.0, 'bearing':0.0, 'type':'satellite'}, ...}

        Returns:
            x, y: current belief in location
        """
        # Look at all the keys in the measurement dictionary.
        Omega = self.Omega
        Xi = self.Xi

        # print ' current list of landmarks --> ', self.knownLandmarks
        for key in measurements.keys():

            # Grab the Type, Bearing and Distance with the key.
            thisType = measurements[key]['type']
            # if Type is a beacon then calculate (landmark)
            # If this is not then ignore it??? --> We dont want to do anything with the sites. Maybe...
            if thisType == 'beacon':
                thisDist = measurements[key]['distance']
                thisBear = self.truncate_angle(measurements[key]['bearing'] + self.heading)

                distance_sigma = 0.05 * thisDist
                bearing_sigma = 0.02 * thisBear

                # distance_noise = random.gauss(0, distance_sigma)
                # bearing_noise = random.gauss(0, bearing_sigma)
                bearing_noise = bearing_sigma
                distance_noise = distance_sigma
                # dx dy = what we actually update the Xi with.
                dx = thisDist * math.cos(thisBear)
                dy = thisDist * math.sin(thisBear)

                # print ' bearingnoise = ', bearing_sigma,
                noiseX = distance_noise * math.cos(bearing_noise)
                noiseY = distance_noise * math.sin(bearing_noise)
                # measurement_noise = [noiseX, noiseY]  #
                measurement_noise = [1.0, 1.0]  #
                # print ' noiseX = ', measurement_noise[0], ' | noiseY = ', measurement_noise[1]

                thismeasurement = [dx, dy]

                # thismeasurement = [1.0, 1.0]

                # Check if the key is in the self.knownlandmarks.
                idx = self.check_knownLM(key)
                currDim = len(Xi.value)
                # print 'printing out initial Xi length ', currDim
                currlist = range(currDim)
                if not idx:

                    # cant find the key? add it and expand the omega and xi
                    self.knownLandmarks.append(key)
                    idx = self.check_knownLM(key)

                    # Automatically expands it with 0.0's so we are good here.
                    Omega = Omega.expand(dimx=currDim+2, dimy=currDim+2, list1=currlist, list2=currlist)
                    Xi = Xi.expand(dimx=currDim+2, dimy=1, list1=currlist, list2=[0])

                # Index to use is 2 + idx (2 = x and y of current measurement)
                # print ' Planning to update this Index! --> ', idx,
                midx = 2*idx + 2
                # print ' in the Omega and Xi matrices it is this index! --> ', midx

                # Update the Omega and Xi matrices at the idx with the measurements in the key.
                for b in range(2):
                    # Update Omega
                    Omega.value[b][b] += 1.0 * (1.0+measurement_noise[b])
                    Omega.value[midx + b][midx + b] += 1.0 * (1.0+measurement_noise[b])
                    Omega.value[b][midx + b] -= 1.0 * (1.0+measurement_noise[b])
                    Omega.value[midx + b][b] -= 1.0 * (1.0+measurement_noise[b])

                    # Update Xi
                    Xi.value[b][0] += -thismeasurement[b] * (1.0+measurement_noise[b])
                    Xi.value[midx + b][0] += thismeasurement[b] * (1.0+measurement_noise[b])

                # print 'Printing each instance of Omega --> '
                # for i in range(len(Omega.value)):
                #     print Omega[i]

            # Finished updating Omega and Xi with all the measurements...
        # print ' FINAL list of landmarks --> ', self.knownLandmarks
        # Update the passthrough Omega and Xi with final Omega and Xi.
        self.Omega = Omega
        self.Xi = Xi

        # Find the final Mu with these two.
        # for i in range(len(Omega)):
        # print ' printing Omega out! \n'
        # print Omega
        mu = Omega.inverse() * Xi

        x = mu.value[0][0]
        y = mu.value[1][0]

        # print 'measurement update: x = ', x, ' and y = ', y
        return x, y

    def process_movement(self, steering, distance, motion_noise=0.001):
        """Process a new movement.

        Args:
            steering(float): amount to turn
            distance(float): distance to move
            motion_noise(float): movement noise

        Returns: idx:
        # cant find the key? add it and expand the omega and xi
        self.knownLandmarks.append(key)
        idx = self.check_knownLM(key)


            x, y: current belief in location
        """

        Omega = self.Omega
        Xi = self.Xi

        thisheading = self.truncate_angle(self.heading + steering)
        self.heading = thisheading

        dx = distance * math.cos(thisheading)
        dy = distance * math.sin(thisheading)
        thismotion = [dx, dy]

        dim = len(Xi.value)

        data_pointer_list = [0, 1] + range(4, dim + 2)

        # Expanding Omega and Xi for the motion update.
        Omega = Omega.expand(dimx=dim+2, dimy=dim+2, list1=data_pointer_list, list2=data_pointer_list)
        Xi = Xi.expand(dimx=dim+2, dimy=1, list1=data_pointer_list, list2=[0])

        # Updating Omega and Xi here now!
        for b in range(4):
            Omega.value[b][b] += 1.0 / motion_noise

        for b in range(2):
            Omega.value[b][b+2] += -1.0 / motion_noise
            Omega.value[b+2][b] += -1.0 / motion_noise
            Xi.value[b][0] += -thismotion[b] / motion_noise
            Xi.value[b+2][0] += thismotion[b] / motion_noise

        # Obtain A B C Omega' and Xi'
        # A
        # dumb way to designate where the data is taken from
        Alist = range(2, dim+2)
        A = Omega.take(list1=[0, 1], list2=Alist)

        # B
        # Always assume that we are taking the first two indices --> 2 dimensions. will screw up if otherwise.
        Blist = [0, 1]
        B = Omega.take(list1=Blist, list2=Blist)

        # C
        # Pretty much always the first two of Xi
        Clist = [0, 1]
        C = Xi.take(list1=Clist, list2=[0])

        # Omega Prime
        # This is the same data points indices as when obtaining A (OPlist = Alist)
        OmegaPrime = Omega.take(list1=Alist, list2=Alist)

        # Xi Prime
        # This is the same data points indices as when obtaining A (XPlist = Alist)
        XiPrime = Xi.take(list1=Alist, list2=[0])

        # Including data from the position data points. and simutaneously updating the data to include only the most current data pos.
        # Omega = Omega' - A^T * B^-1 * A
        Omega = OmegaPrime - A.transpose() * B.inverse() * A

        # Xi = Xi* - A^T * B^-1 * C
        Xi = XiPrime - A.transpose() * B.inverse() * C

        # Update the passThrough Omega and Xi
        self.Omega = Omega
        self.Xi = Xi

        # compute best estimate
        mu = Omega.inverse() * Xi

        x = mu.value[0][0]
        y = mu.value[1][0]

        # print 'measurement update: x = ', x, ' and y = ', y

        return x, y


class WayPointPlanner:
    """Create a planner to navigate the rover to reach all the intended way points from an unknown start position.
    """
    def __init__(self,  max_distance, max_steering):
        """Initialize your planner here.

        Args:
            max_distance(float): the max distance the robot can travel in a single move.
            max_steering(float): the max steering angle the robot can turn in a single move.
        """
        self.max_distance = max_distance
        self.max_steering = max_steering
        self.heading = 0.0
        self.slammer = SLAM()
        self.patrolTurnCount = 2
        self.patrolStraightMax = 1
        self.patrolStraightCount = 1
        self.foundSampleFlag = False
        self.SLAM_meas_offset = [0.0, 0.0, 0.0]
        self.sampleList = []


    def truncate_angle(self, t):
        PI = math.pi
        return ((t + PI) % (2 * PI)) - PI

    def checkMeasurements(self, measurements):
        usekey = []
        useDist = []
        useBear = []
        for key in measurements.keys():
            # Grab the Type, Bearing and Distance with the key.
            thisType = measurements[key]['type']
            # if Type is a beacon then calculate (landmark)
            # If this is not then ignore it??? --> We dont want to do anything with the sites. Maybe...
            minDist = 9999999.0
            if thisType == 'site':
                thisDist = measurements[key]['distance']
                thisBear = self.truncate_angle(measurements[key]['bearing'] + self.heading)
                if thisDist < minDist:
                    usekey = key
                    useDist = thisDist
                    useBear = thisBear
        return usekey, useDist, useBear

    def generatePatrolAction(self):
        # This will be the bulk code for finding the first sample.
        # Jump the furthest distance then look in all 4 direcitons
        # before every turn only need to look in 2 directions before
        distance = self.max_distance * 0.75

        self.patrolStraightCount -= 1
        if self.patrolStraightCount == 0:
            steering = self.max_steering
            self.patrolTurnCount -= 1
            if self.patrolTurnCount == 0:
                self.patrolTurnCount = 2
                self.patrolStraightMax += 1
            self.patrolStraightCount = self.patrolStraightMax
        else:
            steering = 0.0
        action = 'move ' + str(steering) + ' ' + str(distance)
        self.heading += steering

        return action, steering, distance

    def checkSampled(self, new_sample_todo):
        # Check the new sample list versus the old one and see if there is anything new.
        foundSamples = []

        if not not self.sampleList:
            if len(self.sampleList) != len(new_sample_todo):
                for i in range(len(self.sampleList)):
                    thisOld = self.sampleList[i]
                    count = 0
                    for j in range(len(new_sample_todo)):
                        thisNew = new_sample_todo[j]
                        if thisOld[0] == thisNew[0] and thisOld[1] == thisNew[1]:
                            count += 1
                    if count != len(new_sample_todo):
                        foundSamples = thisOld

        return foundSamples

    def findnextSample(self, currPos, samplelist):
        minDist = 999999.0
        nextSample = samplelist[0]
        for i in range(len(samplelist)):
            distance = math.sqrt((currPos[0] - samplelist[i][0])**2 + (currPos[1] - samplelist[i][1])**2)
            if distance < minDist:
                nextSample = samplelist[i]

        return nextSample

    def compute_distance(self, p, q):
        x1, y1 = p
        x2, y2 = q

        dx = x2 - x1
        dy = y2 - y1

        return math.sqrt(dx ** 2 + dy ** 2)

    def compute_bearing(self, p, q):
        x1, y1 = p
        x2, y2 = q

        dx = x2 - x1
        dy = y2 - y1

        return math.atan2(dy, dx)

    def next_move(self, sample_todo, measurements):
        """Next move based on the current set of measurements.

        Args:
            sample_todo(list): Set of locations remaining still needing a sample to be taken.
            measurements(dict): Collection of measurements from satellites and test sites in range.
                in the format {'landmark id':{'distance':0.0, 'bearing':0.0, 'type':'satellite'}, ...}

        Return:
            Next command to execute on the rover.
                allowed:
                    'move 1.570963 1.0' - turn left 90 degrees and move 1.0 distance
                    'take sample' - take sample (will succeed if within tolerance of intended sample site)
        """

        # We need to sample all the sites in the todo list, the order doesnt matter.
        # However, not all sites will be visible in measurements due to the horizon distance.
        # If a site is visible from the very first measurement, then the task becomes easy.
        # You just navigate to the first site using the bearing and distance values from the measurement with type = site.
        # Once you sample that site, it disappears from the todo list (the todo list has the absolute coordinates of the sites to be sampled).
        # Therefore, we now know which site was sampled and we can determine the relative coordinates of all other sites in the todo list.
        # We can simply now beeline to these remaining sites and sample them.
        #
        # This becomes trickier when we dont see any site measurements from the beginning.
        # In this case, it looks like students are l moving the robot in a squared spiral until a site becomes visible( or performing a random walk).
        # Once the first site is visible, we can beeline to it, map other sites relative coordinates and sample them as well.
        #
        # LOOK HERE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # Check the measurement --> if we see one... if we see two... take the smallest distance one.
        # print ' STARTING NEXT_MOVE CALC:'

        # print 'self.sampleList = ', self.sampleList, ' ................... ', 'sample_todo = ', sample_todo
        thiskey, thisDistance, thisBearing = self.checkMeasurements(measurements)
        # print measurements
        # print ' found thiskey: ', thiskey, ' | thisDistance: ', thisDistance, ' | thisBearing: ', thisBearing


        if not self.sampleList:
            # print ' adding the initial sample list to record. '
            self.sampleList = list(sample_todo)

        if not self.foundSampleFlag:
            foundSamples = self.checkSampled(sample_todo)
            # print ' checking the samples --> old versus new. See if we have sampled. '
            if not not foundSamples:
                self.foundSampleFlag = True
                self.SLAM_meas_offset = foundSamples
                self.slammer = SLAM(initialX=foundSamples[0], initialY=foundSamples[1], heading=self.heading)
                # print ' initializing slam with '
                # self.slammer = SLAM(heading=self.heading)
        # else:
            # print 'found sample at: ', self.SLAM_meas_offset

        if not self.foundSampleFlag:
            # We didn't find any samples yet! --> gotta just do patrol movement.

            # if we do not see any sites from measurement --> then do patrol movement.
            # print ' foundSampleFlag = false '

            #   activate slammer
            slammer = self.slammer
            # measurex, measurey = slammer.process_measurements(measurements)

            if not thiskey:
                # !!!!!!!!!!!!!!!!!!!!!!! IMPLEMENT PATROL HERE !!!!!!!!!!!!!!!!!!!!!!!!!!!
                action, steering, distance = self.generatePatrolAction() # updates heading inside function.
                # print ' setting action to patrol. action = ', action
            else:
                # if this is within distance to pick up the sample
                if thisDistance < 0.25:
                    action = 'sample'
                    steering = []
                    distance = []
                    # print ' taking sample because distance < 0.25 units. action = ', action
                else:
                    useDistance = thisDistance
                    useBearing = self.truncate_angle(thisBearing + self.heading)

                    if useDistance > self.max_distance:
                        useDistance = self.max_distance

                    if math.fabs(useBearing) > self.max_steering:
                        useBearing = self.max_steering * (useBearing/math.fabs(useBearing))

                    action = 'move ' + str(useBearing) + ' ' + str(useDistance)
                    self.heading = self.truncate_angle(self.heading + useBearing)
                    steering = useBearing
                    distance = useDistance

                    # resetting the patrol positions.
                    self.patrolTurnCount = 2
                    self.patrolStraightMax = 1
                    self.patrolStraightCount = 1

                    # print ' we found the sample! gonna bee-line to it... action = ', action

            # if not not steering:
                # get the position after movement
                # motionx, motiony = slammer.process_movement(steering, distance)

            # self.slammer = slammer

        else:
            # We have already found a sample! --> will use SLAM with offset as initialx and initialy
            # print ' foundSampleFlag = true , should happen after taking sample at least once!!! '
            if thisDistance < 0.25:
                action = 'sample'
                # print ' taking sample because distance < 0.25 units. action = ', action
            else:
                #   activate slammer
                slammer = self.slammer
                measurex, measurey = slammer.process_measurements(measurements) # Comes out with x, y from initialx and initialy --> found form the first sample
                # print ' we are going to figure out where we are... measurex = ', measurex, ' measurey = ', measurey
                # print 'measurex = ', measurex, ' | measurey = ', measurey

                # This adds on the offset onto SLAM measurex and measurey
                # measurex += self.SLAM_meas_offset[0]
                # measurey += self.SLAM_meas_offset[1]

                # find the next nearest available sample.
                nextSample = self.findnextSample([measurex, measurey], sample_todo)
                # print ' we are finding the next nearest available sample: nextSample = ', nextSample

                # process the movement required for this next sample
                # steering = self.truncate_angle(self.compute_bearing([measurex, measurey], nextSample) + self.heading)
                # distance = self.compute_distance([measurex, measurey], nextSample)
                dx = nextSample[0] - measurex
                dy = nextSample[1] - measurey

                distance = math.sqrt(dx**2 + dy**2)
                steering = self.truncate_angle(math.atan2(dy, dx) - self.heading)
                # print ' try to go to steering = ', steering, ' | distance = ', distance

                # if steering > self.max_steering:
                #     useBearing = self.max_steering
                # else:
                #     useBearing = steering
                if math.fabs(steering) > self.max_steering:
                    useBearing = self.max_steering * (steering / math.fabs(steering))
                    useDistance = self.max_distance * 0.001
                else:
                    useBearing = steering
                    # Cap the distance and bearing to
                    if distance > self.max_distance:
                        useDistance = self.max_distance
                    else:
                        useDistance = distance

                # get the position after movement
                # print 'self.max_steering = ', self.max_steering, ' | self.max_distance = ', self.max_distance
                motionx, motiony = slammer.process_movement(useBearing, useDistance)
                self.heading = self.truncate_angle(self.heading + useBearing)

                # print ' we are going to figure out where we will be if we move. motion x = ', motionx, ' motion y = ', motiony

                # Find the new steering and distance based off the x y position after motion
                # useBearing = self.truncate_angle(self.compute_bearing([motionx, motiony], [measurex, measurey]) + self.heading)
                # useDistance = self.compute_distance([motionx, motiony], [measurex, measurey])
                # print ' will use steering = ', useBearing, ' | distance = ', useDistance


                # print ' capped off steering = ', useBearing, ' | distance = ', useDistance
                self.slammer = slammer

                action = 'move ' + str(useBearing) + ' ' + str(useDistance)
                # print ' taking action: ', action

        # print ' final action = ', action


        return action
