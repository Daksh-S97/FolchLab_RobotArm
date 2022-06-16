from deps.alarm_servo import alarm_servo_list
from deps.alarm_controller import alarm_controller_list
import json
import numpy as np

modes_dict = {1: 'ROBOT_MODE_INIT',
2:	'ROBOT_MODE_BRAKE_OPEN', 	
4:	'ROBOT_MODE_DISABLED',
5:	'ROBOT_MODE_ENABLE',
6:	'ROBOT_MODE_BACKDRIVE',	
7:	'ROBOT_MODE_RUNNING',
8:	'ROBOT_MODE_RECORDING',	
9:	'ROBOT_MODE_ERROR',
10: 'ROBOT_MODE_PAUSE',
11: 'ROBOT_MODE_JOG'}

def report_mode(dash) -> None:
    """Report the current Robot mode according to modes_dict.

    Args:
        dash (DobotApiDashboard): Dashboard class object currently 
        connected to the robot.
    """    
    mode = dash.RobotMode()
    keys = list(modes_dict.keys())
    for key in keys:
        if f'{key}' in mode:
            print(f'Mode: {modes_dict[key]}')

def report_error(dash) -> None:
    """Report the current error message.

    Args:
        dash (DobotApiDashboard): Dashboard class object currently 
        connected to the robot.
    """    
    resp = dash.GetErrorID()
    resp = resp.replace('\t', '').replace('\n','')
    resp = resp.split('{')[1].split('}')[0]
    resp = json.loads(resp)

    for code in resp[0]:
        print(f'\n|CODE: {code}|')
        for dic in alarm_servo_list:
            if dic['id'] == code:
                print('_'*50)
                desc = dic['en']['description']
                print(f'Possible reasons from ALARM SERVO:\n{desc}.')
        for dic in alarm_controller_list:
            if dic['id'] == code:
                print('_'*50)
                desc = dic['en']['description']
                print(f'Possible reasons from ALARM CONTROLLER:\n{desc}.')

def get_pose(dash) -> np.ndarray:
    """Get the current arm position in format X,Y,Z,r.

    Args:
        dash (DobotApiDashboard): Dashboard class object currently 
        connected to the robot.

    Returns:
        np.ndarray: Numpy array with 4 etries: X,Y,Z,r.
    """    
    resp = dash.GetPose()
    coords = resp.split('{')[1].split('}')[0].split(',')
    coords = [float(coord) for coord in coords[:4]]
    print(f'X = {coords[0]}\nY = {coords[1]}\nZ = {coords[2]}\nr = {coords[3]}')
    return np.array(coords)