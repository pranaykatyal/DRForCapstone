from math import pi, sin, cos
import matplotlib.pyplot as plt

def dist_ccw(theta_1,theta_2):
    # distance from theta_1 to theta_2, in the ccw direction,
    # assuming both angles are measured in the ccw direction
    d_ccw = theta_2 - theta_1
    return constrain_angle(d_ccw)

def dist_cw(theta_1,theta_2):
    # distance from theta_1 to theta_2, in the cw direction,
    # assuming both angles are measured in the ccw direction
    t_1_cw = 2*pi - theta_1
    t_2_cw = 2*pi - theta_2
    d_cw = t_2_cw - t_1_cw
    return constrain_angle(d_cw)

def constrain_angle(theta):
    if theta < 0:
        return theta + 2*pi
    elif theta > 2*pi:
        pass
    else:
        return theta


def plot_agents(agents):

    figure, axes = plt.subplots()
    axes.set_aspect( 1 )
    axes.set_xlim(-1.5,1.5)
    axes.set_ylim(-1.5,1.5)

    x = [agent.val[0] for agent in agents]
    y = [agent.val[1] for agent in agents]
    plt.scatter(x,y)

    # plt.show()
    return figure, axes

def plot_agents_circle(agents):

    figure, axes = plt.subplots()
    circ = plt.Circle((0,0),1,fill=False)
    axes.set_aspect( 1 )
    axes.add_artist( circ )
    axes.set_xlim(-1.5,1.5)
    axes.set_ylim(-1.5,1.5)

    x = [cos(agent.theta) for agent in agents]
    y = [sin(agent.theta) for agent in agents]
    plt.scatter(x,y)

    # plt.show()
    return figure, axes