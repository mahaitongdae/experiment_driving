import matplotlib.pyplot as plt
from utils.plot_new.plot_utils.load_record import load_data
import os


def single_plot(data_all, keys=['Deceleration'], **kwargs):
    if 'fig_num' in kwargs.keys():
        plt.figure(kwargs['fig_num'])
    else:
        plt.figure()
    if kwargs['highlight'] == True:
        min_index = data_all['VehicleMode'].index(1.0)
        max_index = min_index + data_all['VehicleMode'].count(1.0)

    labels = []
    for key in keys:
        if isinstance(key, tuple):
            plt.plot(data_all[key[0]], data_all[key[1]])
            labels.append(key[1])
        else:
            plt.plot(data_all['Time'], data_all[key])

    if isinstance(keys[0], tuple):
        plt.xlabel(keys[0][0])
    else:
        plt.xlabel('time /s')
        labels = keys
        if kwargs['highlight'] == True:
            axes = plt.gca()
            ylim = axes.get_ylim()
            plt.plot([data_all['Time'][min_index], data_all['Time'][min_index]], ylim, c='red', linestyle='--')
            plt.plot([data_all['Time'][max_index], data_all['Time'][max_index]], ylim, c='red', linestyle='--')
    plt.legend(labels=labels, loc='best')
    plt.grid()

    if 'x_lim' in kwargs.keys():
        plt.xlim(kwargs['x_lim'])
    if 'y_lim' in kwargs.keys():
        plt.ylim(kwargs['y_lim'])
    if 'title' in kwargs.keys():
        plt.title(kwargs['title'])
    if 'path' in kwargs.keys():
        proj_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        fig_path = proj_root_dir + '/record/' + kwargs['path'] + '/figure/'
        data_fig_path = proj_root_dir + '/record/' + kwargs['path'] + '/figure/data_fig/'
        if not os.path.exists(fig_path):
            os.mkdir(fig_path)
            os.mkdir(data_fig_path)
        name = fig_path + kwargs['title'] +'.jpg'
        plt.savefig(name)
    else:
        plt.show()

def single_plot_time_series(data_all, AutoMode = True):
    highlight = AutoMode
    single_plot(data_all, ['SteerAngleAct', 'SteerAngleAim'], title='Steering Act', path=exp_index, highlight=highlight)
    single_plot(data_all, ['front_wheel_deg'], title='Front wheel deg', path=exp_index, highlight=highlight)
    single_plot(data_all, ['Dec_flag', 'BrkOn'], title='Deceleration flag', path=exp_index, highlight=highlight)
    single_plot(data_all, ['accActual', 'a_x'], title='Deceleration acc', path=exp_index, highlight=highlight)
    # single_plot(data_all, ['BrkOn'], title='', path=exp_index, highlight=highlight)
    single_plot(data_all, ['LongitudinalAcc', 'LateralAcc'], title='Long and lat acc', path=exp_index, highlight=highlight)

    # single_plot(data_all, ['v_light'], title='V light', path=exp_index, highlight=highlight)

    single_plot(data_all, ['GaussX', 'GaussY'], title='Position', path=exp_index, highlight=highlight)
    single_plot(data_all, ['ego_vx', 'ego_vy'], title='Obs ego velocity', path=exp_index, highlight=highlight)
    # single_plot(data_all, ['ego_v_y'], title='Obs ego velocity', path=exp_index, highlight=highlight)

    single_plot(data_all, ['ego_x'], title='Obs ego position', path=exp_index, highlight=highlight)
    single_plot(data_all, ['ego_phi'], title='Obs ego phi', path=exp_index, highlight=highlight)
    single_plot(data_all, ['tracking_delta_y'], title='Obs ego delta y', path=exp_index, highlight=highlight)
    single_plot(data_all, ['tracking_delta_phi'], title='Obs ego delta phi', path=exp_index, highlight=highlight)
    single_plot(data_all, ['tracking_delta_v'], title='Obs ego delta v', path=exp_index, highlight=highlight)
    # single_plot(data_all, ['ego_r'], title='Obs ego r', path=exp_index, highlight=highlight)
    # single_plot(data_all, ['delta_y'], title='Obs ego delta y', path=exp_index, highlight=highlight)
    # single_plot(data_all, ['delta_phi_rad'], title='Obs ego x', path=exp_index, highlight=highlight)

    single_plot(data_all, ['Heading'], title='Heading', path=exp_index, highlight=highlight)
    single_plot(data_all, ['accActual'], title='Acc actual', path=exp_index, highlight=highlight)
    single_plot(data_all, ['NorthVelocity', 'EastVelocity'], title='Velocity', path=exp_index, highlight=highlight)
    single_plot(data_all, ['VehicleMode'], title='Mode', path=exp_index, highlight=highlight)
    single_plot(data_all, ['GpsSpeed', 'VehicleSPeedAct'], title='Speed', path=exp_index, highlight=highlight)
    single_plot(data_all, ['Throttle'], title='Throttle', path=exp_index, highlight=highlight)
    single_plot(data_all, ['YawRate'], title='Yaw rate', path=exp_index, highlight=highlight)

    single_plot(data_all, ['AutoGear'], title='AutoGear', path=exp_index, highlight=highlight)

def single_plot_other_series(data_all):
    single_plot(data_all, [('GaussX', 'GaussY')], title='Trajectory', path=exp_index, x_lim=[-11, 11], highlight=False)

def single_plot_other_vehicles(data_all, exp_index, highlight=True):
    for i in range(6):
        for real_key in ['delta_x','delta_y','delta_v','delta_phi']:
            key = 'other' + str(i) + '_' + real_key
            title = 'obs ' + key
            single_plot(data_all, [key], title=title,  path=exp_index, highlight=highlight)


if __name__ == '__main__':
    exp_index = 'left/case0_noise0_20210103_152139'
    data_all, keys_for_data = load_data(exp_index)
    print(keys_for_data)

    # single_plot_time_series(data_all, AutoMode=False) # if not switch into auto mode, add kwargs: VehicleMode=False
    # single_plot_other_series(data_all)
    single_plot_other_vehicles(data_all, exp_index)


    # # single_plot(data_all, ['YawRate'])
