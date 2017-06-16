import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score


class diffusion:

    class Bass:

        def __init__(self, filename = 'NA'):

            if filename == 'NA':

                print('Error: must input filename.')

            try:

                if filename[-3:] == 'csv':
                    delimiter = ','
                else:
                    delimiter = '\t'

                self.data = np.genfromtxt(filename, delimiter = delimiter)

            except OSError:

                print('Error: file not found. Please input valid filename.')

            try:

                self.time = self.data[:,0]
                self.response = self.data[:,1]
                self.title = filename[:-4]

            except IndexError:

                print('Error: incorrect file format. Please provide either csv or txt.')

            self.popt = None
            self.pcov = None
            self.y_predicted = None
            self.y_pred_plot = None
            self.diff_y_pred = None
            self.imitation = None
            self.innovation = None
            self.maximum_potential = None
            self.perr = None

        def __bass_model__(self, t, p, q, k):

            np.seterr(over='ignore', invalid='ignore')

            return k * ((1 - np.exp(-1 * (p + q) * t)) / (1 + (q / p) * np.exp(-1 * (p + q) * t)))

        def fit(self):

            self.popt, self.pcov = curve_fit(self.__bass_model__, self.time, self.response)

            self.innovation = self.popt[0]
            self.imitation = self.popt[1]
            self.maximum_potential = self.popt[2]

            return self.innovation, self.imitation, self.maximum_potential

        def predict(self):

            self.y_predicted = self.__bass_model__(self.time, *self.popt)
            self.y_pred_plot = self.__bass_model__(np.linspace(-1, len(self.time), 1000), *self.popt)

        def __differentiate__(self):

            self.lst1 = [self.response[0]]

            for i in range(1, len(self.response)):

                self.lst1.append(self.response[i] - self.response[i - 1])

            p, q, k = self.popt
            t = np.linspace(-1, max(self.time), 1000)

            self.diff_y_pred = ( k * p * (p + q)**2 * np.exp(t*(p+q))) / ((p*np.exp(t*(p+q))+q)**2)

        def plot_saturation(self, conf_region = True):

            plt.plot(self.time, self.response, 'o', label='data', color = 'black')
            plt.plot(np.linspace(-1, max(self.time), 1000), self.y_pred_plot, label='fit', color = 'black')
            plt.ylim(0, max(self.response) * 1.1)
            plt.legend(loc='best')
            plt.xlabel('Time')
            plt.ylabel('Saturation')
            plt.title('Bass Diffusion Model for {}'.format(self.title))

            if conf_region:

                self.__error__()

                plt.fill_between(np.linspace(-1, max(self.time), 1000), self.y_pred_lower, self.y_pred_upper,
                                 color='gray', alpha=0.6)

            plt.show()

        def plot_cdf(self):

            plt.plot(np.linspace(-1, max(self.time), 1000), self.y_pred_plot/self.popt[2], label='CProb', color='black')
            plt.ylim(0, 1.1)
            plt.legend(loc='best')
            plt.xlabel('Time')
            plt.ylabel('Cumulative Probability')
            plt.title('Cumulative Probability Density Over Time')
            plt.show()

        def plot_diff(self, conf_region = True):

            self.__differentiate__()
            plt.plot(self.time, self.lst1, 'o', label='data', color='black')
            plt.plot(np.linspace(-1, max(self.time), 1000), self.diff_y_pred, label='fit', color='black')
            plt.legend(loc='best')
            plt.xlabel('Time')
            plt.ylabel('Saturation by Time-Step')
            plt.title('Bass Diffusion Model Derivative for {}'.format(self.title))

            if conf_region:

                self.__error__(cumu=False)

                plt.fill_between(np.linspace(-1,max(self.time),1000), self.diff_y_pred_lower, self.diff_y_pred_upper, color='gray', alpha=0.6)
                plt.fill_between(np.linspace(-1, max(self.time), 1000), self.diff_y_pred, self.diff_y_pred_upper,color='gray', alpha=0.6)
                plt.fill_between(np.linspace(-1, max(self.time), 1000), self.diff_y_pred, self.diff_y_pred_lower,color='gray', alpha=0.6)

            plt.show()

        def diagnostics(self):

            print('-'*68)
            print('Bass Diffusion Model Diagnostics:\n')

            print('\n{:<24}{:<10}'.format('Metric',
                                         'Value'))

            print('\n{:<24}{:<10.3f}'.format('RMSE',
                                         np.sqrt(mean_squared_error(self.response, self.y_predicted))))

            print('\n{:<24}{:<10.4f}'.format('R-Squared',
                                             r2_score(self.response, self.y_predicted)))

            print('\n{:<24}{:<10.4f}'.format('Adj. R-Squared',
                                             1 - ((1 - r2_score(self.response, self.y_predicted)) *
                                                  (len(self.time) - 1) / (len(self.time) - 3 - 1))))

            print('\n{:<24}{:<10.0f}'.format('Df',
                                             len(self.time)-4))

        def results(self):

            self.__error__()

            low_est = self.popt - self.perr
            up_est = self.popt + self.perr

            print('-'*68)
            print('Bass Diffusion Model Results:\n')

            print('\n{:<24}{:<10}{:<10}{:<12}{:<12}'.format('Term','Est','Std Dev','Lower Lim','Upper Lim'))

            print('\n{:<24}{:<10.5f}{:<10.5f}{:<12.5f}{:<12.5f}'.format('Coef. of Innovation',
                                                                           self.popt[0],
                                                                           self.perr[0],
                                                                           low_est[0],
                                                                           up_est[0]))

            print('\n{:<24}{:<10.5f}{:<10.5f}{:<12.5f}{:<12.5f}'.format('Coef. of Imitation',
                                                                           self.popt[1],
                                                                           self.perr[1],
                                                                           low_est[1],
                                                                           up_est[1]))

            print('\n{:<24}{:<10.2f}{:<10}{:<12.2f}{:<12.2f}'.format('Peak Adoption Time',
                                                                           (np.log(self.popt[1]) - np.log(self.popt[0])) /
                                                                               (self.popt[0] + self.popt[1]),
                                                                           '',
                                                                           (np.log(low_est[1]) - np.log(low_est[0])) /
                                                                               (low_est[0] + low_est[1]),
                                                                           (np.log(up_est[1]) - np.log(up_est[0])) /
                                                                               (up_est[0] + up_est[1])))

            print('\n{:<24}{:<10.2f}{:<10.3f}{:<12.2f}{:<12.2f}'.format('Max. Adopters',
                                                                           self.popt[2],
                                                                           self.perr[2],
                                                                           low_est[2],
                                                                           up_est[2]))

            print('-'*68)

        def download(self, filename='bass_results', filetype='csv', differentiated=False):

            if not self.y_predicted:

                self.fit()
                self.predict()

            if filetype == 'csv':
                dlim = ','
            elif filetype == 'tsv' or filetype == 'txt':
                dlim = '\t'
            else:
                print('Warning: bass.download() only recognizes csv, txt, and tsv file types. Default will be space'
                      'delimited txt file.')
                dlim = ' '

            filename += ('.'+filetype)

            if dlim == ' ':
                filename += ('.'+'txt')

            ofile = open(filename, 'w')

            header = 'Time'+dlim+'Observed'+dlim+'Expected\n'

            ofile.write(header)

            for i in range(len(self.time)):

                # Make change here so that the last line does not have carriage return
                # End at range(len - 1) and then have separate mess for index [-1]

                mess = str(self.time[i])+dlim+str(self.response[i])+dlim+str(self.y_predicted[i])+'\n'

                ofile.write(mess)

            ofile.close()

        def __error__(self, cumu = True):

            self.perr = np.sqrt(np.diag(self.pcov))

            if cumu:

                self.y_pred_lower = self.__bass_model__(np.linspace(-1, len(self.time), 1000), *(self.popt - self.perr))
                self.y_pred_upper = self.__bass_model__(np.linspace(-1, len(self.time), 1000), *(self.popt + self.perr))

            else:

                p, q, k = self.popt - self.perr

                t = np.linspace(-1, max(self.time), 1000)

                self.diff_y_pred_lower = (k * p * (p + q) ** 2 * np.exp(t * (p + q))) / ((p * np.exp(t * (p + q)) + q) ** 2)

                p, q, k = self.popt + self.perr

                self.diff_y_pred_upper = (k * p * (p + q) ** 2 * np.exp(t * (p + q))) / ((p * np.exp(t * (p + q)) + q) ** 2)





model = diffusion.Bass(filename = 'ALS.csv')
model.fit()
model.predict()
model.plot_saturation()
model.diagnostics()
model.results()
