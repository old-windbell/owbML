#/usr/bin/python

import numpy
import Gnuplot, Gnuplot.funcutils

samples = []
labels = []

def load_data(data_file):
	f = open(data_file, 'r')
	for line in f:
		if line[0] == 'a':
			values = line.split(':')
			values = values[1]
			vlist = values.split(',')
			vlist = vlist[:-1]
			alpha = [float(i) for i in vlist]
			#print alpha
		elif line[0] == 'b':
			value = line.split(':')
			b = float(value[1])
		else:
			vlist = line.split(',')
			labels.append(int(vlist[-2]))
			samples.append([int(i) for i in vlist[:-2]])
	f.close()
	return alpha, b

def wait(str = None, prompt = 'ooo...'):
	if str is not None:
		print str
	raw_input(prompt)

def my_plot(alpha, b):
	g = Gnuplot.Gnuplot(persist = 1)
	g.clear()
	#wait('mmm')
	try:
		g.title('svm test')
		g.xlabel('x')
		g.ylabel('y')
		g.set_range('xrange', (0,100))
		g.set_range('yrange', (0,100))
		
		g('set pointsize 3')
		g.plot([[0,0]])
		sa, sb = [], []
		for i in xrange(len(samples)):
			if labels[i] is 1:
				sa.append(samples[i])
			else:
				sb.append(samples[i])
		print sa
		print sb
		g.replot(sa)
		g.replot(sb)
		
		
		w1, w2 = 0.0, 0.0
		for i in xrange(len(samples)):
			w1 += alpha[i] * labels[i] * samples[i][0]
			w2 += alpha[i] * labels[i] * samples[i][1]
		print 'sssss'	
		w_ = - w1 / w2
		b_ = - b / w2
		
		r_ = 1 / w2
		
		xs = [10, 90]
		ys = []
		ys_up = []
		ys_down = []
		for xj in xs:
			ys.append(w_ * xj + b_)
			ys_up.append(w_ * xj + b_ + r_)
			ys_down.append(w_ * xj + b_ - r_)
			
		line1 = [[xs[0], ys[0]], [xs[1], ys[1]]]
		print line1
		plot1 = Gnuplot.PlotItems.Data(line1, with_ = "lines lt rgb 'red' lw 2")
		
		line2 = [[xs[0], ys_up[0]], [xs[1], ys_up[1]]]
		plot2 = Gnuplot.PlotItems.Data(line2, with_ = "lines lt rgb 'blue' lw 2")

		line3 = [[xs[0], ys_down[0]], [xs[1], ys_down[1]]]
		plot3 = Gnuplot.PlotItems.Data(line3, with_ = "lines lt rgb 'green' lw 2")

		#g.replot(plot1, plot2, plot3)
		g.replot(plot1, plot2, plot3)
		wait('==')
	except:
		print 'error'

        
if __name__ == '__main__':
	
	alpha, b = load_data('result')
	#print alpha, '\n', len(alpha)
	#print samples
	#print alpha
	#print b
	
	my_plot(alpha, b)
