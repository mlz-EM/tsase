# xyz in tkinter

import os
import math
import time
import io
import sys

import tkinter as tk
import tkinter.filedialog as filenav
import tkinter.messagebox as msgbox

import ase
import tsase
from PIL import Image, ImageTk
from tsase.data import *
from ase.io import vasp
from ase.neighborlist import NeighborList

import numpy as np
#Todo: movie, resize, help section
_standard_font = '"Sans Sarif" '
_window_width = 960
_window_height = 540
_path = None

class queueitem:
	def __init__(self, kind):
		self.kind = kind

class Viewer(tk.Frame):
	# Main Frame within the root tk Window
	# Widgets are packed into Viewer which is then placed in main window
	
	def __init__(self, master=None, to_open=None):
		super().__init__(master)
		self.master = master
		self.master.title('xyz-viewer')
		self.images = []
		self.OptionsFrame = tk.Frame(self, borderwidth=2, relief='ridge')
		self.ButtonFrame = tk.Frame(self.OptionsFrame)
		self.EntryFrame = tk.Frame(self.OptionsFrame)
		self.draw_buttons()
		self.init_buttons()
		self.create_tk_variables()
		self.init_variables()
		self.draw_entries()
		self.draw_canvas()
		self.draw_fps_frame()
		self.create_menu()
		self.bind_shortcuts()
		self.bind_events()
		self.draw_main()
		if to_open is not None:
			self.open_init(to_open)

	def init_variables(self):
		self.trajectory = None
		self.bond_width = 4
		self.rotation = np.identity(3)
		self.translate = np.array([0.0, 0.0, 16.0])
		self.last_draw = 0.0
		self.mouselast = (None, None)
		self.screenatoms = None
		self.playing = False
		
	def create_menu(self):
		self.master.option_add('*tearOff', False)
		self.Menubar = tk.Menu(self.master)
		self.mb_File = tk.Menu(self.Menubar)
		self.mb_Help = tk.Menu(self.Menubar)
		self.Menubar.add_cascade(menu=self.mb_File, label='File')
		self.Menubar.add_cascade(menu=self.mb_Help, label='Help')
		self.mb_File.add_command(label='Open', accelerator='Ctrl+O', command=self.open_file)
		self.mb_File.add_command(label='Open View', accelerator='Ctrl+O', command=self.open_file)
		self.mb_File.add_command(label='Open Colors')
		self.mb_File.add_command(label='Save As', accelerator='Ctrl+S', command=self.save_canvas)
		self.mb_File.add_command(label='Save View')
		self.mb_File.add_command(label='Export')
		self.mb_File.add_command(label='Exit', accelerator='Ctrl+Q', command=lambda:self.master.destroy())
		self.mb_Help.add_command(label='Documentation')
		self.master['menu'] = self.Menubar
		
	def draw_main(self):
		self.ButtonFrame.grid(row=1, column=1, sticky='W')
		self.EntryFrame.grid(row=2, column=1, sticky='W')
		self.EntryFrame.config(pady=10)
		self.OptionsFrame.grid(row=1, column=1, sticky='NSW', rowspan=2)
		self.DrawingArea.grid(row=1, column=2, sticky='NSEW', columnspan=2)
		self.FPSFrame.grid(row=3,column=3, sticky='NSEW')
		self.AtomLabel.grid(row=3,column=1, columnspan=2, sticky='NSEW')
		
	def draw_canvas(self):
		self.DrawingArea = tk.Canvas(self, background='#ffffff')
		self.DrawingArea.grid_configure(rowspan=2)
		
	def draw_buttons(self):
		self.BoxButton = self.button_with_image(self.ButtonFrame, _path + 'boxicon.png')
		self.BoxButton.grid(row=1, column=1)
		self.AxisButton = self.button_with_image(self.ButtonFrame, _path + 'axisicon.png')
		self.AxisButton.grid(row=1, column=2)
		self.FrozenButton = self.button_with_image(self.ButtonFrame, _path + 'frozenicon.png')
		self.FrozenButton.grid(row=1, column=3)
		self.BondsButton = self.button_with_image(self.ButtonFrame, _path + 'bondsicon.png')
		self.BondsButton.grid(row=1, column=4)
		self.RefreshButton = self.button_with_image(self.ButtonFrame, _path + 'refresh.png')
		self.RefreshButton.grid(row=2, column=1)
		self.MoveButton = self.button_with_image(self.ButtonFrame, _path + 'moveicon.png')
		self.MoveButton.grid(row=2, column=2)
		
	def init_buttons(self):
		self.BoxButton.configure(command=lambda:self.toggle_button_state(self.BoxButton))
		self.BoxButton.state = False
		self.AxisButton.configure(command=lambda:self.toggle_button_state(self.AxisButton))
		self.AxisButton.state = False
		self.FrozenButton.configure(command=lambda:self.toggle_button_state(self.FrozenButton))
		self.FrozenButton.state = False
		self.BondsButton.configure(command=lambda:self.toggle_button_state(self.BondsButton))
		self.BondsButton.state = False
		self.RefreshButton.configure(command=self.reset_structure)
		self.set_button_state(self.BoxButton, True)
		self.set_button_state(self.AxisButton, True)
		self.set_button_state(self.FrozenButton, True)
		
		for child in self.ButtonFrame.winfo_children():
			child.grid_configure(padx=3, pady=3)
			
	def create_tk_variables(self):
		self.repeat_x = tk.StringVar()
		self.repeat_y = tk.StringVar()
		self.repeat_z = tk.StringVar()
		self.radius = tk.StringVar()
		self.zoom = tk.StringVar()
		self.fps = tk.StringVar()
		self.mouseover_atom = tk.StringVar()
		self.zoom.set(8.0)
		self.radius.set(1.5)
		self.fps.set(5)
		self.repeat_x.trace_add('write', self.trace_entries)
		self.repeat_y.trace_add('write', self.trace_entries)
		self.repeat_z.trace_add('write', self.trace_entries)
		self.radius.trace_add('write', self.trace_entries)
		self.zoom.trace_add('write', self.trace_entries)

	def save_canvas(self, event=None):
		if self.trajectory is None:
			msgbox.showinfo('Error', message='Can\'t save an empty canvas.')
		else:
			save_loc = filenav.asksaveasfilename(defaultextension='.jpg')
			save_data = self.DrawingArea.postscript(colormode='color')
			image = Image.open(io.BytesIO(save_data.encode('utf-8')))
			try:
				if ('.jpg' not in save_loc):
					image.save(save_loc + '.jpg')
				else:
					image.save(save_loc)
			except (KeyError, TypeError):
				return
			msgbox.showinfo('Success', message='Image saved successfully.')


	def trace_entries(self, a ,b ,c):
		# a, b, c unused here.
		# they are trace variables
		try:
			self.draw_structure()
		except ValueError:
			pass
	
	def standard_label(self, parent, text, size='14'):
		if type(size) is not str:
			size = str(size)
		l = tk.Label(parent, text=text, font=_standard_font + size)
		
		return l

	def reset_structure(self, event=None):
		self.radius.set(1.5)
		self.zoom.set(8.0)
		self.rotation = np.identity(3)
		self.translate = np.array([0.0, 0.0, 16.0])
		self.draw_structure()
		
	def update_button_gfx(self, button):
		try:
			if self.isButtonActive(button):
				button.configure(relief='sunken')
			else:
				button.configure(relief='raised')
		except KeyError:
			print('Invalid button press.')
			
	def toggle_button_state(self, b):
		b.state = not b.state
		self.update_button_gfx(b)
		self.draw_structure()
		
	def set_button_state(self, b, state):
		b.state = state
		self.update_button_gfx(b)
			
	def isButtonActive(self, button):
		return button.state
		
	def button_with_image(self, parent, path, width=25, height=25):
		try:
			i = tk.PhotoImage(file=path)
		except:
			i = ImageTk.PhotoImage(Image.open(path))
		self.images.append(i)
		return tk.Button(parent, image=self.images[len(self.images) - 1], width=25, height=25)
		
	def draw_entries(self):
		self.XLabel = self.standard_label(self.EntryFrame, 'Repeat       x')
		self.YLabel = self.standard_label(self.EntryFrame, '                  y')
		self.ZLabel = self.standard_label(self.EntryFrame, '                  z')
		self.ZoomLabel = self.standard_label(self.EntryFrame, 'Zoom           ')
		self.RadiusLabel = self.standard_label(self.EntryFrame, 'Radius         ')
		self.AtomLabel = tk.Label(self, textvariable=self.mouseover_atom, font=_standard_font + '12', borderwidth=2, relief='ridge')
		self.SpinX = tk.Spinbox(self.EntryFrame, from_=1.0, to=10.0, textvariable=self.repeat_x, font=_standard_font + '14', width=7)
		self.SpinY = tk.Spinbox(self.EntryFrame, from_=1.0, to=10.0, textvariable=self.repeat_y, font=_standard_font + '14', width=7)
		self.SpinZ = tk.Spinbox(self.EntryFrame, from_=1.0, to=10.0, textvariable=self.repeat_z, font=_standard_font + '14', width=7)
		self.SpinZoom = tk.Spinbox(self.EntryFrame, from_=1.0, to=100.0, increment=0.5, textvariable=self.zoom, font=_standard_font + '14', width=7)
		self.SpinRadius = tk.Spinbox(self.EntryFrame, from_=1.0, to=100.0, increment=0.1, textvariable=self.radius, font=_standard_font + '14', width=7)
		self.XLabel.grid(row=1, column=1)
		self.YLabel.grid(row=2, column=1)
		self.ZLabel.grid(row=3, column=1)
		self.ZoomLabel.grid(row=4, column=1)
		self.SpinX.grid(row=1, column=2)
		self.SpinY.grid(row=2, column=2)
		self.SpinZ.grid(row=3, column=2)
		self.SpinZoom.grid(row=4, column=2)
		self.RadiusLabel.grid(row=5, column=1)
		self.SpinRadius.grid(row=5, column=2)

	def draw_fps_frame(self):
		self.FPSFrame = tk.Frame(self, borderwidth=2, relief='ridge')
		self.FPSSlider = tk.Scale(self.FPSFrame, orient='horizontal', length=200, from_=0, to=60, state='disabled', font=_standard_font)
		self.FPSSlider.grid(row=1, column=1)
		self.FPSLabel = tk.Label(self.FPSFrame, text='    FPS  ', font=_standard_font + '14')
		self.FPSEntry = tk.Entry(self.FPSFrame, textvariable=self.fps, width=3, font=_standard_font + '14')
		self.FPSLabel.grid(row=1, column=4, sticky='S')
		self.FPSEntry.grid(row=1, column=5, sticky='S')
		
	def bind_shortcuts(self):
		# Bind shortcuts such as Ctrl+O and Ctrl+Q (Open & Quit)
		self.master.bind('<Control-o>', self.open_file)
		self.master.bind('<Control-q>', lambda e:self.master.destroy())
		self.master.bind('<Control-r>', self.reset_structure)
		self.master.bind('<Control-s>', self.save_canvas)

	def bind_events(self):
		self.DrawingArea.bind('<Motion>', self.try_mouse_atom)
		self.DrawingArea.bind('<Button-1>', lambda e:self.DrawingArea.focus_set())
		self.DrawingArea.bind('<B1-Motion>', lambda e:self.event_mouse_move(e))
		self.DrawingArea.bind('<B3-Motion>', self.translate_structure)
		self.DrawingArea.bind('<Button-4>', lambda e:self.update_zoom(e, True))
		self.DrawingArea.bind('<Button-5>', lambda e:self.update_zoom(e, False))
		self.DrawingArea.bind('<Configure>', lambda e:self.draw_structure())
		self.DrawingArea.bind('<MouseWheel>', self.update_zoom_wheel)
		self.master.bind('<space>', self.toggle_movie)

	def toggle_movie(self, event=None):
		if len(self.trajectory) == 1 or self.trajectory is None:
			return
		if self.playing:
			#...
			self.playing = False
		else:
			#...
			self.playing = True
			try:
				time = int(1000 / int(self.fps.get()))
			except (ValueError, ZeroDivisionError):
				self.playing = False
				return
			self.DrawingArea.after(time, self.update_slider)

	def update_slider(self):
		current = self.FPSSlider.get()
		try:
			time = int(1000 / int(self.fps.get()))
		except (ValueError, ZeroDivisionError):
			self.playing = False
			return
		if current == (len(self.trajectory) - 1):
			self.FPSSlider.set(0)
		else:
			self.FPSSlider.set(current + 1)
		if self.playing:
			self.DrawingArea.after(time, self.update_slider)

	def try_mouse_atom(self, event):
		if self.screenatoms is None:
			return
		x_loc = self.DrawingArea.canvasx(event.x)
		y_loc = self.DrawingArea.canvasy(event.y)
		self.mouselast = (x_loc, y_loc)
		atomid = self.get_mouse_atom()
		if atomid is not None:
			atom = self.get_frame_atoms()[atomid]
			r = atom.position
			self.mouseover_atom.set("Atom %d, %s (%.3fx %.3fy %.3fz)" %
									 (atomid, atom.symbol, r[0], r[1], r[2]))
		else:
			self.mouseover_atom.set("")
		return True

	def update_zoom(self, event, up):
		if float(self.zoom.get()) < 80 and up:
			self.zoom.set(round(float(self.zoom.get()) * 1.1, 2))
		elif float(self.zoom.get()) > 2 and not up:
			self.zoom.set(round(float(self.zoom.get()) * 0.9, 2))
		self.update_idletasks()

	def update_zoom_wheel(self, event):
		self.zoom.set(float(self.zoom.get()) + event.delta / 180)
		self.update_idletasks()

	def open_init(self, location):
		try:
			data = tsase.io.read(location, skip=0, every=1)
		except:
			msgbox.showinfo('Error', message='Could not open file.')
			return
		if type(data) is not list:
			data = [data]
		self.load_data(data)
		self.draw_structure()

	def open_file(self, event=None):
		# Largely taken from xyz
		# Read in a file and verify data
		location = filenav.askopenfilename()
		if location:
			s = location.split('/')
			try:
				data = tsase.io.read(location, skip=0, every=1)
			except:
				msgbox.showinfo('Error', message='Could not open file.')
				return
		else:
			return
		if type(data) is not list:
				data = [data]
		self.master.title(s[len(s) - 1])
		self.load_data(data)
		self.draw_structure()

	def load_data(self, data):
			# Largely taken from xyz
			# Handle movie properties
			self.trajectory = data
			if len(self.trajectory) > 1:
				self.FPSSlider.configure(from_=0, to=len(data) - 1, state='normal', command=lambda e:self.draw_structure())
			self.center_atoms()

	def center_atoms(self):
		# Taken from xyz
		if self.trajectory is None:
			return
		try:
			ra = self.trajectory[0].repeat((int(self.reapeat_x.get()),
											int(self.reapeat_y.get()),
											int(self.repeat_z.get())))
		except:
			ra = self.trajectory[0]
		r = ra.get_positions()
		minx = min(r[:, 0])
		miny = min(r[:, 1])
		minz = min(r[:, 2])
		maxx = max(r[:, 0])
		maxy = max(r[:, 1])
		maxz = max(r[:, 2])
		midx = minx + (maxx - minx) / 2
		midy = miny + (maxy - miny) / 2
		midz = minz + (maxz - minz) / 2
		self.center = np.array([midx, midy, midz])

	def draw_structure(self):
		# Draws the loaded structure
		if self.trajectory is None:
			return
		self.queue = []
		self.queue_atoms()
		if self.isButtonActive(self.BondsButton):
			self.queue_bonds()
		if self.isButtonActive(self.BoxButton):
			self.queue_box()
		self.transform_queue()
		self.sort_queue()
		self.clear_canvas()
		self.draw_queue()
		if self.isButtonActive(self.AxisButton):
			self.draw_axes()
		self.last_draw = time.time()

	def clear_canvas(self):
		self.DrawingArea.delete('all')

	def sort_queue(self):
		def cmp_queue(a):
			return a.depth
		self.queue = sorted(self.queue, key=cmp_queue)

	def draw_queue(self):
		self.screenatoms = []
		s2 = float(self.zoom.get()) * 2
		r2 = float(self.radius.get())
		w2 = int(self.DrawingArea.winfo_width()) * 0.5
		h2 = int(self.DrawingArea.winfo_height()) * 0.5
		dr = math.cos(math.pi*0.25)
		for q in self.queue:
			if q.kind == "atom":
				r = q.r
				rad = int(q.radius * s2 * r2) / 2
				x = int(r[0] * s2 + w2)
				y = int(-r[1] * s2 + h2)
				self.draw_circle(x, y, rad, q.color)
				if self.isButtonActive(self.FrozenButton):
					if q.constrained:
						self.draw_line(x-rad*dr, y-rad*dr, x+rad*dr, y+rad*dr)
						self.draw_line(x-rad*dr, y+rad*dr, x+rad*dr, y-rad*dr)
				self.screenatoms.append([x, y, rad, q.id])
			else:
				q.r1[0] = q.r1[0] * s2 + w2
				q.r1[1] = -q.r1[1] * s2 + h2
				q.r2[0] = q.r2[0] * s2 + w2
				q.r2[1] = -q.r2[1] * s2 + h2
				self.draw_line(q.r1[0], q.r1[1], q.r2[0], q.r2[1], q.width, q.color)

	def draw_line(self, x1, y1, x2, y2, width=1, color='black'):
		if type(color) is not str:
			color = '#{0:02x}{1:02x}{2:02x}'.format(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
		self.DrawingArea.create_line((x1, y1, x2, y2), width=width, fill=color)

	def draw_circle(self, x, y, r, color='red'):
		if type(color) is not str:
			color = '#{0:02x}{1:02x}{2:02x}'.format(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
		self.DrawingArea.create_oval((x-r, y-r, x+r, y+r), fill=color)

	def queue_atoms(self):
		try:
			ra = self.get_frame_atoms().repeat((int(self.repeat_x.get()),
												int(self.repeat_y.get()),
												int(self.repeat_z.get())))
		except:
			ra = self.get_frame_atoms()
		r = ra.get_positions()
		symbols = ra.get_chemical_symbols()
		for i in range(len(r)):
			atom = queueitem("atom")
			atom.id = i % len(self.get_frame_atoms())
			atom.r = np.copy(r[i])
			atom.radius = elements[symbols[i]]['radius']
			tc = self.get_frame_colors()
			if tc != None:
				atom.color = tc[atom.id]
			else:
				atom.color = elements[symbols[i]]['color']
			atom.depth = 0
			atom.constrained = False
			try:
				if len(ra.constraints) > 0:
					if i in ra.constraints[0].index:
						atom.constrained = True
			except:
				pass
			self.queue.append(atom)

	def get_mouse_atom(self):
		mx, my = self.mouselast
		atomid = None
		for a in self.screenatoms:
			d2 = (a[0] - mx)**2 + (a[1] - my)**2
			if d2 < a[2]**2:
				atomid = a[3]
		return atomid

	def draw_axes(self):
		axes = np.identity(3) * 16
		axes[0] = np.dot(self.rotation, axes[0])
		axes[1] = np.dot(self.rotation, axes[1])
		axes[2] = np.dot(self.rotation, axes[2])
		x0 = 60
		y0 = self.DrawingArea.winfo_height() - 60
		self.draw_line(x0, y0, x0 + axes[0][0], y0 - axes[0][1], color='black')
		self.draw_line(x0, y0, x0 + axes[1][0], y0 - axes[1][1], color='black')
		self.draw_line(x0, y0, x0 + axes[2][0], y0 - axes[2][1], color='black')
		self.DrawingArea.create_text((x0 + axes[0][0] - 3, y0 - axes[0][1] - 6), fill='red', text='x')
		self.DrawingArea.create_text((x0 + axes[1][0] - 1, y0 - axes[1][1] - 9), fill='green', text='y')
		self.DrawingArea.create_text((x0 + axes[2][0] - 3, y0 - axes[2][1] - 6), fill='blue', text='z')


	def queue_bonds(self):
		fa = self.get_frame_atoms()
		ra = fa.repeat((int(self.repeat_x.get()),
						int(self.repeat_y.get()),
						int(self.repeat_z.get())))
		if not 'nlist' in fa.__dict__:
			cutoffs = [1.5 * elements[i.symbol]['radius'] for i in ra]
			fa.nlist = NeighborList(cutoffs, skin=0, self_interaction=False, bothways=True)
		if len(fa.nlist.nl.cutoffs) != len(ra):
			cutoffs = [1.5 * elements[i.symbol]['radius'] for i in ra]
			fa.nlist = NeighborList(cutoffs, skin=0, self_interaction=False, bothways=True)
		fa.nlist.update(ra)
		for a in range(len(ra)):
			element = elements[ra[a].symbol]
			indices, offsets = fa.nlist.get_neighbors(a)
			for i, o in zip(indices, offsets):
				r = ra.positions[i] + np.dot(o, ra.get_cell())
				v = r - ra.positions[a]
				vm = np.linalg.norm(v)
				vunit = v/vm
				rad = 0.5 * element['radius'] * float(self.radius.get())
				p1 = ra.positions[a] + vunit * rad
				p2 = ra.positions[a] + v * 0.5
				self.queue_line(p1, p2, [c*0.85 for c in element['color']], width=self.bond_width)

	def queue_line(self, r1, r2, color, width = 1):
		line = queueitem("line")
		line.r1 = np.copy(r1)
		line.r2 = np.copy(r2)
		line.color = color
		line.depth = (r1 + r2) / 2.0
		line.width = width
		self.queue.append(line)

	def queue_box(self):
		try:
			self.get_frame_atoms().cell
		except:
			return
		bx = self.get_frame_atoms().cell
		b = np.array([[0, 0, 0], [bx[1][0], bx[1][1], bx[1][2]], [bx[1][0] +
					 bx[0][0], bx[1][1] + bx[0][1], bx[1][2] + bx[0][2]],
					 [bx[0][0], bx[0][1], bx[0][2]], [bx[2][0], bx[2][1],
					 bx[2][2]], [bx[2][0] + bx[1][0], bx[2][1] + bx[1][1],
					 bx[2][2] + bx[1][2]], [bx[2][0] + bx[1][0] + bx[0][0],
					 bx[2][1] + bx[1][1] + bx[0][1], bx[2][2] + bx[1][2] +
					 bx[0][2]], [bx[2][0] + bx[0][0], bx[2][1] + bx[0][1],
					 bx[2][2] + bx[0][2]]])
		index = [[0, 1], [0, 3], [0, 4], [7, 3], [7, 4], [7, 6], [5, 1], [5, 4],
				 [5, 6], [2, 6], [2, 3], [2, 1]]
		for i in index:
			r1 = b[i[0]]
			r2 = b[i[1]]
			boxsteps = 4
			for l in range(boxsteps):
				self.queue_line(r1 + (r2 - r1) * float(l) / boxsteps,
									r1 + (r2 - r1) * float(l + 1) /
									boxsteps, [0, 0, 0])

	def transform_queue(self):
		for i in range(len(self.queue)):
			q = self.queue[i]
			if q.kind == "atom":
				q.r -= self.center
				q.r = np.dot(self.rotation, q.r)
				q.r += self.translate
				q.depth = q.r[2]
			else:
				q.r1 -= self.center
				q.r2 -= self.center
				q.depth -= self.center
				q.r1 = np.dot(self.rotation, q.r1)
				q.r2 = np.dot(self.rotation, q.r2)
				q.depth = np.dot(self.rotation, q.depth)
				q.r1 += self.translate
				q.r2 += self.translate
				q.depth += self.translate
				q.depth = q.depth[2]

	def get_frame_atoms(self):
		if self.trajectory is None:
			return None
		drawpoint = self.trajectory[0]
		if len(self.trajectory) > 1:
			drawpoint = self.trajectory[int(self.FPSSlider.get())]
		return drawpoint

	def get_frame_colors(self):
		#todo: frame colors
		return None

	def event_mouse_move(self, event):
		if self.mouselast[0] is None:
			return
		x_loc = self.DrawingArea.canvasx(event.x)
		y_loc = self.DrawingArea.canvasy(event.y)
		dx = x_loc - self.mouselast[0]
		dy = y_loc - self.mouselast[1]
		self.mouselast = (x_loc, y_loc)
		self.rotate_x(dy * 0.009)
		self.rotate_y(dx * 0.009)
		self.draw_structure()

	def translate_structure(self, event):
		x_loc = self.DrawingArea.canvasx(event.x)
		y_loc = self.DrawingArea.canvasy(event.y)
		if self.mouselast[0] is None:
			self.mouselast = (x_loc, y_loc)
		else:
			dx = x_loc - self.mouselast[0]
			dy = y_loc - self.mouselast[1]
			self.mouselast = (x_loc, y_loc)
		self.translate += np.array([dx, -dy, 0]) / float(self.zoom.get()) / 2
		self.draw_structure()

	def rotate_x(self, theta):
		ct = math.cos(theta)
		st = math.sin(theta)
		m = np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]])
		self.rotation = np.dot(m, self.rotation)


	def rotate_y(self, theta):
		ct = math.cos(theta)
		st = math.sin(theta)
		m = np.array([[ct, 0, st], [0, 1, 0], [-st, 0, ct]])
		self.rotation = np.dot(m, self.rotation)
	
def post_configure(widget):
	# Necessary post-configuration after geometry manager has
	# Allocated space for widgets
	#widget.AtomLabel.configure(wraplength=widget.OptionsFrame.winfo_width())
	pass

def get_image_path():
	path = __file__.split('/')
	ret = ''
	for i in range(len(path) - 1):
		ret += path[i] + '/'

	return ret

def main():
	initial = None
	if (len(sys.argv) > 1):
		initial = sys.argv[1]
	global _path
	_path = get_image_path()
	main = tk.Tk()
	main.minsize(_window_width, _window_height)
	xyz_viewer = Viewer(main, initial)
	main.update()
	#post_configure(xyz_viewer)
	xyz_viewer.grid(row=0,column=0, sticky='NSEW')
	main.rowconfigure(0, weight=1)
	main.columnconfigure(0, weight=1)
	xyz_viewer.columnconfigure(2, weight=1)
	xyz_viewer.rowconfigure(1, weight=5)
	xyz_viewer.rowconfigure(2, weight=5)
	xyz_viewer.rowconfigure(3, weight=1)
	return main

if __name__ == '__main__':
    xyz = main()
    while True:
        try:
            xyz.mainloop()
            break
        except UnicodeDecodeError:
            pass
