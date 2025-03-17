
"""

"""

from collections import namedtuple
from heapq import heappush, heappop
import operator
import math
import random

class InvalidHex(ValueError):
    pass

class Hex(namedtuple("Hex", "x y")):
    _neighbours = ((2, 0), (1, 1), (-1, 1), (-2, 0), (-1, -1), (1, -1))      

    def __new__(cls, x, y):
        if (x + y) % 2 != 0:
            raise InvalidHex("x and y coordinate must sum to an even number")
        return super().__new__(cls, x, y)

    def neighbours(self):
        x, y = self
        return [Hex(x+dx, y+dy) for dx, dy in self._neighbours]

    def random_neighbour(self, random=random):
        x, y = self
        dx, dy = random.choice(self._neighbours)
        return Hex(x+dx, y+dy)

    def right_neighbour(self, random=random):
        x, y = self
        dx, dy = self._neighbours[0]
        return Hex(x+dx, y+dy)

    def down_right_neighbour(self, random=random):
        x, y = self
        dx, dy = self._neighbours[1]
        return Hex(x+dx, y+dy)

    def down_left_neighbour(self, random=random):
        x, y = self
        dx, dy = self._neighbours[2]
        return Hex(x+dx, y+dy)

    def left_neighbour(self, random=random):
        x, y = self
        dx, dy = self._neighbours[3]
        return Hex(x+dx, y+dy)

    def up_left_neighbour(self, random=random):
        x, y = self
        dx, dy = self._neighbours[4]
        return Hex(x+dx, y+dy)
    
    def up_right_neighbour(self, random=random):
        x, y = self
        dx, dy = self._neighbours[5]
        return Hex(x+dx, y+dy)

    
    def random_walk(self, N, random=random):
        position = self
        yield position
        for i in range(N):
            position = position.random_neighbour(random)
            yield position

    def square_grid(self, M, N):
        position = self
        yield position
        while M>0:
          yield position  
          for j in range(N-1):
            position = position.right_neighbour()
            yield position

          M=M-1
          if M==0:
            return 
          position = position.down_right_neighbour()
          yield position

          for j in range(N-1):
            position = position.left_neighbour()
            yield position
          position = position.down_left_neighbour()
          M=M-1
          
    def __add__(self, other):
        x1, y1 = self
        x2, y2 = other
        return Hex(x1+x2, y1+y2)

    def __sub__(self, other):
        x1, y1 = self
        x2, y2 = other
        return Hex(x1-x2, y1-y2)

    def __neg__(self):
        x, y = self
        return Hex(-x, -y)

    def distance(self, other):
        x1, y1 = self
        x2, y2 = other
        dx = abs(x1 - x2)
        dy = abs(y1 - y2)
        return dy + max(0, (dx - dy)//2)

    def rotate_left(self):
        x, y = self
        return Hex((x - 3 * y) >> 1, (x + y) >> 1)

    def rotate_right(self):
        x, y = self
        return Hex((x + 3 * y) >> 1, (y - x) >> 1)

    def field_of_view(self, transparent, max_distance, visible=None):
        if visible is None:
            visible = {}
        visible[self] = all_directions
        for direction in range(6):
            _fovtree._field_of_view(self, direction, transparent, max_distance, visible)
        return visible

    def find_path(self, destination, passable, cost=lambda pos: 1):
        pathfinder = HexPathFinder(self, destination, passable, cost)
        pathfinder.run()
        return pathfinder.path

all_directions = (1 << 6) - 1
origin = Hex(0, 0)

Hex.rotations = (
        lambda x: x,
        operator.methodcaller("rotate_left"),
        lambda x: -x.rotate_right(),
        operator.neg,
        lambda x: -x.rotate_left(),
        operator.methodcaller("rotate_right")
        )

class _FovTree:
    _corners = ((0, -2), (1, -1), (1, 1), (0, 2))
    _neighbours = (Hex(1, -1), Hex(2, 0), Hex(1, 1))
    _cached_successors = None

    def __init__(self, hexagon, direction, angle1, angle2):
        self.hexagon = hexagon
        self.angle1 = angle1
        self.angle2 = angle2
        self.direction = direction
        self.hexagons = [rot(hexagon) for rot in Hex.rotations]
        self.distance = hexagon.distance(origin)

    def get_angle(self, corner):
        cx, cy = corner
        x, y = self.hexagon
        return (3*y + cy)/float(x + cx)

    def _field_of_view(self, offset, direction, transparent, max_distance, visible):
        if self.distance > max_distance:
            return
        hexagon = offset + self.hexagons[direction]
        if transparent(hexagon):
            visible[hexagon] = all_directions
            for succ in self.successors():
                succ._field_of_view(offset, direction, transparent, max_distance, visible)
        else:
            directions = 1 << ((self.direction + direction) % 6)
            visible[hexagon] = directions | visible.get(hexagon, 0)

    def successors(self):
        _cached_successors = self._cached_successors
        if _cached_successors is None:
            _cached_successors = []
            angles = [self.get_angle(c) for c in self._corners]
            hexagon = self.hexagon
            for i in range(3):
                c1 = max(self.angle1, angles[i])
                c2 = min(self.angle2, angles[i+1])
                if c1 < c2:
                    nb = self._neighbours[i]
                    _cached_successors.append(_FovTree(hexagon + nb, (i-1) % 6, c1, c2))
            self._cached_successors = _cached_successors

        return _cached_successors

_fovtree = _FovTree(Hex(2, 0), 0, -1.0, 1.0)

class Rectangle(namedtuple("Rectangle", "x y width height")):
    
    pass

def _tiled_range(lo, hi, tile_size):
    return range(lo // tile_size, (hi + tile_size - 1) // tile_size) 

def _make_range(x, width, bloat, grid_size):
    return _tiled_range(x + grid_size - 1 - bloat, x + width + bloat, grid_size)

class HexGrid(namedtuple("HexGrid", "width height")):
   
    _hex_factor = math.sqrt(1.0/3.0)
    _corners = ((1, 1), (0, 2), (-1, 1), (-1, -1), (0, -2), (1, -1))

    def __new__(cls, width, height=None):
        if height is None:
            height = round(cls._hex_factor * width)
        return super().__new__(cls, width, height)

    def corners(self, hex):
        width, height = self
        x0, y0 = hex
        y0 *= 3
        return [(width * (x + x0), height * (y + y0)) for x, y in self._corners]

    def center(self, hex):
        width, height = self
        x, y = hex
        return (x*width, 3*height*y)

    def bounding_box(self, hex):
        width, height = self
        xc, yc = self.center(hex)
        return Rectangle(xc - width, yc - 2*height, 2*width, 4*height)
 
    def hex_at_coordinate(self, x, y):
        width, height = self
        x0 = x // width
        δx = x % width
        y0 = y // (3 * height)
        δy = y % (3 * height)

        if (x0 + y0) % 2 == 0:
            if width * δy < height * (2 * width - δx):
                return Hex(x0, y0)
            else:
                return Hex(x0 + 1, y0 + 1)
        elif width * δy < height * (width + δx):
            return Hex(x0 + 1, y0)
        else:
            return Hex(x0, y0 + 1)

    def hexes_in_rectangle(self, rectangle):
        rx, ry, r_width, r_height = rectangle
        width, height = self
        x_range = _make_range(rx, r_width, width, width)
        y_range = _make_range(ry, r_height, 2*height, 3*height)
        return (Hex(x, y) for y in y_range for x in x_range if (x + y) % 2 == 0)
