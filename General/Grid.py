class Grid:
    """
    Represents a multi-dimensional grid in space.
    """

    def __init__(self, bounds, sizes):
        """
        :param bounds: List of tuples each containing bounds by single axis
        :param sizes: List of integers containing point counts for corresponding axes
        """
        if len(bounds) != len(sizes):
            raise ValueError('Grid dimension could not be acquired')
        for size in sizes:
            if not size >= 1:
                raise ValueError('Grid size must be at least 1 for every axis')
        for bound in bounds:
            if len(bound) != 2:
                raise ValueError('Invalid bound parameter detected, must have length 2')
            if bound[0] >= bound[1]:
                raise ValueError('Invalid bound parameter detected, be ascending')
        self.bounds = bounds
        self.sizes = sizes

    def grid_step(self, axis):
        """
        Obtain grid step on specified axis

        :param axis: Index of the axis specified
        :return: Step
        """
        if self.sizes[axis] != 1:
            return (self.bounds[axis][1] - self.bounds[axis][0]) / (self.sizes[axis] - 1)

    def mesh(self, initial_obj=None):
        """
        Return empty multi-dimensional array with parameters corresponding to grid constraints

        :return: Mesh
        """
        mesh = initial_obj if initial_obj else 0.
        for size in reversed(self.sizes):
            mesh = [mesh] * size
        return mesh

    def index(self, point):
        """
        Match any grid point to an unique index

        :param point: Grid point
        :return: Index
        """
        if len(point) != len(self.sizes):
            raise ValueError('Point dimension does not match grid dimension')
        index = 0
        size_prod = 1
        for i in range(len(self.sizes)):
            if not 0 <= point[i] < self.sizes[i]:
                raise ValueError('Point is out of bounds of the grid')
            if type(point[i]) != int:
                raise TypeError('Point indices must be int, not {}'.format(type(point[i])))
            index += point[i] * size_prod
            size_prod *= self.sizes[i]
        return index

    def point_to_absolute(self, point):
        if len(point) != len(self.sizes):
            raise ValueError('Point dimension does not match grid dimension')
        return [self.bounds[dim][0] + self.grid_step(dim) * point[dim] for dim in range(len(self.sizes))]

    def point_from_absolute(self, point):
        if len(point) != len(self.sizes):
            raise ValueError('Point dimension does not match grid dimension')
        return [int((point[dim] - self.bounds[dim][0]) / (self.bounds[dim][1] - self.bounds[dim][0]) * self.sizes[dim])
                for dim in range(len(self.sizes))]

    def shift_point(self, point, axis, offset):
        if not 0 <= axis < len(point):
            raise IndexError('Point has no axis #{}'.format(axis))
        if len(point) != len(self.sizes):
            raise ValueError('Point dimension does not match grid dimension')
        return tuple((point[i] + offset if i == axis else point[i]) % self.sizes[i] for i in range(len(self)))

    def points_inside(self):
        for i in range(len(self)):
            pt = self[i]
            for dim in range(self.dimensions()):
                if pt[dim] == 0 or pt[dim] == self.sizes[dim] - 1:
                    break
            else:
                yield pt

    def __getitem__(self, item):
        """
        Obtain grid point by specified index

        :param item: Index
        :return: Grid point
        """
        if type(item) != int:
            raise TypeError('Grid index must be int, not {}'.format(type(item)))
        if not 0 <= item < len(self):
            raise IndexError('Grid index out of range')
        point = []
        for i in range(len(self.sizes)):
            point.append(item % self.sizes[i])
            item //= self.sizes[i]
        return tuple(point)

    def __contains__(self, item):
        """
        Check if point is in bounds of the grid

        :param item: Point
        :return: True if point is inside the grid, False otherwise
        """
        if len(item) != len(self.sizes):
            raise ValueError('Point dimension does not match grid dimension')
        for i in range(len(self.sizes)):
            if not 1 <= item[i] < self.sizes[i] - 1:
                return False
        return True

    def __iter__(self):
        """
        Returns an iterator of all points of the grid
        """
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        """
        Counts points in the grid
        """
        a = 1
        for size in self.sizes:
            a *= size
        return a

    def dimensions(self):
        """
        :return: Dimension count
        """
        return len(self.sizes)
