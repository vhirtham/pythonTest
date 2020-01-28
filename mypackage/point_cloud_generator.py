"""Contains methods and classes to generate 3d point clouds."""

import mypackage.geometry as geo
import numpy as np
import copy
import mypackage.transformations as tf


# Helper functions ------------------------------------------------------------

def to_list(var):
    if isinstance(var, list):
        return var
    if var is None:
        return []
    return [var]


# Profile class ---------------------------------------------------------------

class Profile:
    """Defines a 2d profile."""

    def __init__(self, shapes):
        """
        Construct profile class.

        :param: shapes: Instance or list of geo.Shape2D class(es)
        """
        self._shapes = []
        self.add_shapes(shapes)

    @property
    def num_shapes(self):
        """
        Get the number of shapes of the profile.

        :return: Number of shapes
        """
        return len(self._shapes)

    def add_shapes(self, shapes):
        """
        Add shapes to the profile.

        :param shapes: Instance or list of geo.Shape2D class(es)
        :return: ---
        """
        if not isinstance(shapes, list):
            shapes = [shapes]

        if not all(isinstance(shape, geo.Shape2D) for shape in shapes):
            raise TypeError(
                "Only instances or lists of Shape2d objects are accepted.")

        self._shapes += shapes

    def rasterize(self, raster_width):
        """
        Rasterize the profile.

        :param: raster_width: Raster width
        :return: Raster data
        """
        raster_data = np.empty([2, 0])
        for shape in self._shapes:
            shape_data = shape.rasterize(raster_width)
            raster_data.shape
            raster_data = np.hstack(
                (raster_data, shape.rasterize(raster_width)))

        return raster_data

    @property
    def shapes(self):
        """
        Get the profiles shapes.

        :return: Shapes
        """
        return self._shapes


# Trace segment classes -------------------------------------------------------

class LinearHorizontalTraceSegment:
    """Trace segment with a linear path and constant z-component."""

    def __init__(self, length):
        """
        Constructor.

        :param length: Length of the segment
        """
        if length <= 0:
            raise ValueError("'length' must have a positive value.")
        self._length = length

    @property
    def length(self):
        """
        Get the length of the segment.

        :return: Length of the segment
        """
        return self._length

    def local_coordinate_system(self, relative_position):
        """
        Calculate a local coordinate system along the trace segment.

        :param relative_position: Relative position on the trace [0 .. 1]
        :return: Local coordinate system
        """
        relative_position = np.clip(relative_position, 0, 1)

        origin = np.array([0, 1, 0]) * relative_position * self._length
        return tf.CartesianCoordinateSystem3d(origin=origin)


class RadialHorizontalTraceSegment:
    """Trace segment describing an arc with constant z-component."""

    def __init__(self, radius, angle, clockwise=False):
        """
        Constructor.

        :param radius: Radius of the arc
        :param angle: Angle of the arc
        :param clockwise: If True, the rotation is clockwise. Otherwise it
        is counter-clockwise.
        """
        if radius <= 0:
            raise ValueError("'radius' must have a positive value.")
        if angle <= 0:
            raise ValueError("'angle' must have a positive value.")
        self._radius = radius
        self._angle = angle
        self._length = self._arc_length(radius, angle)
        if clockwise:
            self._sign_winding = -1
        else:
            self._sign_winding = 1

    @staticmethod
    def _arc_length(radius, angle):
        """
        Calculate the arc length
        :param radius: Radius
        :param angle: Angle (rad)
        :return: Arc length
        """
        return angle * radius

    @property
    def angle(self):
        """
        Get the angle of the segment.

        :return: Angle of the segment
        """
        return self._angle

    @property
    def length(self):
        """
        Get the length of the segment.

        :return: Length of the segment
        """
        return self._length

    @property
    def radius(self):
        """
        Get the radius of the segment.

        :return: Radius of the segment
        """
        return self._radius

    def is_clockwise(self):
        return self._sign_winding < 0

    def local_coordinate_system(self, relative_position):
        """
        Calculate a local coordinate system along the trace segment.

        :param relative_position: Relative position on the trace [0 .. 1]
        :return: Local coordinate system
        """
        relative_position = np.clip(relative_position, 0, 1)

        basis = tf.rotation_matrix_z(
            self._angle * relative_position * self._sign_winding)
        translation = np.array([1, 0, 0]) * self._radius * self._sign_winding

        origin = np.matmul(basis, translation) - translation
        return tf.CartesianCoordinateSystem3d(basis, origin)


# Trace class -----------------------------------------------------------------

class Trace:
    """Defines a 3d trace."""

    def __init__(self, segments,
                 coordinate_system=tf.CartesianCoordinateSystem3d()):
        """
        Constructor.

        :param segments: Single segment or list of segments
        :param coordinate_system: Coordinate system of the trace
        """
        if not isinstance(coordinate_system, tf.CartesianCoordinateSystem3d):
            raise TypeError(
                "'coordinate_system' must be of type "
                "transformations.CartesianCoordinateSystem3d")

        self._segments = to_list(segments)
        self._create_lookups(coordinate_system)

        if self.length <= 0:
            raise Exception("Trace has no length.")

    def _create_lookups(self, coordinate_system_start):
        """
        Create lookup tables.

        :param coordinate_system_start: Coordinate system at the start of
        the trace.
        :return: ---
        """
        self._coordinate_system_lookup = [coordinate_system_start]
        self._total_length_lookup = [0]
        self._segment_length_lookup = []

        segments = self._segments

        # Fill coordinate system lookup
        for i in range(len(segments) - 1):
            lcs_segment_end = segments[i].local_coordinate_system(1)
            cs = self._coordinate_system_lookup[i] + lcs_segment_end
            self._coordinate_system_lookup += [cs]

        # Fill length lookups
        total_length = 0
        for i in range(len(segments)):
            segment_length = segments[i].length
            total_length += segment_length
            self._segment_length_lookup += [segment_length]
            self._total_length_lookup += [total_length]

    def _get_segment_index(self, position):
        """
        Get the segment index for a certain position.

        :param position: Position
        :return: Segment index
        """
        position = np.clip(position, 0, self.length)
        for i in range(len(self._total_length_lookup) - 1):
            if position <= self._total_length_lookup[i + 1]:
                return i

    @property
    def coordinate_system(self):
        """
        Get the trace's coordinate system.

        :return: Coordinate system of the trace
        """
        return self._coordinate_system_lookup[0]

    @property
    def length(self):
        """
        Get the length of the trace.

        :return: Length of the trace.
        """
        return self._total_length_lookup[-1]

    @property
    def segments(self):
        """
        Get the trace's segments.

        :return: Segments of the trace
        """
        return self._segments

    def num_segments(self):
        """
        Get the number of segments.

        :return: Number of segments
        """
        return len(self._segments)

    def local_coordinate_system(self, position):
        """
        Get the local coordinate system at a specific position on the trace.

        :param position: Position
        :return: Local coordinate system
        """
        idx = self._get_segment_index(position)

        total_length_start = self._total_length_lookup[idx]
        segment_length = self._segment_length_lookup[idx]
        weight = (position - total_length_start) / segment_length

        local_segment_cs = self.segments[idx].local_coordinate_system(weight)
        segment_start_cs = self._coordinate_system_lookup[idx]

        return segment_start_cs + local_segment_cs


# Linear profile interpolation class ------------------------------------------

class LinearProfileInterpolationSBS:
    """Linear segment by segment interpolation class for profiles."""

    @staticmethod
    def interpolate(a, b, weight):
        """
        Interpolate 2 profiles.

        :param a: First profile
        :param b: Second profile
        :param weight: Weighting factor [0 .. 1]. If 0, the profile is
        identical to 'a' and if 1, it is identical to b.
        :return: Interpolated profile
        """
        weight = np.clip(weight, 0, 1)
        if not len(a.shapes) == len(b.shapes):
            raise Exception("Number of profile shapes do not match.")

        shapes_c = []
        for i in range(a.num_shapes):
            shapes_c += [geo.Shape2D.linear_interpolation(a.shapes[i],
                                                          b.shapes[i],
                                                          weight)]

        return Profile(shapes_c)


# Varying profile class -------------------------------------------------------

class VaryingProfile:
    def __init__(self, profiles, locations, interpolation_schemes):
        locations = to_list(locations)
        interpolation_schemes = to_list(interpolation_schemes)

        if not locations[0] == 0:
            locations = [0] + locations

        if not len(profiles) == len(locations):
            raise Exception(
                "Invalid list of locations. See function description.")

        if not len(interpolation_schemes) == len(profiles) - 1:
            raise Exception(
                "Number of interpolations must be 1 less than number of "
                "profiles.")

        for i in range(len(profiles) - 1):
            if locations[i] >= locations[i + 1]:
                raise Exception(
                    "Locations need to be sorted in ascending order.")

        self._profiles = profiles
        self._locations = locations
        self._interpolation_schemes = interpolation_schemes

    def _segment_index(self, location):
        idx = 0
        while location > self._locations[idx + 1]:
            idx += 1
        return idx

    @property
    def locations(self):
        return self._locations

    @property
    def max_location(self):
        return self._locations[-1]

    @property
    def num_interpolation_schemes(self):
        return len(self._interpolation_schemes)

    @property
    def num_locations(self):
        return len(self._locations)

    @property
    def num_profiles(self):
        return len(self._profiles)

    def local_profile(self, location):
        location = np.clip(location, 0, self.max_location)

        idx = self._segment_index(location)
        segment_length = self._locations[idx + 1] - self._locations[idx]
        weight = (location - self._locations[idx]) / segment_length

        return self._interpolation_schemes[idx].interpolate(
            self._profiles[idx], self._profiles[idx + 1], weight)


#  Geometry class -------------------------------------------------------------


class Geometry:
    """Define the experimental geometry"""

    def __init__(self, profile, trace):
        """
        Constructor.

        :param profile: Constant or variable profile.
        :param trace: Trace
        """
        self._check_inputs(profile, trace)
        self._profile = profile
        self._trace = trace

    def _check_inputs(self, profile, trace):
        """
        Check the inputs to the constructor.

        :param profile: Constant or variable profile.
        :param trace: Trace
        :return: ---
        """
        if not (isinstance(profile, Profile) or
                isinstance(profile, VaryingProfile)):
            raise TypeError(
                "'profile' must be a 'Profile' or 'VariableProfile' class")

        if not isinstance(trace, Trace):
            raise TypeError(
                "'trace' must be a 'Trace' class")

    def _get_local_profile_data(self, trace_location, raster_width):
        profile_location = trace_location / self._trace.length * \
                           self._profile.max_location
        profile = self._profile.local_profile(profile_location)
        return self._profile_data_3d(profile, raster_width)

    def _get_trace_locations(self, raster_width):
        num_raster_segments = int(np.round(self._trace.length / raster_width))
        raster_width_eff = self._trace.length / num_raster_segments
        locations = np.arange(0,
                              self._trace.length - raster_width_eff / 2,
                              raster_width_eff)
        return np.hstack([locations, self._trace.length])

    def _get_transformed_profile_data(self, profile_data, location):
        local_cs = self._trace.local_coordinate_system(location)
        local_data = np.matmul(local_cs.basis, profile_data)
        return local_data + local_cs.origin[:, np.newaxis]

    def _profile_data_3d(self, profile, raster_width):
        profile_data = profile.rasterize(raster_width)
        return np.insert(profile_data, 1, 0, axis=0)

    def _rasterize_constant_profile(self, raster_width):

        profile_data = self._profile_data_3d(self._profile, raster_width)

        locations = self._get_trace_locations(raster_width)
        raster_data = np.empty([3, 0])
        for i in range(len(locations)):
            local_data = self._get_transformed_profile_data(profile_data,
                                                            locations[i])
            raster_data = np.hstack([raster_data, local_data])

        return raster_data

    def _rasterize_variable_profile(self, raster_width):
        locations = self._get_trace_locations(raster_width)
        raster_data = np.empty([3, 0])
        for i in range(len(locations)):
            profile_data = self._get_local_profile_data(locations[i],
                                                        raster_width)

            local_data = self._get_transformed_profile_data(profile_data,
                                                            locations[i])
            raster_data = np.hstack([raster_data, local_data])

        return raster_data

    def rasterize(self, raster_width):
        if isinstance(self._profile, Profile):
            return self._rasterize_constant_profile(raster_width)
        else:
            return self._rasterize_variable_profile(raster_width)


# Section class ---------------------------------------------------------------

class Section:
    """Defines a section"""

    def __init__(self, profiles, trace, profile_interpolations=None,
                 profile_positions=None):

        profiles = to_list(profiles)
        profile_positions = to_list(profile_positions)
        profile_interpolations = to_list(profile_interpolations)

        if not all(isinstance(profile, Profile) for profile in profiles):
            raise TypeError(
                "Only instances of lists or Shape2d objects are accepted.")

        if len(profiles) > 1 and len(profile_interpolations) != len(
                profiles) - 1:
            raise Exception(
                "Number of interpolations must be one less than number "
                "of profiles")

        if len(profiles) > 2 and len(profile_positions) != len(profiles) - 2:
            raise Exception(
                "If more than two profiles are used, the positions of the "
                "profiles between the first and last one need to be "
                "specified in a list of size: num_profiles -2.")

        self._profiles = profiles
        self._interpolations = profile_interpolations
        self._positions = [0] + profile_positions + [trace.length]
        self.trace = trace

    def _profile_segment_index(self, position):
        position = np.clip(position, 0, self._positions[-1])
        idx = 0
        while position > self._positions[idx + 1]:
            idx += 1
        return idx

    def _interpolated_profile(self, position):
        if len(self._profiles) == 1:
            return self._profiles[0]
        else:
            idx = self._profile_segment_index(position)
            weight = (position - self._positions[idx]) / (
                    self._positions[idx + 1] - self._positions[idx])
            profile = self._interpolations[idx].interpolate(
                self._profiles[idx], self._profiles[idx + 1], weight)

            return profile

    def rasterize(self, raster_width):

        raster_data = np.empty([3, 0])
        trace = self.trace
        global_cs = tf.CartesianCoordinateSystem3d()
        for position in np.arange(0, trace.length + 0.01, 0.5):
            profile = self._interpolated_profile(position)
            profile_raster_data = profile.rasterize(raster_width)
            profile_raster_data = np.insert(profile_raster_data, 1, 0, axis=0)
            trace_cs = trace.local_coordinate_system(position)

            rotation = tf.change_of_basis_rotation(trace_cs, global_cs)
            translation = tf.change_of_basis_translation(trace_cs, global_cs)

            profile_raster_data = np.matmul(rotation,
                                            profile_raster_data)

            profile_raster_data += translation[:, np.newaxis]

            raster_data = np.hstack((raster_data, profile_raster_data))

        return raster_data.transpose()
