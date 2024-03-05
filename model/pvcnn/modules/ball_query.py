import torch
import torch.nn as nn

from . import functional as F

__all__ = ['BallQuery']


class BallQuery(nn.Module):
    def __init__(self, radius, num_neighbors, include_coordinates=True):
        super().__init__()
        self.radius = radius
        self.num_neighbors = num_neighbors
        self.include_coordinates = include_coordinates

    def forward(self, points_coords, centers_coords, temb, points_features=None):
        points_coords = points_coords.contiguous()
        centers_coords = centers_coords.contiguous()
        neighbor_indices = F.ball_query(centers_coords, points_coords, self.radius, self.num_neighbors)
        neighbor_coordinates = F.grouping(points_coords, neighbor_indices)
        neighbor_coordinates = neighbor_coordinates - centers_coords.unsqueeze(-1)

        if points_features is None:
            assert self.include_coordinates, 'No Features For Grouping'
            neighbor_features = neighbor_coordinates
        else:
            neighbor_features = F.grouping(points_features, neighbor_indices) # return [B, C, M, U] C=feat dim, M=# centers, U=# neighbours
            if self.include_coordinates:
                neighbor_features = torch.cat([neighbor_coordinates, neighbor_features], dim=1)
        return neighbor_features, F.grouping(temb, neighbor_indices)

    def extra_repr(self):
        return 'radius={}, num_neighbors={}{}'.format(
            self.radius, self.num_neighbors, ', include coordinates' if self.include_coordinates else '')


class BallQueryHO(nn.Module):
    "no point feature, but only relative and abs coordinate"
    def __init__(self, radius, num_neighbors, include_relative=False):
        super().__init__()
        self.radius = radius
        self.num_neighbors = num_neighbors
        self.include_relative = include_relative

    def forward(self, points_coords, centers_coords, points_features=None):
        """
        if not enough points inside the given radius, the entries will be zero
        if too many points inside the radius, the order is random??? (not sure)
        :param points_coords: (B, 3, N)
        :param centers_coords: (B, 3, M)
        :param points_features: None
        :return:
        """
        points_coords = points_coords.contiguous()
        centers_coords = centers_coords.contiguous()
        neighbor_indices = F.ball_query(centers_coords, points_coords, self.radius, self.num_neighbors)
        neighbor_coordinates = F.grouping(points_coords, neighbor_indices) # (B, 3, M, U)
        if self.include_relative:
            neighbor_coordinates_rela = neighbor_coordinates - centers_coords.unsqueeze(-1)
            neighbor_coordinates = torch.cat([neighbor_coordinates, neighbor_coordinates_rela], 1) # (B, 6, M, U)
        # flatten the coordinate
        neighbor_coordinates = neighbor_coordinates.permute(0, 1, 3, 2) # (B, 3/6, U, M)
        neighbor_coordinates = torch.flatten(neighbor_coordinates, 1, 2) # (B, 3*U, M)
        return neighbor_coordinates

    def extra_repr(self):
        return 'radius={}, num_neighbors={}{}'.format(
            self.radius, self.num_neighbors, ', include relative' if self.include_relative else '')

