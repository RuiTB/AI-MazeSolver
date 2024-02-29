from abc import ABCMeta, abstractmethod
from typing import *
import collections
from enum import Enum
from matplotlib import cm
from PIL import Image, ImageDraw

T = TypeVar("T")


class Frontire(Generic[T], metaclass=ABCMeta):
    def __init__(self) -> None:
        self.frontire: collections.deque[T] = collections.deque()

    def add(self, node: T) -> None:
        self.frontire.append(node)

    @abstractmethod
    def remove(self) -> T:
        pass

    def empty(self) -> bool:
        return len(self.frontire) == 0

    def contains(self, node: T) -> bool:
        return any(i == node for i in self.frontire)


class StackFrontire(Frontire[T]):
    def remove(self) -> T:
        if self.empty():
            raise Exception("empty frontire")
        node = self.frontire.pop()
        return node


class QueueFrontire(Frontire[T]):
    def remove(self) -> T:
        if self.empty():
            raise Exception("empty frontire")
        node = self.frontire.popleft()
        return node


class Block(Enum):
    WALL = "#"
    ROAD = "."
    START = "@"
    GOAL = "$"
    # for visual only
    EXPLORED = "*"
    PATH = "+"


class Action(Enum):
    LEFT = (0, -1)
    UP = (-1, 0)
    RIGHT = (0, 1)
    DOWN = (1, 0)


class Node:
    def __init__(self, state: tuple[int], parent: "Node", action: Action) -> None:
        self.state = state
        self.parent = parent
        self.action = action


type Maze = tuple[tuple[Block]]


class MazePrinter:
    def __init__(self, maze: Maze):
        self.maze = maze
        self.height = len(maze)
        self.width = len(maze[0])

    def __repr__(self):
        return "\n".join(
            "".join(str(block.value) for block in row) for row in self.maze
        )

    def _maze_with_explored(self, explored: set[tuple[int]], maze: Maze) -> Maze:
        new_maze: list[tuple[Block]] = []
        for i, row in enumerate(maze):
            new_row: list[Block] = []
            for j, block in enumerate(row):
                if (
                    (i, j) in explored
                    and block is not Block.START
                    and block is not Block.GOAL
                ):
                    new_row.append(Block.EXPLORED)
                else:
                    new_row.append(block)

            new_maze.append(tuple(new_row))
        return tuple(new_maze)

    def _maze_with_path(self, path: list[Node], maze: Maze) -> Maze:
        pathStates = {node.state for node in path}

        new_maze: list[tuple[Block]] = []
        for i, row in enumerate(maze):
            new_row: list[Block] = []
            for j, block in enumerate(row):
                if (
                    (i, j) in pathStates
                    and block is not Block.START
                    and block is not Block.GOAL
                ):
                    new_row.append(Block.PATH)
                else:
                    new_row.append(block)
            new_maze.append(tuple(new_row))
        return tuple(new_maze)

    def create_image(
        self,
        color_map={
            Block.WALL: "black",
            Block.ROAD: "white",
            Block.START: "green",
            Block.GOAL: "red",
            Block.EXPLORED: "yellow",
            Block.PATH: "blue",
        },
        path: list[Node] = None,
        explored: set[tuple[int]] = None,
    ):
        # Create an empty image
        image = Image.new("RGB", (self.width * 20, self.height * 20))
        draw = ImageDraw.Draw(image)
        maze = self.maze
        if explored is not None:
            maze = self._maze_with_explored(explored, maze)

        if path is not None:
            maze = self._maze_with_path(path, maze)

        # Draw colored squares based on maze blocks
        for y in range(self.height):
            for x in range(self.width):
                block = maze[y][x]
                color = color_map.get(
                    block, "black"
                )  # Default to black for unknown blocks
                draw.rectangle(
                    ((x * 20, y * 20), (x * 20 + 20, y * 20 + 20)), fill=color
                )

        return image


class MazeSolver:
    def __init__(
        self,
        frontire: Frontire[Node],
        maze: Maze,
    ) -> None:
        self.frontire = frontire
        self.maze = maze

    def solve(self) -> list[Node]:
        startingNode: Node = Node(
            state=self._startState(),
            action=None,
            parent=None,
        )

        self.frontire.add(startingNode)
        self.explored: set[Node] = set()
        goalState = self._goalState()
        while True:
            if self.frontire.empty():
                raise Exception("no solution found")

            node = self.frontire.remove()
            self.explored.add(node.state)
            if node.state == goalState:
                path: list[Node] = [node]
                while node.parent is not None:
                    path.append(node.parent)
                    node = node.parent
                return reversed(path)

            for action in self._actions(node):
                result = self._result(node, action)
                if result.state not in self.explored:
                    self.frontire.add(result)

    def _actions(self, node: Node) -> list[Action]:
        actions: set[Action] = set()
        valid_blocks: set[Block] = {
            Block.GOAL,
            Block.ROAD,
        }
        state = node.state
        if (state[1] - 1) >= 0 and self.maze[state[0]][state[1] - 1] in valid_blocks:
            actions.add(Action.LEFT)

        if (state[0] - 1) >= 0 and self.maze[state[0] - 1][state[1]] in valid_blocks:
            actions.add(Action.UP)

        if (state[1] + 1) < len(self.maze[state[0]]) and self.maze[state[0]][
            state[1] + 1
        ] in valid_blocks:
            actions.add(Action.RIGHT)

        if (state[0] + 1) < len(self.maze) and self.maze[state[0] + 1][
            state[1]
        ] in valid_blocks:
            actions.add(Action.DOWN)

        return actions

    def _startState(self) -> tuple[int]:
        for i, row in enumerate(self.maze):
            for j, block in enumerate(row):
                if block == Block.START:
                    return (i, j)
        raise Exception("no starting block was found")

    def _goalState(self) -> tuple[int]:
        for i, row in enumerate(self.maze):
            for j, block in enumerate(row):
                if block == Block.GOAL:
                    return (i, j)
        raise Exception("no goal block was found")

    def _result(self, state: Node, action: Action):
        return Node(
            action=action,
            parent=state,
            state=(state.state[0] + action.value[0], state.state[1] + action.value[1]),
        )


class MazeParser:
    @staticmethod
    def parseFile(file: TextIO) -> Maze:
        maze: list[tuple[Block]] = []
        for line in file.readlines():
            blockLine: list[Block] = []
            for char in line:
                if char.isspace():
                    continue
                blockLine.append(Block(char))
            maze.append(tuple(blockLine))
        return tuple(maze)


def main():
    with open("maze3.txt") as mazeFile:
        maze = MazeParser.parseFile(mazeFile)
        solver = MazeSolver(StackFrontire(), maze)
        path = solver.solve()
        maze_printer = MazePrinter(maze)
        image = maze_printer.create_image(explored=solver.explored, path=path)
        image.save("mazeImage.png")


if __name__ == "__main__":
    main()
