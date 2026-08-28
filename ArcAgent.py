import numpy as np

from ArcProblem import ArcProblem
from typing import List, Tuple, Optional, Dict, Callable
from ArcData import ArcData
from ArcSet import ArcSet


class ArcAgent:

    def allPossibleRotationAndFlipVariants(self, x: np.ndarray) -> List[Callable[[np.ndarray], np.ndarray]]:
        return [
            lambda g: g,  # 0 deg
            lambda g: np.rot90(g, 1),  # 90 deg
            lambda g: np.rot90(g, 2),  # 180 deg
            lambda g: np.rot90(g, 3),  # 270 deg
            lambda g: np.flip(g, axis=1),  # around x
            lambda g: np.flip(g, axis=0),  # around y
            lambda g: g.T.copy(),
        ]

    def addBorder(self,x: np.ndarray, top, bottom, left, right, color: int) -> np.ndarray:
        H, W = x.shape
        y = np.full((H + top + bottom, W + left + right), color, dtype=x.dtype)
        y[top:top + H, left:left + W] = x
        return y

    def removeBorder(self, x: np.ndarray, top, bottom, left, right) -> np.ndarray:
        H, W = x.shape
        return x[top:H - bottom, left:W - right]

    def checkForConstantBorder(self,train_pairs):
        #hopefully this finds the comon pattern in borders and removes them
        params = None
        for x, y in train_pairs:
            Hx, Wx = x.shape;
            Hy, Wy = y.shape
            if Hy >= Hx and Wy >= Wx:
                dH, dW = Hy - Hx, Wy - Wx
                if dH % 2 or dW % 2: return None
                top = dH // 2;
                bottom = dH - top
                left = dW // 2;
                right = dW - left
                color = None
                if top > 0:
                    band = y[:top, :]
                    if band.size and np.all(band == band.flat[0]): color = int(band.flat[0])
                if color is None and left > 0:
                    band = y[:, :left]
                    if band.size and np.all(band == band.flat[0]): color = int(band.flat[0])
                if color is None:
                    edges = np.concatenate([y[0, :], y[-1, :], y[:, 0], y[:, -1]])
                    color = int(np.bincount(edges, minlength=10).argmax())
                cand = ("add", top, bottom, left, right, color)

                if not np.array_equal(self.addBorder(x, top, bottom, left, right, color), y):
                    return None
            elif Hy <= Hx and Wy <= Wx:
                dH, dW = Hx - Hy, Wx - Wy
                if dH % 2 or dW % 2: return None
                top = dH // 2;
                bottom = dH - top
                left = dW // 2;
                right = dW - left
                cand = ("remove", top, bottom, left, right, 0)
                # validate
                if not np.array_equal(self.removeBorder(x, top, bottom, left, right), y):
                    return None
            else:
                return None

            if params is None:
                params = cand
            elif params != cand:
                return None

        kind, top, bottom, left, right, color = params
        if kind == "add":
            return lambda img: self.addBorder(img, top, bottom, left, right, color)
        else:
            return lambda img: self.removeBorder(img, top, bottom, left, right)

    def isSameShape(self, a: np.ndarray, b: np.ndarray) -> bool:
        return a.shape == b.shape

    def mode_color(self,x: np.ndarray) -> int:
        vals, counts = np.unique(x, return_counts=True)
        return int(vals[np.argmax(counts)])


    def detectTransformation(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]]
                             ) -> Optional[Callable[[np.ndarray], np.ndarray]]:
        #try to figure out if there's a single transformation
        candidates = self.allPossibleRotationAndFlipVariants(np.zeros((1, 1), dtype=int))
        for f in candidates:
            ok = True
            for x, y in train_pairs:
                yy = f(x)
                if yy.shape != y.shape or not np.array_equal(yy, y):
                    ok = False
                    break
            if ok:
                return f
        return None

    def cropToObject(self, img: np.ndarray) :
        bg = int(np.bincount(img.reshape(-1), minlength=10).argmax())
        mask = (img != bg)
        if not mask.any():
            return img.copy()
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        r0, r1 = rows[0], rows[-1]
        c0, c1 = cols[0], cols[-1]
        return img[r0:r1 + 1, c0:c1 + 1].copy()

    def cropConsistent(self,train_pairs):

        for x, y in train_pairs:
            if not np.array_equal(self.cropToObject(x), y):
                return None
        return self.cropToObject

    def compose(f, g):
        return lambda img: g(f(img))

    def inferColourPattern(self, train_pairs):
        #if same try by colour
        cmap = {}
        for x, y in train_pairs:
            if not self.isSameShape(x, y):
                return None
            xs = x.reshape(-1);
            ys = y.reshape(-1)
            for a, b in zip(xs, ys):
                a = int(a);
                b = int(b)
                if a in cmap and cmap[a] != b:
                    return None
                cmap[a] = b
        for c in range(10):
            if c not in cmap:
                cmap[c] = c
        return cmap

    def applyColourMap(self, x: np.ndarray, cmap: Dict[int, int]) :
        lut = np.arange(10, dtype=x.dtype)
        for k, v in cmap.items():
            if 0 <= k <= 9 and 0 <= v <= 9:
                lut[k] = v
        return lut[x]


    def __init__(self):
        #defining here so i don't forget
        self.max_predictions = 5
        pass

    def loadTrainPairs(self, arc_problem: ArcProblem) -> List[Tuple[np.ndarray, np.ndarray]]:
        pairs: List[Tuple[np.ndarray, np.ndarray]] = []
        for aset in arc_problem.training_set():
            x = aset.get_input_data().data()
            y = aset.get_output_data().data()
            pairs.append((x, y))
        return pairs

    def validateCandidate(self, fn: Callable[[np.ndarray], np.ndarray],
                          train_pairs) :
        for x, y in train_pairs:
            yy = fn(x)
            if yy.shape != y.shape or not np.array_equal(yy, y):
                return False
        return True

    def dihedralTransforms(self) -> List[Callable[[np.ndarray], np.ndarray]]:
#this is a superset of all transformations
        def rot(g, k): return np.rot90(g, k)

        def fliplr(g): return np.fliplr(g)

#todo merge this with allPossibleRotationAndFlipVariants
        return [
            lambda g: g,
            lambda g: rot(g, 1),
            lambda g: rot(g, 2),
            lambda g: rot(g, 3),
            lambda g: fliplr(g),
            lambda g: rot(fliplr(g), 1),
            lambda g: rot(fliplr(g), 2),
            lambda g: rot(fliplr(g), 3),
        ]

    def seeDihedral(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]]
                    ) -> Optional[Callable[[np.ndarray], np.ndarray]]:

        for f in self.dihedralTransforms():
            ok = True
            for x, y in train_pairs:
                yy = f(x)
                if yy.shape != y.shape or not np.array_equal(yy, y):
                    ok = False
                    break
            if ok:
                return f
        return None

    def repeatPixel(self, img: np.ndarray, sy: int, sx: int):

        if sy <= 0 or sx <= 0:
            return img
        return np.kron(img, np.ones((sy, sx), dtype=img.dtype))

    def mostFrequentColour(self, img: np.ndarray, sy: int, sx: int):
        H, W = img.shape
        if H % sy != 0 or W % sx != 0:
            return None
        out_h, out_w = H // sy, W // sx
        out = np.empty((out_h, out_w), dtype=img.dtype)
        for r in range(out_h):
            r0 = r * sy
            for c in range(out_w):
                c0 = c * sx
                block = img[r0:r0 + sy, c0:c0 + sx]
                vals, cnts = np.unique(block, return_counts=True)
                nz = vals != 0
                if nz.any():
                    out[r, c] = vals[nz][np.argmax(cnts[nz])]
                else:
                    out[r, c] = 0
        return out

    def detectColourScaling(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]]
                            ):

        def infer_up(x, y):
            Hx, Wx = x.shape;
            Hy, Wy = y.shape
            if Hx == 0 or Wx == 0: return None
            if Hy % Hx or Wy % Wx: return None
            return (Hy // Hx, Wy // Wx)

        def infer_down(x, y):
            Hx, Wx = x.shape;
            Hy, Wy = y.shape
            if Hy == 0 or Wy == 0: return None
            if Hx % Hy or Wx % Wy: return None
            return (Hx // Hy, Wx // Wy)

        #UPSCALE
        up_factors = None
        ok = True
        for x, y in train_pairs:
            f = infer_up(x, y)
            if f is None: ok = False; break
            if up_factors is None:
                up_factors = f
            elif f != up_factors:
                ok = False; break
        if ok and up_factors is not None:
            sy, sx = up_factors

            def up_fn(img):
                return self.repeatPixel(img, sy, sx)

            if self.validateCandidate(up_fn, train_pairs):
                return up_fn

        #DOWNSCALE
        down_factors = None
        ok = True
        for x, y in train_pairs:
            f = infer_down(x, y)
            if f is None: ok = False; break
            if down_factors is None:
                down_factors = f
            elif f != down_factors:
                ok = False; break
        if ok and down_factors is not None:
            sy, sx = down_factors

            def down_fn(img):
                out = self.mostFrequentColour(img, sy, sx)
                return out if out is not None else img

            if self.validateCandidate(down_fn, train_pairs):
                return down_fn

        return None

    def detectImageTiling(self, train_pairs):
        # this is the diagnoal -> vertical patters
        def is_tiling(x, y):
            Hx, Wx = x.shape;
            Hy, Wy = y.shape
            if Hy % Hx or Wy % Wx: return None
            sy, sx = Hy // Hx, Wy // Wx
            tiled = np.tile(x, (sy, sx))
            return (sy, sx) if np.array_equal(tiled, y) else None

        # same factor
        factors = None
        for x, y in train_pairs:
            f = is_tiling(x, y)
            if f is None: return None
            if factors is None:
                factors = f
            elif f != factors:
                return None

        sy, sx = factors

        def tile_fn(img):
            return np.tile(img, (sy, sx))

        return tile_fn if self.validateCandidate(tile_fn, train_pairs) else None

    def detectMirroredTiling(self, train_pairs):
        # combo from the top one
        def factors(x, y):
            Hx, Wx = x.shape
            Hy, Wy = y.shape
            if Hy % Hx or Wy % Wx: return None
            return Hy // Hx, Wy // Wx

        fac = None
        for x, y in train_pairs:
            f = factors(x, y)
            if f is None: return None
            if fac is None:
                fac = f
            elif f != fac:
                return None

            sy, sx = f
            Hx, Wx = x.shape

            for i in range(sy):
                for j in range(sx):
                    t = x.copy()  # ← ADD .copy() HERE
                    if i % 2 == 1: t = np.flipud(t)
                    if j % 2 == 1: t = np.fliplr(t)
                    y_block = y[i * Hx:(i + 1) * Hx, j * Wx:(j + 1) * Wx]
                    if not np.array_equal(t, y_block):
                        return None

        sy, sx = fac

        def mirror_tile_fn(img):
            Hx, Wx = img.shape
            out = np.empty((Hx * sy, Wx * sx), dtype=img.dtype)
            for i in range(sy):
                for j in range(sx):
                    t = img.copy()  # ← ADD .copy() HERE TOO
                    if i % 2 == 1: t = np.flipud(t)
                    if j % 2 == 1: t = np.fliplr(t)
                    out[i * Hx:(i + 1) * Hx, j * Wx:(j + 1) * Wx] = t
            return out

        return mirror_tile_fn if self.validateCandidate(mirror_tile_fn, train_pairs) else None

    def detectFillEqualEnds(self, train_pairs):

        # try to refactor this in another function this is too pattern specific
        def fn(img: np.ndarray) -> np.ndarray:
            out = img.copy()
            H, W = img.shape
            for r in range(H):
                a, b = int(img[r, 0]), int(img[r, W - 1])
                if a != 0 and a == b:
                    out[r, :] = a
            for c in range(W):
                a, b = int(img[0, c]), int(img[H - 1, c])
                if a != 0 and a == b:
                    out[:, c] = a

            return out

        return fn if self.validateCandidate(fn, train_pairs) else None

    def detectDiagonalRayFrom2x2(self, train_pairs):
        # Find 2x2 monochrome ones
        def seeds(img):
            H, W = img.shape
            out = []
            for r in range(H - 1):
                for c in range(W - 1):
                    block = img[r:r + 2, c:c + 2]
                    vals = np.unique(block)
                    if len(vals) == 1 and vals[0] != 0:
                        out.append((r, c, int(vals[0])))
            return out

        #find the NW/SE diagonal cells outside the 2x2
        def diag_cells(H, W, r, c, dir_):
            cells = []
            if dir_ == "NW":
                rr, cc = r - 1, c - 1
                while rr >= 0 and cc >= 0:
                    cells.append((rr, cc))
                    rr -= 1;
                    cc -= 1
            else:  #so  SE
                rr, cc = r + 2, c + 2
                while rr < H and cc < W:
                    cells.append((rr, cc))
                    rr += 1;
                    cc += 1
            return cells

        color_dir = {}
        for x, y in train_pairs:
            if x.shape != y.shape:
                return None
            H, W = x.shape
            for r, c, col in seeds(x):
                nw = any(y[rr, cc] == col for rr, cc in diag_cells(H, W, r, c, "NW"))
                se = any(y[rr, cc] == col for rr, cc in diag_cells(H, W, r, c, "SE"))
                if nw == se:
                    return None
                want = "NW" if nw else "SE"
                if col in color_dir and color_dir[col] != want:
                    return None
                color_dir[col] = want

        if not color_dir:
            return None

        def fn(img: np.ndarray) -> np.ndarray:
            H, W = img.shape
            out = img.copy()
            for r, c, col in seeds(img):
                dir_ = color_dir.get(col)
                if dir_ is None:
                    continue
                for rr, cc in diag_cells(H, W, r, c, dir_):
                    out[rr, cc] = col
            return out

        return fn if self.validateCandidate(fn, train_pairs) else None


    def detectIdentity(self, train_pairs):
        for x, y in train_pairs:
            if x.shape != y.shape or not np.array_equal(x, y):
                return None
        return lambda img: img.copy()



    def make_predictions(self, arc_problem: ArcProblem) -> list[np.ndarray]:
        train_pairs = self.loadTrainPairs(arc_problem)
        x_test = arc_problem.test_set().get_input_data().data()

        # 3 techniques:
        # 1) flip/rotate/transpose and see if it works
        # 2) check for any colour permutations (global recolor)
        # 2.5) (maybe) compose pose + recolor if both fit training
        # 3) shape fixes: constant border add/remove, crop-to-object
        # Fallbacks: identity + majority fill

        predictions: list[np.ndarray] = []

        f_id = self.detectIdentity(train_pairs)
        if f_id is not None:
            return [f_id(x_test)]


        f_xform = self.detectTransformation(train_pairs)
        if f_xform is not None:
            return [f_xform(x_test)]


        #just fill if there nothing there
        f_fill = self.detectFillEqualEnds(train_pairs)
        if f_fill is not None:
            predictions.append(f_fill(x_test))
            if len(predictions) >= self.max_predictions:
                return predictions[:self.max_predictions]



        #1 Mirrored tiling
        f_mtile = self.detectMirroredTiling(train_pairs)
        if f_mtile is not None:
            predictions.append(f_mtile(x_test))
            if len(predictions) >= self.max_predictions:
                return predictions[:self.max_predictions]

        #2basic tiling
        f_tile = self.detectImageTiling(train_pairs)
        if f_tile is not None:
            predictions.append(f_tile(x_test))
            if len(predictions) >= self.max_predictions:
                return predictions[:self.max_predictions]


        #3 Scaling
        f_scale = self.detectColourScaling(train_pairs)
        if f_scale is not None:
            predictions.append(f_scale(x_test))
            if len(predictions) >= self.max_predictions:
                return predictions[:self.max_predictions]



        #4Border operations
        f_border = self.checkForConstantBorder(train_pairs)
        if f_border is not None:
            predictions.append(f_border(x_test))
            if len(predictions) >= self.max_predictions:
                return predictions[:self.max_predictions]

        # 5 Crop to object -removes background
        f_crop = self.cropConsistent(train_pairs)
        if f_crop is not None:
            predictions.append(f_crop(x_test))
            if len(predictions) >= self.max_predictions:
                return predictions[:self.max_predictions]


        # 6 Dihedral transformations (rotation/flip - preserves size, simpler)
        f_pose = self.seeDihedral(train_pairs)
        if f_pose is not None:
            predictions.append(f_pose(x_test))
            if len(predictions) >= self.max_predictions:
                return predictions[:self.max_predictions]

        # Diagonal rays from 2x2 seeds
        f_diag = self.detectDiagonalRayFrom2x2(train_pairs)
        if f_diag is not None:
            predictions.append(f_diag(x_test))
            if len(predictions) >= self.max_predictions:
                return predictions[:self.max_predictions]



        # 7 Color permutation
        cmap = self.inferColourPattern(train_pairs)
        if cmap is not None:
            predictions.append(self.applyColourMap(x_test, cmap))
            if len(predictions) >= self.max_predictions:
                return predictions[:self.max_predictions]

        # 8 Combos
        if f_pose is not None and cmap is not None and len(predictions) < self.max_predictions:
            predictions.append(self.applyColourMap(f_pose(x_test), cmap))
            if len(predictions) >= self.max_predictions:
                return predictions[:self.max_predictions]

        # Failsafes
        predictions.append(x_test.copy())
        c = self.mode_color(x_test)
        predictions.append(np.full_like(x_test, c))

        return predictions[:self.max_predictions]

