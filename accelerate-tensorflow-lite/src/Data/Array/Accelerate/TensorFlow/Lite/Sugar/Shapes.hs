{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications    #-}
{-# LANGUAGE TypeFamilies        #-}
-- |
-- Module      : Data.Array.Accelerate.TensorFlow.Lite.Sugar.Shapes
-- Copyright   : [2021..2022] The Accelerate Team
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <trevor.mcdonell@gmail.com>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.TensorFlow.Lite.Sugar.Shapes
  where

-- import Data.Array.Accelerate.Representation.Type
import Data.Array.Accelerate.Sugar.Array
-- import Data.Array.Accelerate.Sugar.Elt
-- import Data.Array.Accelerate.Sugar.Shape

-- import qualified Data.Array.Accelerate.TensorFlow.Lite.Representation.Shapes as R


type family Shapes a
type instance Shapes () = ()
type instance Shapes (Array sh e) = sh
type instance Shapes (a, b) = (Shapes a, Shapes b)
type instance Shapes (a, b, c) = (Shapes a, Shapes b, Shapes c)
type instance Shapes (a, b, c, d) = (Shapes a, Shapes b, Shapes c, Shapes d)
type instance Shapes (a, b, c, d, e) = (Shapes a, Shapes b, Shapes c, Shapes d, Shapes e)
type instance Shapes (a, b, c, d, e, f) = (Shapes a, Shapes b, Shapes c, Shapes d, Shapes e, Shapes f)
type instance Shapes (a, b, c, d, e, f, g) = (Shapes a, Shapes b, Shapes c, Shapes d, Shapes e, Shapes f, Shapes g)
type instance Shapes (a, b, c, d, e, f, g, h) = (Shapes a, Shapes b, Shapes c, Shapes d, Shapes e, Shapes f, Shapes g, Shapes h)
type instance Shapes (a, b, c, d, e, f, g, h, i) = (Shapes a, Shapes b, Shapes c, Shapes d, Shapes e, Shapes f, Shapes g, Shapes h, Shapes i)
type instance Shapes (a, b, c, d, e, f, g, h, i, j) = (Shapes a, Shapes b, Shapes c, Shapes d, Shapes e, Shapes f, Shapes g, Shapes h, Shapes i, Shapes j)
type instance Shapes (a, b, c, d, e, f, g, h, i, j, k) = (Shapes a, Shapes b, Shapes c, Shapes d, Shapes e, Shapes f, Shapes g, Shapes h, Shapes i, Shapes j, Shapes k)
type instance Shapes (a, b, c, d, e, f, g, h, i, j, k, l) = (Shapes a, Shapes b, Shapes c, Shapes d, Shapes e, Shapes f, Shapes g, Shapes h, Shapes i, Shapes j, Shapes k, Shapes l)
type instance Shapes (a, b, c, d, e, f, g, h, i, j, k, l, m) = (Shapes a, Shapes b, Shapes c, Shapes d, Shapes e, Shapes f, Shapes g, Shapes h, Shapes i, Shapes j, Shapes k, Shapes l, Shapes m)
type instance Shapes (a, b, c, d, e, f, g, h, i, j, k, l, m, n) = (Shapes a, Shapes b, Shapes c, Shapes d, Shapes e, Shapes f, Shapes g, Shapes h, Shapes i, Shapes j, Shapes k, Shapes l, Shapes m, Shapes n)
type instance Shapes (a, b, c, d, e, f, g, h, i, j, k, l, m, n, o) = (Shapes a, Shapes b, Shapes c, Shapes d, Shapes e, Shapes f, Shapes g, Shapes h, Shapes i, Shapes j, Shapes k, Shapes l, Shapes m, Shapes n, Shapes o)
type instance Shapes (a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p) = (Shapes a, Shapes b, Shapes c, Shapes d, Shapes e, Shapes f, Shapes g, Shapes h, Shapes i, Shapes j, Shapes k, Shapes l, Shapes m, Shapes n, Shapes o, Shapes p)


{--
class Shapes a where
  type ShapesR a
  shapesR    :: R.ShapesR' (EltR (ShapesR a))
  fromShapes :: ShapesR a -> EltR (ShapesR a)

instance Shapes () where
  type ShapesR () = ()
  shapesR    = TupRunit
  fromShapes = id

instance Shape sh => Shapes (Array sh e) where
  type ShapesR (Array sh e) = sh
  shapesR    = TupRsingle (shapeR @sh)
  fromShapes = fromElt

instance (Shapes a, Shapes b) => Shapes (a, b) where
  type ShapesR (a, b) = (ShapesR a, ShapesR b)
  shapesR = TupRunit `TupRpair` shapesR @a `TupRpair` shapesR @b
  fromShapes (a, b) = (((), fromShapes @a a), fromShapes @b b)

instance (Shapes a, Shapes b, Shapes c) => Shapes (a, b, c) where
  type ShapesR (a, b, c) = (ShapesR a, ShapesR b, ShapesR c)
  shapesR = TupRunit `TupRpair` shapesR @a `TupRpair` shapesR @b `TupRpair` shapesR @c
  fromShapes (a, b, c) = ((((), fromShapes @a a), fromShapes @b b), fromShapes @c c)
--}

