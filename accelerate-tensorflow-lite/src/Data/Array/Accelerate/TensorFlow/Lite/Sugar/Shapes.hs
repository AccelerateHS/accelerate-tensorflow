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

import Data.Kind

import Data.Array.Accelerate.Sugar.Array
import Data.Array.Accelerate.Sugar.Elt
import Data.Array.Accelerate.Sugar.Shape
import Data.Array.Accelerate.Representation.Type

import qualified Data.Array.Accelerate.TensorFlow.Lite.Representation.Shapes as R



class Arrays a => HasShapes a where
  type Shapes a :: Type
  shapesR :: R.ShapesR (ArraysR a)
  fromShapes :: Shapes a -> R.Shapes (ArraysR a)

instance HasShapes () where
  type Shapes () = ()
  shapesR = TupRunit
  fromShapes = id

instance (Shape sh, Elt e) => HasShapes (Array sh e) where
  type Shapes (Array sh e) = sh
  shapesR = TupRsingle (shapeR @sh)
  fromShapes = fromElt

instance (HasShapes a, HasShapes b) => HasShapes (a, b) where
  type Shapes (a, b) = (Shapes a, Shapes b)
  shapesR = TupRunit `TupRpair` shapesR @a `TupRpair` shapesR @b
  fromShapes (a, b) = (((), fromShapes @a a), fromShapes @b b)

instance (HasShapes a, HasShapes b, HasShapes c) => HasShapes (a, b, c) where
  type Shapes (a, b, c) = (Shapes a, Shapes b, Shapes c)
  shapesR = TupRunit `TupRpair` shapesR @a `TupRpair` shapesR @b `TupRpair` shapesR @c
  fromShapes (a, b, c) = ((((), fromShapes @a a), fromShapes @b b), fromShapes @c c)

instance (HasShapes a, HasShapes b, HasShapes c, HasShapes d) => HasShapes (a, b, c, d) where
  type Shapes (a, b, c, d) = (Shapes a, Shapes b, Shapes c, Shapes d)
  shapesR = TupRunit `TupRpair` shapesR @a `TupRpair` shapesR @b `TupRpair` shapesR @c `TupRpair` shapesR @d
  fromShapes (a, b, c, d) = (((((), fromShapes @a a), fromShapes @b b), fromShapes @c c), fromShapes @d d)

-- TODO: more instances I suppose, or generics

