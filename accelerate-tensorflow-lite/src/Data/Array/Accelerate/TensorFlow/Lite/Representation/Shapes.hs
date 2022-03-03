{-# LANGUAGE TypeFamilies #-}
-- |
-- Module      : Data.Array.Accelerate.TensorFlow.Lite.Representation.Args
-- Copyright   : [2021..2022] The Accelerate Team
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <trevor.mcdonell@gmail.com>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.TensorFlow.Lite.Representation.Shapes
  where

import Data.Array.Accelerate.Representation.Array
import Data.Array.Accelerate.Representation.Shape
import Data.Array.Accelerate.Representation.Type


type ShapesR a = TupR ShapeR (Shapes a)

type family Shapes t where
  Shapes ()           = ()
  Shapes (Array sh e) = sh
  Shapes (a, b)       = (Shapes a, Shapes b)

