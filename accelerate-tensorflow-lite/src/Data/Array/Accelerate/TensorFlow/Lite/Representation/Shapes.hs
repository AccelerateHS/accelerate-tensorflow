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
import Data.Array.Accelerate.Representation.Type


type ShapesR = TupR ShapeR

type family Shapes t where
  Shapes ()           = ()
  Shapes (Array sh e) = sh
  Shapes (a, b)       = (Shapes a, Shapes b)

-- shapes :: ArraysR a -> a -> ShapesR a
-- shapes TupRunit              ()           = ()
-- shapes (TupRpair aR bR)      (a, b)       = (shapes aR a, shapes bR b)
-- shapes (TupRsingle ArrayR{}) (Array sh _) = sh

