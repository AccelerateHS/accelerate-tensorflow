{-# LANGUAGE GADTs #-}
-- |
-- Module      : Data.Array.Accelerate.TensorFlow.Lite.CodeGen.Tensor
-- Copyright   : [2021] The Accelerate Team
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <trevor.mcdonell@gmail.com>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.TensorFlow.Lite.CodeGen.Tensor
  where

import Data.Array.Accelerate.Array.Data
import Data.Array.Accelerate.Type

import qualified TensorFlow.Core                                    as TF


type TensorShape sh    = TF.Tensor TF.Build Int64
type TensorArrayData e = GArrayDataR (TF.Tensor TF.Build) e

data Tensor sh e where
  Tensor :: TensorShape sh
         -> TensorArrayData e
         -> Tensor sh e

