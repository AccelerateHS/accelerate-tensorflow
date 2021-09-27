{-# LANGUAGE GADTs #-}
{-# OPTIONS_HADDOCK hide #-}
-- |
-- Module      : Data.Array.Accelerate.TensorFlow.CodeGen.AST
-- Copyright   : [2021] The Accelerate Team
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <trevor.mcdonell@gmail.com>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.TensorFlow.CodeGen.AST
  where

import Data.Array.Accelerate.TensorFlow.CodeGen.Tensor

import Data.Array.Accelerate.AST
import Data.Array.Accelerate.Representation.Array


type Tfun = OpenTfun ()

data OpenTfun aenv a where
  Tbody :: ArraysR a -> Tensors a                         -> OpenTfun aenv a
  Tlam  :: ALeftHandSide a aenv aenv' -> OpenTfun aenv' b -> OpenTfun aenv (a -> b)

