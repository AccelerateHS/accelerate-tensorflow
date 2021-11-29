{-# LANGUAGE GADTs #-}
{-# OPTIONS_HADDOCK hide #-}
-- |
-- Module      : Data.Array.Accelerate.TensorFlow.CodeGen.Foreign
-- Copyright   : [2021] The Accelerate Team
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <trevor.mcdonell@gmail.com>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.TensorFlow.CodeGen.Foreign
  where

import Data.Array.Accelerate.TensorFlow.CodeGen.Tensor

import Data.Array.Accelerate.Sugar.Foreign                          as A

data ForeignAcc f where
  ForeignAcc :: String
             -> (Tensors a -> Tensors b)
             -> ForeignAcc (a -> b)

instance Foreign ForeignAcc where
  strForeign (ForeignAcc str _) = str

