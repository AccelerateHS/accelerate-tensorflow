{-# LANGUAGE GADTs               #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications    #-}
-- |
-- Module      : Data.Array.Accelerate.TensorFlow.Lite.Sugar.Args
-- Copyright   : [2021..2022] The Accelerate Team
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <trevor.mcdonell@gmail.com>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.TensorFlow.Lite.Sugar.Args
  where

import Data.Array.Accelerate.Sugar.Array
import Data.Array.Accelerate.Trafo.Sharing
import qualified Data.Array.Accelerate.Representation.Array                   as R

import Data.Array.Accelerate.TensorFlow.Lite.Sugar.Shapes
import qualified Data.Array.Accelerate.TensorFlow.Lite.Representation.Args    as R


infixr 0 :->

data Args f where
  (:->)  :: Arrays a => a -> Args b -> Args (a -> b)
  Result :: HasShapes a => Shapes a -> Args a

fromArgs :: AfunctionRepr f (AfunctionR f) (ArraysFunctionR f)
         -> Args (AfunctionR f)
         -> R.Args (ArraysFunctionR f)
fromArgs = go
  where
    go :: forall g.
          AfunctionRepr g (AfunctionR g) (ArraysFunctionR g)
       -> Args (AfunctionR g)
       -> R.Args (ArraysFunctionR g)
    go (AfunctionReprLam lamR) (x :-> xs)  = R.Aparam (arraysR' x) (fromArr x) (go lamR xs)
    go AfunctionReprBody       (Result sh) = R.Aresult (shapesR @(AfunctionR g)) (fromShapes @(AfunctionR g) sh)
    go _ _ = error "impossible"

    arraysR' :: forall a. Arrays a => a -> R.ArraysR (ArraysR a)
    arraysR' _ = arraysR @a

