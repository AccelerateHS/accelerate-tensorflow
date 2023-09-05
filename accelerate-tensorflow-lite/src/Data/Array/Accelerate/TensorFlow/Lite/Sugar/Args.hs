{-# LANGUAGE GADTs                #-}
{-# LANGUAGE ScopedTypeVariables  #-}
{-# LANGUAGE StandaloneDeriving   #-}
{-# LANGUAGE TypeApplications     #-}
{-# LANGUAGE TypeFamilies         #-}
{-# LANGUAGE UndecidableInstances #-}
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

-- | Arguments to feed to a tensor computation. This is used to capture the
-- representative data for a computation, as well as indicate the sizes of
-- the input and output tensors, both of which are required for compilation
-- to the EdgeTPU hardware.
--
-- For example, given a tensor computation:
--
-- > f :: (Arrays a, Arrays b, Arrays c) => Acc a -> Acc b -> Acc c
-- > f = ...
--
-- The arguments that will be passed to that function can be collected into
-- this data type as:
--
-- > args :: (Arrays a, Arrays b, Arrays c) => a -> b -> Shapes c -> Args (a -> b -> c)
-- > args a b c = a :-> b :-> Result c
--
-- Note that @a@ and @b@ here are "real" arrays, not embedded Accelerate
-- ('Data.Array.Accelerate.Acc') arrays. For the output type @c@ we only
-- need the extents ('Shapes') of each of the output arrays. For example:
--
-- @
-- 'Shapes' ('Array' 'Data.Array.Accelerate.DIM1' Float) = 'Data.Array.Accelerate.DIM1'
-- 'Shapes' ('Array' 'Data.Array.Accelerate.DIM1' Int8, 'Array' 'Data.Array.Accelerate.DIM2' Float) = ('Data.Array.Accelerate.DIM1', 'Data.Array.Accelerate.DIM2')
-- @
--
data Args f where
  (:->)  :: Arrays a => a -> Args b -> Args (a -> b)
  Result :: HasShapes a => Shapes a -> Args a

type family AllShowFun f where
  AllShowFun (a -> b) = (Show a, AllShowFun b)
  AllShowFun b = ()

instance AllShowFun f => Show (Args f) where
  showsPrec d (x :-> as) = showParen (d > 0) $
    showsPrec 1 x . showString " :-> " . showsPrec 0 as
  showsPrec _ (Result _) = showString "<shapes>"

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

