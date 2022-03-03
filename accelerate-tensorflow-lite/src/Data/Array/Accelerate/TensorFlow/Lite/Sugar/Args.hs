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

import Data.Array.Accelerate.Trafo.Sharing
import Data.Array.Accelerate.Sugar.Array
import Data.Array.Accelerate.TensorFlow.Lite.Sugar.Shapes

import qualified Data.Array.Accelerate.TensorFlow.Lite.Representation.Args    as R


data Args f where
  (:->)  :: Arrays a => a -> Args b -> Args (a -> b)
  Result :: Arrays a => Shapes a    -> Args a


fromArgs :: forall f.
            AfunctionRepr f (AfunctionR f) (ArraysFunctionR f)
         -> Args f
         -> R.Args (ArraysFunctionR f)
fromArgs = undefined
-- fromArgs AfunctionReprBody (Result sh) = R.Aresult undefined sh

