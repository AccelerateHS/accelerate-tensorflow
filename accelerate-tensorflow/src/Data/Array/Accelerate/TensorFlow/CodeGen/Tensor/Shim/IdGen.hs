-- Necessary to ensure that the global IORef isn't messed with
{-# OPTIONS -fno-full-laziness -fno-cse #-}

-- |
-- Module      : Data.Array.Accelerate.TensorFlow.CodeGen.Tensor.Shim
-- Copyright   : [2023] The Accelerate Team
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <trevor.mcdonell@gmail.com>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.TensorFlow.CodeGen.Tensor.Shim.IdGen (
  genId,
) where

import Data.IORef
import System.IO.Unsafe


{-# NOINLINE counter #-}
counter :: IORef Int
counter = unsafePerformIO $ newIORef 0

{-# NOINLINE genId #-}
-- | The argument is unused except for evaluation to WHNF, but allows
-- generating more than 1 ID in the presence of common subexpression
-- elimination.
genId :: a -> Int
genId x = unsafePerformIO $ x `seq` atomicModifyIORef counter (\i -> (i + 1, i))
