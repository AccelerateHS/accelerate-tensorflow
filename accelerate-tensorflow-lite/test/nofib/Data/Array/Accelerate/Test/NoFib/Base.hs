{-# LANGUAGE BangPatterns      #-}
{-# LANGUAGE DefaultSignatures #-}
{-# LANGUAGE TypeOperators     #-}
-- |
-- Module      : Data.Array.Accelerate.Test.NoFib.Base
-- Copyright   : [2022] The Accelerate Team
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <trevor.mcdonell@gmail.com>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.Test.NoFib.Base where

import Data.Array.Accelerate.Sugar.Array
import Data.Array.Accelerate.Sugar.Elt
import Data.Array.Accelerate.Sugar.Shape

import Hedgehog
import Hedgehog.Internal.Source                                     ( HasCallStack, withFrozenCallStack )
import qualified Hedgehog.Gen                                       as Gen
import qualified Hedgehog.Range                                     as Range

import Control.Monad
import Prelude                                                      hiding ( (!!) )


-- | Generate random values of a given type
--
dim0 :: Gen DIM0
dim0 = return Z

dim1 :: Gen DIM1
dim1 = (Z :.) <$> Gen.int (Range.linear 0 1024)

dim2 :: Gen DIM2
dim2 = do
  x <- Gen.int (Range.linear 0 128)
  y <- Gen.int (Range.linear 0 48)
  return (Z :. y :. x)

dim3 :: Gen DIM3
dim3 = do
  x <- Gen.int (Range.linear 0 64)
  y <- Gen.int (Range.linear 0 32)
  z <- Gen.int (Range.linear 0 16)
  return (Z :. z :. y :. x)

array :: (Shape sh, Elt e) => sh -> Gen e -> Gen (Array sh e)
array sh gen = fromList sh <$> Gen.list (Range.singleton (size sh)) gen

f32 :: Gen Float
f32 = Gen.float (Range.linearFracFrom 0 (-1) 1)

except :: Gen e -> (e -> Bool) -> Gen e
except gen f  = do
  v <- gen
  when (f v) Gen.discard
  return v


-- | Fails the test if the two arguments are not equal, allowing for a small
-- amount of floating point inaccuracy.
--
infix 4 ~~~
(~~~) :: (MonadTest m, Similar a, Show a, HasCallStack) => a -> a -> m ()
a ~~~ b = withFrozenCallStack $ Sim a === Sim b

newtype Sim a = Sim a

instance Similar a => Eq (Sim a) where
  Sim a == Sim b = a ~= b

instance Show a => Show (Sim a) where
  show (Sim a) = show a

-- | A class of things that support almost-equality, so that we can disregard
-- small amounts of floating-point round-off error.
--
class Similar a where
  {-# INLINE (~=) #-}
  (~=) :: a -> a -> Bool
  default (~=) :: Eq a => a -> a -> Bool
  (~=) = (==)

infix 4 ~=

instance Similar Z
instance Similar Int
instance (Eq sh, Eq sz) => Similar (sh:.sz)

instance (Shape sh, Eq sh, Elt e, Similar e) => Similar (Array sh e) where
  a1 ~= a2 = shape a1 == shape a2 && go 0
    where
      n = size (shape a1)
      go !i
        | i >= n              = True
        | a1 !! i ~= a2 !! i  = go (i+1)
        | otherwise           = False

instance Similar Float where
  (~=) = absRelTol 0.05 0.5 -- precision on the EdgeTPU is terrible

instance (Similar a, Similar b) => Similar (a, b) where
  (x1, x2) ~= (y1, y2) = x1 ~= y1 && x2 ~= y2

instance (Similar a, Similar b, Similar c) => Similar (a, b, c) where
  (x1, x2, x3) ~= (y1, y2, y3) = x1 ~= y1 && x2 ~= y2 && x3 ~= y3

instance (Similar a, Similar b, Similar c, Similar d) => Similar (a, b, c, d) where
  (x1, x2, x3, x4) ~= (y1, y2, y3, y4) = x1 ~= y1 && x2 ~= y2 && x3 ~= y3 && x4 ~= y4

{-# INLINEABLE absRelTol #-}
absRelTol :: RealFloat a => a -> a -> a -> a -> Bool
absRelTol epsilonAbs epsilonRel u v
  |  isInfinite u
  && isInfinite v          = True
  |  isNaN u
  && isNaN v               = True
  | abs (u-v) < epsilonAbs = True
  | abs u > abs v          = abs ((u-v) / u) < epsilonRel
  | otherwise              = abs ((v-u) / v) < epsilonRel

