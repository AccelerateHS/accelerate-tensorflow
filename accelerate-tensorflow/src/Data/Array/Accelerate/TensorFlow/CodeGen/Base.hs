{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications    #-}
{-# OPTIONS_HADDOCK hide #-}
-- |
-- Module      : Data.Array.Accelerate.TensorFlow.CodeGen.Base
-- Copyright   : [2021] The Accelerate Team
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <trevor.mcdonell@gmail.com>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.TensorFlow.CodeGen.Base
  where

import Data.Proxy
import Data.Typeable
import Text.Printf


infixr 0 $$
($$) :: (b -> a) -> (c -> d -> b) -> c -> d -> a
(f $$ g) x y = f (g x y)

unsupported :: String -> t a
unsupported thing = error (printf "Not supported: %s" thing)

excluded :: forall a t. Typeable a => t a
excluded = error (printf "Excluded type case: %s" (showsTypeRep (typeRep (Proxy @a)) ""))

