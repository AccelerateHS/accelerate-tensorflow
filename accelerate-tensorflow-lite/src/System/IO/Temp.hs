-- |
-- Module      : System.IO.Temp
-- Copyright   : [2021] The Accelerate Team
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <trevor.mcdonell@gmail.com>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module System.IO.Temp
  where

import Control.Monad
import Control.Exception
import Text.Printf
import System.Directory
import System.FilePath
import System.IO
import System.IO.Error
import System.Random
import Data.Bits


withBinaryTempFile :: FilePath -> String -> (FilePath -> Handle -> IO a) -> IO a
withBinaryTempFile tmp_dir name k =
  let finalise f h = do
        open <- hIsOpen h
        when open $ hClose h
        removeFile f
  in
  bracket (openBinaryTempFile tmp_dir name) (uncurry finalise) (uncurry k)

-- | Create a temporary directory.
--
createTempDirectory :: FilePath -> String -> IO FilePath
createTempDirectory dir template =
  let
      wordSize = finiteBitSize (undefined :: Word)

      findTempName = do
        x <- randomIO
        let path = dir </> template ++ printf "-%.*x" (wordSize `div` 4) (x :: Word)
        r <- try $ createDirectory path
        case r of
          Right _ -> return path
          Left  e | isAlreadyExistsError e -> findTempName
                  | otherwise              -> ioError e
  in
  findTempName

