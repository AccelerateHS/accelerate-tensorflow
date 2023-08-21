{-# LANGUAGE LambdaCase #-}
-- |
-- Module      : System.IO.Temp
-- Copyright   : [2021] The Accelerate Team
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <trevor.mcdonell@gmail.com>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module System.IO.Temp (
  withTemporaryDirectory,
) where

import Control.Exception
import System.Directory
import System.Environment
import System.FilePath
import System.Posix.Temp
import System.IO.Unsafe


tmpdir :: String
tmpdir = unsafePerformIO $ do
  lookupEnv "TMPDIR" >>= \case
    Nothing -> return "/tmp"
    Just dir -> return dir

withTemporaryDirectory :: String -> (FilePath -> IO a) -> IO a
withTemporaryDirectory prefix f = do
  lookupEnv "ACCELERATE_TFLITE_PRESERVE_TEMP" >>= \case
    Just s | not (null s) -> do
      dir <- mkdtemp (tmpdir </> prefix)
      f dir
    _ -> do
      bracket (mkdtemp (tmpdir </> prefix))
              removeDirectoryRecursive
              f
