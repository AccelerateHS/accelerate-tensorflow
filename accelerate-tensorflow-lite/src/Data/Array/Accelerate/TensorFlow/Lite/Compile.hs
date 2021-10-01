{-# LANGUAGE GADTs             #-}
{-# LANGUAGE OverloadedStrings #-}
-- |
-- Module      : Data.Array.Accelerate.TensorFlow.Lite.Compile
-- Copyright   : [2021] The Accelerate Team
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <trevor.mcdonell@gmail.com>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.TensorFlow.Lite.Compile (

  compileTfun,

) where

import Data.Array.Accelerate.Type
import Data.Array.Accelerate.Representation.Array
import Data.Array.Accelerate.Representation.Type
import qualified Data.Array.Accelerate.Debug.Internal               as Debug

import Data.Array.Accelerate.TensorFlow.CodeGen.AST
import Data.Array.Accelerate.TensorFlow.CodeGen.Base
import Data.Array.Accelerate.TensorFlow.CodeGen.Tensor

import qualified Proto.Tensorflow.Core.Framework.Graph              as TF
import qualified Proto.Tensorflow.Core.Framework.Graph_Fields       as TF
import qualified Proto.Tensorflow.Core.Framework.NodeDef_Fields     as TF
import qualified TensorFlow.Build                                   as TF
import qualified TensorFlow.Core                                    as TF

import Control.DeepSeq
import Control.Exception
import Data.Functor.Identity
import Data.ProtoLens
import Formatting
import Lens.Family2
import System.Directory
import System.Exit
import System.FilePath
import System.IO
import System.Process
import System.Process.Extra
import Text.Printf
import qualified Data.ByteString                                    as B
import qualified Data.Text                                          as T



compileTfun :: Tfun f -> IO FilePath
compileTfun = compileOpenTfun

compileOpenTfun :: OpenTfun aenv f -> IO FilePath
compileOpenTfun (Tlam _ f)   = compileOpenTfun f
compileOpenTfun (Tbody bR b) = do
  tflite <- tflite_model (graph_of_model bR b)
  edge   <- edgetpu_compile tflite
  return edge

edgetpu_compile :: FilePath -> IO FilePath
edgetpu_compile path = do
  let
      cp    = (proc "edgetpu_compiler" flags) { std_in = NoStream, std_out = CreatePipe, std_err = CreatePipe }
      flags = [ "--show_operations"
              , "--out_dir=" ++ dropFileName path
              , path
              ]
      edgetpu_file = dropExtension path ++ "_edgetpu" <.> "tflite"

  -- Invoke 'edgetpu_compile' to convert the tflite file into something
  -- suitable for the edge tpu
  withCreateProcess cp $ \Nothing (Just outh) (Just errh) ph -> do

    -- fork off threads to start consuming stdout and stderr
    out <- hGetContents outh
    err <- hGetContents errh
    withForkWait (evaluate (rnf out)) $ \waitOut -> do
      withForkWait (evaluate (rnf err)) $ \waitErr -> do
        waitOut
        hClose outh

        waitErr
        hClose errh

    -- wait on the process
    ex <- waitForProcess ph
    case ex of
      ExitFailure r -> error $ printf "edgetpu_compiler %s (exit %d)\n%s" (unwords flags) r err
      ExitSuccess   -> return ()

    Debug.traceM Debug.dump_cc ("cc: edgetpu_compiler\n" % reindented 2 string) out

  return edgetpu_file


tflite_model :: TF.GraphDef -> IO FilePath
tflite_model graph = do
  tmp_dir         <- getTemporaryDirectory
  (pb_file, pb_h) <- openBinaryTempFile tmp_dir "model.pb"      -- TODO: be exception safe
  (tf_file, tf_h) <- openBinaryTempFile tmp_dir "model.tflite"  -- TODO: be exception safe
  --
  B.hPut pb_h (encodeMessage graph)
  hClose pb_h
  hClose tf_h
  --
  let names   = map (view TF.name) (graph ^. TF.node)
      inputs  = filter (T.isPrefixOf "input") names
      outputs = filter (T.isPrefixOf "output") names
      --
      cp      = (proc "python3" flags) { std_in = NoStream, std_out = NoStream, std_err = CreatePipe }
      flags   = [ "converter.py"
                , "--graph_def_file=" ++ pb_file
                , "--output_file=" ++ tf_file
                , "--input_arrays=" ++ T.unpack (T.intercalate "," inputs)
                , "--output_arrays=" ++ T.unpack (T.intercalate "," outputs)
                ]

  -- Invoke 'tflite_convert' to convert the protobuf file to the tflite representation
  withCreateProcess cp $ \Nothing Nothing (Just errh) ph -> do

    -- fork off threads to start consuming stdout and stderr
    err <- hGetContents errh
    withForkWait (evaluate (rnf err)) $ \waitErr -> do
      waitErr
      hClose errh

    -- wait on the process
    ex <- waitForProcess ph
    case ex of
      ExitFailure r -> error $ printf "python3 %s (exit %d)\n%s" (unwords flags) r err
      ExitSuccess   -> return ()

  removeFile pb_file
  return tf_file

graph_of_model :: ArraysR arrs -> Tensors arrs -> TF.GraphDef
graph_of_model arrR model =
  let
      go :: TF.MonadBuild m => ArraysR a -> Tensors a -> m ()
      go TupRunit              ()                         = return ()
      go (TupRpair aR bR)      (a, b)                     = go aR a >> go bR b
      go (TupRsingle ArrayR{}) (Tensor (ArrayR _ aR) _ a) = array aR a

      array :: TF.MonadBuild m => TypeR t -> TensorArrayData t -> m ()
      array TupRunit         ()     = return ()
      array (TupRpair aR bR) (a, b) = array aR a >> array bR b
      array (TupRsingle aR)  a      = scalar aR a

      scalar :: TF.MonadBuild m => ScalarType t -> TensorArrayData t -> m ()
      scalar (SingleScalarType t) = single t
      scalar (VectorScalarType _) = unsupported "SIMD-vector types"

      single :: TF.MonadBuild m => SingleType t -> TensorArrayData t -> m ()
      single (NumSingleType t) = num t

      num :: TF.MonadBuild m => NumType t -> TensorArrayData t -> m ()
      num (IntegralNumType t) = integral t
      num (FloatingNumType t) = floating t

      integral :: TF.MonadBuild m => IntegralType t -> TensorArrayData t -> m ()
      integral TypeInt8   = render
      integral TypeInt16  = render
      integral TypeInt32  = render
      integral TypeInt64  = render
      integral TypeWord8  = render
      integral TypeWord16 = render
      integral TypeWord32 = render
      integral TypeWord64 = render
      integral TypeInt    = render
      integral TypeWord   = render

      floating :: TF.MonadBuild m => FloatingType t -> TensorArrayData t -> m ()
      floating TypeFloat  = render
      floating TypeDouble = render
      floating TypeHalf   = unsupported "half-precision floating point"

      render :: TF.MonadBuild m => TF.Tensor TF.Build a -> m ()
      render t = TF.render t >> return ()

      nodes = runIdentity $ TF.evalBuildT (go arrR model >> TF.flushNodeBuffer)
      graph = defMessage & TF.node .~ nodes :: TF.GraphDef
  in
  graph

