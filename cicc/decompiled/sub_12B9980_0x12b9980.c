// Function: sub_12B9980
// Address: 0x12b9980
//
const char *__fastcall sub_12B9980(int a1)
{
  const char *result; // rax

  switch ( a1 )
  {
    case 0:
      result = "NVVM_SUCCESS";
      break;
    case 1:
      result = "NVVM_ERROR_OUT_OF_MEMORY";
      break;
    case 2:
      result = "NVVM_ERROR_PROGRAM_CREATION_FAILURE";
      break;
    case 3:
      result = "NVVM_ERROR_IR_VERSION_MISMATCH";
      break;
    case 4:
      result = "NVVM_ERROR_INVALID_INPUT";
      break;
    case 5:
      result = "NVVM_ERROR_INVALID_PROGRAM";
      break;
    case 6:
      result = "NVVM_ERROR_INVALID_IR";
      break;
    case 7:
      result = "NVVM_ERROR_INVALID_OPTION";
      break;
    case 8:
      result = "NVVM_ERROR_NO_MODULE_IN_PROGRAM";
      break;
    case 9:
      result = "NVVM_ERROR_COMPILATION";
      break;
    default:
      result = "NVVM_ERROR not found";
      break;
  }
  return result;
}
