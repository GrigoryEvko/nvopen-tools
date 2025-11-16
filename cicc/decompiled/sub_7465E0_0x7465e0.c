// Function: sub_7465E0
// Address: 0x7465e0
//
char *__fastcall sub_7465E0(char a1, int a2)
{
  char *result; // rax

  switch ( a1 )
  {
    case 0:
      result = "_Float16";
      break;
    case 1:
      result = "__fp16";
      break;
    case 2:
      result = "float";
      break;
    case 3:
      result = "_Float32x";
      break;
    case 4:
      result = "double";
      break;
    case 5:
      result = "_Float64x";
      break;
    case 6:
      result = "long double";
      break;
    case 7:
      result = "__float80";
      break;
    case 8:
      result = "__float128";
      break;
    case 9:
      result = "__bf16";
      if ( !a2 )
      {
        if ( unk_4D04190 )
          result = "std::bfloat16_t";
      }
      break;
    case 10:
      result = "_Float16";
      if ( !a2 )
        result = "std::float16_t";
      break;
    case 11:
      result = "_Float32";
      if ( !a2 )
        result = "std::float32_t";
      break;
    case 12:
      result = "_Float64";
      if ( !a2 )
        result = "std::float64_t";
      break;
    case 13:
      result = "_Float128";
      if ( !a2 )
        result = "std::float128_t";
      break;
    default:
      result = "**BAD-FLOAT-KIND**";
      break;
  }
  return result;
}
