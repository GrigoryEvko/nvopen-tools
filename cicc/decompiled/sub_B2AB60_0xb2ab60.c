// Function: sub_B2AB60
// Address: 0xb2ab60
//
const char *__fastcall sub_B2AB60(int a1)
{
  const char *result; // rax

  switch ( a1 )
  {
    case 1:
      result = "__gnat_eh_personality";
      break;
    case 2:
      result = "__gcc_personality_v0";
      break;
    case 3:
      result = "__gcc_personality_sj0";
      break;
    case 4:
      result = "__gxx_personality_v0";
      break;
    case 5:
      result = "__gxx_personality_sj0";
      break;
    case 6:
      result = "__objc_personality_v0";
      break;
    case 7:
      result = "_except_handler3";
      break;
    case 8:
      result = "__C_specific_handler";
      break;
    case 9:
      result = "__CxxFrameHandler3";
      break;
    case 10:
      result = "ProcessCLRException";
      break;
    case 11:
      result = "rust_eh_personality";
      break;
    case 12:
      result = "__gxx_wasm_personality_v0";
      break;
    case 13:
      result = "__xlcxx_personality_v1";
      break;
    case 14:
      result = "__zos_cxx_personality_v2";
      break;
    default:
      BUG();
  }
  return result;
}
