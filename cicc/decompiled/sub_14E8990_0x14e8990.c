// Function: sub_14E8990
// Address: 0x14e8990
//
const char *__fastcall sub_14E8990(unsigned int a1)
{
  const char *result; // rax

  if ( a1 > 0xCB )
  {
    result = "DW_CC_GDB_IBM_OpenCL";
    if ( a1 == 255 )
      return result;
    return 0;
  }
  if ( !a1 )
    return 0;
  switch ( a1 )
  {
    case 1u:
      result = "DW_CC_normal";
      break;
    case 2u:
      result = "DW_CC_program";
      break;
    case 3u:
      result = "DW_CC_nocall";
      break;
    case 4u:
      result = "DW_CC_pass_by_reference";
      break;
    case 5u:
      result = "DW_CC_pass_by_value";
      break;
    case 0x40u:
      result = "DW_CC_GNU_renesas_sh";
      break;
    case 0x41u:
      result = "DW_CC_GNU_borland_fastcall_i386";
      break;
    case 0xB0u:
      result = "DW_CC_BORLAND_safecall";
      break;
    case 0xB1u:
      result = "DW_CC_BORLAND_stdcall";
      break;
    case 0xB2u:
      result = "DW_CC_BORLAND_pascal";
      break;
    case 0xB3u:
      result = "DW_CC_BORLAND_msfastcall";
      break;
    case 0xB4u:
      result = "DW_CC_BORLAND_msreturn";
      break;
    case 0xB5u:
      result = "DW_CC_BORLAND_thiscall";
      break;
    case 0xB6u:
      result = "DW_CC_BORLAND_fastcall";
      break;
    case 0xC0u:
      result = "DW_CC_LLVM_vectorcall";
      break;
    case 0xC1u:
      result = "DW_CC_LLVM_Win64";
      break;
    case 0xC2u:
      result = "DW_CC_LLVM_X86_64SysV";
      break;
    case 0xC3u:
      result = "DW_CC_LLVM_AAPCS";
      break;
    case 0xC4u:
      result = "DW_CC_LLVM_AAPCS_VFP";
      break;
    case 0xC5u:
      result = "DW_CC_LLVM_IntelOclBicc";
      break;
    case 0xC6u:
      result = "DW_CC_LLVM_SpirFunction";
      break;
    case 0xC7u:
      result = "DW_CC_LLVM_OpenCLKernel";
      break;
    case 0xC8u:
      result = "DW_CC_LLVM_Swift";
      break;
    case 0xC9u:
      result = "DW_CC_LLVM_PreserveMost";
      break;
    case 0xCAu:
      result = "DW_CC_LLVM_PreserveAll";
      break;
    case 0xCBu:
      result = "DW_CC_LLVM_X86RegCall";
      break;
    default:
      return 0;
  }
  return result;
}
