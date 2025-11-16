// Function: sub_E0BA50
// Address: 0xe0ba50
//
const char *__fastcall sub_E0BA50(int a1)
{
  const char *result; // rax

  switch ( a1 )
  {
    case 1:
      result = "DW_CC_normal";
      break;
    case 2:
      result = "DW_CC_program";
      break;
    case 3:
      result = "DW_CC_nocall";
      break;
    case 4:
      result = "DW_CC_pass_by_reference";
      break;
    case 5:
      result = "DW_CC_pass_by_value";
      break;
    case 64:
      result = "DW_CC_GNU_renesas_sh";
      break;
    case 65:
      result = "DW_CC_GNU_borland_fastcall_i386";
      break;
    case 176:
      result = "DW_CC_BORLAND_safecall";
      break;
    case 177:
      result = "DW_CC_BORLAND_stdcall";
      break;
    case 178:
      result = "DW_CC_BORLAND_pascal";
      break;
    case 179:
      result = "DW_CC_BORLAND_msfastcall";
      break;
    case 180:
      result = "DW_CC_BORLAND_msreturn";
      break;
    case 181:
      result = "DW_CC_BORLAND_thiscall";
      break;
    case 182:
      result = "DW_CC_BORLAND_fastcall";
      break;
    case 192:
      result = "DW_CC_LLVM_vectorcall";
      break;
    case 193:
      result = "DW_CC_LLVM_Win64";
      break;
    case 194:
      result = "DW_CC_LLVM_X86_64SysV";
      break;
    case 195:
      result = "DW_CC_LLVM_AAPCS";
      break;
    case 196:
      result = "DW_CC_LLVM_AAPCS_VFP";
      break;
    case 197:
      result = "DW_CC_LLVM_IntelOclBicc";
      break;
    case 198:
      result = "DW_CC_LLVM_SpirFunction";
      break;
    case 199:
      result = "DW_CC_LLVM_OpenCLKernel";
      break;
    case 200:
      result = "DW_CC_LLVM_Swift";
      break;
    case 201:
      result = "DW_CC_LLVM_PreserveMost";
      break;
    case 202:
      result = "DW_CC_LLVM_PreserveAll";
      break;
    case 203:
      result = "DW_CC_LLVM_X86RegCall";
      break;
    case 204:
      result = "DW_CC_LLVM_M68kRTD";
      break;
    case 205:
      result = "DW_CC_LLVM_PreserveNone";
      break;
    case 206:
      result = "DW_CC_LLVM_RISCVVectorCall";
      break;
    case 207:
      result = "DW_CC_LLVM_SwiftTail";
      break;
    case 208:
      result = "DW_CC_LLVM_RISCVVLSCall";
      break;
    case 255:
      result = "DW_CC_GDB_IBM_OpenCL";
      break;
    default:
      result = 0;
      break;
  }
  return result;
}
