// Function: sub_14E77F0
// Address: 0x14e77f0
//
const char *__fastcall sub_14E77F0(unsigned int a1)
{
  const char *result; // rax

  if ( a1 > 0x36 )
  {
    switch ( a1 )
    {
      case 0x8E57u:
        return "DW_LANG_GOOGLE_RenderScript";
      case 0xB000u:
        return "DW_LANG_BORLAND_Delphi";
      case 0x8001u:
        return "DW_LANG_Mips_Assembler";
    }
    return 0;
  }
  if ( !a1 )
    return 0;
  switch ( a1 )
  {
    case 1u:
      result = "DW_LANG_C89";
      break;
    case 2u:
      result = "DW_LANG_C";
      break;
    case 3u:
      result = "DW_LANG_Ada83";
      break;
    case 4u:
      result = "DW_LANG_C_plus_plus";
      break;
    case 5u:
      result = "DW_LANG_Cobol74";
      break;
    case 6u:
      result = "DW_LANG_Cobol85";
      break;
    case 7u:
      result = "DW_LANG_Fortran77";
      break;
    case 8u:
      result = "DW_LANG_Fortran90";
      break;
    case 9u:
      result = "DW_LANG_Pascal83";
      break;
    case 0xAu:
      result = "DW_LANG_Modula2";
      break;
    case 0xBu:
      result = "DW_LANG_Java";
      break;
    case 0xCu:
      result = "DW_LANG_C99";
      break;
    case 0xDu:
      result = "DW_LANG_Ada95";
      break;
    case 0xEu:
      result = "DW_LANG_Fortran95";
      break;
    case 0xFu:
      result = "DW_LANG_PLI";
      break;
    case 0x10u:
      result = "DW_LANG_ObjC";
      break;
    case 0x11u:
      result = "DW_LANG_ObjC_plus_plus";
      break;
    case 0x12u:
      result = "DW_LANG_UPC";
      break;
    case 0x13u:
      result = "DW_LANG_D";
      break;
    case 0x14u:
      result = "DW_LANG_Python";
      break;
    case 0x15u:
      result = "DW_LANG_OpenCL";
      break;
    case 0x16u:
      result = "DW_LANG_Go";
      break;
    case 0x17u:
      result = "DW_LANG_Modula3";
      break;
    case 0x18u:
      result = "DW_LANG_Haskell";
      break;
    case 0x19u:
      result = "DW_LANG_C_plus_plus_03";
      break;
    case 0x1Au:
      result = "DW_LANG_C_plus_plus_11";
      break;
    case 0x1Bu:
      result = "DW_LANG_OCaml";
      break;
    case 0x1Cu:
      result = "DW_LANG_Rust";
      break;
    case 0x1Du:
      result = "DW_LANG_C11";
      break;
    case 0x1Eu:
      result = "DW_LANG_Swift";
      break;
    case 0x1Fu:
      result = "DW_LANG_Julia";
      break;
    case 0x20u:
      result = "DW_LANG_Dylan";
      break;
    case 0x21u:
      result = "DW_LANG_C_plus_plus_14";
      break;
    case 0x22u:
      result = "DW_LANG_Fortran03";
      break;
    case 0x23u:
      result = "DW_LANG_Fortran08";
      break;
    case 0x24u:
      result = "DW_LANG_RenderScript";
      break;
    case 0x25u:
      result = "DW_LANG_BLISS";
      break;
    case 0x34u:
      result = "DW_LANG_GLSL";
      break;
    case 0x35u:
      result = "DW_LANG_GLSL_ES";
      break;
    case 0x36u:
      result = "DW_LANG_HLSL";
      break;
    default:
      return 0;
  }
  return result;
}
