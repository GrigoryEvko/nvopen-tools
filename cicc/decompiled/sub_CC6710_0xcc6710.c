// Function: sub_CC6710
// Address: 0xcc6710
//
char *__fastcall sub_CC6710(int a1)
{
  char *result; // rax

  switch ( a1 )
  {
    case 0:
      result = (char *)byte_3F871B3;
      break;
    case 1:
      result = "coff";
      break;
    case 2:
      result = "dxcontainer";
      break;
    case 3:
      result = "elf";
      break;
    case 4:
      result = "goff";
      break;
    case 5:
      result = "macho";
      break;
    case 6:
      result = "spirv";
      break;
    case 7:
      result = "wasm";
      break;
    case 8:
      result = "xcoff";
      break;
    default:
      BUG();
  }
  return result;
}
