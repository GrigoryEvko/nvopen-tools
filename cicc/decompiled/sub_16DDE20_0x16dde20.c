// Function: sub_16DDE20
// Address: 0x16dde20
//
char *__fastcall sub_16DDE20(int a1)
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
      result = "elf";
      break;
    case 3:
      result = "macho";
      break;
    case 4:
      result = "wasm";
      break;
  }
  return result;
}
