// Function: sub_23DBFF0
// Address: 0x23dbff0
//
char *__fastcall sub_23DBFF0(__int64 a1)
{
  char *result; // rax

  switch ( *(_DWORD *)(a1 + 100) )
  {
    case 1:
      result = ".ASAN$GL";
      break;
    case 2:
    case 4:
    case 6:
    case 7:
    case 8:
      sub_C64ED0("ModuleAddressSanitizer not implemented for object file format", 1u);
    case 3:
      result = "asan_globals";
      break;
    case 5:
      result = "__DATA,__asan_globals,regular";
      break;
    default:
      BUG();
  }
  return result;
}
