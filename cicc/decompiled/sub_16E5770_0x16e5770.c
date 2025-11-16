// Function: sub_16E5770
// Address: 0x16e5770
//
const char *__fastcall sub_16E5770(__int64 a1, __int64 a2, __int64 a3, _BYTE *a4)
{
  if ( a2 != 4 )
  {
    if ( a2 == 5 && *(_DWORD *)a1 == 1936482662 && *(_BYTE *)(a1 + 4) == 101 )
    {
      *a4 = 0;
      return 0;
    }
    return "invalid boolean";
  }
  if ( *(_DWORD *)a1 != 1702195828 )
    return "invalid boolean";
  *a4 = 1;
  return 0;
}
