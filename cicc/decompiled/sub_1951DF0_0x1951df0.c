// Function: sub_1951DF0
// Address: 0x1951df0
//
__int64 __fastcall sub_1951DF0(__int64 a1, int a2)
{
  char v2; // dl
  __int64 result; // rax

  if ( !a1 )
    return 0;
  v2 = *(_BYTE *)(a1 + 16);
  result = a1;
  if ( v2 == 9 )
    return result;
  if ( a2 != 1 )
  {
    if ( v2 == 13 )
      return result;
    return 0;
  }
  result = sub_1649C60(a1);
  if ( *(_BYTE *)(result + 16) != 4 )
    return 0;
  return result;
}
