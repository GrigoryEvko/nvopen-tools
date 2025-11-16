// Function: sub_1B29A30
// Address: 0x1b29a30
//
char __fastcall sub_1B29A30(__int64 a1, __int64 a2, __int64 a3)
{
  char result; // al

  if ( a2 && *(_BYTE *)(a2 + 16) == 17 )
  {
    result = 1;
    if ( a3 )
    {
      if ( *(_BYTE *)(a3 + 16) == 17 )
        return *(_DWORD *)(a2 + 32) < *(_DWORD *)(a3 + 32);
    }
  }
  else
  {
    if ( !a3 )
      return sub_1B298A0(a1, a2, a3);
    result = 0;
    if ( *(_BYTE *)(a3 + 16) != 17 )
      return sub_1B298A0(a1, a2, a3);
  }
  return result;
}
