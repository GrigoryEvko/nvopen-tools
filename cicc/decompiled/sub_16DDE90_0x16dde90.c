// Function: sub_16DDE90
// Address: 0x16dde90
//
__int64 __fastcall sub_16DDE90(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  if ( a2 == 3 )
  {
    if ( *(_WORD *)a1 == 28770 )
    {
      if ( *(_BYTE *)(a1 + 2) != 102 )
        return 0;
      return 7;
    }
    return 0;
  }
  if ( a2 == 6 )
  {
    if ( *(_DWORD *)a1 == 1600548962 && *(_WORD *)(a1 + 4) == 25954 )
      return 8;
    if ( *(_DWORD *)a1 == 1600548962 )
    {
      if ( *(_WORD *)(a1 + 4) != 25964 )
        return 0;
      return 7;
    }
    return 0;
  }
  result = 0;
  if ( a2 == 5 )
  {
    if ( *(_DWORD *)a1 == 1701212258 && *(_BYTE *)(a1 + 4) == 98 )
      return 8;
    if ( *(_DWORD *)a1 == 1701212258 && *(_BYTE *)(a1 + 4) == 108 )
      return 7;
    return 0;
  }
  return result;
}
