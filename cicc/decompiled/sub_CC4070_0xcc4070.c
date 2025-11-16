// Function: sub_CC4070
// Address: 0xcc4070
//
__int64 __fastcall sub_CC4070(__int64 a1, unsigned __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rax
  _DWORD *v4; // rax
  unsigned __int64 v6; // rsi
  __int64 v7; // rsi

  if ( a2 <= 4 )
  {
    if ( a2 != 4 )
    {
      if ( a2 == 3 && *(_WORD *)a1 == 27749 && *(_BYTE *)(a1 + 2) == 102 )
        return 3;
      return 0;
    }
  }
  else
  {
    v2 = a1 + a2 - 5;
    if ( *(_DWORD *)v2 == 1718575992 && *(_BYTE *)(v2 + 4) == 102 )
      return 8;
  }
  if ( *(_DWORD *)(a1 + a2 - 4) == 1717989219 )
    return 1;
  v3 = a1 + a2 - 3;
  if ( *(_WORD *)v3 == 27749 && *(_BYTE *)(v3 + 2) == 102 )
    return 3;
  v4 = (_DWORD *)(a1 + a2 - 4);
  if ( *v4 == 1717989223 )
    return 4;
  if ( a2 == 4 )
  {
    if ( *(_DWORD *)a1 == 1836278135 )
      return 7;
    return 0;
  }
  v6 = a2 - 5;
  if ( *(_DWORD *)(a1 + v6) != 1751343469 || *(_BYTE *)(a1 + v6 + 4) != 111 )
  {
    if ( *v4 == 1836278135 )
      return 7;
    v7 = a1 + v6;
    if ( *(_DWORD *)v7 == 1919512691 && *(_BYTE *)(v7 + 4) == 118 )
      return 6;
    return 0;
  }
  return 5;
}
