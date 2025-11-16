// Function: sub_3140D50
// Address: 0x3140d50
//
__int64 __fastcall sub_3140D50(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int8 v3; // al
  __int64 v5; // rsi

  if ( a3 == a2 )
    return 1;
  if ( a3 )
  {
    while ( a2 )
    {
      if ( a3 == a2 )
        return 1;
      v3 = *(_BYTE *)(a2 - 16);
      if ( (v3 & 2) != 0 )
      {
        if ( *(_DWORD *)(a2 - 24) != 2 )
          return 0;
        v5 = *(_QWORD *)(a2 - 32);
      }
      else
      {
        if ( ((*(_WORD *)(a2 - 16) >> 6) & 0xF) != 2 )
          return 0;
        v5 = a2 - 16 - 8LL * ((v3 >> 2) & 0xF);
      }
      a2 = *(_QWORD *)(v5 + 8);
    }
  }
  return 0;
}
