// Function: sub_B10DA0
// Address: 0xb10da0
//
__int64 __fastcall sub_B10DA0(__int64 *a1)
{
  __int64 i; // rax
  unsigned __int8 v2; // cl
  __int64 v3; // rsi
  __int64 v5; // rsi
  __int64 v6; // rsi

  for ( i = *a1; ; i = v6 )
  {
    v2 = *(_BYTE *)(i - 16);
    if ( (v2 & 2) != 0 )
    {
      v3 = *(_QWORD *)(i - 32);
      if ( *(_DWORD *)(i - 24) != 2 )
        return **(_QWORD **)(i - 32);
    }
    else
    {
      v5 = i - 16;
      if ( ((*(_WORD *)(i - 16) >> 6) & 0xF) != 2 )
        return *(_QWORD *)(v5 - 8LL * ((v2 >> 2) & 0xF));
      v3 = -8LL * ((v2 >> 2) & 0xF) + v5;
    }
    v6 = *(_QWORD *)(v3 + 8);
    if ( !v6 )
      break;
  }
  v5 = i - 16;
  if ( (*(_BYTE *)(i - 16) & 2) != 0 )
    return **(_QWORD **)(i - 32);
  return *(_QWORD *)(v5 - 8LL * ((v2 >> 2) & 0xF));
}
