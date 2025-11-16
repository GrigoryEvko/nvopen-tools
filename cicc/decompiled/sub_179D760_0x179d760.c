// Function: sub_179D760
// Address: 0x179d760
//
__int64 __fastcall sub_179D760(__int64 a1, __int64 a2)
{
  int v2; // eax
  unsigned int v3; // r8d
  __int64 *v5; // rax
  __int64 v6; // rax
  __int64 v7; // rsi
  __int64 v8; // rax

  v2 = *(unsigned __int8 *)(a2 + 16);
  if ( (unsigned __int8)v2 <= 0x17u )
  {
    v3 = 0;
    if ( (_BYTE)v2 == 5 && (unsigned int)*(unsigned __int16 *)(a2 + 18) - 24 <= 1 )
    {
      v8 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
      if ( v8 )
      {
        **(_QWORD **)a1 = v8;
        LOBYTE(v3) = *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF))) == *(_QWORD *)(a1 + 8);
      }
    }
    return v3;
  }
  v3 = 0;
  if ( (unsigned int)(v2 - 48) > 1 )
    return v3;
  v5 = (*(_BYTE *)(a2 + 23) & 0x40) != 0
     ? *(__int64 **)(a2 - 8)
     : (__int64 *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v6 = *v5;
  v3 = 0;
  if ( !v6 )
    return v3;
  **(_QWORD **)a1 = v6;
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v7 = *(_QWORD *)(a2 - 8);
  else
    v7 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
  LOBYTE(v3) = *(_QWORD *)(a1 + 8) == *(_QWORD *)(v7 + 24);
  return v3;
}
