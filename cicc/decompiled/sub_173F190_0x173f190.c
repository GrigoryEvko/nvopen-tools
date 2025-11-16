// Function: sub_173F190
// Address: 0x173f190
//
__int64 __fastcall sub_173F190(__int64 a1, __int64 a2)
{
  int v2; // eax
  __int64 *v4; // rdx
  __int64 v5; // rcx
  int v6; // eax
  __int64 *v7; // rsi
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rax
  int v11; // eax
  unsigned __int16 v12; // dx
  __int64 v13; // rdx
  __int64 v14; // rax
  int v15; // eax

  v2 = *(unsigned __int8 *)(a2 + 16);
  if ( (unsigned __int8)v2 > 0x17u )
  {
    if ( (_BYTE)v2 != 61 )
    {
LABEL_11:
      v6 = v2 - 24;
      goto LABEL_12;
    }
LABEL_5:
    if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
      v4 = *(__int64 **)(a2 - 8);
    else
      v4 = (__int64 *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
    v5 = *v4;
    if ( (unsigned __int8)(*(_BYTE *)(*v4 + 16) - 75) <= 1u )
    {
      v13 = *(_QWORD *)(v5 - 48);
      if ( v13 )
      {
        **(_QWORD **)(a1 + 8) = v13;
        v14 = *(_QWORD *)(v5 - 24);
        if ( v14 )
        {
          **(_QWORD **)(a1 + 16) = v14;
          v15 = *(unsigned __int16 *)(v5 + 18);
          BYTE1(v15) &= ~0x80u;
          **(_DWORD **)a1 = v15;
          return 1;
        }
        v2 = *(unsigned __int8 *)(a2 + 16);
      }
    }
    if ( (unsigned __int8)v2 <= 0x17u )
    {
      if ( (_BYTE)v2 != 5 )
        return 0;
      v6 = *(unsigned __int16 *)(a2 + 18);
      goto LABEL_12;
    }
    goto LABEL_11;
  }
  if ( (_BYTE)v2 != 5 )
    return 0;
  v12 = *(_WORD *)(a2 + 18);
  if ( v12 == 37 )
    goto LABEL_5;
  v6 = v12;
LABEL_12:
  if ( v6 != 38 )
    return 0;
  v7 = (*(_BYTE *)(a2 + 23) & 0x40) != 0
     ? *(__int64 **)(a2 - 8)
     : (__int64 *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v8 = *v7;
  if ( (unsigned __int8)(*(_BYTE *)(*v7 + 16) - 75) > 1u )
    return 0;
  v9 = *(_QWORD *)(v8 - 48);
  if ( !v9 )
    return 0;
  **(_QWORD **)(a1 + 32) = v9;
  v10 = *(_QWORD *)(v8 - 24);
  if ( !v10 )
    return 0;
  **(_QWORD **)(a1 + 40) = v10;
  v11 = *(unsigned __int16 *)(v8 + 18);
  BYTE1(v11) &= ~0x80u;
  **(_DWORD **)(a1 + 24) = v11;
  return 1;
}
