// Function: sub_175DFA0
// Address: 0x175dfa0
//
__int64 __fastcall sub_175DFA0(_QWORD **a1, __int64 a2)
{
  char v2; // al
  __int64 v4; // rdx
  int v5; // eax
  int v6; // eax
  __int64 *v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rdx
  int v10; // eax
  int v11; // eax
  _QWORD *v12; // rdx
  __int64 v13; // rdx
  int v14; // eax
  int v15; // eax
  __int64 *v16; // rdx
  __int64 v17; // rdx

  v2 = *(_BYTE *)(a2 + 16);
  if ( v2 == 39 )
  {
    v4 = *(_QWORD *)(a2 - 48);
    v5 = *(unsigned __int8 *)(v4 + 16);
    if ( (unsigned __int8)v5 > 0x17u )
    {
      v6 = v5 - 24;
    }
    else
    {
      if ( (_BYTE)v5 != 5 )
        return 0;
      v6 = *(unsigned __int16 *)(v4 + 18);
    }
    if ( v6 != 37 )
      return 0;
    v7 = (*(_BYTE *)(v4 + 23) & 0x40) != 0
       ? *(__int64 **)(v4 - 8)
       : (__int64 *)(v4 - 24LL * (*(_DWORD *)(v4 + 20) & 0xFFFFFFF));
    v8 = *v7;
    if ( !v8 )
      return 0;
    **a1 = v8;
    v9 = *(_QWORD *)(a2 - 24);
    v10 = *(unsigned __int8 *)(v9 + 16);
    if ( (unsigned __int8)v10 <= 0x17u )
    {
LABEL_29:
      if ( (_BYTE)v10 == 5 )
      {
        v11 = *(unsigned __int16 *)(v9 + 18);
        goto LABEL_14;
      }
      return 0;
    }
  }
  else
  {
    if ( v2 != 5 || *(_WORD *)(a2 + 18) != 15 )
      return 0;
    v13 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
    v14 = *(unsigned __int8 *)(v13 + 16);
    if ( (unsigned __int8)v14 > 0x17u )
    {
      v15 = v14 - 24;
    }
    else
    {
      if ( (_BYTE)v14 != 5 )
        return 0;
      v15 = *(unsigned __int16 *)(v13 + 18);
    }
    if ( v15 != 37 )
      return 0;
    v16 = (*(_BYTE *)(v13 + 23) & 0x40) != 0
        ? *(__int64 **)(v13 - 8)
        : (__int64 *)(v13 - 24LL * (*(_DWORD *)(v13 + 20) & 0xFFFFFFF));
    v17 = *v16;
    if ( !v17 )
      return 0;
    **a1 = v17;
    v9 = *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
    v10 = *(unsigned __int8 *)(v9 + 16);
    if ( (unsigned __int8)v10 <= 0x17u )
      goto LABEL_29;
  }
  v11 = v10 - 24;
LABEL_14:
  if ( v11 != 37 )
    return 0;
  v12 = (*(_BYTE *)(v9 + 23) & 0x40) != 0
      ? *(_QWORD **)(v9 - 8)
      : (_QWORD *)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF));
  if ( !*v12 )
    return 0;
  *a1[1] = *v12;
  return 1;
}
