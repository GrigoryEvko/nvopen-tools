// Function: sub_17230F0
// Address: 0x17230f0
//
__int64 __fastcall sub_17230F0(_QWORD **a1, __int64 a2)
{
  char v2; // al
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rdx
  __int64 v16; // rdx

  v2 = *(_BYTE *)(a2 + 16);
  if ( v2 != 35 )
  {
    if ( v2 != 5 || *(_WORD *)(a2 + 18) != 11 )
      return 0;
    v10 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
    v11 = *(_QWORD *)(a2 - 24 * v10);
    if ( v11 )
    {
      **a1 = v11;
      v12 = *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
      if ( *(_BYTE *)(v12 + 16) != 79 || (v15 = *(_QWORD *)(v12 - 72)) == 0 )
      {
LABEL_14:
        **a1 = v12;
        v6 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
        if ( *(_BYTE *)(v6 + 16) == 79 )
          goto LABEL_8;
        return 0;
      }
      *a1[1] = v15;
      v16 = *(_QWORD *)(v12 - 48);
      if ( v16 )
      {
        *a1[2] = v16;
        v9 = *(_QWORD *)(v12 - 24);
        if ( v9 )
          goto LABEL_19;
      }
      v10 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
    }
    v12 = *(_QWORD *)(a2 + 24 * (1 - v10));
    if ( !v12 )
      return 0;
    goto LABEL_14;
  }
  v4 = *(_QWORD *)(a2 - 48);
  if ( v4 )
  {
    **a1 = v4;
    v5 = *(_QWORD *)(a2 - 24);
    if ( *(_BYTE *)(v5 + 16) != 79 )
      goto LABEL_7;
    v13 = *(_QWORD *)(v5 - 72);
    if ( !v13 )
      goto LABEL_7;
    *a1[1] = v13;
    v14 = *(_QWORD *)(v5 - 48);
    if ( v14 )
    {
      *a1[2] = v14;
      v9 = *(_QWORD *)(v5 - 24);
      if ( v9 )
        goto LABEL_19;
    }
  }
  v5 = *(_QWORD *)(a2 - 24);
  if ( !v5 )
    return 0;
LABEL_7:
  **a1 = v5;
  v6 = *(_QWORD *)(a2 - 48);
  if ( *(_BYTE *)(v6 + 16) != 79 )
    return 0;
LABEL_8:
  v7 = *(_QWORD *)(v6 - 72);
  if ( !v7 )
    return 0;
  *a1[1] = v7;
  v8 = *(_QWORD *)(v6 - 48);
  if ( !v8 )
    return 0;
  *a1[2] = v8;
  v9 = *(_QWORD *)(v6 - 24);
  if ( !v9 )
    return 0;
LABEL_19:
  *a1[3] = v9;
  return 1;
}
