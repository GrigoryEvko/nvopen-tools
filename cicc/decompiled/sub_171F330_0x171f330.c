// Function: sub_171F330
// Address: 0x171f330
//
char __fastcall sub_171F330(_QWORD **a1, __int64 a2)
{
  __int64 v2; // rax
  char v3; // dl
  __int64 v4; // rax
  char v5; // dl
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rax
  char v9; // dl
  __int64 v11; // rdx
  __int64 v12; // rax
  char v13; // dl
  __int16 v14; // dx
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // r8
  __int64 v24; // rsi
  __int64 v25; // rax
  __int64 v26; // r8
  __int64 v27; // rsi
  __int64 v28; // rax

  if ( !a2 )
    return 0;
  v2 = *(_QWORD *)(a2 - 48);
  v3 = *(_BYTE *)(v2 + 16);
  if ( v3 != 51 )
  {
    if ( v3 != 5 )
      goto LABEL_5;
    if ( *(_WORD *)(v2 + 18) != 27 )
      goto LABEL_5;
    v19 = *(_QWORD *)(v2 - 24LL * (*(_DWORD *)(v2 + 20) & 0xFFFFFFF));
    if ( !v19 )
      goto LABEL_5;
    **a1 = v19;
    v12 = *(_QWORD *)(v2 + 24 * (1LL - (*(_DWORD *)(v2 + 20) & 0xFFFFFFF)));
    if ( !v12 )
      goto LABEL_5;
LABEL_14:
    *a1[1] = v12;
    v4 = *(_QWORD *)(a2 - 24);
    v13 = *(_BYTE *)(v4 + 16);
    if ( v13 == 50 )
    {
      v16 = *(_QWORD *)(v4 - 48);
      v17 = *(_QWORD *)(v4 - 24);
      v18 = *a1[2];
      return v16 == v18 && *a1[3] == v17 || v18 == v17 && v16 == *a1[3];
    }
    if ( v13 != 5 )
    {
      if ( v13 != 51 )
        return 0;
      goto LABEL_6;
    }
    v14 = *(_WORD *)(v4 + 18);
    if ( v14 == 26 )
    {
      v26 = *(_QWORD *)(v4 - 24LL * (*(_DWORD *)(v4 + 20) & 0xFFFFFFF));
      v27 = *a1[2];
      v28 = *(_QWORD *)(v4 + 24 * (1LL - (*(_DWORD *)(v4 + 20) & 0xFFFFFFF)));
      return v26 == v27 && *a1[3] == v28 || v27 == v28 && v26 == *a1[3];
    }
LABEL_17:
    if ( v14 != 27 )
      return 0;
    v15 = *(_QWORD *)(v4 - 24LL * (*(_DWORD *)(v4 + 20) & 0xFFFFFFF));
    if ( !v15 )
      return 0;
    **a1 = v15;
    v7 = *(_QWORD *)(v4 + 24 * (1LL - (*(_DWORD *)(v4 + 20) & 0xFFFFFFF)));
    if ( !v7 )
      return 0;
    goto LABEL_8;
  }
  v11 = *(_QWORD *)(v2 - 48);
  if ( v11 )
  {
    **a1 = v11;
    v12 = *(_QWORD *)(v2 - 24);
    if ( v12 )
      goto LABEL_14;
  }
LABEL_5:
  v4 = *(_QWORD *)(a2 - 24);
  v5 = *(_BYTE *)(v4 + 16);
  if ( v5 != 51 )
  {
    if ( v5 != 5 )
      return 0;
    v14 = *(_WORD *)(v4 + 18);
    goto LABEL_17;
  }
LABEL_6:
  v6 = *(_QWORD *)(v4 - 48);
  if ( !v6 )
    return 0;
  **a1 = v6;
  v7 = *(_QWORD *)(v4 - 24);
  if ( !v7 )
    return 0;
LABEL_8:
  *a1[1] = v7;
  v8 = *(_QWORD *)(a2 - 48);
  v9 = *(_BYTE *)(v8 + 16);
  if ( v9 == 50 )
  {
    v20 = *(_QWORD *)(v8 - 48);
    v21 = *(_QWORD *)(v8 - 24);
    v22 = *a1[2];
    if ( v20 != v22 || *a1[3] != v21 )
    {
      if ( v22 == v21 )
        return *a1[3] == v20;
      return 0;
    }
    return 1;
  }
  if ( v9 != 5 || *(_WORD *)(v8 + 18) != 26 )
    return 0;
  v23 = *(_QWORD *)(v8 - 24LL * (*(_DWORD *)(v8 + 20) & 0xFFFFFFF));
  v24 = *a1[2];
  v25 = *(_QWORD *)(v8 + 24 * (1LL - (*(_DWORD *)(v8 + 20) & 0xFFFFFFF)));
  if ( v23 == v24 && *a1[3] == v25 )
    return 1;
  if ( v24 != v25 )
    return 0;
  return *a1[3] == v23;
}
