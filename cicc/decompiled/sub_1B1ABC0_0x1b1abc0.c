// Function: sub_1B1ABC0
// Address: 0x1b1abc0
//
__int64 __fastcall sub_1B1ABC0(_QWORD **a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int v4; // r12d
  char v6; // al
  __int64 v8; // rax
  _BYTE *v9; // rdi
  unsigned __int8 v10; // al
  __int64 v11; // rax
  __int64 v12; // rcx
  __int64 v13; // rdx
  _BYTE *v14; // rdi
  unsigned __int8 v15; // al
  __int64 v16; // rdx
  unsigned __int8 v17; // al
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax

  v6 = *(_BYTE *)(a2 + 16);
  if ( v6 == 50 )
  {
    v8 = *(_QWORD *)(a2 - 48);
    if ( *(_BYTE *)(v8 + 16) > 0x17u )
    {
      **a1 = v8;
      v9 = *(_BYTE **)(a2 - 24);
      v10 = v9[16];
      if ( v10 == 13 )
        goto LABEL_8;
      v16 = *(_QWORD *)v9;
      LOBYTE(v4) = v10 <= 0x10u && *(_BYTE *)(*(_QWORD *)v9 + 8LL) == 16;
      if ( !(_BYTE)v4 )
        goto LABEL_13;
      LOBYTE(v16) = v10 <= 0x10u;
      v18 = sub_15A1020(v9, a2, v16, a4);
      if ( v18 && *(_BYTE *)(v18 + 16) == 13 )
        goto LABEL_18;
    }
    v9 = *(_BYTE **)(a2 - 24);
    v10 = v9[16];
LABEL_13:
    if ( v10 <= 0x17u )
      return 0;
    **a1 = v9;
    v9 = *(_BYTE **)(a2 - 48);
    v17 = v9[16];
    if ( v17 == 13 )
      goto LABEL_8;
    LOBYTE(v4) = v17 <= 0x10u && *(_BYTE *)(*(_QWORD *)v9 + 8LL) == 16;
    if ( !(_BYTE)v4 )
      return 0;
    v18 = sub_15A1020(v9, a2, *(_QWORD *)v9, a4);
    if ( !v18 || *(_BYTE *)(v18 + 16) != 13 )
      return 0;
LABEL_18:
    *a1[1] = v18 + 24;
    return v4;
  }
  if ( v6 != 5 || *(_WORD *)(a2 + 18) != 26 )
    return 0;
  v11 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  v12 = 4 * v11;
  v13 = *(_QWORD *)(a2 - 24 * v11);
  if ( *(_BYTE *)(v13 + 16) <= 0x17u )
  {
    v14 = *(_BYTE **)(a2 + 24 * (1 - v11));
    v15 = v14[16];
  }
  else
  {
    v4 = 1;
    **a1 = v13;
    v14 = *(_BYTE **)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
    v15 = v14[16];
    if ( v15 == 13 )
    {
      *a1[1] = v14 + 24;
      return v4;
    }
    if ( *(_BYTE *)(*(_QWORD *)v14 + 8LL) == 16 )
    {
      v20 = sub_15A1020(v14, a2, *(_QWORD *)v14, v12);
      if ( v20 && *(_BYTE *)(v20 + 16) == 13 )
        goto LABEL_28;
      v14 = *(_BYTE **)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
      v15 = v14[16];
    }
  }
  if ( v15 <= 0x17u )
    return 0;
  **a1 = v14;
  v19 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  v9 = *(_BYTE **)(a2 - 24 * v19);
  if ( v9[16] != 13 )
  {
    if ( *(_BYTE *)(*(_QWORD *)v9 + 8LL) == 16 )
    {
      v20 = sub_15A1020(v9, a2, 4 * v19, v12);
      if ( v20 )
      {
        if ( *(_BYTE *)(v20 + 16) == 13 )
        {
LABEL_28:
          v4 = 1;
          *a1[1] = v20 + 24;
          return v4;
        }
      }
    }
    return 0;
  }
LABEL_8:
  *a1[1] = v9 + 24;
  return 1;
}
