// Function: sub_176AD30
// Address: 0x176ad30
//
__int64 __fastcall sub_176AD30(__int64 a1, __int64 a2)
{
  unsigned int v2; // r12d
  __int64 v4; // rbx
  char v5; // al
  __int64 v7; // rax
  char v8; // dl
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // rdx
  _BYTE *v12; // rdi
  __int64 v13; // rax
  __int64 v14; // rax
  char v15; // dl
  __int64 v16; // rdx
  __int64 v17; // rcx
  _BYTE *v18; // rdi
  __int64 v19; // rdx
  _BYTE *v20; // rdi
  __int64 v21; // rax
  __int64 v22; // rdx
  unsigned __int8 v23; // al
  unsigned __int8 v24; // al
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rax

  v4 = a2;
  v5 = *(_BYTE *)(a2 + 16);
  if ( v5 == 49 )
  {
    v7 = *(_QWORD *)(a2 - 48);
    v8 = *(_BYTE *)(v7 + 16);
    if ( v8 == 47 )
    {
      v22 = *(_QWORD *)(v7 - 48);
      if ( !v22 )
        return 0;
      v10 = *(_QWORD *)a1;
      **(_QWORD **)a1 = v22;
      v12 = *(_BYTE **)(v7 - 24);
      v23 = v12[16];
      if ( v23 != 13 )
      {
        v11 = *(_QWORD *)v12;
        if ( *(_BYTE *)(*(_QWORD *)v12 + 8LL) != 16 || v23 > 0x10u )
          return 0;
LABEL_12:
        v13 = sub_15A1020(v12, a2, v11, v10);
        if ( !v13 || *(_BYTE *)(v13 + 16) != 13 )
          return 0;
        **(_QWORD **)(a1 + 8) = v13 + 24;
        goto LABEL_29;
      }
    }
    else
    {
      if ( v8 != 5 )
        return 0;
      if ( *(_WORD *)(v7 + 18) != 23 )
        return 0;
      v9 = *(_QWORD *)(v7 - 24LL * (*(_DWORD *)(v7 + 20) & 0xFFFFFFF));
      if ( !v9 )
        return 0;
      **(_QWORD **)a1 = v9;
      v10 = *(_DWORD *)(v7 + 20) & 0xFFFFFFF;
      v11 = 3 * (1 - v10);
      v12 = *(_BYTE **)(v7 + 24 * (1 - v10));
      if ( v12[16] != 13 )
      {
        if ( *(_BYTE *)(*(_QWORD *)v12 + 8LL) != 16 )
          return 0;
        goto LABEL_12;
      }
    }
    **(_QWORD **)(a1 + 8) = v12 + 24;
LABEL_29:
    v20 = *(_BYTE **)(a2 - 24);
    v24 = v20[16];
    if ( v24 != 13 )
    {
      LOBYTE(v2) = v24 <= 0x10u && *(_BYTE *)(*(_QWORD *)v20 + 8LL) == 16;
      if ( (_BYTE)v2 )
      {
        v25 = sub_15A1020(v20, a2, *(_QWORD *)v20, v10);
        if ( v25 )
        {
          if ( *(_BYTE *)(v25 + 16) == 13 )
          {
            **(_QWORD **)(a1 + 16) = v25 + 24;
            return v2;
          }
        }
      }
      return 0;
    }
    goto LABEL_30;
  }
  if ( v5 != 5 || *(_WORD *)(a2 + 18) != 25 )
    return 0;
  v14 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v15 = *(_BYTE *)(v14 + 16);
  if ( v15 == 47 )
  {
    v26 = *(_QWORD *)(v14 - 48);
    if ( !v26 )
      return 0;
    **(_QWORD **)a1 = v26;
    a2 = *(_QWORD *)(v14 - 24);
    if ( !(unsigned __int8)sub_13D2630((_QWORD **)(a1 + 8), (_BYTE *)a2) )
      return 0;
  }
  else
  {
    if ( v15 != 5 )
      return 0;
    if ( *(_WORD *)(v14 + 18) != 23 )
      return 0;
    v16 = *(_QWORD *)(v14 - 24LL * (*(_DWORD *)(v14 + 20) & 0xFFFFFFF));
    if ( !v16 )
      return 0;
    **(_QWORD **)a1 = v16;
    v17 = *(_DWORD *)(v14 + 20) & 0xFFFFFFF;
    v18 = *(_BYTE **)(v14 + 24 * (1 - v17));
    if ( v18[16] == 13 )
    {
      **(_QWORD **)(a1 + 8) = v18 + 24;
    }
    else
    {
      if ( *(_BYTE *)(*(_QWORD *)v18 + 8LL) != 16 )
        return 0;
      v27 = sub_15A1020(v18, a2, 3 * (1 - v17), v17);
      if ( !v27 || *(_BYTE *)(v27 + 16) != 13 )
        return 0;
      **(_QWORD **)(a1 + 8) = v27 + 24;
    }
  }
  v19 = *(_DWORD *)(v4 + 20) & 0xFFFFFFF;
  v20 = *(_BYTE **)(v4 + 24 * (1 - v19));
  if ( v20[16] != 13 )
  {
    if ( *(_BYTE *)(*(_QWORD *)v20 + 8LL) == 16 )
    {
      v21 = sub_15A1020(v20, a2, v19, v17);
      if ( v21 )
      {
        if ( *(_BYTE *)(v21 + 16) == 13 )
        {
          v2 = 1;
          **(_QWORD **)(a1 + 16) = v21 + 24;
          return v2;
        }
      }
    }
    return 0;
  }
LABEL_30:
  **(_QWORD **)(a1 + 16) = v20 + 24;
  return 1;
}
