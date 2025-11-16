// Function: sub_F137C0
// Address: 0xf137c0
//
bool __fastcall sub_F137C0(__int64 a1, char *a2)
{
  unsigned __int8 v2; // al
  bool result; // al
  __int64 v5; // rdi
  __int64 v6; // rcx
  __int64 v7; // r12
  __int64 v8; // rdx
  __int64 v9; // r13
  __int16 v10; // ax
  int v11; // eax
  __int64 v12; // r12
  __int64 v13; // r13
  __int16 v14; // ax
  int v15; // eax
  __int64 v16; // r13
  __int64 v17; // r14
  __int16 v18; // ax
  __int64 v19; // rdx
  __int64 v20; // r8
  __int64 v21; // r12
  __int64 v22; // r13
  __int16 v23; // ax
  int v24; // eax
  __int64 v25; // rcx
  __int64 v26; // rax
  _QWORD *v27; // rdx
  __int64 v28; // r9
  __int64 v29; // rcx
  _QWORD *v30; // rax
  __int64 v31; // rcx
  __int64 v32; // rax
  __int64 v33; // r9
  __int64 v34; // rcx
  _QWORD *v35; // rax

  v2 = *a2;
  if ( (unsigned __int8)*a2 <= 0x1Cu )
    return 0;
  if ( v2 != 85 )
  {
    if ( v2 != 86 )
      goto LABEL_23;
    v5 = *((_QWORD *)a2 - 12);
    if ( *(_BYTE *)v5 != 82 )
      return 0;
    v6 = *((_QWORD *)a2 - 8);
    v7 = *(_QWORD *)(v5 - 64);
    v8 = *((_QWORD *)a2 - 4);
    v9 = *(_QWORD *)(v5 - 32);
    if ( v6 == v7 && v8 == v9 )
    {
      v10 = *(_WORD *)(v5 + 2);
    }
    else
    {
      if ( v6 != v9 || v8 != v7 )
        goto LABEL_17;
      v10 = *(_WORD *)(v5 + 2);
      if ( v6 != v7 )
      {
        v11 = sub_B52870(v10 & 0x3F);
        goto LABEL_11;
      }
    }
    v11 = v10 & 0x3F;
LABEL_11:
    result = (unsigned int)(v11 - 38) <= 1 && v7 != 0;
    if ( result )
    {
      **(_QWORD **)a1 = v7;
      if ( v9 )
      {
        **(_QWORD **)(a1 + 8) = v9;
        return result;
      }
    }
    goto LABEL_12;
  }
  v19 = *((_QWORD *)a2 - 4);
  v20 = v19;
  if ( !v19 )
    goto LABEL_40;
  if ( *(_BYTE *)v19
    || *(_QWORD *)(v19 + 24) != *((_QWORD *)a2 + 10)
    || (*(_BYTE *)(v19 + 33) & 0x20) == 0
    || *(_DWORD *)(v19 + 36) != 329 )
  {
LABEL_37:
    v20 = v19;
    if ( !v19 )
      goto LABEL_40;
    goto LABEL_38;
  }
  v33 = *(_QWORD *)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
  v34 = *(_QWORD *)&a2[32 * (1LL - (*((_DWORD *)a2 + 1) & 0x7FFFFFF))];
  if ( !v33 )
  {
LABEL_38:
    if ( !*(_BYTE *)v19
      && *(_QWORD *)(v19 + 24) == *((_QWORD *)a2 + 10)
      && (*(_BYTE *)(v19 + 33) & 0x20) != 0
      && *(_DWORD *)(v19 + 36) == 330 )
    {
      v28 = *(_QWORD *)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
      v29 = *(_QWORD *)&a2[32 * (1LL - (*((_DWORD *)a2 + 1) & 0x7FFFFFF))];
      if ( !v28 )
      {
LABEL_41:
        if ( *(_BYTE *)v19
          || *(_QWORD *)(v19 + 24) != *((_QWORD *)a2 + 10)
          || (*(_BYTE *)(v19 + 33) & 0x20) == 0
          || *(_DWORD *)(v19 + 36) != 365 )
        {
          goto LABEL_43;
        }
        v25 = *(_QWORD *)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
        v26 = *(_QWORD *)&a2[32 * (1LL - (*((_DWORD *)a2 + 1) & 0x7FFFFFF))];
        if ( !v25 )
        {
LABEL_65:
          if ( *(_BYTE *)v20 )
            return 0;
          if ( *(_QWORD *)(v20 + 24) != *((_QWORD *)a2 + 10) )
            return 0;
          if ( (*(_BYTE *)(v20 + 33) & 0x20) == 0 )
            return 0;
          if ( *(_DWORD *)(v20 + 36) != 366 )
            return 0;
          v31 = *(_QWORD *)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
          v32 = *(_QWORD *)&a2[32 * (1LL - (*((_DWORD *)a2 + 1) & 0x7FFFFFF))];
          if ( !v31 )
            return 0;
          **(_QWORD **)(a1 + 48) = v31;
          if ( !v32 )
            return 0;
          **(_QWORD **)(a1 + 56) = v32;
          return 1;
        }
        v27 = *(_QWORD **)(a1 + 32);
        if ( v26 )
        {
          *v27 = v25;
          **(_QWORD **)(a1 + 40) = v26;
          return 1;
        }
        *v27 = v25;
        goto LABEL_79;
      }
      v30 = *(_QWORD **)(a1 + 16);
      if ( v29 )
      {
        *v30 = v28;
        **(_QWORD **)(a1 + 24) = v29;
        return 1;
      }
      *v30 = v28;
      goto LABEL_22;
    }
LABEL_40:
    v19 = *((_QWORD *)a2 - 4);
    v20 = v19;
    if ( !v19 )
      goto LABEL_43;
    goto LABEL_41;
  }
  v35 = *(_QWORD **)a1;
  if ( v34 )
  {
    *v35 = v33;
    **(_QWORD **)(a1 + 8) = v34;
    return 1;
  }
  *v35 = v33;
LABEL_12:
  v2 = *a2;
  if ( (unsigned __int8)*a2 <= 0x1Cu )
    return 0;
  if ( v2 == 85 )
  {
    v19 = *((_QWORD *)a2 - 4);
    goto LABEL_37;
  }
  if ( v2 != 86 )
    goto LABEL_23;
  v5 = *((_QWORD *)a2 - 12);
  if ( *(_BYTE *)v5 != 82 )
    return 0;
  v8 = *((_QWORD *)a2 - 4);
  v6 = *((_QWORD *)a2 - 8);
LABEL_17:
  v12 = *(_QWORD *)(v5 - 64);
  v13 = *(_QWORD *)(v5 - 32);
  if ( v6 == v12 && v8 == v13 )
  {
    v14 = *(_WORD *)(v5 + 2);
    goto LABEL_20;
  }
  if ( v6 == v13 && v8 == v12 )
  {
    v14 = *(_WORD *)(v5 + 2);
    if ( v6 != v12 )
    {
      v15 = sub_B52870(v14 & 0x3F);
      goto LABEL_21;
    }
LABEL_20:
    v15 = v14 & 0x3F;
LABEL_21:
    result = (unsigned int)(v15 - 40) <= 1 && v12 != 0;
    if ( result )
    {
      **(_QWORD **)(a1 + 16) = v12;
      if ( v13 )
      {
        **(_QWORD **)(a1 + 24) = v13;
        return result;
      }
    }
LABEL_22:
    v2 = *a2;
    if ( (unsigned __int8)*a2 <= 0x1Cu )
      return 0;
LABEL_23:
    if ( v2 != 85 )
    {
      if ( v2 != 86 )
        goto LABEL_43;
      v5 = *((_QWORD *)a2 - 12);
      if ( *(_BYTE *)v5 != 82 )
        return 0;
      v8 = *((_QWORD *)a2 - 4);
      v6 = *((_QWORD *)a2 - 8);
      goto LABEL_27;
    }
    goto LABEL_40;
  }
LABEL_27:
  v16 = *(_QWORD *)(v5 - 64);
  v17 = *(_QWORD *)(v5 - 32);
  if ( v6 == v16 && v8 == v17 )
  {
    v18 = *(_WORD *)(v5 + 2);
LABEL_30:
    if ( (v18 & 0x3Fu) - 34 > 1 || !v16 )
      goto LABEL_47;
    goto LABEL_32;
  }
  if ( v6 != v17 || v8 != v16 )
    goto LABEL_47;
  v18 = *(_WORD *)(v5 + 2);
  if ( v6 == v16 )
    goto LABEL_30;
  if ( (unsigned int)sub_B52870(*(_WORD *)(v5 + 2) & 0x3F) - 34 <= 1 && v16 )
  {
LABEL_32:
    **(_QWORD **)(a1 + 32) = v16;
    if ( v17 )
    {
      **(_QWORD **)(a1 + 40) = v17;
      return 1;
    }
  }
LABEL_79:
  v2 = *a2;
  if ( (unsigned __int8)*a2 <= 0x1Cu )
    return 0;
LABEL_43:
  if ( v2 == 85 )
  {
    v20 = *((_QWORD *)a2 - 4);
    if ( !v20 )
      return 0;
    goto LABEL_65;
  }
  if ( v2 != 86 )
    return 0;
  v5 = *((_QWORD *)a2 - 12);
  if ( *(_BYTE *)v5 != 82 )
    return 0;
  v8 = *((_QWORD *)a2 - 4);
  v6 = *((_QWORD *)a2 - 8);
LABEL_47:
  v21 = *(_QWORD *)(v5 - 64);
  v22 = *(_QWORD *)(v5 - 32);
  if ( v6 == v21 && v8 == v22 )
  {
    v23 = *(_WORD *)(v5 + 2);
LABEL_50:
    v24 = v23 & 0x3F;
    goto LABEL_51;
  }
  if ( v6 != v22 || v8 != v21 )
    return 0;
  v23 = *(_WORD *)(v5 + 2);
  if ( v6 == v21 )
    goto LABEL_50;
  v24 = sub_B52870(v23 & 0x3F);
LABEL_51:
  result = (unsigned int)(v24 - 36) <= 1 && v21 != 0;
  if ( !result )
    return 0;
  **(_QWORD **)(a1 + 48) = v21;
  if ( !v22 )
    return 0;
  **(_QWORD **)(a1 + 56) = v22;
  return result;
}
