// Function: sub_1111940
// Address: 0x1111940
//
bool __fastcall sub_1111940(char *a1)
{
  unsigned __int8 v1; // al
  __int64 v4; // rax
  __int16 v5; // si
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // rdx
  __int64 v9; // rdx
  __int64 v10; // rsi
  __int64 v11; // rdi
  __int16 v12; // ax
  int v13; // eax
  __int64 v14; // rax
  __int64 v15; // rsi
  __int64 v16; // rdi
  __int16 v17; // ax
  __int64 v18; // rdi
  __int64 v19; // rsi
  __int64 v20; // rdi
  __int64 v21; // rsi
  __int16 v22; // si

  v1 = *a1;
  if ( (unsigned __int8)*a1 <= 0x1Cu )
    return 0;
  if ( v1 == 85 )
  {
    v8 = *((_QWORD *)a1 - 4);
    if ( !v8 )
      goto LABEL_22;
    if ( !*(_BYTE *)v8
      && *(_QWORD *)(v8 + 24) == *((_QWORD *)a1 + 10)
      && (*(_BYTE *)(v8 + 33) & 0x20) != 0
      && *(_DWORD *)(v8 + 36) == 329 )
    {
      return 1;
    }
    goto LABEL_19;
  }
  if ( v1 != 86 )
    goto LABEL_12;
  v4 = *((_QWORD *)a1 - 12);
  if ( *(_BYTE *)v4 != 82 )
    return 0;
  v7 = *((_QWORD *)a1 - 8);
  v15 = *(_QWORD *)(v4 - 64);
  v6 = *((_QWORD *)a1 - 4);
  v16 = *(_QWORD *)(v4 - 32);
  if ( v7 == v15 && v6 == v16 )
  {
    v17 = *(_WORD *)(v4 + 2);
    goto LABEL_43;
  }
  if ( v7 == v16 && v6 == v15 )
  {
    v17 = *(_WORD *)(v4 + 2);
    if ( v7 == v15 )
    {
LABEL_43:
      if ( (v17 & 0x3Fu) - 38 <= 1 )
        return 1;
LABEL_44:
      v4 = *((_QWORD *)a1 - 12);
      if ( *(_BYTE *)v4 != 82 )
        return 0;
      v6 = *((_QWORD *)a1 - 4);
      v7 = *((_QWORD *)a1 - 8);
      goto LABEL_46;
    }
    if ( (unsigned int)sub_B52870(v17 & 0x3F) - 38 <= 1 )
      return 1;
    v1 = *a1;
    if ( (unsigned __int8)*a1 <= 0x1Cu )
      return 0;
    if ( v1 != 85 )
    {
      if ( v1 != 86 )
        goto LABEL_12;
      goto LABEL_44;
    }
    v8 = *((_QWORD *)a1 - 4);
LABEL_19:
    if ( v8
      && !*(_BYTE *)v8
      && *(_QWORD *)(v8 + 24) == *((_QWORD *)a1 + 10)
      && (*(_BYTE *)(v8 + 33) & 0x20) != 0
      && *(_DWORD *)(v8 + 36) == 330 )
    {
      return 1;
    }
LABEL_22:
    v9 = *((_QWORD *)a1 - 4);
    if ( v9 && !*(_BYTE *)v9 && *(_QWORD *)(v9 + 24) == *((_QWORD *)a1 + 10) && (*(_BYTE *)(v9 + 33) & 0x20) != 0 )
    {
      if ( *(_DWORD *)(v9 + 36) == 365 )
        return 1;
      goto LABEL_34;
    }
LABEL_25:
    if ( v1 != 85 )
    {
      if ( v1 != 86 )
        return 0;
      v4 = *((_QWORD *)a1 - 12);
      if ( *(_BYTE *)v4 != 82 )
        return 0;
      v6 = *((_QWORD *)a1 - 4);
      v7 = *((_QWORD *)a1 - 8);
      goto LABEL_29;
    }
LABEL_34:
    v14 = *((_QWORD *)a1 - 4);
    return v14
        && !*(_BYTE *)v14
        && *(_QWORD *)(v14 + 24) == *((_QWORD *)a1 + 10)
        && (*(_BYTE *)(v14 + 33) & 0x20) != 0
        && *(_DWORD *)(v14 + 36) == 366;
  }
LABEL_46:
  v18 = *(_QWORD *)(v4 - 64);
  v19 = *(_QWORD *)(v4 - 32);
  if ( v7 == v18 && v6 == v19 )
  {
    v5 = *(_WORD *)(v4 + 2);
LABEL_49:
    if ( (v5 & 0x3Fu) - 40 <= 1 )
      return 1;
    goto LABEL_50;
  }
  if ( v7 != v19 || v6 != v18 )
    goto LABEL_50;
  v5 = *(_WORD *)(v4 + 2);
  if ( v7 == v18 )
    goto LABEL_49;
  if ( (unsigned int)sub_B52870(*(_WORD *)(v4 + 2) & 0x3F) - 40 <= 1 )
    return 1;
  v1 = *a1;
  if ( (unsigned __int8)*a1 <= 0x1Cu )
    return 0;
LABEL_12:
  if ( v1 == 85 )
    goto LABEL_22;
  if ( v1 != 86 )
    goto LABEL_25;
  v4 = *((_QWORD *)a1 - 12);
  if ( *(_BYTE *)v4 != 82 )
    return 0;
  v6 = *((_QWORD *)a1 - 4);
  v7 = *((_QWORD *)a1 - 8);
LABEL_50:
  v20 = *(_QWORD *)(v4 - 64);
  v21 = *(_QWORD *)(v4 - 32);
  if ( v7 == v20 && v6 == v21 )
  {
    v22 = *(_WORD *)(v4 + 2);
    goto LABEL_53;
  }
  if ( v7 != v21 || v6 != v20 )
  {
LABEL_29:
    v10 = *(_QWORD *)(v4 - 64);
    v11 = *(_QWORD *)(v4 - 32);
    if ( v7 == v10 && v6 == v11 )
    {
      v12 = *(_WORD *)(v4 + 2);
    }
    else
    {
      if ( v7 != v11 || v6 != v10 )
        return 0;
      v12 = *(_WORD *)(v4 + 2);
      if ( v7 != v10 )
      {
        v13 = sub_B52870(v12 & 0x3F);
        return (unsigned int)(v13 - 36) <= 1;
      }
    }
    v13 = v12 & 0x3F;
    return (unsigned int)(v13 - 36) <= 1;
  }
  v22 = *(_WORD *)(v4 + 2);
  if ( v7 != v20 )
  {
    if ( (unsigned int)sub_B52870(*(_WORD *)(v4 + 2) & 0x3F) - 34 <= 1 )
      return 1;
    v1 = *a1;
    if ( (unsigned __int8)*a1 <= 0x1Cu )
      return 0;
    goto LABEL_25;
  }
LABEL_53:
  if ( (v22 & 0x3Fu) - 34 > 1 )
    goto LABEL_29;
  return 1;
}
