// Function: sub_10079C0
// Address: 0x10079c0
//
char *__fastcall sub_10079C0(unsigned int a1, char *a2, char *a3)
{
  char *result; // rax
  __int64 v7; // rax
  char *v8; // r15
  int v9; // r14d
  char *v10; // rdx
  unsigned __int8 v11; // al
  __int64 v12; // rax
  char *v13; // rsi
  char *v14; // r8
  char *v15; // rcx
  char *v16; // r9
  __int16 v17; // ax
  char *v18; // r8
  char *v19; // r9
  __int16 v20; // di
  char *v21; // r8
  char *v22; // r9
  __int16 v23; // di
  char *v24; // r8
  char *v25; // r9
  __int16 v26; // ax
  int v27; // eax
  int v28; // r8d
  __int64 v29; // rcx
  __int64 v30; // r8
  char *v31; // rsi
  char *v32; // rax
  char *v33; // rsi
  char *v34; // rax
  char *v35; // r9
  char *v36; // rsi
  char *v37; // r9
  char *v38; // rsi
  int v39; // eax
  int v40; // eax
  int v41; // eax
  char *v42; // [rsp+8h] [rbp-98h]
  char *v43; // [rsp+8h] [rbp-98h]
  char *v44; // [rsp+8h] [rbp-98h]
  char *v45; // [rsp+8h] [rbp-98h]
  char *v46; // [rsp+10h] [rbp-90h]
  char *v47; // [rsp+10h] [rbp-90h]
  char *v48; // [rsp+10h] [rbp-90h]
  char *v49; // [rsp+10h] [rbp-90h]
  char *v50; // [rsp+18h] [rbp-88h]
  char *v51; // [rsp+18h] [rbp-88h]
  char *v52; // [rsp+18h] [rbp-88h]
  char *v53; // [rsp+18h] [rbp-88h]
  char *v54; // [rsp+20h] [rbp-80h] BYREF
  char *v55; // [rsp+28h] [rbp-78h] BYREF
  _QWORD *v56[2]; // [rsp+30h] [rbp-70h] BYREF
  _QWORD *v57[2]; // [rsp+40h] [rbp-60h] BYREF
  _QWORD *v58[2]; // [rsp+50h] [rbp-50h] BYREF
  _QWORD *v59[8]; // [rsp+60h] [rbp-40h] BYREF

  v56[0] = &v54;
  v56[1] = &v55;
  v57[0] = &v54;
  v57[1] = &v55;
  v58[0] = &v54;
  v58[1] = &v55;
  v59[0] = &v54;
  v59[1] = &v55;
  if ( !sub_1007480(v56, a2) && !sub_10075D0(v57, a2) && !sub_1007720(v58, a2) && !sub_1007870(v59, a2) )
    return 0;
  if ( *a2 != 85 )
    return 0;
  v7 = *((_QWORD *)a2 - 4);
  if ( !v7 || *(_BYTE *)v7 || *(_QWORD *)(v7 + 24) != *((_QWORD *)a2 + 10) || (*(_BYTE *)(v7 + 33) & 0x20) == 0 )
    return 0;
  v8 = v54;
  v9 = *(_DWORD *)(v7 + 36);
  if ( v54 == a3 )
    goto LABEL_67;
  v10 = v55;
  if ( v55 == a3 )
    goto LABEL_67;
  v11 = *a3;
  if ( (unsigned __int8)*a3 <= 0x1Cu )
    return 0;
  if ( v11 == 85 )
  {
    v29 = *((_QWORD *)a3 - 4);
    v30 = v29;
    if ( !v29 )
      goto LABEL_74;
    if ( !*(_BYTE *)v29
      && *(_QWORD *)(v29 + 24) == *((_QWORD *)a3 + 10)
      && (*(_BYTE *)(v29 + 33) & 0x20) != 0
      && *(_DWORD *)(v29 + 36) == 329 )
    {
      v35 = *(char **)&a3[-32 * (*((_DWORD *)a3 + 1) & 0x7FFFFFF)];
      v36 = *(char **)&a3[32 * (1LL - (*((_DWORD *)a3 + 1) & 0x7FFFFFF))];
      if ( v55 == v36 && v54 == v35 || v54 == v36 && v55 == v35 )
        goto LABEL_67;
LABEL_98:
      if ( !*(_BYTE *)v29
        && *(_QWORD *)(v29 + 24) == *((_QWORD *)a3 + 10)
        && (*(_BYTE *)(v29 + 33) & 0x20) != 0
        && *(_DWORD *)(v29 + 36) == 330 )
      {
        v37 = *(char **)&a3[-32 * (*((_DWORD *)a3 + 1) & 0x7FFFFFF)];
        v38 = *(char **)&a3[32 * (1LL - (*((_DWORD *)a3 + 1) & 0x7FFFFFF))];
        if ( v8 == v37 && v10 == v38 || v8 == v38 && v10 == v37 )
          goto LABEL_67;
LABEL_75:
        if ( !*(_BYTE *)v29
          && *(_QWORD *)(v29 + 24) == *((_QWORD *)a3 + 10)
          && (*(_BYTE *)(v29 + 33) & 0x20) != 0
          && *(_DWORD *)(v29 + 36) == 365 )
        {
          v31 = *(char **)&a3[-32 * (*((_DWORD *)a3 + 1) & 0x7FFFFFF)];
          v32 = *(char **)&a3[32 * (1LL - (*((_DWORD *)a3 + 1) & 0x7FFFFFF))];
          if ( v10 == v32 && v8 == v31 || v8 == v32 && v10 == v31 )
            goto LABEL_67;
          goto LABEL_83;
        }
        goto LABEL_54;
      }
LABEL_74:
      v29 = *((_QWORD *)a3 - 4);
      v30 = v29;
      if ( !v29 )
        goto LABEL_54;
      goto LABEL_75;
    }
LABEL_73:
    v30 = v29;
    if ( !v29 )
      goto LABEL_74;
    goto LABEL_98;
  }
  if ( v11 != 86 )
    goto LABEL_41;
  v12 = *((_QWORD *)a3 - 12);
  if ( *(_BYTE *)v12 != 82 )
    return 0;
  v13 = (char *)*((_QWORD *)a3 - 8);
  v14 = *(char **)(v12 - 64);
  v15 = (char *)*((_QWORD *)a3 - 4);
  v16 = *(char **)(v12 - 32);
  if ( v13 == v14 && v15 == v16 )
  {
    v17 = *(_WORD *)(v12 + 2);
  }
  else
  {
    if ( v13 != v16 || v15 != v14 )
      goto LABEL_32;
    v17 = *(_WORD *)(v12 + 2);
    if ( v13 != v14 )
    {
      v42 = v16;
      v46 = v14;
      v50 = v55;
      v39 = sub_B52870(v17 & 0x3F);
      v10 = v50;
      v14 = v46;
      v16 = v42;
      if ( (unsigned int)(v39 - 38) > 1 )
      {
LABEL_27:
        v11 = *a3;
        if ( (unsigned __int8)*a3 <= 0x1Cu )
          return 0;
        if ( v11 != 85 )
        {
          if ( v11 != 86 )
            goto LABEL_41;
          goto LABEL_30;
        }
        v29 = *((_QWORD *)a3 - 4);
        goto LABEL_73;
      }
LABEL_23:
      if ( v8 == v14 && v10 == v16 || v8 == v16 && v10 == v14 )
        goto LABEL_67;
      goto LABEL_27;
    }
  }
  if ( (v17 & 0x3Fu) - 38 <= 1 )
    goto LABEL_23;
LABEL_30:
  v12 = *((_QWORD *)a3 - 12);
  if ( *(_BYTE *)v12 != 82 )
    return 0;
  v15 = (char *)*((_QWORD *)a3 - 4);
  v13 = (char *)*((_QWORD *)a3 - 8);
LABEL_32:
  v18 = *(char **)(v12 - 64);
  v19 = *(char **)(v12 - 32);
  if ( v13 == v18 && v15 == v19 )
  {
    v20 = *(_WORD *)(v12 + 2);
  }
  else
  {
    if ( v13 != v19 || v15 != v18 )
      goto LABEL_45;
    v20 = *(_WORD *)(v12 + 2);
    if ( v13 != v18 )
    {
      v44 = *(char **)(v12 - 32);
      v48 = *(char **)(v12 - 64);
      v52 = v10;
      v41 = sub_B52870(*(_WORD *)(v12 + 2) & 0x3F);
      v10 = v52;
      v18 = v48;
      v19 = v44;
      if ( (unsigned int)(v41 - 40) > 1 )
        goto LABEL_40;
      goto LABEL_36;
    }
  }
  if ( (v20 & 0x3Fu) - 40 > 1 )
    goto LABEL_45;
LABEL_36:
  if ( v8 == v18 && v10 == v19 || v10 == v18 && v8 == v19 )
    goto LABEL_67;
LABEL_40:
  v11 = *a3;
  if ( (unsigned __int8)*a3 <= 0x1Cu )
    return 0;
LABEL_41:
  if ( v11 == 85 )
    goto LABEL_74;
  if ( v11 != 86 )
    goto LABEL_54;
  v12 = *((_QWORD *)a3 - 12);
  if ( *(_BYTE *)v12 != 82 )
    return 0;
  v15 = (char *)*((_QWORD *)a3 - 4);
  v13 = (char *)*((_QWORD *)a3 - 8);
LABEL_45:
  v21 = *(char **)(v12 - 64);
  v22 = *(char **)(v12 - 32);
  if ( v13 == v21 && v15 == v22 )
  {
    v23 = *(_WORD *)(v12 + 2);
  }
  else
  {
    if ( v13 != v22 || v15 != v21 )
      goto LABEL_58;
    v23 = *(_WORD *)(v12 + 2);
    if ( v13 != v21 )
    {
      v43 = *(char **)(v12 - 32);
      v47 = *(char **)(v12 - 64);
      v51 = v10;
      v40 = sub_B52870(*(_WORD *)(v12 + 2) & 0x3F);
      v10 = v51;
      v21 = v47;
      v22 = v43;
      if ( (unsigned int)(v40 - 34) > 1 )
        goto LABEL_53;
      goto LABEL_49;
    }
  }
  if ( (v23 & 0x3Fu) - 34 > 1 )
    goto LABEL_58;
LABEL_49:
  if ( v10 == v22 && v8 == v21 || v8 == v22 && v10 == v21 )
    goto LABEL_67;
LABEL_53:
  v11 = *a3;
  if ( (unsigned __int8)*a3 <= 0x1Cu )
    return 0;
LABEL_54:
  if ( v11 == 85 )
  {
    v30 = *((_QWORD *)a3 - 4);
    if ( v30 )
    {
LABEL_83:
      if ( *(_BYTE *)v30
        || *(_QWORD *)(v30 + 24) != *((_QWORD *)a3 + 10)
        || (*(_BYTE *)(v30 + 33) & 0x20) == 0
        || *(_DWORD *)(v30 + 36) != 366 )
      {
        return 0;
      }
      v33 = *(char **)&a3[-32 * (*((_DWORD *)a3 + 1) & 0x7FFFFFF)];
      v34 = *(char **)&a3[32 * (1LL - (*((_DWORD *)a3 + 1) & 0x7FFFFFF))];
      if ( (v8 != v33 || v10 != v34) && (v10 != v33 || v8 != v34) )
        return 0;
      goto LABEL_67;
    }
    return 0;
  }
  if ( v11 != 86 )
    return 0;
  v12 = *((_QWORD *)a3 - 12);
  if ( *(_BYTE *)v12 != 82 )
    return 0;
  v15 = (char *)*((_QWORD *)a3 - 4);
  v13 = (char *)*((_QWORD *)a3 - 8);
LABEL_58:
  v24 = *(char **)(v12 - 64);
  v25 = *(char **)(v12 - 32);
  if ( v13 == v24 && v15 == v25 )
  {
    v26 = *(_WORD *)(v12 + 2);
  }
  else
  {
    if ( v13 != v25 || v15 != v24 )
      return 0;
    v26 = *(_WORD *)(v12 + 2);
    if ( v13 != v24 )
    {
      v45 = v25;
      v49 = v24;
      v53 = v10;
      v27 = sub_B52870(v26 & 0x3F);
      v25 = v45;
      v24 = v49;
      v10 = v53;
      goto LABEL_62;
    }
  }
  v27 = v26 & 0x3F;
LABEL_62:
  if ( (unsigned int)(v27 - 36) > 1 || (v8 != v24 || v10 != v25) && (v8 != v25 || v10 != v24) )
    return 0;
LABEL_67:
  result = a2;
  if ( a1 != v9 )
  {
    v28 = sub_9905C0(a1);
    result = a3;
    if ( v28 != v9 )
      return 0;
  }
  return result;
}
