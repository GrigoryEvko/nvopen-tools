// Function: sub_18B80E0
// Address: 0x18b80e0
//
_QWORD *__fastcall sub_18B80E0(_QWORD *a1, _QWORD *a2, __int64 *a3)
{
  _QWORD *v6; // rax
  __int64 v7; // rdx
  _QWORD *v8; // r12
  unsigned __int64 v9; // r14
  __int64 v10; // rax
  char *v11; // r13
  signed __int64 v12; // r8
  char *v13; // rcx
  _QWORD *v14; // r9
  _QWORD *v15; // r10
  char *v16; // rdx
  char *v17; // rdi
  signed __int64 v18; // rsi
  char *v19; // r11
  char *v20; // rax
  __int64 v21; // rax
  char *v22; // rsi
  _QWORD *v23; // r11
  char *v24; // rax
  char *v25; // rdx
  _QWORD *v26; // rax
  _QWORD *v27; // rax
  _QWORD *v28; // rdx
  bool v29; // al
  _BOOL8 v30; // rdi
  char *v32; // r11
  char *v33; // rdi
  char *v34; // rax
  __int64 v35; // rax
  char *v36; // rdi
  char *v37; // rdx
  char *v38; // r10
  char *v39; // rax
  char *v40; // rsi
  char *v41; // rax
  char *v42; // rdx
  _QWORD *v43; // rax
  char *v44; // [rsp+0h] [rbp-70h]
  char *v45; // [rsp+8h] [rbp-68h]
  signed __int64 v46; // [rsp+10h] [rbp-60h]
  char *v47; // [rsp+20h] [rbp-50h]
  char *v48; // [rsp+20h] [rbp-50h]
  char *v49; // [rsp+20h] [rbp-50h]
  signed __int64 v50; // [rsp+28h] [rbp-48h]
  signed __int64 v51; // [rsp+28h] [rbp-48h]
  __int64 v52; // [rsp+30h] [rbp-40h]
  __int64 v53; // [rsp+38h] [rbp-38h]
  size_t v54; // [rsp+38h] [rbp-38h]
  char *v55; // [rsp+38h] [rbp-38h]
  _QWORD *v56; // [rsp+38h] [rbp-38h]
  _QWORD *v57; // [rsp+38h] [rbp-38h]
  _QWORD *v58; // [rsp+38h] [rbp-38h]

  v6 = (_QWORD *)sub_22077B0(112);
  v7 = *a3;
  v8 = v6;
  v9 = *(_QWORD *)(*a3 + 8) - *(_QWORD *)*a3;
  v52 = (__int64)(v6 + 4);
  v6[4] = 0;
  v6[5] = 0;
  v6[6] = 0;
  if ( v9 )
  {
    if ( v9 > 0x7FFFFFFFFFFFFFF8LL )
      sub_4261EA(112, a2, v7);
    v53 = v7;
    v10 = sub_22077B0(v9);
    v7 = v53;
    v11 = (char *)v10;
  }
  else
  {
    v9 = 0;
    v11 = 0;
  }
  v8[4] = v11;
  v8[5] = v11;
  v8[6] = &v11[v9];
  v12 = *(_QWORD *)(v7 + 8) - *(_QWORD *)v7;
  if ( *(_QWORD *)(v7 + 8) != *(_QWORD *)v7 )
  {
    v54 = *(_QWORD *)(v7 + 8) - *(_QWORD *)v7;
    memmove(v11, *(const void **)v7, v54);
    v12 = v54;
  }
  v13 = &v11[v12];
  v14 = a1 + 1;
  v15 = a2;
  *(_OWORD *)(v8 + 9) = 0;
  v8[5] = &v11[v12];
  v8[13] = 0;
  *((_BYTE *)v8 + 80) = 1;
  *(_OWORD *)(v8 + 7) = 0;
  *(_OWORD *)(v8 + 11) = 0;
  if ( a1 + 1 == a2 )
  {
    if ( a1[5] )
    {
      v23 = (_QWORD *)a1[4];
      v40 = (char *)v23[5];
      v41 = (char *)v23[4];
      if ( v12 < v40 - v41 )
        v40 = &v41[v12];
      v42 = v11;
      if ( v41 != v40 )
      {
        while ( *(_QWORD *)v41 >= *(_QWORD *)v42 )
        {
          if ( *(_QWORD *)v41 > *(_QWORD *)v42 )
            goto LABEL_27;
          v41 += 8;
          v42 += 8;
          if ( v40 == v41 )
            goto LABEL_77;
        }
LABEL_64:
        v29 = 0;
        goto LABEL_30;
      }
LABEL_77:
      if ( v13 != v42 )
      {
        v29 = 0;
LABEL_30:
        if ( v14 == v23 || v29 )
          goto LABEL_32;
        v37 = (char *)v23[4];
        v55 = (char *)v23[5];
        v18 = v55 - v37;
        goto LABEL_66;
      }
    }
LABEL_27:
    v48 = v13;
    v51 = v12;
    v56 = v14;
    v27 = sub_18B5D90((__int64)a1, v52);
    v14 = v56;
    v12 = v51;
    v13 = v48;
    v15 = v27;
    v23 = v28;
    goto LABEL_28;
  }
  v16 = (char *)a2[4];
  v55 = (char *)a2[5];
  v17 = v16;
  v18 = v55 - v16;
  v19 = &v11[v55 - v16];
  if ( v12 <= v55 - v16 )
    v19 = &v11[v12];
  if ( v11 != v19 )
  {
    v20 = v11;
    while ( *(_QWORD *)v20 >= *(_QWORD *)v17 )
    {
      if ( *(_QWORD *)v20 > *(_QWORD *)v17 )
        goto LABEL_39;
      v20 += 8;
      v17 += 8;
      if ( v19 == v20 )
        goto LABEL_38;
    }
    goto LABEL_14;
  }
LABEL_38:
  if ( v55 != v17 )
  {
LABEL_14:
    if ( (_QWORD *)a1[3] == a2 )
    {
LABEL_23:
      v26 = a2;
LABEL_24:
      v23 = a2;
      v15 = v26;
LABEL_29:
      v29 = v15 != 0;
      goto LABEL_30;
    }
    v47 = &v11[v12];
    v50 = v12;
    v21 = sub_220EF80(a2);
    v12 = v50;
    v14 = a1 + 1;
    v22 = *(char **)(v21 + 40);
    v23 = (_QWORD *)v21;
    v24 = *(char **)(v21 + 32);
    v13 = v47;
    if ( v50 < v22 - v24 )
      v22 = &v24[v50];
    v25 = v11;
    if ( v24 != v22 )
    {
      while ( *(_QWORD *)v24 >= *(_QWORD *)v25 )
      {
        if ( *(_QWORD *)v24 > *(_QWORD *)v25 )
          goto LABEL_27;
        v24 += 8;
        v25 += 8;
        if ( v22 == v24 )
          goto LABEL_74;
      }
LABEL_22:
      if ( v23[3] )
        goto LABEL_23;
      goto LABEL_64;
    }
LABEL_74:
    if ( v47 != v25 )
      goto LABEL_22;
    goto LABEL_27;
  }
LABEL_39:
  v32 = &v16[v12];
  if ( v12 >= v18 )
    v32 = (char *)a2[5];
  v33 = v11;
  if ( v16 == v32 )
  {
LABEL_34:
    if ( v13 == v33 )
      goto LABEL_35;
  }
  else
  {
    v34 = (char *)a2[4];
    while ( *(_QWORD *)v34 >= *(_QWORD *)v33 )
    {
      if ( *(_QWORD *)v34 > *(_QWORD *)v33 )
        goto LABEL_35;
      v34 += 8;
      v33 += 8;
      if ( v32 == v34 )
        goto LABEL_34;
    }
  }
  v49 = (char *)a2[4];
  if ( (_QWORD *)a1[4] == a2 )
  {
    v26 = 0;
    goto LABEL_24;
  }
  v45 = &v11[v12];
  v46 = v12;
  v35 = sub_220EEE0(a2);
  v12 = v46;
  v13 = v45;
  v23 = (_QWORD *)v35;
  v14 = a1 + 1;
  v36 = *(char **)(v35 + 32);
  v44 = *(char **)(v35 + 40);
  v37 = v49;
  v38 = &v11[v44 - v36];
  if ( v46 <= v44 - v36 )
    v38 = v45;
  if ( v11 != v38 )
  {
    v39 = v11;
    while ( *(_QWORD *)v39 >= *(_QWORD *)v36 )
    {
      if ( *(_QWORD *)v39 > *(_QWORD *)v36 )
        goto LABEL_81;
      v39 += 8;
      v36 += 8;
      if ( v38 == v39 )
        goto LABEL_80;
    }
    goto LABEL_54;
  }
LABEL_80:
  if ( v44 != v36 )
  {
LABEL_54:
    if ( a2[3] )
      goto LABEL_32;
    v23 = a2;
LABEL_66:
    if ( v12 > v18 )
      v13 = &v11[v18];
    if ( v11 == v13 )
    {
LABEL_79:
      v30 = v37 != v55;
      goto LABEL_33;
    }
    while ( *(_QWORD *)v11 >= *(_QWORD *)v37 )
    {
      if ( *(_QWORD *)v11 > *(_QWORD *)v37 )
      {
        v30 = 0;
        goto LABEL_33;
      }
      v11 += 8;
      v37 += 8;
      if ( v13 == v11 )
        goto LABEL_79;
    }
LABEL_32:
    v30 = 1;
LABEL_33:
    sub_220F040(v30, v8, v23, v14);
    ++a1[5];
    return v8;
  }
LABEL_81:
  v43 = sub_18B5D90((__int64)a1, v52);
  v13 = v45;
  v12 = v46;
  v14 = a1 + 1;
  v15 = v43;
  v23 = v28;
LABEL_28:
  if ( v28 )
    goto LABEL_29;
LABEL_35:
  if ( v11 )
  {
    v57 = v15;
    j_j___libc_free_0(v11, v9);
    v15 = v57;
  }
  v58 = v15;
  j_j___libc_free_0(v8, 112);
  return v58;
}
