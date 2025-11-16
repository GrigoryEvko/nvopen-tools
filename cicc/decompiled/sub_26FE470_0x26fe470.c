// Function: sub_26FE470
// Address: 0x26fe470
//
_QWORD *__fastcall sub_26FE470(_QWORD *a1, unsigned __int8 *a2)
{
  _QWORD *v2; // r13
  __int64 v4; // rax
  bool v5; // zf
  int v7; // edx
  __int64 v8; // r13
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // r14
  int v12; // r14d
  __int64 v13; // rax
  __int64 v14; // rdx
  unsigned __int64 v15; // rdi
  int v16; // edx
  __int64 v17; // r13
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // r14
  __int64 v21; // rdx
  int v22; // r14d
  __int64 v23; // rax
  __int64 v24; // rdx
  unsigned __int8 *v25; // r13
  unsigned __int8 *i; // rbx
  _BYTE *v27; // rax
  __int64 v28; // rax
  _BYTE *v29; // r8
  _QWORD *v30; // r8
  _QWORD *v31; // rbx
  _BYTE *v32; // r11
  _QWORD *v33; // rsi
  _BYTE *v34; // r9
  char *v35; // rcx
  char *v36; // rax
  _BYTE *v37; // rdx
  _QWORD *v38; // rdx
  __int64 v39; // rax
  _QWORD *v40; // rax
  unsigned __int64 v41; // rax
  unsigned __int64 *v42; // [rsp+8h] [rbp-48h] BYREF
  unsigned __int64 v43; // [rsp+10h] [rbp-40h] BYREF
  _BYTE *v44; // [rsp+18h] [rbp-38h]
  _BYTE *v45; // [rsp+20h] [rbp-30h]

  v2 = a1;
  v4 = *((_QWORD *)a2 + 1);
  v44 = 0;
  v45 = 0;
  v5 = *(_BYTE *)(v4 + 8) == 12;
  v43 = 0;
  if ( !v5 || *(_DWORD *)(v4 + 8) > 0x40FFu )
    return v2;
  v7 = *a2;
  if ( v7 == 40 )
  {
    v8 = -32 - 32LL * (unsigned int)sub_B491D0((__int64)a2);
  }
  else
  {
    v8 = -32;
    if ( v7 != 85 )
    {
      v8 = -96;
      if ( v7 != 34 )
LABEL_72:
        BUG();
    }
  }
  if ( (a2[7] & 0x80u) == 0 )
    goto LABEL_12;
  v9 = sub_BD2BC0((__int64)a2);
  v11 = v9 + v10;
  if ( (a2[7] & 0x80u) == 0 )
  {
    if ( (unsigned int)(v11 >> 4) )
      goto LABEL_69;
  }
  else if ( (unsigned int)((v11 - sub_BD2BC0((__int64)a2)) >> 4) )
  {
    if ( (a2[7] & 0x80u) != 0 )
    {
      v12 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
      if ( (a2[7] & 0x80u) == 0 )
        BUG();
      v13 = sub_BD2BC0((__int64)a2);
      v8 -= 32LL * (unsigned int)(*(_DWORD *)(v13 + v14 - 4) - v12);
      goto LABEL_12;
    }
LABEL_69:
    BUG();
  }
LABEL_12:
  if ( !(32LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF) + v8) )
  {
    v15 = v43;
    v2 = a1;
    goto LABEL_14;
  }
  v16 = *a2;
  if ( v16 == 40 )
  {
    v17 = -32 - 32LL * (unsigned int)sub_B491D0((__int64)a2);
  }
  else
  {
    v17 = -32;
    if ( v16 != 85 )
    {
      v17 = -96;
      if ( v16 != 34 )
        goto LABEL_72;
    }
  }
  if ( (a2[7] & 0x80u) != 0 )
  {
    v18 = sub_BD2BC0((__int64)a2);
    v20 = v18 + v19;
    v21 = 0;
    if ( (a2[7] & 0x80u) != 0 )
      v21 = sub_BD2BC0((__int64)a2);
    if ( (unsigned int)((v20 - v21) >> 4) )
    {
      if ( (a2[7] & 0x80u) == 0 )
        BUG();
      v22 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
      if ( (a2[7] & 0x80u) == 0 )
        BUG();
      v23 = sub_BD2BC0((__int64)a2);
      v17 -= 32LL * (unsigned int)(*(_DWORD *)(v23 + v24 - 4) - v22);
    }
  }
  v25 = &a2[v17];
  for ( i = &a2[32 * (1LL - (*((_DWORD *)a2 + 1) & 0x7FFFFFF))]; v25 != i; i += 32 )
  {
    v27 = *(_BYTE **)i;
    if ( **(_BYTE **)i != 17 || *((_DWORD *)v27 + 8) > 0x40u )
    {
      v15 = v43;
      v2 = a1;
      goto LABEL_14;
    }
    v28 = *((_QWORD *)v27 + 3);
    v29 = v44;
    v42 = (unsigned __int64 *)v28;
    if ( v44 == v45 )
    {
      sub_A235E0((__int64)&v43, v44, &v42);
    }
    else
    {
      if ( v44 )
      {
        *(_QWORD *)v44 = v28;
        v29 = v44;
      }
      v44 = v29 + 8;
    }
  }
  v30 = (_QWORD *)a1[12];
  v31 = a1 + 11;
  if ( !v30 )
  {
    v33 = a1 + 11;
    goto LABEL_58;
  }
  v32 = v44;
  v15 = v43;
  v33 = a1 + 11;
  v34 = &v44[-v43];
  do
  {
    v35 = (char *)v30[5];
    v36 = (char *)v30[4];
    if ( v35 - v36 > (__int64)v34 )
      v35 = &v34[(_QWORD)v36];
    v37 = (_BYTE *)v43;
    if ( v36 != v35 )
    {
      while ( *(_QWORD *)v36 >= *(_QWORD *)v37 )
      {
        if ( *(_QWORD *)v36 > *(_QWORD *)v37 )
          goto LABEL_64;
        v36 += 8;
        v37 += 8;
        if ( v35 == v36 )
          goto LABEL_63;
      }
LABEL_48:
      v30 = (_QWORD *)v30[3];
      continue;
    }
LABEL_63:
    if ( v44 != v37 )
      goto LABEL_48;
LABEL_64:
    v33 = v30;
    v30 = (_QWORD *)v30[2];
  }
  while ( v30 );
  if ( v31 == v33 )
    goto LABEL_58;
  v38 = (_QWORD *)v33[4];
  v39 = v33[5] - (_QWORD)v38;
  if ( (__int64)v34 > v39 )
    v32 = (_BYTE *)(v43 + v39);
  if ( (_BYTE *)v43 == v32 )
  {
LABEL_66:
    if ( v38 != (_QWORD *)v33[5] )
      goto LABEL_58;
  }
  else
  {
    v40 = (_QWORD *)v43;
    while ( *v40 >= *v38 )
    {
      if ( *v40 > *v38 )
        goto LABEL_59;
      ++v40;
      ++v38;
      if ( v32 == (_BYTE *)v40 )
        goto LABEL_66;
    }
LABEL_58:
    v42 = &v43;
    v41 = sub_26F9D50(a1 + 10, v33, (__int64 *)&v42);
    v15 = v43;
    v33 = (_QWORD *)v41;
  }
LABEL_59:
  v2 = v33 + 7;
LABEL_14:
  if ( !v15 )
    return v2;
  j_j___libc_free_0(v15);
  return v2;
}
