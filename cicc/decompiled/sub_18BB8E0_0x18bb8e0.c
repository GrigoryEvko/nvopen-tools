// Function: sub_18BB8E0
// Address: 0x18bb8e0
//
_QWORD *__fastcall sub_18BB8E0(_QWORD *a1, __int64 a2)
{
  _QWORD *v2; // r12
  __int64 v3; // rax
  unsigned __int64 v6; // rax
  _BYTE *v7; // rdi
  _BYTE *v8; // rsi
  unsigned __int64 v9; // r13
  unsigned __int64 i; // r12
  __int64 v11; // rax
  __int64 v12; // rax
  _BYTE *v13; // r8
  _QWORD *v14; // rsi
  _QWORD *v15; // r11
  _BYTE *v16; // r10
  _QWORD *v17; // r12
  _BYTE *v18; // r8
  char *v19; // rcx
  char *v20; // rax
  _BYTE *v21; // rdx
  _QWORD *v22; // rdx
  __int64 v23; // rax
  _QWORD *v24; // rax
  _QWORD *v25; // rax
  _QWORD v26[2]; // [rsp+8h] [rbp-58h] BYREF
  _QWORD *v27; // [rsp+18h] [rbp-48h] BYREF
  _BYTE *v28; // [rsp+20h] [rbp-40h] BYREF
  _BYTE *v29; // [rsp+28h] [rbp-38h]
  _BYTE *v30; // [rsp+30h] [rbp-30h]

  v2 = a1;
  v29 = 0;
  v30 = 0;
  v26[0] = a2;
  v28 = 0;
  v3 = *(_QWORD *)(a2 & 0xFFFFFFFFFFFFFFF8LL);
  if ( *(_BYTE *)(v3 + 8) != 11 || *(_DWORD *)(v3 + 8) > 0x40FFu )
    return v2;
  v6 = sub_1389B50(v26);
  if ( v6 == (v26[0] & 0xFFFFFFFFFFFFFFF8LL) - 24LL * (*(_DWORD *)((v26[0] & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF) )
  {
    v7 = v28;
    v8 = (_BYTE *)(v30 - v28);
    goto LABEL_6;
  }
  v9 = sub_1389B50(v26);
  for ( i = (v26[0] & 0xFFFFFFFFFFFFFFF8LL)
          + 24 * (1LL - (*(_DWORD *)((v26[0] & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF)); v9 != i; i += 24LL )
  {
    v11 = *(_QWORD *)i;
    if ( *(_BYTE *)(*(_QWORD *)i + 16LL) != 13 || *(_DWORD *)(v11 + 32) > 0x40u )
    {
      v7 = v28;
      v2 = a1;
      v8 = (_BYTE *)(v30 - v28);
      goto LABEL_6;
    }
    v12 = *(_QWORD *)(v11 + 24);
    v13 = v29;
    v27 = (_QWORD *)v12;
    if ( v29 == v30 )
    {
      sub_A235E0((__int64)&v28, v29, &v27);
    }
    else
    {
      if ( v29 )
      {
        *(_QWORD *)v29 = v12;
        v13 = v29;
      }
      v29 = v13 + 8;
    }
  }
  v14 = (_QWORD *)a1[9];
  v15 = a1 + 8;
  if ( !v14 )
  {
    v17 = a1 + 8;
    goto LABEL_35;
  }
  v16 = v29;
  v7 = v28;
  v17 = a1 + 8;
  v18 = (_BYTE *)(v29 - v28);
  do
  {
    v19 = (char *)v14[5];
    v20 = (char *)v14[4];
    if ( v19 - v20 > (__int64)v18 )
      v19 = &v18[(_QWORD)v20];
    v21 = v28;
    if ( v20 != v19 )
    {
      while ( *(_QWORD *)v20 >= *(_QWORD *)v21 )
      {
        if ( *(_QWORD *)v20 > *(_QWORD *)v21 )
          goto LABEL_39;
        v20 += 8;
        v21 += 8;
        if ( v19 == v20 )
          goto LABEL_38;
      }
LABEL_25:
      v14 = (_QWORD *)v14[3];
      continue;
    }
LABEL_38:
    if ( v21 != v29 )
      goto LABEL_25;
LABEL_39:
    v17 = v14;
    v14 = (_QWORD *)v14[2];
  }
  while ( v14 );
  if ( v15 == v17 )
    goto LABEL_35;
  v22 = (_QWORD *)v17[4];
  v23 = v17[5] - (_QWORD)v22;
  if ( (__int64)v18 > v23 )
    v16 = &v28[v23];
  if ( v28 == v16 )
  {
LABEL_41:
    if ( (_QWORD *)v17[5] != v22 )
      goto LABEL_35;
  }
  else
  {
    v24 = v28;
    while ( *v24 >= *v22 )
    {
      if ( *v24 > *v22 )
        goto LABEL_36;
      ++v24;
      ++v22;
      if ( v16 == (_BYTE *)v24 )
        goto LABEL_41;
    }
LABEL_35:
    v27 = &v28;
    v25 = sub_18B80E0(a1 + 7, v17, (__int64 *)&v27);
    v7 = v28;
    v17 = v25;
  }
LABEL_36:
  v2 = v17 + 7;
  v8 = (_BYTE *)(v30 - v7);
LABEL_6:
  if ( !v7 )
    return v2;
  j_j___libc_free_0(v7, v8);
  return v2;
}
