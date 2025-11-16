// Function: sub_25104F0
// Address: 0x25104f0
//
_QWORD *__fastcall sub_25104F0(_QWORD *a1, _QWORD *a2)
{
  _QWORD *v2; // r13
  _QWORD *v5; // r14
  _BYTE *v6; // r8
  __int64 v7; // r10
  _BYTE *v8; // r11
  __int64 v9; // rdi
  _BYTE *v10; // rsi
  _BYTE *v11; // rcx
  _BYTE *v12; // r9
  _BYTE *v13; // rdx
  _BYTE *v14; // rax
  _QWORD *v15; // rax
  char v16; // dl
  _QWORD *v17; // r15
  _BOOL4 v18; // r9d
  __int64 v19; // rax
  unsigned __int64 v20; // r8
  __int64 v21; // r9
  _QWORD *v22; // r14
  void *v23; // rax
  __int64 v25; // rax
  _BYTE *v26; // rsi
  size_t v27; // rdx
  __int64 v28; // rdx
  __int64 v29; // rdi
  _BYTE *v30; // rax
  bool v31; // cc
  _BYTE *v32; // r8
  _BYTE *v33; // rcx
  _BYTE *v34; // rdx
  __int64 v35; // rax
  char v36; // [rsp+4h] [rbp-3Ch]
  bool v37; // [rsp+4h] [rbp-3Ch]
  _BOOL4 v38; // [rsp+8h] [rbp-38h]
  unsigned __int64 v39; // [rsp+8h] [rbp-38h]
  unsigned __int64 v40; // [rsp+8h] [rbp-38h]

  v2 = a1 + 1;
  v5 = (_QWORD *)a1[2];
  if ( !v5 )
  {
    v5 = a1 + 1;
    goto LABEL_27;
  }
  v6 = (_BYTE *)*a2;
  v7 = a2[1];
  v8 = (_BYTE *)(*a2 + v7);
  while ( 1 )
  {
    v9 = v5[5];
    v10 = (_BYTE *)v5[4];
    v11 = &v6[v9];
    v12 = &v10[v9];
    v13 = v10;
    if ( v9 >= v7 )
      v11 = v8;
    if ( v6 == v11 )
      break;
    v14 = v6;
    while ( *v14 >= *v13 )
    {
      if ( *v14 > *v13 )
        goto LABEL_13;
      ++v14;
      ++v13;
      if ( v11 == v14 )
        goto LABEL_12;
    }
LABEL_10:
    v15 = (_QWORD *)v5[2];
    v16 = 1;
    if ( !v15 )
      goto LABEL_14;
LABEL_11:
    v5 = v15;
  }
LABEL_12:
  if ( v12 != v13 )
    goto LABEL_10;
LABEL_13:
  v15 = (_QWORD *)v5[3];
  v16 = 0;
  if ( v15 )
    goto LABEL_11;
LABEL_14:
  v17 = v5;
  if ( !v16 )
    goto LABEL_15;
LABEL_27:
  if ( (_QWORD *)a1[3] == v5 )
  {
    v17 = v5;
    v18 = 1;
    if ( v2 == v5 )
      goto LABEL_24;
LABEL_37:
    v28 = a2[1];
    v29 = v17[5];
    v30 = (_BYTE *)*a2;
    v31 = v28 <= v29;
    v32 = (_BYTE *)(*a2 + v28);
    v33 = (_BYTE *)(*a2 + v29);
    v34 = (_BYTE *)v17[4];
    if ( v31 )
      v33 = v32;
    if ( v30 == v33 )
    {
LABEL_48:
      v18 = v34 != (_BYTE *)(v29 + v17[4]);
    }
    else
    {
      while ( 1 )
      {
        if ( *v30 < *v34 )
        {
          v18 = 1;
          goto LABEL_24;
        }
        if ( *v30 > *v34 )
          break;
        ++v30;
        ++v34;
        if ( v33 == v30 )
          goto LABEL_48;
      }
      v18 = 0;
    }
LABEL_24:
    v38 = v18;
    v19 = sub_22077B0(0x58u);
    v20 = a2[1];
    v21 = v38;
    v22 = (_QWORD *)v19;
    v23 = (void *)(v19 + 56);
    v22[4] = v23;
    v22[5] = 0;
    v22[6] = 32;
    if ( v20 && v22 + 4 != a2 )
    {
      v26 = a2 + 3;
      if ( (_QWORD *)*a2 == a2 + 3 )
      {
        v27 = v20;
        if ( v20 <= 0x20 )
          goto LABEL_34;
        v37 = v38;
        v40 = v20;
        sub_C8D290((__int64)(v22 + 4), v23, v20, 1u, v20, v21);
        v27 = a2[1];
        v23 = (void *)v22[4];
        v26 = (_BYTE *)*a2;
        v20 = v40;
        LOBYTE(v21) = v37;
        if ( v27 )
        {
LABEL_34:
          v36 = v21;
          v39 = v20;
          memcpy(v23, v26, v27);
          LOBYTE(v21) = v36;
          v20 = v39;
        }
        v22[5] = v20;
        a2[1] = 0;
      }
      else
      {
        v35 = a2[2];
        v22[4] = *a2;
        v22[5] = v20;
        v22[6] = v35;
        *a2 = v26;
        a2[2] = 0;
        a2[1] = 0;
      }
    }
    sub_220F040(v21, (__int64)v22, v17, v2);
    ++a1[5];
    return v22;
  }
  else
  {
    v17 = v5;
    v25 = sub_220EF80((__int64)v5);
    v6 = (_BYTE *)*a2;
    v7 = a2[1];
    v10 = *(_BYTE **)(v25 + 32);
    v9 = *(_QWORD *)(v25 + 40);
    v5 = (_QWORD *)v25;
    v8 = (_BYTE *)(*a2 + v7);
    v12 = &v10[v9];
LABEL_15:
    if ( v7 < v9 )
      v12 = &v10[v7];
    if ( v10 == v12 )
    {
LABEL_29:
      if ( v6 == v8 )
        return v5;
    }
    else
    {
      while ( *v10 >= *v6 )
      {
        if ( *v10 > *v6 )
          return v5;
        ++v10;
        ++v6;
        if ( v12 == v10 )
          goto LABEL_29;
      }
    }
    if ( v17 )
    {
      v18 = 1;
      if ( v2 == v17 )
        goto LABEL_24;
      goto LABEL_37;
    }
    return 0;
  }
}
