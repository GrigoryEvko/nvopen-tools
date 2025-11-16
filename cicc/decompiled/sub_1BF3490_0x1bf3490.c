// Function: sub_1BF3490
// Address: 0x1bf3490
//
__int64 __fastcall sub_1BF3490(__int64 *a1, __int64 a2)
{
  __int64 *v3; // rbx
  __int64 v4; // rax
  __int64 v5; // r14
  _QWORD *v6; // rax
  _QWORD *v7; // rbx
  int v8; // r13d
  _QWORD *v9; // r15
  _QWORD *v10; // rax
  __int64 v11; // rcx
  __int64 v12; // rsi
  __int64 v13; // rbx
  _QWORD *v14; // r15
  char *v15; // rax
  _QWORD *v16; // rbx
  _QWORD *v17; // r13
  _QWORD *v18; // rdi
  __int64 v19; // rbx
  __int64 v20; // rbx
  _QWORD *v21; // r13
  char *v22; // rax
  _QWORD *v23; // rbx
  _QWORD *v24; // r12
  _QWORD *v25; // rdi
  __int64 v27; // rbx
  _QWORD *v28; // r15
  char *v29; // rax
  _QWORD *v30; // rbx
  _QWORD *v31; // r13
  _QWORD *v32; // rdi
  __int64 v33; // rbx
  _QWORD *v34; // r15
  char *v35; // rax
  _QWORD *v36; // rbx
  _QWORD *v37; // r13
  _QWORD *v38; // rdi
  _QWORD *v39; // rcx
  __int64 v40; // rax
  __int64 v41; // r13
  char v43; // [rsp+Eh] [rbp-222h]
  unsigned __int8 v44; // [rsp+Fh] [rbp-221h]
  __int64 v45; // [rsp+10h] [rbp-220h]
  __int64 v46; // [rsp+18h] [rbp-218h]
  _QWORD v47[11]; // [rsp+20h] [rbp-210h] BYREF
  _QWORD *v48; // [rsp+78h] [rbp-1B8h]
  unsigned int v49; // [rsp+80h] [rbp-1B0h]
  _BYTE v50[424]; // [rsp+88h] [rbp-1A8h] BYREF

  v3 = (__int64 *)a1[7];
  v4 = sub_15E0530(*v3);
  v43 = 1;
  if ( !sub_1602790(v4) )
  {
    v40 = sub_15E0530(*v3);
    v41 = sub_16033E0(v40);
    if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, __int64))(*(_QWORD *)v41 + 32LL))(
           v41,
           "loop-vectorize",
           14)
      || (*(unsigned __int8 (__fastcall **)(__int64, const char *, __int64))(*(_QWORD *)v41 + 40LL))(
           v41,
           "loop-vectorize",
           14)
      || (v43 = (*(__int64 (__fastcall **)(__int64, const char *, __int64))(*(_QWORD *)v41 + 24LL))(
                  v41,
                  "loop-vectorize",
                  14)) != 0 )
    {
      v43 = 1;
    }
  }
  v44 = 1;
  if ( !sub_13FC520(a2) )
  {
    v33 = *a1;
    v34 = (_QWORD *)a1[7];
    v35 = sub_1BF18B0(a1[58]);
    sub_1BF1750((__int64)v47, (__int64)v35, (__int64)"CFGNotUnderstood", 16, v33, 0);
    sub_15CAB20((__int64)v47, "loop control flow is not understood by vectorizer", 0x31u);
    sub_143AA50(v34, (__int64)v47);
    v36 = v48;
    v47[0] = &unk_49ECF68;
    v37 = &v48[11 * v49];
    if ( v48 != v37 )
    {
      do
      {
        v37 -= 11;
        v38 = (_QWORD *)v37[4];
        if ( v38 != v37 + 6 )
          j_j___libc_free_0(v38, v37[6] + 1LL);
        if ( (_QWORD *)*v37 != v37 + 2 )
          j_j___libc_free_0(*v37, v37[2] + 1LL);
      }
      while ( v36 != v37 );
      v37 = v48;
    }
    if ( v37 != (_QWORD *)v50 )
      _libc_free((unsigned __int64)v37);
    v44 = 0;
    if ( !v43 )
      return 0;
  }
  v5 = *(_QWORD *)(**(_QWORD **)(a2 + 32) + 8LL);
  if ( !v5 )
  {
LABEL_25:
    v13 = *a1;
    v14 = (_QWORD *)a1[7];
    v15 = sub_1BF18B0(a1[58]);
    sub_1BF1750((__int64)v47, (__int64)v15, (__int64)"CFGNotUnderstood", 16, v13, 0);
    sub_15CAB20((__int64)v47, "loop control flow is not understood by vectorizer", 0x31u);
    sub_143AA50(v14, (__int64)v47);
    v16 = v48;
    v47[0] = &unk_49ECF68;
    v17 = &v48[11 * v49];
    if ( v48 != v17 )
    {
      do
      {
        v17 -= 11;
        v18 = (_QWORD *)v17[4];
        if ( v18 != v17 + 6 )
          j_j___libc_free_0(v18, v17[6] + 1LL);
        if ( (_QWORD *)*v17 != v17 + 2 )
          j_j___libc_free_0(*v17, v17[2] + 1LL);
      }
      while ( v16 != v17 );
      v17 = v48;
    }
    if ( v17 != (_QWORD *)v50 )
      _libc_free((unsigned __int64)v17);
    if ( !v43 )
      return 0;
    v44 = 0;
    if ( sub_13F9E70(a2) )
      goto LABEL_36;
    goto LABEL_51;
  }
  while ( 1 )
  {
    v6 = sub_1648700(v5);
    if ( (unsigned __int8)(*((_BYTE *)v6 + 16) - 25) <= 9u )
      break;
    v5 = *(_QWORD *)(v5 + 8);
    if ( !v5 )
      goto LABEL_25;
  }
  v7 = *(_QWORD **)(a2 + 72);
  v8 = 0;
  v45 = a2 + 56;
LABEL_12:
  v11 = v6[5];
  v10 = *(_QWORD **)(a2 + 64);
  if ( v7 == v10 )
  {
    v12 = *(unsigned int *)(a2 + 84);
    v9 = &v7[v12];
    if ( v7 == v9 )
    {
      v39 = v7;
    }
    else
    {
      do
      {
        if ( v11 == *v10 )
          break;
        ++v10;
      }
      while ( v9 != v10 );
      v39 = &v7[v12];
    }
    goto LABEL_20;
  }
  v46 = v11;
  v9 = &v7[*(unsigned int *)(a2 + 80)];
  v10 = sub_16CC9F0(v45, v11);
  if ( v46 == *v10 )
  {
    v7 = *(_QWORD **)(a2 + 72);
    if ( v7 == *(_QWORD **)(a2 + 64) )
      v39 = &v7[*(unsigned int *)(a2 + 84)];
    else
      v39 = &v7[*(unsigned int *)(a2 + 80)];
LABEL_20:
    while ( v39 != v10 && *v10 >= 0xFFFFFFFFFFFFFFFELL )
      ++v10;
    goto LABEL_9;
  }
  v7 = *(_QWORD **)(a2 + 72);
  if ( v7 == *(_QWORD **)(a2 + 64) )
  {
    v10 = &v7[*(unsigned int *)(a2 + 84)];
    v39 = v10;
    goto LABEL_20;
  }
  v10 = &v7[*(unsigned int *)(a2 + 80)];
LABEL_9:
  v8 += v9 != v10;
  while ( 1 )
  {
    v5 = *(_QWORD *)(v5 + 8);
    if ( !v5 )
      break;
    v6 = sub_1648700(v5);
    if ( (unsigned __int8)(*((_BYTE *)v6 + 16) - 25) <= 9u )
      goto LABEL_12;
  }
  if ( v8 != 1 )
    goto LABEL_25;
  if ( sub_13F9E70(a2) )
    goto LABEL_36;
LABEL_51:
  v27 = *a1;
  v28 = (_QWORD *)a1[7];
  v29 = sub_1BF18B0(a1[58]);
  sub_1BF1750((__int64)v47, (__int64)v29, (__int64)"CFGNotUnderstood", 16, v27, 0);
  sub_15CAB20((__int64)v47, "loop control flow is not understood by vectorizer", 0x31u);
  sub_143AA50(v28, (__int64)v47);
  v30 = v48;
  v47[0] = &unk_49ECF68;
  v31 = &v48[11 * v49];
  if ( v48 != v31 )
  {
    do
    {
      v31 -= 11;
      v32 = (_QWORD *)v31[4];
      if ( v32 != v31 + 6 )
        j_j___libc_free_0(v32, v31[6] + 1LL);
      if ( (_QWORD *)*v31 != v31 + 2 )
        j_j___libc_free_0(*v31, v31[2] + 1LL);
    }
    while ( v30 != v31 );
    v31 = v48;
  }
  if ( v31 != (_QWORD *)v50 )
    _libc_free((unsigned __int64)v31);
  if ( !v43 )
    return 0;
  v44 = 0;
LABEL_36:
  v19 = sub_13F9E70(a2);
  if ( v19 != sub_13FCB50(a2) )
  {
    v20 = *a1;
    v21 = (_QWORD *)a1[7];
    v22 = sub_1BF18B0(a1[58]);
    sub_1BF1750((__int64)v47, (__int64)v22, (__int64)"CFGNotUnderstood", 16, v20, 0);
    sub_15CAB20((__int64)v47, "loop control flow is not understood by vectorizer", 0x31u);
    sub_143AA50(v21, (__int64)v47);
    v23 = v48;
    v47[0] = &unk_49ECF68;
    v24 = &v48[11 * v49];
    if ( v48 != v24 )
    {
      do
      {
        v24 -= 11;
        v25 = (_QWORD *)v24[4];
        if ( v25 != v24 + 6 )
          j_j___libc_free_0(v25, v24[6] + 1LL);
        if ( (_QWORD *)*v24 != v24 + 2 )
          j_j___libc_free_0(*v24, v24[2] + 1LL);
      }
      while ( v23 != v24 );
      v24 = v48;
    }
    if ( v24 != (_QWORD *)v50 )
      _libc_free((unsigned __int64)v24);
    return 0;
  }
  return v44;
}
