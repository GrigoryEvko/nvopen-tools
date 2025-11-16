// Function: sub_1E2D160
// Address: 0x1e2d160
//
__int64 __fastcall sub_1E2D160(char **a1, char *a2, _QWORD *a3)
{
  char *v4; // r15
  char *v5; // r12
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rdi
  char *v8; // r8
  char *v9; // r14
  bool v10; // cf
  unsigned __int64 v11; // rax
  signed __int64 v12; // r9
  __int64 v13; // rbx
  _QWORD *v14; // rax
  __int64 v15; // rsi
  _QWORD *v16; // rbx
  char *v17; // rax
  __int64 v18; // rsi
  __int64 v19; // rsi
  void *v20; // r8
  __int64 v21; // rsi
  __int64 v22; // rax
  __int64 v23; // rax
  char *v24; // rax
  __int64 v25; // rdx
  __int64 v27; // rbx
  __int64 v28; // rax
  _QWORD *v29; // [rsp+8h] [rbp-58h]
  __int64 v30; // [rsp+10h] [rbp-50h]
  char *v31; // [rsp+18h] [rbp-48h]
  char *v32; // [rsp+18h] [rbp-48h]
  _QWORD *v33; // [rsp+20h] [rbp-40h]
  _QWORD *v34; // [rsp+28h] [rbp-38h]
  char *v35; // [rsp+28h] [rbp-38h]
  void *v36; // [rsp+28h] [rbp-38h]
  char *v37; // [rsp+28h] [rbp-38h]

  v4 = a1[1];
  v5 = *a1;
  v6 = 0xCCCCCCCCCCCCCCCDLL * ((v4 - *a1) >> 3);
  if ( v6 == 0x333333333333333LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v7 = 1;
  v8 = a2;
  if ( v6 )
    v7 = 0xCCCCCCCCCCCCCCCDLL * ((v4 - v5) >> 3);
  v9 = a2;
  v10 = __CFADD__(v7, v6);
  v11 = v7 - 0x3333333333333333LL * ((v4 - v5) >> 3);
  v12 = a2 - v5;
  if ( v10 )
  {
    v27 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v11 )
    {
      v30 = 0;
      v13 = 40;
      v33 = 0;
      goto LABEL_7;
    }
    if ( v11 > 0x333333333333333LL )
      v11 = 0x333333333333333LL;
    v27 = 40 * v11;
  }
  v29 = a3;
  v28 = sub_22077B0(v27);
  v12 = a2 - v5;
  v8 = a2;
  v33 = (_QWORD *)v28;
  a3 = v29;
  v30 = v28 + v27;
  v13 = v28 + 40;
LABEL_7:
  v14 = (_QWORD *)((char *)v33 + v12);
  if ( (_QWORD *)((char *)v33 + v12) )
  {
    v15 = *a3;
    v14[1] = 2;
    v14[2] = 0;
    v14[3] = v15;
    if ( v15 != -8 && v15 != 0 && v15 != -16 )
    {
      v31 = v8;
      v34 = (_QWORD *)((char *)v33 + v12);
      sub_164C220((__int64)(v14 + 1));
      v8 = v31;
      v14 = v34;
    }
    v14[4] = 0;
    *v14 = &unk_49FBDD8;
  }
  if ( v8 != v5 )
  {
    v16 = v33;
    v17 = v5;
    while ( 1 )
    {
      if ( v16 )
      {
        v18 = *((_QWORD *)v17 + 1);
        v16[2] = 0;
        v16[1] = v18 & 6;
        v19 = *((_QWORD *)v17 + 3);
        v16[3] = v19;
        if ( v19 != 0 && v19 != -8 && v19 != -16 )
        {
          v32 = v8;
          v35 = v17;
          sub_1649AC0(v16 + 1, *((_QWORD *)v17 + 1) & 0xFFFFFFFFFFFFFFF8LL);
          v8 = v32;
          v17 = v35;
        }
        *v16 = &unk_49FBDD8;
        v16[4] = *((_QWORD *)v17 + 4);
      }
      v17 += 40;
      if ( v8 == v17 )
        break;
      v16 += 5;
    }
    v13 = (__int64)(v16 + 10);
  }
  if ( v8 != v4 )
  {
    v20 = &unk_49FBDD8;
    do
    {
      v21 = *((_QWORD *)v9 + 1);
      *(_QWORD *)(v13 + 16) = 0;
      *(_QWORD *)(v13 + 8) = v21 & 6;
      v22 = *((_QWORD *)v9 + 3);
      *(_QWORD *)(v13 + 24) = v22;
      if ( v22 != -8 && v22 != 0 && v22 != -16 )
      {
        v36 = v20;
        sub_1649AC0((unsigned __int64 *)(v13 + 8), v21 & 0xFFFFFFFFFFFFFFF8LL);
        v20 = v36;
      }
      v23 = *((_QWORD *)v9 + 4);
      v9 += 40;
      *(_QWORD *)v13 = v20;
      v13 += 40;
      *(_QWORD *)(v13 - 8) = v23;
    }
    while ( v4 != v9 );
  }
  if ( v5 != v4 )
  {
    v24 = v5;
    do
    {
      v25 = *((_QWORD *)v24 + 3);
      *(_QWORD *)v24 = &unk_49EE2B0;
      if ( v25 != -8 && v25 != 0 && v25 != -16 )
      {
        v37 = v24;
        sub_1649B30((_QWORD *)v24 + 1);
        v24 = v37;
      }
      v24 += 40;
    }
    while ( v24 != v4 );
  }
  if ( v5 )
    j_j___libc_free_0(v5, a1[2] - v5);
  a1[1] = (char *)v13;
  *a1 = (char *)v33;
  a1[2] = (char *)v30;
  return v30;
}
