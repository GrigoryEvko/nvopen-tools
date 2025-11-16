// Function: sub_29B8A50
// Address: 0x29b8a50
//
void __fastcall sub_29B8A50(unsigned __int64 *a1, __int64 *a2, _QWORD *a3)
{
  unsigned __int64 v4; // r12
  __int64 v5; // rax
  __int64 v6; // rbx
  __int64 v7; // rax
  double v8; // xmm0_8
  __int64 v9; // rax
  _QWORD *v10; // rax
  __int64 v11; // rdx
  unsigned __int64 v12; // r14
  __int64 v13; // rbx
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // rdi
  bool v16; // cf
  unsigned __int64 v17; // rax
  __int64 v18; // rcx
  char *v19; // rbx
  __int64 v20; // rax
  __int64 v21; // r13
  __int64 v22; // rax
  double v23; // xmm0_8
  __int64 v24; // rax
  _QWORD *v25; // rax
  _QWORD *v26; // r13
  _QWORD *v27; // rbx
  __int64 v28; // rdx
  unsigned __int64 v29; // rdi
  unsigned __int64 v30; // rdi
  __int64 v31; // rsi
  unsigned __int64 v32; // r13
  __int64 v33; // rax
  _QWORD *v34; // [rsp+0h] [rbp-50h]
  __int64 v35; // [rsp+8h] [rbp-48h]
  __int64 v36; // [rsp+8h] [rbp-48h]
  unsigned __int64 v37; // [rsp+10h] [rbp-40h]
  _QWORD *v38; // [rsp+18h] [rbp-38h]

  v4 = a1[1];
  if ( v4 != a1[2] )
  {
    if ( v4 )
    {
      v5 = *a2;
      v6 = *a3;
      *(_QWORD *)(v4 + 8) = 0;
      *(_QWORD *)v4 = v5;
      v7 = *(_QWORD *)(v6 + 24);
      if ( v7 < 0 )
      {
        v11 = *(_QWORD *)(v6 + 24) & 1LL | (*(_QWORD *)(v6 + 24) >> 1);
        v8 = (double)(int)v11 + (double)(int)v11;
      }
      else
      {
        v8 = (double)(int)v7;
      }
      *(double *)(v4 + 16) = v8;
      v9 = *(_QWORD *)(v6 + 16);
      *(_QWORD *)(v4 + 32) = 0;
      *(_QWORD *)(v4 + 24) = v9;
      *(_QWORD *)(v4 + 40) = 0;
      *(_QWORD *)(v4 + 48) = 0;
      v10 = (_QWORD *)sub_22077B0(8u);
      *(_QWORD *)(v4 + 32) = v10;
      *(_QWORD *)(v4 + 48) = v10 + 1;
      *v10 = v6;
      *(_QWORD *)(v4 + 40) = v10 + 1;
      *(_QWORD *)(v4 + 56) = 0;
      *(_QWORD *)(v4 + 64) = 0;
      *(_QWORD *)(v4 + 72) = 0;
      v4 = a1[1];
    }
    a1[1] = v4 + 80;
    return;
  }
  v12 = *a1;
  v13 = v4 - *a1;
  v14 = 0xCCCCCCCCCCCCCCCDLL * (v13 >> 4);
  if ( v14 == 0x199999999999999LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v15 = 1;
  if ( v14 )
    v15 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)(v4 - v12) >> 4);
  v16 = __CFADD__(v15, v14);
  v17 = v15 - 0x3333333333333333LL * ((__int64)(v4 - v12) >> 4);
  if ( v16 )
  {
    v32 = 0x7FFFFFFFFFFFFFD0LL;
LABEL_36:
    v34 = a3;
    v33 = sub_22077B0(v32);
    a3 = v34;
    v38 = (_QWORD *)v33;
    v18 = v33 + 80;
    v37 = v33 + v32;
    goto LABEL_14;
  }
  if ( v17 )
  {
    if ( v17 > 0x199999999999999LL )
      v17 = 0x199999999999999LL;
    v32 = 80 * v17;
    goto LABEL_36;
  }
  v37 = 0;
  v18 = 80;
  v38 = 0;
LABEL_14:
  v19 = (char *)v38 + v13;
  if ( v19 )
  {
    v20 = *a2;
    v21 = *a3;
    *((_QWORD *)v19 + 1) = 0;
    *(_QWORD *)v19 = v20;
    v22 = *(_QWORD *)(v21 + 24);
    if ( v22 < 0 )
    {
      v31 = *(_QWORD *)(v21 + 24) & 1LL | (*(_QWORD *)(v21 + 24) >> 1);
      v23 = (double)(int)v31 + (double)(int)v31;
    }
    else
    {
      v23 = (double)(int)v22;
    }
    *((double *)v19 + 2) = v23;
    v24 = *(_QWORD *)(v21 + 16);
    *((_QWORD *)v19 + 4) = 0;
    *((_QWORD *)v19 + 3) = v24;
    *((_QWORD *)v19 + 5) = 0;
    *((_QWORD *)v19 + 6) = 0;
    v35 = v18;
    v25 = (_QWORD *)sub_22077B0(8u);
    *((_QWORD *)v19 + 7) = 0;
    v18 = v35;
    *((_QWORD *)v19 + 4) = v25;
    *((_QWORD *)v19 + 6) = v25 + 1;
    *v25 = v21;
    *((_QWORD *)v19 + 5) = v25 + 1;
    *((_QWORD *)v19 + 8) = 0;
    *((_QWORD *)v19 + 9) = 0;
  }
  if ( v4 != v12 )
  {
    v26 = v38;
    v27 = (_QWORD *)v12;
    while ( 1 )
    {
      if ( v26 )
      {
        *v26 = *v27;
        v26[1] = v27[1];
        v26[2] = v27[2];
        v26[3] = v27[3];
        v26[4] = v27[4];
        v26[5] = v27[5];
        v26[6] = v27[6];
        v28 = v27[7];
        v27[6] = 0;
        v27[5] = 0;
        v27[4] = 0;
        v26[7] = v28;
        v26[8] = v27[8];
        v26[9] = v27[9];
        v27[9] = 0;
        v27[7] = 0;
      }
      else
      {
        v30 = v27[7];
        if ( v30 )
          j_j___libc_free_0(v30);
      }
      v29 = v27[4];
      if ( v29 )
        j_j___libc_free_0(v29);
      v27 += 10;
      if ( (_QWORD *)v4 == v27 )
        break;
      v26 += 10;
    }
    v18 = (__int64)(v26 + 20);
  }
  if ( v12 )
  {
    v36 = v18;
    j_j___libc_free_0(v12);
    v18 = v36;
  }
  a1[1] = v18;
  *a1 = (unsigned __int64)v38;
  a1[2] = v37;
}
