// Function: sub_2900050
// Address: 0x2900050
//
void __fastcall sub_2900050(unsigned __int64 *a1, __int64 a2)
{
  unsigned __int64 v4; // rbx
  __int64 v5; // rax
  __int64 v6; // rax
  unsigned __int64 v7; // r14
  __int64 v8; // rsi
  __int64 v9; // rax
  __int64 v10; // rdi
  bool v11; // cf
  unsigned __int64 v12; // rax
  __int64 v13; // r12
  __int64 v14; // rax
  __int64 v15; // rsi
  __int64 v16; // rsi
  __int64 v17; // r12
  unsigned __int64 v18; // r15
  __int64 v19; // rcx
  __int64 v20; // rcx
  _QWORD *v21; // r15
  __int64 v22; // rcx
  __int64 v23; // rcx
  unsigned __int64 v24; // r12
  __int64 v25; // [rsp+8h] [rbp-48h]
  __int64 v26; // [rsp+8h] [rbp-48h]
  unsigned __int64 v27; // [rsp+10h] [rbp-40h]
  __int64 v28; // [rsp+18h] [rbp-38h]

  v4 = a1[1];
  if ( v4 != a1[2] )
  {
    if ( v4 )
    {
      *(_QWORD *)v4 = 0;
      v5 = *(_QWORD *)(a2 + 16);
      *(_QWORD *)(v4 + 8) = 0;
      *(_QWORD *)(v4 + 16) = v5;
      if ( v5 != 0 && v5 != -4096 && v5 != -8192 )
        sub_BD6050((unsigned __int64 *)v4, *(_QWORD *)a2 & 0xFFFFFFFFFFFFFFF8LL);
      *(_QWORD *)(v4 + 24) = 0;
      v6 = *(_QWORD *)(a2 + 40);
      *(_QWORD *)(v4 + 32) = 0;
      *(_QWORD *)(v4 + 40) = v6;
      if ( v6 != 0 && v6 != -4096 && v6 != -8192 )
        sub_BD6050((unsigned __int64 *)(v4 + 24), *(_QWORD *)(a2 + 24) & 0xFFFFFFFFFFFFFFF8LL);
      *(_BYTE *)(v4 + 48) = *(_BYTE *)(a2 + 48);
      v4 = a1[1];
    }
    a1[1] = v4 + 56;
    return;
  }
  v7 = *a1;
  v8 = v4 - *a1;
  v9 = 0x6DB6DB6DB6DB6DB7LL * (v8 >> 3);
  if ( v9 == 0x249249249249249LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v10 = 1;
  if ( v9 )
    v10 = 0x6DB6DB6DB6DB6DB7LL * ((__int64)(v4 - v7) >> 3);
  v11 = __CFADD__(v10, v9);
  v12 = v10 + v9;
  if ( v11 )
  {
    v24 = 0x7FFFFFFFFFFFFFF8LL;
LABEL_51:
    v8 = v4 - v7;
    v28 = sub_22077B0(v24);
    v27 = v28 + v24;
    v13 = v28 + 56;
    goto LABEL_17;
  }
  if ( v12 )
  {
    if ( v12 > 0x249249249249249LL )
      v12 = 0x249249249249249LL;
    v24 = 56 * v12;
    goto LABEL_51;
  }
  v27 = 0;
  v13 = 56;
  v28 = 0;
LABEL_17:
  v14 = v28 + v8;
  if ( v28 + v8 )
  {
    v15 = *(_QWORD *)(a2 + 16);
    *(_QWORD *)v14 = 0;
    *(_QWORD *)(v14 + 8) = 0;
    *(_QWORD *)(v14 + 16) = v15;
    if ( v15 != -4096 && v15 != 0 && v15 != -8192 )
    {
      v25 = v14;
      sub_BD6050((unsigned __int64 *)v14, *(_QWORD *)a2 & 0xFFFFFFFFFFFFFFF8LL);
      v14 = v25;
    }
    v16 = *(_QWORD *)(a2 + 40);
    *(_QWORD *)(v14 + 24) = 0;
    *(_QWORD *)(v14 + 32) = 0;
    *(_QWORD *)(v14 + 40) = v16;
    if ( v16 != -4096 && v16 != 0 && v16 != -8192 )
    {
      v26 = v14;
      sub_BD6050((unsigned __int64 *)(v14 + 24), *(_QWORD *)(a2 + 24) & 0xFFFFFFFFFFFFFFF8LL);
      v14 = v26;
    }
    *(_BYTE *)(v14 + 48) = *(_BYTE *)(a2 + 48);
  }
  if ( v4 != v7 )
  {
    v17 = v28;
    v18 = v7;
    while ( 1 )
    {
      if ( v17 )
      {
        *(_QWORD *)v17 = 0;
        *(_QWORD *)(v17 + 8) = 0;
        v19 = *(_QWORD *)(v18 + 16);
        *(_QWORD *)(v17 + 16) = v19;
        if ( v19 != -4096 && v19 != 0 && v19 != -8192 )
          sub_BD6050((unsigned __int64 *)v17, *(_QWORD *)v18 & 0xFFFFFFFFFFFFFFF8LL);
        *(_QWORD *)(v17 + 24) = 0;
        *(_QWORD *)(v17 + 32) = 0;
        v20 = *(_QWORD *)(v18 + 40);
        *(_QWORD *)(v17 + 40) = v20;
        if ( v20 != -4096 && v20 != 0 && v20 != -8192 )
          sub_BD6050((unsigned __int64 *)(v17 + 24), *(_QWORD *)(v18 + 24) & 0xFFFFFFFFFFFFFFF8LL);
        *(_BYTE *)(v17 + 48) = *(_BYTE *)(v18 + 48);
      }
      v18 += 56LL;
      if ( v4 == v18 )
        break;
      v17 += 56;
    }
    v13 = v17 + 112;
    v21 = (_QWORD *)v7;
    do
    {
      v22 = v21[5];
      if ( v22 != 0 && v22 != -4096 && v22 != -8192 )
        sub_BD60C0(v21 + 3);
      v23 = v21[2];
      if ( v23 != -4096 && v23 != 0 && v23 != -8192 )
        sub_BD60C0(v21);
      v21 += 7;
    }
    while ( (_QWORD *)v4 != v21 );
  }
  if ( v7 )
    j_j___libc_free_0(v7);
  a1[1] = v13;
  *a1 = v28;
  a1[2] = v27;
}
