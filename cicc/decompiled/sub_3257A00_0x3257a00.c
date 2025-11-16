// Function: sub_3257A00
// Address: 0x3257a00
//
void __fastcall sub_3257A00(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // r13
  char *v4; // r14
  __int64 v5; // rbx
  unsigned __int64 v7; // r15
  __int64 v8; // rax
  char *v9; // r8
  size_t v10; // rdx
  unsigned __int64 v11; // rax
  __int64 v12; // rsi
  bool v13; // cf
  unsigned __int64 v14; // rax
  size_t v15; // r10
  char *v16; // r11
  char *v17; // r9
  char *v18; // rax
  char *v19; // rdi
  __int64 v20; // rax
  char *v21; // rax
  char *v22; // rbx
  unsigned __int64 v23; // rsi
  __int64 v24; // rax
  size_t v25; // [rsp+0h] [rbp-60h]
  char *v26; // [rsp+8h] [rbp-58h]
  char *v27; // [rsp+8h] [rbp-58h]
  size_t v28; // [rsp+18h] [rbp-48h]
  char *v29; // [rsp+18h] [rbp-48h]
  char *v30; // [rsp+20h] [rbp-40h]
  size_t v31; // [rsp+20h] [rbp-40h]
  char *v32; // [rsp+20h] [rbp-40h]
  unsigned __int64 v33; // [rsp+28h] [rbp-38h]

  v2 = *(_QWORD *)(a2 + 392);
  v3 = *(_QWORD *)(a2 + 384);
  if ( v3 == v2 )
    return;
  v4 = (char *)a1[3];
  v5 = v2 - v3;
  v7 = v5 >> 3;
  if ( a1[4] - (_QWORD)v4 >= (unsigned __int64)v5 )
  {
    if ( v5 > 0 )
    {
      v8 = 0;
      do
      {
        *(_QWORD *)&v4[8 * v8] = *(_QWORD *)(v3 + 8 * v8);
        ++v8;
      }
      while ( (__int64)(v7 - v8) > 0 );
      v4 = (char *)a1[3];
    }
    a1[3] = &v4[v5];
    return;
  }
  v9 = (char *)a1[2];
  v10 = v4 - v9;
  v11 = (v4 - v9) >> 3;
  if ( v7 > 0xFFFFFFFFFFFFFFFLL - v11 )
    sub_4262D8((__int64)"vector::_M_range_insert");
  v12 = (v4 - v9) >> 3;
  if ( v7 >= v11 )
    v12 = v5 >> 3;
  v13 = __CFADD__(v12, v11);
  v14 = v12 + v11;
  v15 = v14;
  if ( v13 )
  {
    v23 = 0x7FFFFFFFFFFFFFF8LL;
LABEL_27:
    v24 = sub_22077B0(v23);
    v16 = (char *)a1[3];
    v9 = (char *)a1[2];
    v17 = (char *)v24;
    v33 = v23 + v24;
    v10 = v4 - v9;
    v15 = v16 - v4;
    goto LABEL_15;
  }
  if ( v14 )
  {
    if ( v14 > 0xFFFFFFFFFFFFFFFLL )
      v14 = 0xFFFFFFFFFFFFFFFLL;
    v23 = 8 * v14;
    goto LABEL_27;
  }
  v33 = 0;
  v16 = (char *)a1[3];
  v17 = 0;
LABEL_15:
  if ( v4 != v9 )
  {
    v25 = v15;
    v26 = v16;
    v28 = v10;
    v30 = v9;
    v18 = (char *)memmove(v17, v9, v10);
    v15 = v25;
    v16 = v26;
    v10 = v28;
    v17 = v18;
    v9 = v30;
  }
  v19 = &v17[v10];
  if ( v5 > 0 )
  {
    v20 = 0;
    do
    {
      *(_QWORD *)&v19[8 * v20] = *(_QWORD *)(v3 + 8 * v20);
      ++v20;
    }
    while ( (__int64)(v7 - v20) > 0 );
    v19 += v5;
  }
  if ( v4 != v16 )
  {
    v27 = v17;
    v29 = v9;
    v31 = v15;
    v21 = (char *)memcpy(v19, v4, v15);
    v17 = v27;
    v9 = v29;
    v15 = v31;
    v19 = v21;
  }
  v22 = &v19[v15];
  if ( v9 )
  {
    v32 = v17;
    j_j___libc_free_0((unsigned __int64)v9);
    v17 = v32;
  }
  a1[2] = v17;
  a1[3] = v22;
  a1[4] = v33;
}
