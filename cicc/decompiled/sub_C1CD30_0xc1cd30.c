// Function: sub_C1CD30
// Address: 0xc1cd30
//
_QWORD *__fastcall sub_C1CD30(_QWORD *a1, const __m128i *a2)
{
  __int64 v3; // r12
  unsigned __int64 v4; // r13
  __int64 v5; // r14
  __int64 *v6; // rax
  __int64 v7; // rax
  __int64 v9; // rax
  _QWORD *v10; // r12
  __m128i v11; // xmm0
  __int64 v12; // rdx
  _QWORD *v13; // rdi
  __int64 v14; // rsi
  char v15; // al
  unsigned __int64 v16; // rdx
  __int64 v17; // rcx
  unsigned __int64 v18; // r15
  _QWORD *v19; // r8
  __int64 v20; // r14
  _QWORD **v21; // rax
  _QWORD *v22; // rdx
  size_t v23; // r14
  void *v24; // rax
  _QWORD *v25; // rax
  _QWORD *v26; // r10
  _QWORD *v27; // rsi
  unsigned __int64 v28; // rdi
  _QWORD *v29; // rcx
  unsigned __int64 v30; // rdx
  _QWORD **v31; // rax
  __int64 v32; // rdx
  _QWORD *v33; // [rsp+8h] [rbp-E8h]
  unsigned __int64 v34; // [rsp+10h] [rbp-E0h] BYREF
  _BYTE v35[208]; // [rsp+20h] [rbp-D0h] BYREF

  v3 = a2->m128i_i64[0];
  v4 = a2->m128i_u64[1];
  if ( a2->m128i_i64[0] )
  {
    sub_C7D030(v35);
    sub_C7D280(v35, v3, v4);
    sub_C7D290(v35, &v34);
    v4 = v34;
  }
  v5 = v4 % a1[1];
  v6 = sub_C1CC80(a1, v5, (__int64)a2, v4);
  if ( v6 )
  {
    v7 = *v6;
    if ( v7 )
      return (_QWORD *)(v7 + 24);
  }
  v9 = sub_22077B0(40);
  v10 = (_QWORD *)v9;
  if ( v9 )
    *(_QWORD *)v9 = 0;
  v11 = _mm_loadu_si128(a2);
  v12 = a1[3];
  v13 = a1 + 4;
  *(_QWORD *)(v9 + 24) = 0;
  v14 = a1[1];
  *(__m128i *)(v9 + 8) = v11;
  v15 = sub_222DA10(a1 + 4, v14, v12, 1);
  v18 = v16;
  if ( v15 )
  {
    if ( v16 == 1 )
    {
      v19 = a1 + 6;
      a1[6] = 0;
      v26 = a1 + 6;
    }
    else
    {
      if ( v16 > 0xFFFFFFFFFFFFFFFLL )
        sub_4261EA(v13, v14, v16, v17);
      v23 = 8 * v16;
      v24 = (void *)sub_22077B0(8 * v16);
      v25 = memset(v24, 0, v23);
      v26 = a1 + 6;
      v19 = v25;
    }
    v27 = (_QWORD *)a1[2];
    a1[2] = 0;
    if ( !v27 )
    {
LABEL_23:
      if ( (_QWORD *)*a1 != v26 )
      {
        v33 = v19;
        j_j___libc_free_0(*a1, 8LL * a1[1]);
        v19 = v33;
      }
      a1[1] = v18;
      *a1 = v19;
      v5 = v4 % v18;
      goto LABEL_10;
    }
    v28 = 0;
    while ( 1 )
    {
      while ( 1 )
      {
        v29 = v27;
        v27 = (_QWORD *)*v27;
        v30 = v29[4] % v18;
        v31 = (_QWORD **)&v19[v30];
        if ( !*v31 )
          break;
        *v29 = **v31;
        **v31 = v29;
LABEL_19:
        if ( !v27 )
          goto LABEL_23;
      }
      *v29 = a1[2];
      a1[2] = v29;
      *v31 = a1 + 2;
      if ( !*v29 )
      {
        v28 = v30;
        goto LABEL_19;
      }
      v19[v28] = v29;
      v28 = v30;
      if ( !v27 )
        goto LABEL_23;
    }
  }
  v19 = (_QWORD *)*a1;
LABEL_10:
  v20 = v5;
  v10[4] = v4;
  v21 = (_QWORD **)&v19[v20];
  v22 = (_QWORD *)v19[v20];
  if ( v22 )
  {
    *v10 = *v22;
    **v21 = v10;
  }
  else
  {
    v32 = a1[2];
    a1[2] = v10;
    *v10 = v32;
    if ( v32 )
    {
      v19[*(_QWORD *)(v32 + 32) % a1[1]] = v10;
      v21 = (_QWORD **)(v20 * 8 + *a1);
    }
    *v21 = a1 + 2;
  }
  ++a1[3];
  return v10 + 3;
}
