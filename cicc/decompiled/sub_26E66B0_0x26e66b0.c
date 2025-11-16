// Function: sub_26E66B0
// Address: 0x26e66b0
//
unsigned __int64 *__fastcall sub_26E66B0(unsigned __int64 *a1, __m128i *a2)
{
  unsigned __int64 v3; // r13
  unsigned __int64 v4; // r14
  unsigned __int64 **v5; // rax
  unsigned __int64 *v6; // r12
  unsigned __int64 *v8; // rax
  __int64 v9; // rax
  __m128i v10; // xmm0
  _QWORD *v11; // rdi
  __int64 v12; // rdx
  __int64 v13; // rsi
  char v14; // al
  unsigned __int64 v15; // rdx
  unsigned __int64 v16; // r15
  _QWORD *v17; // r8
  __int64 v18; // r14
  unsigned __int64 ***v19; // rax
  unsigned __int64 *v20; // rdx
  size_t v21; // r14
  void *v22; // rax
  _QWORD *v23; // rax
  _QWORD *v24; // r10
  _QWORD *v25; // rsi
  unsigned __int64 v26; // rdi
  _QWORD *v27; // rcx
  unsigned __int64 v28; // rdx
  _QWORD **v29; // rax
  unsigned __int64 v30; // rdx
  _QWORD *v31; // [rsp+8h] [rbp-38h]

  v3 = sub_26E11E0(a2->m128i_i64, (__int64)&a2->m128i_i64[1]);
  v4 = v3 % a1[1];
  v5 = (unsigned __int64 **)sub_26E65E0(a1, v3 % a1[1], a2, v3);
  if ( v5 )
  {
    v6 = *v5;
    if ( *v5 )
      return v6 + 4;
  }
  v8 = (unsigned __int64 *)sub_22077B0(0x30u);
  v6 = v8;
  if ( v8 )
    *v8 = 0;
  v9 = a2[1].m128i_i64[0];
  v10 = _mm_loadu_si128(a2);
  *((_BYTE *)v6 + 32) = 0;
  v11 = a1 + 4;
  v12 = a1[3];
  v13 = a1[1];
  v6[3] = v9;
  *(__m128i *)(v6 + 1) = v10;
  v14 = sub_222DA10((__int64)(a1 + 4), v13, v12, 1);
  v16 = v15;
  if ( v14 )
  {
    if ( v15 == 1 )
    {
      v17 = a1 + 6;
      a1[6] = 0;
      v24 = a1 + 6;
    }
    else
    {
      if ( v15 > 0xFFFFFFFFFFFFFFFLL )
        sub_4261EA(v11, v13, v15);
      v21 = 8 * v15;
      v22 = (void *)sub_22077B0(8 * v15);
      v23 = memset(v22, 0, v21);
      v24 = a1 + 6;
      v17 = v23;
    }
    v25 = (_QWORD *)a1[2];
    a1[2] = 0;
    if ( !v25 )
    {
LABEL_21:
      if ( v24 != (_QWORD *)*a1 )
      {
        v31 = v17;
        j_j___libc_free_0(*a1);
        v17 = v31;
      }
      a1[1] = v16;
      *a1 = (unsigned __int64)v17;
      v4 = v3 % v16;
      goto LABEL_8;
    }
    v26 = 0;
    while ( 1 )
    {
      while ( 1 )
      {
        v27 = v25;
        v25 = (_QWORD *)*v25;
        v28 = v27[5] % v16;
        v29 = (_QWORD **)&v17[v28];
        if ( !*v29 )
          break;
        *v27 = **v29;
        **v29 = v27;
LABEL_17:
        if ( !v25 )
          goto LABEL_21;
      }
      *v27 = a1[2];
      a1[2] = (unsigned __int64)v27;
      *v29 = a1 + 2;
      if ( !*v27 )
      {
        v26 = v28;
        goto LABEL_17;
      }
      v17[v26] = v27;
      v26 = v28;
      if ( !v25 )
        goto LABEL_21;
    }
  }
  v17 = (_QWORD *)*a1;
LABEL_8:
  v18 = v4;
  v6[5] = v3;
  v19 = (unsigned __int64 ***)&v17[v18];
  v20 = (unsigned __int64 *)v17[v18];
  if ( v20 )
  {
    *v6 = *v20;
    **v19 = v6;
  }
  else
  {
    v30 = a1[2];
    a1[2] = (unsigned __int64)v6;
    *v6 = v30;
    if ( v30 )
    {
      v17[*(_QWORD *)(v30 + 40) % a1[1]] = v6;
      v19 = (unsigned __int64 ***)(v18 * 8 + *a1);
    }
    *v19 = (unsigned __int64 **)(a1 + 2);
  }
  ++a1[3];
  return v6 + 4;
}
