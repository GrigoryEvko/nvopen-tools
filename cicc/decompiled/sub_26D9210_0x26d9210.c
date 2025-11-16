// Function: sub_26D9210
// Address: 0x26d9210
//
__int64 *__fastcall sub_26D9210(__int64 *a1, __m128i *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r13
  bool v8; // zf
  __int64 v9; // rbx
  _QWORD *v10; // rax
  unsigned __int64 v11; // rdi
  __int64 v12; // rbx
  unsigned __int64 v13; // r15
  __int64 v14; // rax
  const __m128i *v15; // rcx
  unsigned __int64 v16; // r14
  __m128i *v17; // rdx
  const __m128i *v18; // rax
  unsigned __int64 v19; // rax
  unsigned __int64 v21; // [rsp+8h] [rbp-78h]
  __m128i v22; // [rsp+10h] [rbp-70h] BYREF
  char v23; // [rsp+20h] [rbp-60h]
  __int64 v24; // [rsp+30h] [rbp-50h]
  const __m128i *v25; // [rsp+38h] [rbp-48h] BYREF
  const __m128i *v26; // [rsp+40h] [rbp-40h]
  __int64 v27; // [rsp+48h] [rbp-38h]

  v6 = a3;
  v8 = *(_BYTE *)(a3 + 28) == 0;
  v24 = a3;
  v25 = 0;
  v9 = a2->m128i_i64[0];
  v26 = 0;
  v27 = 0;
  if ( v8 )
    goto LABEL_18;
  v10 = *(_QWORD **)(a3 + 8);
  a4 = *(unsigned int *)(a3 + 20);
  a3 = (__int64)&v10[a4];
  if ( v10 == (_QWORD *)a3 )
  {
LABEL_17:
    if ( (unsigned int)a4 < *(_DWORD *)(v6 + 16) )
    {
      *(_DWORD *)(v6 + 20) = a4 + 1;
      *(_QWORD *)a3 = v9;
      ++*(_QWORD *)v6;
LABEL_19:
      a2 = &v22;
      v22.m128i_i64[0] = v9;
      v23 = 0;
      sub_26D91D0((unsigned __int64 *)&v25, &v22);
      goto LABEL_6;
    }
LABEL_18:
    a2 = (__m128i *)v9;
    sub_C8CC70(v6, v9, a3, a4, a5, a6);
    if ( !(_BYTE)a3 )
      goto LABEL_6;
    goto LABEL_19;
  }
  while ( v9 != *v10 )
  {
    if ( (_QWORD *)a3 == ++v10 )
      goto LABEL_17;
  }
LABEL_6:
  v11 = (unsigned __int64)v25;
  v12 = v24;
  v13 = (char *)v26 - (char *)v25;
  if ( v26 == v25 )
  {
    v16 = 0;
LABEL_21:
    v19 = v16;
    goto LABEL_14;
  }
  if ( v13 > 0x7FFFFFFFFFFFFFF8LL )
    sub_4261EA(v25, a2, a3);
  v14 = sub_22077B0((char *)v26 - (char *)v25);
  v15 = v26;
  v11 = (unsigned __int64)v25;
  v16 = v14;
  v13 += v14;
  if ( v26 == v25 )
    goto LABEL_21;
  v17 = (__m128i *)v14;
  v18 = v25;
  do
  {
    if ( v17 )
    {
      *v17 = _mm_loadu_si128(v18);
      v17[1].m128i_i64[0] = v18[1].m128i_i64[0];
    }
    v18 = (const __m128i *)((char *)v18 + 24);
    v17 = (__m128i *)((char *)v17 + 24);
  }
  while ( v18 != v15 );
  v19 = v16 + 8 * (((unsigned __int64)&v18[-2].m128i_u64[1] - v11) >> 3) + 24;
LABEL_14:
  if ( v11 )
  {
    v21 = v19;
    j_j___libc_free_0(v11);
    v19 = v21;
  }
  *a1 = v12;
  a1[1] = v16;
  a1[2] = v19;
  a1[3] = v13;
  a1[4] = v6;
  a1[5] = 0;
  a1[6] = 0;
  a1[7] = 0;
  return a1;
}
