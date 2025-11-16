// Function: sub_3005920
// Address: 0x3005920
//
__int64 *__fastcall sub_3005920(__int64 *a1, __m128i *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r13
  __int64 v8; // rax
  bool v9; // zf
  __int64 v10; // rbx
  _QWORD *v11; // rax
  unsigned __int64 v12; // rdi
  __int64 v13; // rbx
  unsigned __int64 v14; // r15
  __int64 v15; // rax
  const __m128i *v16; // rcx
  unsigned __int64 v17; // r14
  __m128i *v18; // rdx
  const __m128i *v19; // rax
  unsigned __int64 v20; // rax
  unsigned __int64 v22; // [rsp+8h] [rbp-78h]
  __m128i v23; // [rsp+10h] [rbp-70h] BYREF
  char v24; // [rsp+20h] [rbp-60h]
  __int64 v25; // [rsp+30h] [rbp-50h]
  const __m128i *v26; // [rsp+38h] [rbp-48h] BYREF
  const __m128i *v27; // [rsp+40h] [rbp-40h]
  __int64 v28; // [rsp+48h] [rbp-38h]

  v6 = a3;
  v8 = a2->m128i_i64[0];
  v9 = *(_BYTE *)(a3 + 28) == 0;
  v25 = a3;
  v26 = 0;
  v27 = 0;
  v10 = *(_QWORD *)(v8 + 328);
  v28 = 0;
  if ( v9 )
    goto LABEL_18;
  v11 = *(_QWORD **)(a3 + 8);
  a4 = *(unsigned int *)(a3 + 20);
  a3 = (__int64)&v11[a4];
  if ( v11 == (_QWORD *)a3 )
  {
LABEL_17:
    if ( (unsigned int)a4 < *(_DWORD *)(v6 + 16) )
    {
      *(_DWORD *)(v6 + 20) = a4 + 1;
      *(_QWORD *)a3 = v10;
      ++*(_QWORD *)v6;
LABEL_19:
      a2 = &v23;
      v23.m128i_i64[0] = v10;
      v24 = 0;
      sub_30058E0((unsigned __int64 *)&v26, &v23);
      goto LABEL_6;
    }
LABEL_18:
    a2 = (__m128i *)v10;
    sub_C8CC70(v6, v10, a3, a4, a5, a6);
    if ( !(_BYTE)a3 )
      goto LABEL_6;
    goto LABEL_19;
  }
  while ( v10 != *v11 )
  {
    if ( (_QWORD *)a3 == ++v11 )
      goto LABEL_17;
  }
LABEL_6:
  v12 = (unsigned __int64)v26;
  v13 = v25;
  v14 = (char *)v27 - (char *)v26;
  if ( v27 == v26 )
  {
    v17 = 0;
LABEL_21:
    v20 = v17;
    goto LABEL_14;
  }
  if ( v14 > 0x7FFFFFFFFFFFFFF8LL )
    sub_4261EA(v26, a2, a3);
  v15 = sub_22077B0((char *)v27 - (char *)v26);
  v16 = v27;
  v12 = (unsigned __int64)v26;
  v17 = v15;
  v14 += v15;
  if ( v27 == v26 )
    goto LABEL_21;
  v18 = (__m128i *)v15;
  v19 = v26;
  do
  {
    if ( v18 )
    {
      *v18 = _mm_loadu_si128(v19);
      v18[1].m128i_i64[0] = v19[1].m128i_i64[0];
    }
    v19 = (const __m128i *)((char *)v19 + 24);
    v18 = (__m128i *)((char *)v18 + 24);
  }
  while ( v19 != v16 );
  v20 = v17 + 8 * (((unsigned __int64)&v19[-2].m128i_u64[1] - v12) >> 3) + 24;
LABEL_14:
  if ( v12 )
  {
    v22 = v20;
    j_j___libc_free_0(v12);
    v20 = v22;
  }
  *a1 = v13;
  a1[1] = v17;
  a1[2] = v20;
  a1[3] = v14;
  a1[4] = v6;
  a1[5] = 0;
  a1[6] = 0;
  a1[7] = 0;
  return a1;
}
