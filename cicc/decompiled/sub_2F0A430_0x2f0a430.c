// Function: sub_2F0A430
// Address: 0x2f0a430
//
__int64 __fastcall sub_2F0A430(__m128i *a1, int *a2, __int64 a3)
{
  __m128i *v3; // rbx
  __int64 v4; // r15
  __int64 v5; // rax
  __int64 v6; // rax
  int **v7; // r15
  __m128i *v8; // r14
  const __m128i *v9; // r13
  __int64 v10; // r12
  __int64 v11; // rax
  unsigned __int64 v12; // rcx
  __int64 v13; // rax
  unsigned __int64 v14; // r13
  __int64 v15; // rbx
  __m128i *v16; // r15
  unsigned __int64 *i; // rbx
  unsigned __int64 *j; // r13
  unsigned int v20; // [rsp+8h] [rbp-58h]
  unsigned int v21; // [rsp+Ch] [rbp-54h]
  unsigned int v22; // [rsp+18h] [rbp-48h]
  unsigned int v23; // [rsp+1Ch] [rbp-44h]
  unsigned __int64 v24; // [rsp+28h] [rbp-38h]

  v3 = a1;
  v4 = *((_QWORD *)a2 + 2);
  v22 = *a2;
  v20 = a2[1];
  v5 = *((_QWORD *)a2 + 1);
  if ( v4 == v5 )
  {
    v24 = 0;
LABEL_30:
    v8 = (__m128i *)v24;
    goto LABEL_8;
  }
  if ( (unsigned __int64)(v4 - v5) > 0x7FFFFFFFFFFFFFF8LL )
    goto LABEL_34;
  a1 = (__m128i *)(v4 - v5);
  v6 = sub_22077B0(v4 - v5);
  v7 = (int **)*((_QWORD *)a2 + 2);
  v24 = v6;
  if ( v7 == *((int ***)a2 + 1) )
    goto LABEL_30;
  v8 = (__m128i *)v6;
  v9 = (const __m128i *)*((_QWORD *)a2 + 1);
  do
  {
    if ( v8 )
    {
      a1 = v8;
      v8->m128i_i64[0] = (__int64)v8[1].m128i_i64;
      a2 = (int *)v9->m128i_i64[0];
      sub_2F07250(v8->m128i_i64, v9->m128i_i64[0], v9->m128i_i64[0] + v9->m128i_i64[1]);
      v8[2] = _mm_loadu_si128(v9 + 2);
      v8[3].m128i_i16[0] = v9[3].m128i_i16[0];
    }
    v9 = (const __m128i *)((char *)v9 + 56);
    v8 = (__m128i *)((char *)v8 + 56);
  }
  while ( v7 != (int **)v9 );
LABEL_8:
  v10 = v3[1].m128i_i64[0];
  v23 = v3->m128i_i32[0];
  v21 = v3->m128i_u32[1];
  v11 = v3->m128i_i64[1];
  v12 = v10 - v11;
  if ( v10 == v11 )
  {
    v14 = 0;
LABEL_32:
    v16 = (__m128i *)v14;
    LOBYTE(v10) = v23 < v22;
    if ( v23 != v22 )
      goto LABEL_16;
    goto LABEL_33;
  }
  if ( v12 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_34:
    sub_4261EA(a1, a2, a3);
  v13 = sub_22077B0(v12);
  v10 = v3[1].m128i_i64[0];
  v14 = v13;
  if ( v10 == v3->m128i_i64[1] )
    goto LABEL_32;
  v15 = v3->m128i_i64[1];
  v16 = (__m128i *)v13;
  do
  {
    if ( v16 )
    {
      v16->m128i_i64[0] = (__int64)v16[1].m128i_i64;
      sub_2F07250(v16->m128i_i64, *(_BYTE **)v15, *(_QWORD *)v15 + *(_QWORD *)(v15 + 8));
      v16[2] = _mm_loadu_si128((const __m128i *)(v15 + 32));
      v16[3].m128i_i16[0] = *(_WORD *)(v15 + 48);
    }
    v15 += 56;
    v16 = (__m128i *)((char *)v16 + 56);
  }
  while ( v10 != v15 );
  LOBYTE(v10) = v23 < v22;
  if ( v23 == v22 )
LABEL_33:
    LOBYTE(v10) = v21 < v20;
LABEL_16:
  for ( i = (unsigned __int64 *)v14; v16 != (__m128i *)i; i += 7 )
  {
    if ( (unsigned __int64 *)*i != i + 2 )
      j_j___libc_free_0(*i);
  }
  if ( v14 )
    j_j___libc_free_0(v14);
  for ( j = (unsigned __int64 *)v24; j != (unsigned __int64 *)v8; j += 7 )
  {
    if ( (unsigned __int64 *)*j != j + 2 )
      j_j___libc_free_0(*j);
  }
  if ( v24 )
    j_j___libc_free_0(v24);
  return (unsigned int)v10;
}
