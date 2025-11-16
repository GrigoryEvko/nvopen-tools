// Function: sub_A56CB0
// Address: 0xa56cb0
//
void __fastcall sub_A56CB0(__int64 *a1, unsigned __int64 a2)
{
  const __m128i *v4; // r13
  const __m128i *v5; // rsi
  __int64 v6; // rbx
  unsigned __int64 v7; // r12
  unsigned __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rax
  bool v11; // cf
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rdx
  __int64 v14; // rax
  __m128i *v15; // rax
  const __m128i *i; // rbx
  __m128i v17; // xmm0
  const __m128i *v18; // rdi
  const __m128i *v19; // rdx
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // [rsp-50h] [rbp-50h]
  __int64 v23; // [rsp-48h] [rbp-48h]
  __m128i *v24; // [rsp-40h] [rbp-40h]
  __int64 v25; // [rsp-40h] [rbp-40h]

  if ( !a2 )
    return;
  v4 = (const __m128i *)a1[1];
  v5 = (const __m128i *)*a1;
  v6 = (__int64)v4->m128i_i64 - *a1;
  v7 = 0x6DB6DB6DB6DB6DB7LL * (v6 >> 3);
  if ( a2 <= 0x6DB6DB6DB6DB6DB7LL * ((a1[2] - (__int64)v4) >> 3) )
  {
    v8 = a2;
    v9 = a1[1];
    do
    {
      if ( v9 )
      {
        *(_QWORD *)(v9 + 8) = 0;
        *(_QWORD *)v9 = v9 + 16;
        *(_BYTE *)(v9 + 16) = 0;
        *(_DWORD *)(v9 + 32) = 0;
        *(_DWORD *)(v9 + 36) = 0;
        *(_DWORD *)(v9 + 40) = 0;
        *(_DWORD *)(v9 + 44) = 0;
        *(_DWORD *)(v9 + 48) = 0;
      }
      v9 += 56;
      --v8;
    }
    while ( v8 );
    a1[1] = (__int64)&v4->m128i_i64[7 * a2];
    return;
  }
  if ( 0x249249249249249LL - v7 < a2 )
    sub_4262D8((__int64)"vector::_M_default_append");
  v10 = 0x6DB6DB6DB6DB6DB7LL * (((char *)v4 - (char *)v5) >> 3);
  if ( a2 >= v7 )
    v10 = a2;
  v11 = __CFADD__(v7, v10);
  v12 = v7 + v10;
  if ( v11 )
  {
    v20 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v12 )
    {
      v22 = 0;
      v23 = 0;
      goto LABEL_15;
    }
    if ( v12 > 0x249249249249249LL )
      v12 = 0x249249249249249LL;
    v20 = 56 * v12;
  }
  v25 = v20;
  v21 = sub_22077B0(v20);
  v4 = (const __m128i *)a1[1];
  v23 = v21;
  v5 = (const __m128i *)*a1;
  v22 = v21 + v25;
LABEL_15:
  v13 = a2;
  v14 = v6 + v23;
  do
  {
    if ( v14 )
    {
      *(_QWORD *)(v14 + 8) = 0;
      *(_QWORD *)v14 = v14 + 16;
      *(_BYTE *)(v14 + 16) = 0;
      *(_DWORD *)(v14 + 32) = 0;
      *(_DWORD *)(v14 + 36) = 0;
      *(_DWORD *)(v14 + 40) = 0;
      *(_DWORD *)(v14 + 44) = 0;
      *(_DWORD *)(v14 + 48) = 0;
    }
    v14 += 56;
    --v13;
  }
  while ( v13 );
  if ( v5 != v4 )
  {
    v15 = (__m128i *)v23;
    for ( i = v5 + 1; ; i = (const __m128i *)((char *)i + 56) )
    {
      if ( v15 )
      {
        v15->m128i_i64[0] = (__int64)v15[1].m128i_i64;
        v19 = (const __m128i *)i[-1].m128i_i64[0];
        if ( i == v19 )
        {
          v15[1] = _mm_loadu_si128(i);
        }
        else
        {
          v15->m128i_i64[0] = (__int64)v19;
          v15[1].m128i_i64[0] = i->m128i_i64[0];
        }
        v15->m128i_i64[1] = i[-1].m128i_i64[1];
        v17 = _mm_loadu_si128(i + 1);
        i[-1].m128i_i64[0] = (__int64)i;
        i[-1].m128i_i64[1] = 0;
        i->m128i_i8[0] = 0;
        v15[2] = v17;
        v15[3].m128i_i32[0] = i[2].m128i_i32[0];
      }
      v18 = (const __m128i *)i[-1].m128i_i64[0];
      if ( i != v18 )
      {
        v24 = v15;
        j_j___libc_free_0(v18, i->m128i_i64[0] + 1);
        v15 = v24;
      }
      v15 = (__m128i *)((char *)v15 + 56);
      if ( v4 == (const __m128i *)&i[2].m128i_u64[1] )
        break;
    }
    v4 = (const __m128i *)*a1;
  }
  if ( v4 )
    j_j___libc_free_0(v4, a1[2] - (_QWORD)v4);
  *a1 = v23;
  a1[1] = v23 + 56 * (a2 + v7);
  a1[2] = v22;
}
