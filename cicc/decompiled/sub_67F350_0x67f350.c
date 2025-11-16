// Function: sub_67F350
// Address: 0x67f350
//
__int64 __fastcall sub_67F350(const __m128i *a1)
{
  __int64 v1; // r12
  __m128i *v2; // rax
  __int64 result; // rax
  _QWORD *v4; // r12
  __int8 v5; // al
  __int64 *v6; // r14
  __int64 v7; // r15
  __int64 v8; // r13
  __int64 v9; // rsi
  unsigned __int64 v10; // r12
  __int64 v11; // r8
  __m128i *v12; // rdx
  __m128i *v13; // rax
  __m128i v14; // xmm0
  __m128i *v15; // rsi
  _QWORD *v16; // rdx
  bool v17; // zf
  __int64 v18; // r13
  __int64 v19; // r12
  __m128i *v20; // rcx
  __m128i *v21; // rdx
  __m128i *v22; // rax
  __m128i v23; // xmm1
  __int64 v24; // rax

  v1 = qword_4CFFD80;
  if ( *(_QWORD *)(qword_4CFFD80 + 16) )
  {
    v4 = sub_67B720(a1);
    if ( v4 )
    {
      v5 = a1->m128i_i8[8];
      if ( v5 == *((_BYTE *)v4 + 8) && ((*v4 ^ a1->m128i_i64[0]) & 0xFFFFFFFFFFFFLL) == 0 )
      {
        v17 = v5 == 36;
        result = 0;
        if ( v17 )
        {
          if ( (a1->m128i_i8[9] & 1) == 0 || a1[1].m128i_i64[0] == v4[2] )
            return result;
        }
        else if ( a1[1].m128i_i32[0] == *((_DWORD *)v4 + 4) )
        {
          return result;
        }
      }
      v6 = (__int64 *)qword_4CFFD80;
      v7 = *(_QWORD *)qword_4CFFD80;
      v8 = *(_QWORD *)(qword_4CFFD80 + 16);
      v9 = *(_QWORD *)qword_4CFFD80;
      if ( v8 == *(_QWORD *)(qword_4CFFD80 + 8) )
      {
        sub_67F290((const __m128i **)qword_4CFFD80);
        v9 = *v6;
      }
      v10 = 0xAAAAAAAAAAAAAAABLL * (((__int64)v4 - v7) >> 3);
      v11 = (int)v10 + 1;
      if ( v11 < v8 )
      {
        v12 = (__m128i *)(v9 + 24 * v8);
        do
        {
          v13 = v12;
          v12 = (__m128i *)((char *)v12 - 24);
          if ( v13 )
          {
            v14 = _mm_loadu_si128((__m128i *)((char *)v13 - 24));
            v13[1].m128i_i64[0] = v13[-1].m128i_i64[1];
            *v13 = v14;
          }
        }
        while ( v12 != (__m128i *)(v9 + 8 * (3LL * (int)v10 + 3)) );
      }
      v15 = (__m128i *)(24 * v11 + v9);
      if ( v15 )
      {
        *v15 = _mm_loadu_si128(a1);
        v15[1].m128i_i64[0] = a1[1].m128i_i64[0];
      }
      v16 = (_QWORD *)qword_4CFFD80;
      v6[2] = v8 + 1;
      return *v16 + 24 * v11;
    }
    else
    {
      v18 = qword_4CFFD80;
      v19 = *(_QWORD *)(qword_4CFFD80 + 16);
      if ( v19 == *(_QWORD *)(qword_4CFFD80 + 8) )
        sub_67F290((const __m128i **)qword_4CFFD80);
      v20 = *(__m128i **)v18;
      if ( v19 > 0 )
      {
        v21 = (__m128i *)((char *)v20 + 24 * v19);
        do
        {
          v22 = v21;
          v21 = (__m128i *)((char *)v21 - 24);
          if ( v22 )
          {
            v23 = _mm_loadu_si128((__m128i *)((char *)v22 - 24));
            v22[1].m128i_i64[0] = v22[-1].m128i_i64[1];
            *v22 = v23;
          }
        }
        while ( v20 != v21 );
      }
      if ( v20 )
      {
        *v20 = _mm_loadu_si128(a1);
        v20[1].m128i_i64[0] = a1[1].m128i_i64[0];
      }
      v24 = qword_4CFFD80;
      *(_QWORD *)(v18 + 16) = v19 + 1;
      return *(_QWORD *)v24;
    }
  }
  else
  {
    if ( !*(_QWORD *)(qword_4CFFD80 + 8) )
      sub_67F290((const __m128i **)qword_4CFFD80);
    v2 = *(__m128i **)v1;
    if ( *(_QWORD *)v1 )
    {
      *v2 = _mm_loadu_si128(a1);
      v2[1].m128i_i64[0] = a1[1].m128i_i64[0];
    }
    *(_QWORD *)(v1 + 16) = 1;
    return *(_QWORD *)qword_4CFFD80;
  }
}
