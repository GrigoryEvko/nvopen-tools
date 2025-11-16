// Function: sub_B3F040
// Address: 0xb3f040
//
__int64 __fastcall sub_B3F040(const __m128i **a1, __int64 a2)
{
  __int64 result; // rax
  const __m128i *v3; // r14
  __m128i *v4; // r12
  const __m128i *i; // rbx
  const __m128i *v6; // rdx
  __int8 v7; // al
  __int8 v8; // al
  __int64 v9; // rbx
  __int64 v10; // rdx
  __int64 v11; // r15
  __int64 v12; // r14
  __int64 v13; // rax
  _QWORD *v14; // r12
  _QWORD *v15; // r13
  _QWORD *v16; // r15
  _QWORD *v17; // r12
  const __m128i *v18; // [rsp+8h] [rbp-48h]
  __int64 m128i_i64; // [rsp+10h] [rbp-40h]
  __int64 v20; // [rsp+18h] [rbp-38h]

  result = 832LL * *((unsigned int *)a1 + 2);
  v3 = (const __m128i *)((char *)*a1 + result);
  if ( *a1 != v3 )
  {
    v4 = (__m128i *)a2;
    for ( i = *a1 + 1; ; i += 52 )
    {
      if ( v4 )
      {
        v4->m128i_i64[0] = (__int64)v4[1].m128i_i64;
        v6 = (const __m128i *)i[-1].m128i_i64[0];
        if ( i == v6 )
        {
          v4[1] = _mm_loadu_si128(i);
        }
        else
        {
          v4->m128i_i64[0] = (__int64)v6;
          v4[1].m128i_i64[0] = i->m128i_i64[0];
        }
        v4->m128i_i64[1] = i[-1].m128i_i64[1];
        v7 = i[1].m128i_i8[0];
        i[-1].m128i_i64[0] = (__int64)i;
        i[-1].m128i_i64[1] = 0;
        i->m128i_i8[0] = 0;
        v4[2].m128i_i8[0] = v7;
        v8 = i[1].m128i_i8[1];
        v4[3].m128i_i32[0] = 0;
        v4[2].m128i_i8[1] = v8;
        v4[2].m128i_i64[1] = (__int64)&v4[3].m128i_i64[1];
        v4[3].m128i_i32[1] = 4;
        if ( i[2].m128i_i32[0] )
        {
          a2 = (__int64)&i[1].m128i_i64[1];
          sub_B3E030(&v4[2].m128i_i64[1], (__int64)&i[1].m128i_i64[1]);
        }
        v4[51].m128i_i64[1] = i[50].m128i_i64[1];
      }
      v4 += 52;
      if ( v3 == &i[51] )
        break;
    }
    result = *((unsigned int *)a1 + 2);
    v18 = *a1;
    m128i_i64 = (__int64)(*a1)[52 * result].m128i_i64;
    if ( *a1 != (const __m128i *)m128i_i64 )
    {
      do
      {
        m128i_i64 -= 832;
        v20 = *(_QWORD *)(m128i_i64 + 40);
        v9 = v20 + 192LL * *(unsigned int *)(m128i_i64 + 48);
        if ( v20 != v9 )
        {
          do
          {
            v10 = *(unsigned int *)(v9 - 120);
            v11 = *(_QWORD *)(v9 - 128);
            v9 -= 192;
            v12 = v11 + 56 * v10;
            if ( v11 != v12 )
            {
              do
              {
                v13 = *(unsigned int *)(v12 - 40);
                v14 = *(_QWORD **)(v12 - 48);
                v12 -= 56;
                v13 *= 32;
                v15 = (_QWORD *)((char *)v14 + v13);
                if ( v14 != (_QWORD *)((char *)v14 + v13) )
                {
                  do
                  {
                    v15 -= 4;
                    if ( (_QWORD *)*v15 != v15 + 2 )
                    {
                      a2 = v15[2] + 1LL;
                      j_j___libc_free_0(*v15, a2);
                    }
                  }
                  while ( v14 != v15 );
                  v14 = *(_QWORD **)(v12 + 8);
                }
                if ( v14 != (_QWORD *)(v12 + 24) )
                  _libc_free(v14, a2);
              }
              while ( v11 != v12 );
              v11 = *(_QWORD *)(v9 + 64);
            }
            if ( v11 != v9 + 80 )
              _libc_free(v11, a2);
            v16 = *(_QWORD **)(v9 + 16);
            v17 = &v16[4 * *(unsigned int *)(v9 + 24)];
            if ( v16 != v17 )
            {
              do
              {
                v17 -= 4;
                if ( (_QWORD *)*v17 != v17 + 2 )
                {
                  a2 = v17[2] + 1LL;
                  j_j___libc_free_0(*v17, a2);
                }
              }
              while ( v16 != v17 );
              v16 = *(_QWORD **)(v9 + 16);
            }
            if ( v16 != (_QWORD *)(v9 + 32) )
              _libc_free(v16, a2);
          }
          while ( v20 != v9 );
          v20 = *(_QWORD *)(m128i_i64 + 40);
        }
        if ( v20 != m128i_i64 + 56 )
          _libc_free(v20, a2);
        result = m128i_i64 + 16;
        if ( *(_QWORD *)m128i_i64 != m128i_i64 + 16 )
        {
          a2 = *(_QWORD *)(m128i_i64 + 16) + 1LL;
          result = j_j___libc_free_0(*(_QWORD *)m128i_i64, a2);
        }
      }
      while ( (const __m128i *)m128i_i64 != v18 );
    }
  }
  return result;
}
