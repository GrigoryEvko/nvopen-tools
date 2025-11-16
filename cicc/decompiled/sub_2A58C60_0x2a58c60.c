// Function: sub_2A58C60
// Address: 0x2a58c60
//
__int64 __fastcall sub_2A58C60(__int64 a1, __m128i *a2)
{
  __int64 result; // rax
  __int64 v3; // r14
  const __m128i *i; // r12
  __m128i *v7; // rdi
  __int64 v8; // rdx
  __int64 v9; // rax
  unsigned int v10; // r13d
  const __m128i *v11; // rax
  const __m128i *v12; // rsi
  size_t v13; // rdx
  const __m128i *v14; // r12
  __int64 v15; // rbx
  unsigned __int64 v16; // rdi

  result = 13LL * *(unsigned int *)(a1 + 8);
  v3 = *(_QWORD *)a1 + 104LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v3 )
  {
    for ( i = (const __m128i *)(*(_QWORD *)a1 + 48LL); ; i = (const __m128i *)((char *)i + 104) )
    {
      if ( a2 )
      {
        a2[1].m128i_i32[2] = 0;
        v7 = a2 + 3;
        a2->m128i_i64[1] = 0;
        a2[1].m128i_i32[0] = 0;
        a2[1].m128i_i32[1] = 0;
        a2->m128i_i64[0] = 1;
        v8 = i[-3].m128i_i64[1];
        ++i[-3].m128i_i64[0];
        v9 = a2->m128i_i64[1];
        a2->m128i_i64[1] = v8;
        LODWORD(v8) = i[-2].m128i_i32[0];
        i[-3].m128i_i64[1] = v9;
        LODWORD(v9) = a2[1].m128i_i32[0];
        a2[1].m128i_i32[0] = v8;
        LODWORD(v8) = i[-2].m128i_i32[1];
        i[-2].m128i_i32[0] = v9;
        LODWORD(v9) = a2[1].m128i_i32[1];
        a2[1].m128i_i32[1] = v8;
        LODWORD(v8) = i[-2].m128i_i32[2];
        i[-2].m128i_i32[1] = v9;
        LODWORD(v9) = a2[1].m128i_i32[2];
        a2[1].m128i_i32[2] = v8;
        i[-2].m128i_i32[2] = v9;
        a2[2].m128i_i64[0] = (__int64)a2[3].m128i_i64;
        a2[2].m128i_i32[2] = 0;
        a2[2].m128i_i32[3] = 4;
        v10 = i[-1].m128i_u32[2];
        if ( v10 && &a2[2] != &i[-1] )
        {
          v11 = (const __m128i *)i[-1].m128i_i64[0];
          if ( v11 == i )
          {
            v12 = i;
            v13 = 8LL * v10;
            if ( v10 <= 4
              || (sub_C8D5F0((__int64)a2[2].m128i_i64, &a2[3], v10, 8u, (__int64)a2[2].m128i_i64, v10),
                  v7 = (__m128i *)a2[2].m128i_i64[0],
                  v12 = (const __m128i *)i[-1].m128i_i64[0],
                  (v13 = 8LL * i[-1].m128i_u32[2]) != 0) )
            {
              memcpy(v7, v12, v13);
            }
            a2[2].m128i_i32[2] = v10;
            i[-1].m128i_i32[2] = 0;
          }
          else
          {
            a2[2].m128i_i64[0] = (__int64)v11;
            a2[2].m128i_i32[2] = i[-1].m128i_i32[2];
            a2[2].m128i_i32[3] = i[-1].m128i_i32[3];
            i[-1].m128i_i64[0] = (__int64)i;
            i[-1].m128i_i32[3] = 0;
            i[-1].m128i_i32[2] = 0;
          }
        }
        a2[5] = _mm_loadu_si128(i + 2);
        a2[6].m128i_i64[0] = i[3].m128i_i64[0];
      }
      a2 = (__m128i *)((char *)a2 + 104);
      if ( (unsigned __int64 *)v3 == &i[3].m128i_u64[1] )
        break;
    }
    v14 = *(const __m128i **)a1;
    result = 13LL * *(unsigned int *)(a1 + 8);
    v15 = *(_QWORD *)a1 + 104LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v15 )
    {
      do
      {
        v15 -= 104;
        v16 = *(_QWORD *)(v15 + 32);
        if ( v16 != v15 + 48 )
          _libc_free(v16);
        result = sub_C7D6A0(*(_QWORD *)(v15 + 8), 16LL * *(unsigned int *)(v15 + 24), 8);
      }
      while ( (const __m128i *)v15 != v14 );
    }
  }
  return result;
}
