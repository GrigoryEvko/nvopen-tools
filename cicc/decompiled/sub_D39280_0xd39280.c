// Function: sub_D39280
// Address: 0xd39280
//
__int64 __fastcall sub_D39280(unsigned int *a1, const __m128i *a2)
{
  __int64 result; // rax
  __int64 v3; // r13
  __m128i *v5; // r12
  const __m128i *i; // rbx
  const __m128i *v7; // rdx
  const __m128i *v8; // rax
  __int32 v9; // eax
  __int64 v10; // rax
  unsigned __int64 *v11; // rdi
  unsigned int v12; // r15d
  const __m128i *v13; // rax
  size_t v14; // rdx
  _QWORD *v15; // r12
  _QWORD *v16; // rbx
  _QWORD *v17; // rdi
  _QWORD *v18; // rdi
  _QWORD *v19; // rdi

  result = a1[2];
  v3 = *(_QWORD *)a1 + 224 * result;
  if ( *(_QWORD *)a1 != v3 )
  {
    v5 = (__m128i *)a2;
    for ( i = (const __m128i *)(*(_QWORD *)a1 + 200LL); ; i += 14 )
    {
      if ( v5 )
      {
        v10 = i[-13].m128i_i64[1];
        v11 = &v5[1].m128i_u64[1];
        v5[1].m128i_i32[0] = 0;
        v5->m128i_i64[1] = (__int64)&v5[1].m128i_i64[1];
        v5->m128i_i64[0] = v10;
        v5[1].m128i_i32[1] = 8;
        v12 = i[-12].m128i_u32[2];
        if ( v12 && &v5->m128i_u64[1] != (unsigned __int64 *)&i[-12] )
        {
          v13 = (const __m128i *)i[-12].m128i_i64[0];
          a2 = i - 11;
          if ( v13 == &i[-11] )
          {
            v14 = 16LL * v12;
            if ( v12 <= 8
              || (sub_C8D5F0(
                    (__int64)&v5->m128i_i64[1],
                    &v5[1].m128i_u64[1],
                    v12,
                    0x10u,
                    (__int64)&v5->m128i_i64[1],
                    v12),
                  v11 = (unsigned __int64 *)v5->m128i_i64[1],
                  a2 = (const __m128i *)i[-12].m128i_i64[0],
                  (v14 = 16LL * i[-12].m128i_u32[2]) != 0) )
            {
              memcpy(v11, a2, v14);
            }
            v5[1].m128i_i32[0] = v12;
            i[-12].m128i_i32[2] = 0;
          }
          else
          {
            v5->m128i_i64[1] = (__int64)v13;
            v5[1].m128i_i32[0] = i[-12].m128i_i32[2];
            v5[1].m128i_i32[1] = i[-12].m128i_i32[3];
            i[-12].m128i_i64[0] = (__int64)a2;
            i[-12].m128i_i32[3] = 0;
            i[-12].m128i_i32[2] = 0;
          }
        }
        v5[9].m128i_i64[1] = (__int64)&v5[10].m128i_i64[1];
        v7 = (const __m128i *)i[-3].m128i_i64[0];
        if ( v7 == &i[-2] )
        {
          *(__m128i *)((char *)v5 + 168) = _mm_loadu_si128(i - 2);
        }
        else
        {
          v5[9].m128i_i64[1] = (__int64)v7;
          v5[10].m128i_i64[1] = i[-2].m128i_i64[0];
        }
        v5[10].m128i_i64[0] = i[-3].m128i_i64[1];
        i[-3].m128i_i64[0] = (__int64)i[-2].m128i_i64;
        i[-3].m128i_i64[1] = 0;
        i[-2].m128i_i8[0] = 0;
        v5[11].m128i_i64[1] = (__int64)&v5[12].m128i_i64[1];
        v8 = (const __m128i *)i[-1].m128i_i64[0];
        if ( i == v8 )
        {
          *(__m128i *)((char *)v5 + 200) = _mm_loadu_si128(i);
        }
        else
        {
          v5[11].m128i_i64[1] = (__int64)v8;
          v5[12].m128i_i64[1] = i->m128i_i64[0];
        }
        v5[12].m128i_i64[0] = i[-1].m128i_i64[1];
        v9 = i[1].m128i_i32[0];
        i[-1].m128i_i64[0] = (__int64)i;
        i[-1].m128i_i64[1] = 0;
        i->m128i_i8[0] = 0;
        v5[13].m128i_i32[2] = v9;
      }
      v5 += 14;
      if ( (unsigned __int64 *)v3 == &i[1].m128i_u64[1] )
        break;
    }
    result = a1[2];
    v15 = *(_QWORD **)a1;
    v16 = (_QWORD *)(*(_QWORD *)a1 + 224 * result);
    if ( *(_QWORD **)a1 != v16 )
    {
      do
      {
        v16 -= 28;
        v17 = (_QWORD *)v16[23];
        if ( v17 != v16 + 25 )
        {
          a2 = (const __m128i *)(v16[25] + 1LL);
          j_j___libc_free_0(v17, a2);
        }
        v18 = (_QWORD *)v16[19];
        if ( v18 != v16 + 21 )
        {
          a2 = (const __m128i *)(v16[21] + 1LL);
          j_j___libc_free_0(v18, a2);
        }
        v19 = (_QWORD *)v16[1];
        result = (__int64)(v16 + 3);
        if ( v19 != v16 + 3 )
          result = _libc_free(v19, a2);
      }
      while ( v16 != v15 );
    }
  }
  return result;
}
