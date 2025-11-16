// Function: sub_C9A200
// Address: 0xc9a200
//
void __fastcall sub_C9A200(__int64 a1, __m128i *a2)
{
  unsigned __int64 v2; // r14
  const __m128i *i; // rbx
  const __m128i *v5; // rax
  __int32 v6; // eax
  const __m128i *v7; // rdx
  const __m128i *v8; // r12
  const __m128i *v9; // rbx
  const __m128i *v10; // rdi
  const __m128i *v11; // rdi
  const __m128i *v12; // rdi

  v2 = *(_QWORD *)a1 + ((unsigned __int64)*(unsigned int *)(a1 + 8) << 7);
  if ( *(_QWORD *)a1 != v2 )
  {
    for ( i = (const __m128i *)(*(_QWORD *)a1 + 96LL); ; i += 8 )
    {
      if ( a2 )
      {
        a2->m128i_i64[0] = i[-6].m128i_i64[0];
        a2->m128i_i64[1] = i[-6].m128i_i64[1];
        a2[1].m128i_i64[0] = (__int64)a2[2].m128i_i64;
        sub_C95D30(a2[1].m128i_i64, (_BYTE *)i[-5].m128i_i64[0], i[-5].m128i_i64[0] + i[-5].m128i_i64[1]);
        a2[3].m128i_i64[0] = (__int64)a2[4].m128i_i64;
        v7 = (const __m128i *)i[-3].m128i_i64[0];
        if ( v7 == &i[-2] )
        {
          a2[4] = _mm_loadu_si128(i - 2);
        }
        else
        {
          a2[3].m128i_i64[0] = (__int64)v7;
          a2[4].m128i_i64[0] = i[-2].m128i_i64[0];
        }
        a2[3].m128i_i64[1] = i[-3].m128i_i64[1];
        i[-3].m128i_i64[0] = (__int64)i[-2].m128i_i64;
        i[-3].m128i_i64[1] = 0;
        i[-2].m128i_i8[0] = 0;
        a2[5].m128i_i64[0] = (__int64)a2[6].m128i_i64;
        v5 = (const __m128i *)i[-1].m128i_i64[0];
        if ( v5 == i )
        {
          a2[6] = _mm_loadu_si128(i);
        }
        else
        {
          a2[5].m128i_i64[0] = (__int64)v5;
          a2[6].m128i_i64[0] = i->m128i_i64[0];
        }
        a2[5].m128i_i64[1] = i[-1].m128i_i64[1];
        v6 = i[1].m128i_i32[0];
        i[-1].m128i_i64[0] = (__int64)i;
        i[-1].m128i_i64[1] = 0;
        i->m128i_i8[0] = 0;
        a2[7].m128i_i32[0] = v6;
        a2[7].m128i_i32[2] = i[1].m128i_i32[2];
      }
      a2 += 8;
      if ( (const __m128i *)v2 == &i[2] )
        break;
    }
    v8 = *(const __m128i **)a1;
    v9 = (const __m128i *)(*(_QWORD *)a1 + ((unsigned __int64)*(unsigned int *)(a1 + 8) << 7));
    if ( *(const __m128i **)a1 != v9 )
    {
      do
      {
        v9 -= 8;
        v10 = (const __m128i *)v9[5].m128i_i64[0];
        if ( v10 != &v9[6] )
          j_j___libc_free_0(v10, v9[6].m128i_i64[0] + 1);
        v11 = (const __m128i *)v9[3].m128i_i64[0];
        if ( v11 != &v9[4] )
          j_j___libc_free_0(v11, v9[4].m128i_i64[0] + 1);
        v12 = (const __m128i *)v9[1].m128i_i64[0];
        if ( v12 != &v9[2] )
          j_j___libc_free_0(v12, v9[2].m128i_i64[0] + 1);
      }
      while ( v9 != v8 );
    }
  }
}
