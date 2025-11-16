// Function: sub_25A4000
// Address: 0x25a4000
//
__int64 __fastcall sub_25A4000(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __m128i *v6; // r14
  unsigned __int64 *v7; // r12
  const __m128i *i; // r13
  const __m128i *v9; // rdx
  const __m128i *v10; // r13
  __int64 v11; // rax
  unsigned __int64 v12; // r14
  unsigned __int64 *v13; // r15
  int v14; // r13d
  __int64 v16; // [rsp+0h] [rbp-50h]
  unsigned __int64 v17[7]; // [rsp+18h] [rbp-38h] BYREF

  v16 = sub_C8D7D0(a1, a1 + 16, a2, 0xB0u, v17, a6);
  v6 = (__m128i *)v16;
  v7 = (unsigned __int64 *)(*(_QWORD *)a1 + 176LL * *(unsigned int *)(a1 + 8));
  if ( *(unsigned __int64 **)a1 != v7 )
  {
    for ( i = (const __m128i *)(*(_QWORD *)a1 + 16LL); ; i += 11 )
    {
      if ( v6 )
      {
        v6->m128i_i64[0] = (__int64)v6[1].m128i_i64;
        v9 = (const __m128i *)i[-1].m128i_i64[0];
        if ( i == v9 )
        {
          v6[1] = _mm_loadu_si128(i);
        }
        else
        {
          v6->m128i_i64[0] = (__int64)v9;
          v6[1].m128i_i64[0] = i->m128i_i64[0];
        }
        v6->m128i_i64[1] = i[-1].m128i_i64[1];
        i[-1].m128i_i64[0] = (__int64)i;
        i[-1].m128i_i64[1] = 0;
        i->m128i_i8[0] = 0;
        v6[2].m128i_i64[0] = (__int64)v6[3].m128i_i64;
        v6[2].m128i_i32[2] = 0;
        v6[2].m128i_i32[3] = 4;
        if ( i[1].m128i_i32[2] )
          sub_25A3B40((__int64)v6[2].m128i_i64, (__int64)i[1].m128i_i64);
      }
      v6 += 11;
      if ( v7 == (unsigned __int64 *)&i[10] )
        break;
    }
    v10 = *(const __m128i **)a1;
    v7 = (unsigned __int64 *)(*(_QWORD *)a1 + 176LL * *(unsigned int *)(a1 + 8));
    if ( *(unsigned __int64 **)a1 != v7 )
    {
      do
      {
        v11 = *((unsigned int *)v7 - 34);
        v12 = *(v7 - 18);
        v7 -= 22;
        v11 *= 32;
        v13 = (unsigned __int64 *)(v12 + v11);
        if ( v12 != v12 + v11 )
        {
          do
          {
            v13 -= 4;
            if ( (unsigned __int64 *)*v13 != v13 + 2 )
              j_j___libc_free_0(*v13);
          }
          while ( (unsigned __int64 *)v12 != v13 );
          v12 = v7[4];
        }
        if ( (unsigned __int64 *)v12 != v7 + 6 )
          _libc_free(v12);
        if ( (unsigned __int64 *)*v7 != v7 + 2 )
          j_j___libc_free_0(*v7);
      }
      while ( v7 != (unsigned __int64 *)v10 );
      v7 = *(unsigned __int64 **)a1;
    }
  }
  v14 = v17[0];
  if ( (unsigned __int64 *)(a1 + 16) != v7 )
    _libc_free((unsigned __int64)v7);
  *(_DWORD *)(a1 + 12) = v14;
  *(_QWORD *)a1 = v16;
  return v16;
}
