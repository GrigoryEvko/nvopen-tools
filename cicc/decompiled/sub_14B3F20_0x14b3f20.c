// Function: sub_14B3F20
// Address: 0x14b3f20
//
void __fastcall sub_14B3F20(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v3; // rdx
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // rbx
  unsigned __int64 v6; // rax
  __int64 v7; // r14
  const __m128i *v8; // rcx
  unsigned __int64 *v9; // r12
  const __m128i *v10; // rax
  __m128i *v11; // rdx
  __int64 m128i_i64; // rcx
  const __m128i *v13; // rsi
  __m128i v14; // xmm0
  __int64 v15; // rsi
  const __m128i *v16; // r15
  unsigned __int64 *v17; // rdi

  if ( a2 > 0xFFFFFFFF )
    sub_16BD1C0("SmallVector capacity overflow during allocation");
  v3 = ((((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
      | (*(unsigned int *)(a1 + 12) + 2LL)
      | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 4;
  v4 = ((v3
       | (((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
       | (*(unsigned int *)(a1 + 12) + 2LL)
       | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 8)
     | v3
     | (((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
     | (*(unsigned int *)(a1 + 12) + 2LL)
     | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1);
  v5 = a2;
  v6 = (v4 | (v4 >> 16) | HIDWORD(v4)) + 1;
  if ( v6 >= a2 )
    v5 = v6;
  if ( v5 > 0xFFFFFFFF )
    v5 = 0xFFFFFFFFLL;
  v7 = malloc(88 * v5);
  if ( !v7 )
    sub_16BD1C0("Allocation failed");
  v8 = *(const __m128i **)a1;
  v9 = (unsigned __int64 *)(*(_QWORD *)a1 + 88LL * *(unsigned int *)(a1 + 8));
  if ( *(unsigned __int64 **)a1 != v9 )
  {
    v10 = v8 + 3;
    v11 = (__m128i *)v7;
    m128i_i64 = (__int64)v8[1].m128i_i64;
    while ( 1 )
    {
      if ( v11 )
      {
        v11->m128i_i64[0] = (__int64)v11[1].m128i_i64;
        v15 = v10[-3].m128i_i64[0];
        if ( v15 == m128i_i64 )
        {
          v11[1] = _mm_loadu_si128(v10 - 2);
        }
        else
        {
          v11->m128i_i64[0] = v15;
          v11[1].m128i_i64[0] = v10[-2].m128i_i64[0];
        }
        v11->m128i_i64[1] = v10[-3].m128i_i64[1];
        v10[-3].m128i_i64[0] = m128i_i64;
        v10[-3].m128i_i64[1] = 0;
        v10[-2].m128i_i8[0] = 0;
        v11[2].m128i_i64[0] = (__int64)v11[3].m128i_i64;
        v13 = (const __m128i *)v10[-1].m128i_i64[0];
        if ( v13 == v10 )
        {
          v11[3] = _mm_loadu_si128(v10);
        }
        else
        {
          v11[2].m128i_i64[0] = (__int64)v13;
          v11[3].m128i_i64[0] = v10->m128i_i64[0];
        }
        v11[2].m128i_i64[1] = v10[-1].m128i_i64[1];
        v14 = _mm_loadu_si128(v10 + 1);
        v10[-1].m128i_i64[0] = (__int64)v10;
        v10[-1].m128i_i64[1] = 0;
        v10->m128i_i8[0] = 0;
        v11[4] = v14;
        v11[5].m128i_i64[0] = v10[2].m128i_i64[0];
      }
      v11 = (__m128i *)((char *)v11 + 88);
      m128i_i64 += 88;
      if ( v9 == &v10[2].m128i_u64[1] )
        break;
      v10 = (const __m128i *)((char *)v10 + 88);
    }
    v16 = *(const __m128i **)a1;
    v9 = (unsigned __int64 *)(*(_QWORD *)a1 + 88LL * *(unsigned int *)(a1 + 8));
    if ( *(unsigned __int64 **)a1 != v9 )
    {
      do
      {
        v9 -= 11;
        v17 = (unsigned __int64 *)v9[4];
        if ( v17 != v9 + 6 )
          j_j___libc_free_0(v17, v9[6] + 1);
        if ( (unsigned __int64 *)*v9 != v9 + 2 )
          j_j___libc_free_0(*v9, v9[2] + 1);
      }
      while ( v9 != (unsigned __int64 *)v16 );
      v9 = *(unsigned __int64 **)a1;
    }
  }
  if ( v9 != (unsigned __int64 *)(a1 + 16) )
    _libc_free((unsigned __int64)v9);
  *(_QWORD *)a1 = v7;
  *(_DWORD *)(a1 + 12) = v5;
}
