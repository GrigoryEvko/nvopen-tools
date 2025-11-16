// Function: sub_29F7CF0
// Address: 0x29f7cf0
//
void __fastcall sub_29F7CF0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  const __m128i *v7; // rdx
  __m128i *v8; // r12
  unsigned __int64 *v9; // r14
  const __m128i *v10; // rax
  unsigned __int64 v11; // rsi
  __m128i *v12; // rdx
  __int64 v13; // rcx
  const __m128i *v14; // rcx
  unsigned __int64 *v15; // r15
  int v16; // r15d
  unsigned __int64 v17[7]; // [rsp+8h] [rbp-38h] BYREF

  v6 = sub_C8D7D0(a1, a1 + 16, a2, 0x28u, v17, a6);
  v7 = *(const __m128i **)a1;
  v8 = (__m128i *)v6;
  v9 = (unsigned __int64 *)(*(_QWORD *)a1 + 40LL * *(unsigned int *)(a1 + 8));
  if ( *(unsigned __int64 **)a1 != v9 )
  {
    v10 = v7 + 1;
    v11 = (unsigned __int64)&v8[2].m128i_u64[((unsigned __int64)((char *)v9 - (char *)v7 - 40) >> 3) + 1];
    v12 = v8;
    do
    {
      if ( v12 )
      {
        v12->m128i_i64[0] = (__int64)v12[1].m128i_i64;
        v14 = (const __m128i *)v10[-1].m128i_i64[0];
        if ( v14 == v10 )
        {
          v12[1] = _mm_loadu_si128(v10);
        }
        else
        {
          v12->m128i_i64[0] = (__int64)v14;
          v12[1].m128i_i64[0] = v10->m128i_i64[0];
        }
        v12->m128i_i64[1] = v10[-1].m128i_i64[1];
        v13 = v10[1].m128i_i64[0];
        v10[-1].m128i_i64[0] = (__int64)v10;
        v10[-1].m128i_i64[1] = 0;
        v10->m128i_i8[0] = 0;
        v12[2].m128i_i64[0] = v13;
      }
      v12 = (__m128i *)((char *)v12 + 40);
      v10 = (const __m128i *)((char *)v10 + 40);
    }
    while ( v12 != (__m128i *)v11 );
    v15 = *(unsigned __int64 **)a1;
    v9 = (unsigned __int64 *)(*(_QWORD *)a1 + 40LL * *(unsigned int *)(a1 + 8));
    if ( *(unsigned __int64 **)a1 != v9 )
    {
      do
      {
        v9 -= 5;
        if ( (unsigned __int64 *)*v9 != v9 + 2 )
          j_j___libc_free_0(*v9);
      }
      while ( v15 != v9 );
      v9 = *(unsigned __int64 **)a1;
    }
  }
  v16 = v17[0];
  if ( (unsigned __int64 *)(a1 + 16) != v9 )
    _libc_free((unsigned __int64)v9);
  *(_QWORD *)a1 = v8;
  *(_DWORD *)(a1 + 12) = v16;
}
