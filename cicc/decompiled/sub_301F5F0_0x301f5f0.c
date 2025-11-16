// Function: sub_301F5F0
// Address: 0x301f5f0
//
void __fastcall sub_301F5F0(__int64 a1, _BYTE *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rdi
  int v9; // eax
  __int64 *v10; // rdi
  __int64 v11; // r12
  __int64 v12; // rcx
  __int64 *v13; // rdi
  const __m128i *v14; // rax
  unsigned __int64 *v15; // r14
  const __m128i *v16; // rax
  __m128i *v17; // rcx
  __m128i *v18; // rdx
  const __m128i *v19; // rsi
  unsigned __int64 *v20; // r15
  int v21; // r15d
  unsigned __int64 v22[7]; // [rsp+8h] [rbp-38h] BYREF

  v8 = *(unsigned int *)(a1 + 24);
  if ( *(_DWORD *)(a1 + 28) <= (unsigned int)v8 )
  {
    v11 = sub_C8D7D0(a1 + 16, a1 + 32, 0, 0x20u, v22, a6);
    v12 = 2LL * *(unsigned int *)(a1 + 24);
    v13 = (__int64 *)(v12 * 16 + v11);
    if ( v12 * 16 + v11 )
    {
      *v13 = (__int64)(v13 + 2);
      sub_301F430(v13, a2, (__int64)&a2[a3]);
      v12 = 2LL * *(unsigned int *)(a1 + 24);
    }
    v14 = *(const __m128i **)(a1 + 16);
    v15 = (unsigned __int64 *)&v14[v12];
    if ( v14 != &v14[v12] )
    {
      v16 = v14 + 1;
      v17 = (__m128i *)(v11 + v12 * 16);
      v18 = (__m128i *)v11;
      do
      {
        if ( v18 )
        {
          v18->m128i_i64[0] = (__int64)v18[1].m128i_i64;
          v19 = (const __m128i *)v16[-1].m128i_i64[0];
          if ( v16 == v19 )
          {
            v18[1] = _mm_loadu_si128(v16);
          }
          else
          {
            v18->m128i_i64[0] = (__int64)v19;
            v18[1].m128i_i64[0] = v16->m128i_i64[0];
          }
          v18->m128i_i64[1] = v16[-1].m128i_i64[1];
          v16[-1].m128i_i64[0] = (__int64)v16;
          v16[-1].m128i_i64[1] = 0;
          v16->m128i_i8[0] = 0;
        }
        v18 += 2;
        v16 += 2;
      }
      while ( v17 != v18 );
      v20 = *(unsigned __int64 **)(a1 + 16);
      v15 = &v20[4 * *(unsigned int *)(a1 + 24)];
      if ( v20 != v15 )
      {
        do
        {
          v15 -= 4;
          if ( (unsigned __int64 *)*v15 != v15 + 2 )
            j_j___libc_free_0(*v15);
        }
        while ( v20 != v15 );
        v15 = *(unsigned __int64 **)(a1 + 16);
      }
    }
    v21 = v22[0];
    if ( (unsigned __int64 *)(a1 + 32) != v15 )
      _libc_free((unsigned __int64)v15);
    *(_QWORD *)(a1 + 16) = v11;
    *(_DWORD *)(a1 + 28) = v21;
    ++*(_DWORD *)(a1 + 24);
  }
  else
  {
    v9 = v8;
    v10 = (__int64 *)(*(_QWORD *)(a1 + 16) + 32 * v8);
    if ( v10 )
    {
      *v10 = (__int64)(v10 + 2);
      sub_301F430(v10, a2, (__int64)&a2[a3]);
      v9 = *(_DWORD *)(a1 + 24);
    }
    *(_DWORD *)(a1 + 24) = v9 + 1;
  }
}
