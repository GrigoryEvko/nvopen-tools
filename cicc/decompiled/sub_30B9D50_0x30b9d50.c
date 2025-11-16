// Function: sub_30B9D50
// Address: 0x30b9d50
//
void __fastcall sub_30B9D50(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r13
  __int64 v8; // r12
  const __m128i *v9; // rax
  unsigned __int64 v10; // rdi
  __m128i *v11; // rdx
  int v12; // r14d
  unsigned __int64 v13[5]; // [rsp+8h] [rbp-28h] BYREF

  v6 = a1 + 16;
  v8 = sub_C8D7D0(a1, a1 + 16, a2, 0x28u, v13, a6);
  v9 = *(const __m128i **)a1;
  v10 = *(_QWORD *)a1 + 40LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v10 )
  {
    v11 = (__m128i *)v8;
    do
    {
      if ( v11 )
      {
        *v11 = _mm_loadu_si128(v9);
        v11[1] = _mm_loadu_si128(v9 + 1);
        v11[2].m128i_i64[0] = v9[2].m128i_i64[0];
      }
      v9 = (const __m128i *)((char *)v9 + 40);
      v11 = (__m128i *)((char *)v11 + 40);
    }
    while ( (const __m128i *)v10 != v9 );
    v10 = *(_QWORD *)a1;
  }
  v12 = v13[0];
  if ( v6 != v10 )
    _libc_free(v10);
  *(_QWORD *)a1 = v8;
  *(_DWORD *)(a1 + 12) = v12;
}
