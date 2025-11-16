// Function: sub_2851200
// Address: 0x2851200
//
__int64 __fastcall sub_2851200(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  const __m128i *v7; // rbx
  __m128i *v8; // r14
  unsigned __int64 v9; // r12
  const __m128i *v10; // rbx
  int v11; // ebx
  __int64 v13; // [rsp+8h] [rbp-48h]
  unsigned __int64 v14[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = sub_C8D7D0(a1, a1 + 16, a2, 0x50u, v14, a6);
  v7 = *(const __m128i **)a1;
  v13 = v6;
  v8 = (__m128i *)v6;
  v9 = *(_QWORD *)a1 + 80LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v9 )
  {
    do
    {
      if ( v8 )
      {
        v8->m128i_i64[0] = v7->m128i_i64[0];
        v8->m128i_i64[1] = v7->m128i_i64[1];
        sub_C8CF70((__int64)v8[1].m128i_i64, &v8[3], 2, (__int64)v7[3].m128i_i64, (__int64)v7[1].m128i_i64);
        v8[4] = _mm_loadu_si128(v7 + 4);
      }
      v7 += 5;
      v8 += 5;
    }
    while ( (const __m128i *)v9 != v7 );
    v10 = *(const __m128i **)a1;
    v9 = *(_QWORD *)a1 + 80LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v9 )
    {
      do
      {
        while ( 1 )
        {
          v9 -= 80LL;
          if ( !*(_BYTE *)(v9 + 44) )
            break;
          if ( (const __m128i *)v9 == v10 )
            goto LABEL_10;
        }
        _libc_free(*(_QWORD *)(v9 + 24));
      }
      while ( (const __m128i *)v9 != v10 );
LABEL_10:
      v9 = *(_QWORD *)a1;
    }
  }
  v11 = v14[0];
  if ( a1 + 16 != v9 )
    _libc_free(v9);
  *(_DWORD *)(a1 + 12) = v11;
  *(_QWORD *)a1 = v13;
  return v13;
}
