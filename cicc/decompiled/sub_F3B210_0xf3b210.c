// Function: sub_F3B210
// Address: 0xf3b210
//
__int64 __fastcall sub_F3B210(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rbx
  bool v6; // zf
  __int64 result; // rax
  __int64 v8; // rdx
  __int64 i; // rdx
  __int64 v10; // rsi
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __m128i *v14; // rax
  __m128i *v15; // r15
  __int64 v16; // rdx
  __int64 v17; // rdi
  __m128i *v18; // [rsp+8h] [rbp-38h] BYREF

  v5 = a2;
  v6 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v6 )
  {
    result = *(_QWORD *)(a1 + 16);
    v8 = 96LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    result = a1 + 16;
    v8 = 384;
  }
  for ( i = result + v8; i != result; result += 96 )
  {
    if ( result )
    {
      *(_QWORD *)result = 0;
      *(_BYTE *)(result + 24) = 0;
      *(_QWORD *)(result + 32) = 0;
    }
  }
  if ( a2 != a3 )
  {
    do
    {
      if ( *(_QWORD *)v5 || *(_BYTE *)(v5 + 24) && (*(_QWORD *)(v5 + 8) || *(_QWORD *)(v5 + 16)) || *(_QWORD *)(v5 + 32) )
      {
        v10 = v5;
        sub_F386A0(a1, v5, (__int64 *)&v18);
        v14 = v18;
        *v18 = _mm_loadu_si128((const __m128i *)v5);
        v15 = v18;
        v14[1] = _mm_loadu_si128((const __m128i *)(v5 + 16));
        v16 = *(_QWORD *)(v5 + 32);
        v14[2].m128i_i64[0] = v16;
        v15[2].m128i_i64[1] = (__int64)&v15[3].m128i_i64[1];
        v15[3].m128i_i64[0] = 0x400000000LL;
        if ( *(_DWORD *)(v5 + 48) )
        {
          v10 = v5 + 40;
          sub_F334E0((__int64)&v15[2].m128i_i64[1], (char **)(v5 + 40), v16, v11, v12, v13);
        }
        v15[5].m128i_i64[1] = *(_QWORD *)(v5 + 88);
        *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
        v17 = *(_QWORD *)(v5 + 40);
        result = v5 + 56;
        if ( v17 != v5 + 56 )
          result = _libc_free(v17, v10);
      }
      v5 += 96;
    }
    while ( a3 != v5 );
  }
  return result;
}
