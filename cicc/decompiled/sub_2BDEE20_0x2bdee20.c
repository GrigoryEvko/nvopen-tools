// Function: sub_2BDEE20
// Address: 0x2bdee20
//
__m128i *__fastcall sub_2BDEE20(__m128i *a1, _QWORD *a2)
{
  unsigned __int64 v3; // rdi
  __int64 v4; // rax
  __m128i v5; // xmm0
  const __m128i *v7; // rax
  __m128i v8; // xmm1
  __int64 v9; // rax
  __int64 *v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx

  v3 = a2[44];
  if ( v3 == a2[45] )
  {
    v7 = *(const __m128i **)(a2[47] - 8LL);
    v8 = _mm_loadu_si128(v7 + 30);
    v9 = v7[31].m128i_i64[0];
    *a1 = v8;
    a1[1].m128i_i64[0] = v9;
    j_j___libc_free_0(v3);
    v10 = (__int64 *)(a2[47] - 8LL);
    a2[47] = v10;
    v11 = *v10;
    a2[45] = *v10;
    v12 = v11 + 504;
    a2[44] = v11 + 480;
    a2[46] = v12;
    return a1;
  }
  else
  {
    v4 = *(_QWORD *)(v3 - 8);
    v5 = _mm_loadu_si128((const __m128i *)(v3 - 24));
    a2[44] = v3 - 24;
    a1[1].m128i_i64[0] = v4;
    *a1 = v5;
    return a1;
  }
}
