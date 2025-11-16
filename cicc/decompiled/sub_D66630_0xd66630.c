// Function: sub_D66630
// Address: 0xd66630
//
__m128i *__fastcall sub_D66630(__m128i *a1, __int64 a2)
{
  __int64 v2; // r13
  unsigned __int64 v3; // rax
  char v4; // dl
  __int64 v5; // rdx
  __m128i v6; // xmm0
  __m128i v8; // xmm1
  __m128i v9; // [rsp+10h] [rbp-40h] BYREF
  __m128i v10[3]; // [rsp+20h] [rbp-30h] BYREF

  v2 = sub_B43CC0(a2);
  sub_B91FC0(v9.m128i_i64, a2);
  v3 = (unsigned __int64)(sub_9208B0(v2, *(_QWORD *)(*(_QWORD *)(a2 - 64) + 8LL)) + 7) >> 3;
  if ( v4 )
    v3 |= 0x4000000000000000uLL;
  v5 = *(_QWORD *)(a2 - 32);
  v6 = _mm_loadu_si128(&v9);
  a1->m128i_i64[1] = v3;
  v8 = _mm_loadu_si128(v10);
  a1->m128i_i64[0] = v5;
  a1[1] = v6;
  a1[2] = v8;
  return a1;
}
