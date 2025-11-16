// Function: sub_B4EB40
// Address: 0xb4eb40
//
__int64 __fastcall sub_B4EB40(__int64 a1, __int64 a2, void *a3, __int64 a4, __int64 a5, __int64 a6, __int128 a7)
{
  __int64 **v11; // rdi
  __int64 v12; // rax
  __m128i si128; // xmm0
  __m128i v15[4]; // [rsp+0h] [rbp-40h] BYREF

  v11 = *(__int64 ***)(a2 + 8);
  v15[0] = _mm_loadu_si128((const __m128i *)&a7);
  v12 = sub_ACADE0(v11);
  si128 = _mm_load_si128(v15);
  return sub_B4E9E0(a1, a2, v12, a3, a4, a5, si128.m128i_i64[0], si128.m128i_i64[1]);
}
