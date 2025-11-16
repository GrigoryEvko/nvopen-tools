// Function: sub_226AAD0
// Address: 0x226aad0
//
void __fastcall sub_226AAD0(__int64 a1)
{
  __m128i v1; // [rsp+0h] [rbp-10h] BYREF

  v1.m128i_i64[1] = 0;
  *(__m128i *)(a1 + 136) = _mm_loadu_si128(&v1);
}
