// Function: sub_2FB04E0
// Address: 0x2fb04e0
//
__int64 __fastcall sub_2FB04E0(__int64 a1, __int64 a2)
{
  const __m128i *v2; // rdx
  __int64 result; // rax
  __m128i v4; // [rsp+0h] [rbp-30h]

  v2 = (const __m128i *)sub_2E09D00((__int64 *)a2, a1);
  if ( v2 == (const __m128i *)(*(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8))
    || (*(_DWORD *)((v2->m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v2->m128i_i64[0] >> 1) & 3) > (*(_DWORD *)((a1 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(a1 >> 1) & 3) )
  {
    return 1;
  }
  result = 0;
  if ( v2->m128i_i64[1] == (a1 & 0xFFFFFFFFFFFFFFF8LL | 6) )
  {
    v4 = _mm_loadu_si128(v2);
    sub_2E0C3B0(a2, v4.m128i_i64[0], v4.m128i_i64[1], 1);
    return 1;
  }
  return result;
}
