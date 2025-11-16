// Function: sub_37F7FF0
// Address: 0x37f7ff0
//
__int64 __fastcall sub_37F7FF0(__int64 a1)
{
  const __m128i *v1; // rbx
  const __m128i *i; // r12
  __m128i v3; // xmm0
  __int64 result; // rax
  __m128i v5[3]; // [rsp+0h] [rbp-30h] BYREF

  v1 = *(const __m128i **)(a1 + 224);
  for ( i = &v1[*(unsigned int *)(a1 + 232)]; i != v1; result = sub_37F7F20(a1, (__int64)v5) )
  {
    v3 = _mm_loadu_si128(v1++);
    v5[0] = v3;
  }
  return result;
}
