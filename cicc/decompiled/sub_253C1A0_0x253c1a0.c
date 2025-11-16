// Function: sub_253C1A0
// Address: 0x253c1a0
//
__int64 *__fastcall sub_253C1A0(__int64 *a1)
{
  __int64 v1; // rax
  unsigned __int64 v2; // rdx
  __m128i si128; // xmm0
  unsigned __int64 v4; // rax
  __int64 v5; // rdx
  unsigned __int64 v7; // [rsp+8h] [rbp-18h] BYREF

  *a1 = (__int64)(a1 + 2);
  v7 = 20;
  v1 = sub_22409D0((__int64)a1, &v7, 0);
  v2 = v7;
  si128 = _mm_load_si128((const __m128i *)&xmmword_438A6A0);
  *a1 = v1;
  a1[2] = v2;
  *(_DWORD *)(v1 + 16) = 1701277281;
  *(__m128i *)v1 = si128;
  v4 = v7;
  v5 = *a1;
  a1[1] = v7;
  *(_BYTE *)(v5 + v4) = 0;
  return a1;
}
