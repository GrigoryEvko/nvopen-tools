// Function: sub_2508BA0
// Address: 0x2508ba0
//
__int64 *__fastcall sub_2508BA0(__int64 *a1)
{
  _OWORD *v1; // rax
  unsigned __int64 v2; // rdx
  __m128i si128; // xmm0
  unsigned __int64 v4; // rax
  __int64 v5; // rdx
  unsigned __int64 v7; // [rsp+8h] [rbp-18h] BYREF

  *a1 = (__int64)(a1 + 2);
  v7 = 16;
  v1 = (_OWORD *)sub_22409D0((__int64)a1, &v7, 0);
  v2 = v7;
  si128 = _mm_load_si128((const __m128i *)&xmmword_4389C50);
  *a1 = (__int64)v1;
  a1[2] = v2;
  *v1 = si128;
  v4 = v7;
  v5 = *a1;
  a1[1] = v7;
  *(_BYTE *)(v5 + v4) = 0;
  return a1;
}
