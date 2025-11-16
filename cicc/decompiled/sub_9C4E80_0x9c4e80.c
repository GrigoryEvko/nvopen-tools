// Function: sub_9C4E80
// Address: 0x9c4e80
//
__int64 *__fastcall sub_9C4E80(__int64 *a1, __int64 a2, int a3)
{
  __int64 v3; // rbp
  __int64 v4; // rax
  __int64 v5; // rdx
  __m128i si128; // xmm0
  __int64 v7; // rax
  __int64 v8; // rdx
  _QWORD v10[4]; // [rsp-20h] [rbp-20h] BYREF

  if ( a3 != 1 )
    BUG();
  v10[3] = v3;
  *a1 = (__int64)(a1 + 2);
  v10[0] = 17;
  v4 = sub_22409D0(a1, v10, 0);
  v5 = v10[0];
  si128 = _mm_load_si128((const __m128i *)&xmmword_3F222D0);
  *a1 = v4;
  a1[2] = v5;
  *(_BYTE *)(v4 + 16) = 101;
  *(__m128i *)v4 = si128;
  v7 = v10[0];
  v8 = *a1;
  a1[1] = v10[0];
  *(_BYTE *)(v8 + v7) = 0;
  return a1;
}
