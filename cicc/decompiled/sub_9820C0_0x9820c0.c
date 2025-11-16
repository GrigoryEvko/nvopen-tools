// Function: sub_9820C0
// Address: 0x9820c0
//
__int64 __fastcall sub_9820C0(const __m128i *a1, __int64 (__fastcall *a2)(__m128i *, const __m128i *))
{
  const __m128i *v2; // rbx
  __m128i v3; // xmm0
  __m128i v4; // xmm1
  __m128i v5; // xmm2
  __m128i v6; // xmm3
  __int64 result; // rax
  __m128i v8; // xmm5
  __m128i v9; // xmm6
  __m128i v10; // xmm7
  __m128i v11; // [rsp+0h] [rbp-60h] BYREF
  __m128i v12; // [rsp+10h] [rbp-50h] BYREF
  __m128i v13; // [rsp+20h] [rbp-40h] BYREF
  __m128i v14[3]; // [rsp+30h] [rbp-30h] BYREF

  v2 = a1 - 4;
  v11 = _mm_loadu_si128(a1);
  v12 = _mm_loadu_si128(a1 + 1);
  v13 = _mm_loadu_si128(a1 + 2);
  v14[0] = _mm_loadu_si128(a1 + 3);
  while ( 1 )
  {
    result = a2(&v11, v2);
    if ( !(_BYTE)result )
      break;
    v3 = _mm_loadu_si128(v2);
    v4 = _mm_loadu_si128(v2 + 1);
    v2 -= 4;
    v5 = _mm_loadu_si128(v2 + 6);
    v6 = _mm_loadu_si128(v2 + 7);
    v2[8] = v3;
    v2[9] = v4;
    v2[10] = v5;
    v2[11] = v6;
  }
  v8 = _mm_loadu_si128(&v12);
  v9 = _mm_loadu_si128(&v13);
  v10 = _mm_loadu_si128(v14);
  v2[4] = _mm_loadu_si128(&v11);
  v2[5] = v8;
  v2[6] = v9;
  v2[7] = v10;
  return result;
}
