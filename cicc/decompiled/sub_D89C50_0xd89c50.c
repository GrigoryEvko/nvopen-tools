// Function: sub_D89C50
// Address: 0xd89c50
//
__int64 __fastcall sub_D89C50(__int64 a1, __int64 *a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  __m128i v4; // xmm1
  __int64 v5; // rax
  __m128i v6; // xmm0
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 result; // rax

  v2 = *a2;
  v3 = *(_QWORD *)(a1 + 32);
  *(_QWORD *)(a1 + 24) = 0;
  v4 = _mm_loadu_si128((const __m128i *)(a1 + 8));
  *(_QWORD *)a1 = v2;
  v5 = a2[3];
  v6 = _mm_loadu_si128((const __m128i *)(a2 + 1));
  a2[3] = 0;
  *(_QWORD *)(a1 + 24) = v5;
  v7 = a2[4];
  *(__m128i *)(a1 + 8) = v6;
  *(_QWORD *)(a1 + 32) = v7;
  v8 = a2[5];
  a2[4] = v3;
  *(__m128i *)(a2 + 1) = v4;
  *(_QWORD *)(a1 + 40) = v8;
  result = a2[6];
  *(_QWORD *)(a1 + 48) = result;
  a2[6] = 0;
  return result;
}
