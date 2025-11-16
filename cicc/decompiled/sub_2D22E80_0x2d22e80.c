// Function: sub_2D22E80
// Address: 0x2d22e80
//
__int64 __fastcall sub_2D22E80(__int64 a1)
{
  __int64 v1; // rax
  unsigned __int64 v2; // r8
  char v3; // si
  unsigned __int64 v4; // rdx
  unsigned __int64 v5; // rax
  __m128i v6; // xmm0
  __m128i v7; // xmm1
  __m128i *v8; // rcx
  __int64 result; // rax
  __m128i v10; // xmm4
  __m128i v11; // xmm5
  __m128i v12; // [rsp+0h] [rbp-30h] BYREF
  __m128i v13; // [rsp+10h] [rbp-20h] BYREF
  __int64 v14; // [rsp+20h] [rbp-10h]

  v1 = *(_QWORD *)(a1 + 32);
  v2 = *(_QWORD *)(a1 + 8);
  v3 = *(_BYTE *)(a1 + 24);
  v12 = _mm_loadu_si128((const __m128i *)a1);
  v14 = v1;
  v13 = _mm_loadu_si128((const __m128i *)(a1 + 16));
  while ( 1 )
  {
    v8 = (__m128i *)a1;
    v4 = *(_BYTE *)(a1 - 16) ? *(_QWORD *)(a1 - 32) : qword_4F81350[0];
    v5 = v2;
    if ( !v3 )
      v5 = qword_4F81350[0];
    a1 -= 40;
    if ( v5 >= v4 )
      break;
    v6 = _mm_loadu_si128((const __m128i *)a1);
    v7 = _mm_loadu_si128((const __m128i *)(a1 + 16));
    *(_QWORD *)(a1 + 72) = *(_QWORD *)(a1 + 32);
    *(__m128i *)(a1 + 40) = v6;
    *(__m128i *)(a1 + 56) = v7;
  }
  result = v14;
  v10 = _mm_loadu_si128(&v12);
  v11 = _mm_loadu_si128(&v13);
  v8[2].m128i_i64[0] = v14;
  *v8 = v10;
  v8[1] = v11;
  return result;
}
