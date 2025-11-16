// Function: sub_26E16C0
// Address: 0x26e16c0
//
unsigned __int64 __fastcall sub_26E16C0(unsigned __int64 *a1, __int64 a2)
{
  unsigned __int64 v3; // rsi
  unsigned __int64 result; // rax
  __m128i v5; // xmm0

  v3 = a1[1];
  if ( v3 == a1[2] )
    return sub_26E1510(a1, (const __m128i *)v3, a2);
  if ( v3 )
  {
    result = *(_QWORD *)a2;
    v5 = _mm_loadu_si128((const __m128i *)(a2 + 8));
    *(_QWORD *)v3 = *(_QWORD *)a2;
    *(__m128i *)(v3 + 8) = v5;
    v3 = a1[1];
  }
  a1[1] = v3 + 24;
  return result;
}
