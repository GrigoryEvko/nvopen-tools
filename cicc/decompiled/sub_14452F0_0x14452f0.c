// Function: sub_14452F0
// Address: 0x14452f0
//
unsigned __int64 __fastcall sub_14452F0(__int64 *a1, __int64 a2)
{
  __int64 v3; // rsi
  unsigned __int64 result; // rax

  v3 = a1[1];
  if ( v3 == a1[2] )
    return sub_14450F0(a1, v3, a2);
  if ( v3 )
  {
    *(_QWORD *)v3 = *(_QWORD *)a2;
    result = *(unsigned __int8 *)(a2 + 32);
    *(_BYTE *)(v3 + 32) = result;
    if ( (_BYTE)result )
    {
      result = *(_QWORD *)(a2 + 24);
      *(__m128i *)(v3 + 8) = _mm_loadu_si128((const __m128i *)(a2 + 8));
      *(_QWORD *)(v3 + 24) = result;
    }
    v3 = a1[1];
  }
  a1[1] = v3 + 40;
  return result;
}
