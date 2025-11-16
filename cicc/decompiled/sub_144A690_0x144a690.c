// Function: sub_144A690
// Address: 0x144a690
//
__int64 __fastcall sub_144A690(__int64 *a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 result; // rax

  v3 = a1[1];
  if ( v3 == a1[2] )
    return sub_144A4B0(a1, v3, a2);
  if ( v3 )
  {
    *(_QWORD *)v3 = *(_QWORD *)a2;
    result = *(unsigned __int8 *)(a2 + 24);
    *(_BYTE *)(v3 + 24) = result;
    if ( (_BYTE)result )
      *(__m128i *)(v3 + 8) = _mm_loadu_si128((const __m128i *)(a2 + 8));
    v3 = a1[1];
  }
  a1[1] = v3 + 32;
  return result;
}
