// Function: sub_6E5070
// Address: 0x6e5070
//
__int64 __fastcall sub_6E5070(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  if ( (*(_BYTE *)(a2 + 20) & 4) != 0 )
  {
    *(_BYTE *)(a1 + 20) |= 4u;
    *(__m128i *)(a1 + 24) = _mm_loadu_si128((const __m128i *)(a2 + 24));
    *(__m128i *)(a1 + 40) = _mm_loadu_si128((const __m128i *)(a2 + 40));
    result = *(_QWORD *)(a2 + 56);
    *(_QWORD *)(a1 + 56) = result;
  }
  return result;
}
