// Function: sub_770C80
// Address: 0x770c80
//
__int64 __fastcall sub_770C80(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __m128i *a5)
{
  __int64 result; // rax

  result = 0;
  if ( (*(_BYTE *)(a1 + 132) & 1) != 0 )
  {
    result = 1;
    *a5 = _mm_load_si128((const __m128i *)&xmmword_4F08280);
    if ( !**(_QWORD **)(a1 + 72) && (*(_BYTE *)(a1 + 132) & 2) == 0 )
    {
      sub_770C10(a2, a3);
      return 1;
    }
  }
  return result;
}
