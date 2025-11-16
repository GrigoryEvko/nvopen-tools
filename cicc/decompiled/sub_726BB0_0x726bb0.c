// Function: sub_726BB0
// Address: 0x726bb0
//
_BYTE *__fastcall sub_726BB0(unsigned __int8 a1)
{
  _BYTE *result; // rax
  __m128i v2; // xmm0

  result = sub_7246D0(64);
  result[9] &= 0xF0u;
  *(_QWORD *)result = 0;
  result[8] = a1;
  if ( a1 > 2u )
  {
    if ( a1 != 3 )
      sub_721090();
  }
  else
  {
    *((_QWORD *)result + 2) = 0;
  }
  *((_QWORD *)result + 3) = 0;
  *((_QWORD *)result + 4) = 0;
  v2 = _mm_loadu_si128((const __m128i *)&unk_4F07370);
  *((_QWORD *)result + 7) = 0;
  *(__m128i *)(result + 40) = v2;
  return result;
}
