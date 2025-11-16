// Function: sub_824F50
// Address: 0x824f50
//
_QWORD *__fastcall sub_824F50(__m128i *a1)
{
  __m128i *v1; // rsi
  __int64 v2; // rdx
  _QWORD *result; // rax

  if ( (unsigned int)sub_824860(a1[2].m128i_i64[0], 3u) )
  {
    result = (_QWORD *)sub_824390(a1);
    if ( !(_DWORD)result )
    {
      sub_824E10((__int64)a1);
      a1->m128i_i64[0] = qword_4F07320[0];
      qword_4F07320[0] = a1;
      return &qword_4F07280;
    }
  }
  else
  {
    v1 = a1 + 1;
    v2 = *(_QWORD *)(a1[2].m128i_i64[0] + 8);
    if ( !unk_4D047F4 )
      sub_685200(0xBFEu, v1, v2);
    return (_QWORD *)sub_684B10(0xC79u, v1, v2);
  }
  return result;
}
