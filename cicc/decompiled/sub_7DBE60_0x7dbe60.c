// Function: sub_7DBE60
// Address: 0x7dbe60
//
__int64 sub_7DBE60()
{
  __int64 i; // rbx
  unsigned int v1; // edi
  __int64 result; // rax
  __int64 v3; // rbx
  _OWORD *v4; // rax
  __m128i si128; // xmm0
  const __m128i *v6; // rax
  __m128i *v7; // rax
  unsigned __int64 v8; // rsi

  for ( i = 1; i != 11; ++i )
  {
    while ( qword_4F18960[i] )
    {
      if ( ++i == 11 )
        goto LABEL_5;
    }
    v1 = i;
    sub_7DB910(v1, 0);
  }
LABEL_5:
  result = qword_4F18960[0];
  if ( !qword_4F18960[0] )
  {
    qword_4F18960[0] = sub_7E16B0(10);
    sub_7E1CA0(qword_4F18960[0]);
    v3 = qword_4F18960[0];
    v4 = (_OWORD *)sub_7E1510(16);
    si128 = _mm_load_si128((const __m128i *)&xmmword_3C1ADC0);
    *(_QWORD *)(v3 + 8) = v4;
    *v4 = si128;
    *(_BYTE *)(v3 + 88) = *(_BYTE *)(v3 + 88) & 0x8F | 0x20;
    sub_7E1DC0();
    sub_7E1B70("__vptr");
    v6 = (const __m128i *)sub_72BA30(0);
    v7 = sub_73C570(v6, 1);
    v8 = sub_72D2E0(v7);
    sub_7E1B70("__name");
    sub_7E1C00(qword_4F18960[0], v8);
    return qword_4F18960[0];
  }
  return result;
}
