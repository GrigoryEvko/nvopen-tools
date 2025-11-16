// Function: sub_7F7020
// Address: 0x7f7020
//
__m128i *__fastcall sub_7F7020(__int64 a1)
{
  __int64 v1; // rbx
  __m128i *v2; // r12
  _QWORD *v4; // r12
  const __m128i *v5; // rax
  char *v6; // rax
  __m128i *v7; // rax

  v1 = *(_QWORD *)(a1 + 168);
  v2 = *(__m128i **)(v1 + 200);
  if ( !v2 )
  {
    v4 = sub_7259C0(8);
    v5 = (const __m128i *)sub_7E1DC0();
    v4[20] = sub_73C570(v5, 1);
    v6 = (char *)sub_80F670(a1);
    v7 = sub_7E2190(v6, 0, (__int64)v4, 1);
    v7[5].m128i_i8[9] |= 8u;
    v2 = v7;
    v7[10].m128i_i8[8] = *(_BYTE *)(v1 + 109) & 7 | v7[10].m128i_i8[8] & 0xF8;
    *(_QWORD *)(v1 + 200) = v7;
  }
  sub_7E1230(v2, 0, HIDWORD(qword_4D045BC) == 0, 1);
  v2[9].m128i_i8[12] |= 0x40u;
  return v2;
}
