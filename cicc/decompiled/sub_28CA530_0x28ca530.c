// Function: sub_28CA530
// Address: 0x28ca530
//
_BYTE *__fastcall sub_28CA530(__int64 a1, __int64 a2, char a3)
{
  __m128i *v3; // rdx
  __m128i si128; // xmm0
  __int64 v6; // rax

  if ( a3 )
    sub_904010(a2, "ExpressionTypeLoad, ");
  sub_27AFB90(a1, a2, 0);
  sub_904010(a2, " represents Load at ");
  sub_A5BF40(*(unsigned __int8 **)(a1 + 56), a2, 1, 0);
  v3 = *(__m128i **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v3 <= 0x12u )
  {
    v6 = sub_CB6200(a2, " with MemoryLeader ", 0x13u);
    return sub_103D830(*(_QWORD *)(a1 + 48), v6);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_4397BE0);
    v3[1].m128i_i8[2] = 32;
    v3[1].m128i_i16[0] = 29285;
    *v3 = si128;
    *(_QWORD *)(a2 + 32) += 19LL;
    return sub_103D830(*(_QWORD *)(a1 + 48), a2);
  }
}
