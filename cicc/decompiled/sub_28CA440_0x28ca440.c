// Function: sub_28CA440
// Address: 0x28ca440
//
_BYTE *__fastcall sub_28CA440(__int64 a1, __int64 a2, char a3)
{
  __int64 v3; // r12
  __m128i *v4; // rdx
  __m128i si128; // xmm0
  __int64 v6; // rax

  v3 = a2;
  if ( a3 )
    sub_904010(a2, "ExpressionTypeStore, ");
  sub_27AFB90(a1, a2, 0);
  v4 = *(__m128i **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v4 <= 0x12u )
  {
    a2 = sub_CB6200(a2, " represents Store  ", 0x13u);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_4397BD0);
    v4[1].m128i_i8[2] = 32;
    v4[1].m128i_i16[0] = 8293;
    *v4 = si128;
    *(_QWORD *)(a2 + 32) += 19LL;
  }
  sub_A69870(*(_QWORD *)(a1 + 56), (_BYTE *)a2, 0);
  sub_904010(v3, " with StoredValue ");
  sub_A5BF40(*(unsigned __int8 **)(a1 + 64), v3, 1, 0);
  v6 = sub_904010(v3, " and MemoryLeader ");
  return sub_103D830(*(_QWORD *)(a1 + 48), v6);
}
