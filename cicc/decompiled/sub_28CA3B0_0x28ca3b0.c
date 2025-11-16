// Function: sub_28CA3B0
// Address: 0x28ca3b0
//
void __fastcall sub_28CA3B0(__int64 a1, __int64 a2, char a3)
{
  __m128i *v3; // rdx
  __m128i si128; // xmm0

  if ( a3 )
    sub_904010(a2, "ExpressionTypeCall, ");
  sub_27AFB90(a1, a2, 0);
  v3 = *(__m128i **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v3 <= 0x13u )
  {
    sub_CB6200(a2, " represents call at ", 0x14u);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_4397BC0);
    v3[1].m128i_i32[0] = 544497952;
    *v3 = si128;
    *(_QWORD *)(a2 + 32) += 20LL;
  }
  sub_A5BF40(*(unsigned __int8 **)(a1 + 56), a2, 1, 0);
}
