// Function: sub_E58F10
// Address: 0xe58f10
//
_BYTE *__fastcall sub_E58F10(__int64 a1, signed __int64 a2, signed __int64 a3)
{
  __int64 v5; // rdi
  __m128i *v6; // rdx
  __m128i si128; // xmm0
  __int64 v8; // rdi
  _WORD *v9; // rdx

  sub_E9D600();
  v5 = *(_QWORD *)(a1 + 304);
  v6 = *(__m128i **)(v5 + 32);
  if ( *(_QWORD *)(v5 + 24) - (_QWORD)v6 <= 0x10u )
  {
    sub_CB6200(v5, "\t.cfi_rel_offset ", 0x11u);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F7F9B0);
    v6[1].m128i_i8[0] = 32;
    *v6 = si128;
    *(_QWORD *)(v5 + 32) += 17LL;
  }
  sub_E4C9A0(a1, a2);
  v8 = *(_QWORD *)(a1 + 304);
  v9 = *(_WORD **)(v8 + 32);
  if ( *(_QWORD *)(v8 + 24) - (_QWORD)v9 <= 1u )
  {
    v8 = sub_CB6200(v8, (unsigned __int8 *)", ", 2u);
  }
  else
  {
    *v9 = 8236;
    *(_QWORD *)(v8 + 32) += 2LL;
  }
  sub_CB59F0(v8, a3);
  return sub_E4D880(a1);
}
