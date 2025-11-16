// Function: sub_E4EE30
// Address: 0xe4ee30
//
_BYTE *__fastcall sub_E4EE30(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rdi
  __m128i *v6; // rdx
  __m128i si128; // xmm0
  __int64 v8; // rdi
  _WORD *v9; // rdx

  v5 = *(_QWORD *)(a1 + 304);
  v6 = *(__m128i **)(v5 + 32);
  if ( *(_QWORD *)(v5 + 24) - (_QWORD)v6 <= 0x14u )
  {
    sub_CB6200(v5, ".lto_set_conditional ", 0x15u);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F7F840);
    v6[1].m128i_i32[0] = 1818324591;
    v6[1].m128i_i8[4] = 32;
    *v6 = si128;
    *(_QWORD *)(v5 + 32) += 21LL;
  }
  sub_EA12C0(a2, *(_QWORD *)(a1 + 304), *(_QWORD *)(a1 + 312));
  v8 = *(_QWORD *)(a1 + 304);
  v9 = *(_WORD **)(v8 + 32);
  if ( *(_QWORD *)(v8 + 24) - (_QWORD)v9 <= 1u )
  {
    sub_CB6200(v8, (unsigned __int8 *)", ", 2u);
  }
  else
  {
    *v9 = 8236;
    *(_QWORD *)(v8 + 32) += 2LL;
  }
  sub_E7FAD0(a3, *(_QWORD *)(a1 + 304), *(_QWORD *)(a1 + 312), 0);
  return sub_E4D880(a1);
}
