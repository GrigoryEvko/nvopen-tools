// Function: sub_E4EFC0
// Address: 0xe4efc0
//
_BYTE *__fastcall sub_E4EFC0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rdi
  __m128i *v6; // rdx
  __int64 v7; // rax
  _WORD *v8; // rdx
  __int64 v9; // rdi

  sub_E4CF20(a1, a2, a3);
  v5 = *(_QWORD *)(a1 + 304);
  v6 = *(__m128i **)(v5 + 32);
  if ( *(_QWORD *)(v5 + 24) - (_QWORD)v6 <= 0xFu )
  {
    sub_CB6200(v5, ", subfield_reg, ", 0x10u);
  }
  else
  {
    *v6 = _mm_load_si128((const __m128i *)&xmmword_3F7F850);
    *(_QWORD *)(v5 + 32) += 16LL;
  }
  v7 = sub_CB59F0(*(_QWORD *)(a1 + 304), (unsigned __int16)a4);
  v8 = *(_WORD **)(v7 + 32);
  v9 = v7;
  if ( *(_QWORD *)(v7 + 24) - (_QWORD)v8 <= 1u )
  {
    v9 = sub_CB6200(v7, (unsigned __int8 *)", ", 2u);
  }
  else
  {
    *v8 = 8236;
    *(_QWORD *)(v7 + 32) += 2LL;
  }
  sub_CB59D0(v9, HIDWORD(a4));
  return sub_E4D880(a1);
}
