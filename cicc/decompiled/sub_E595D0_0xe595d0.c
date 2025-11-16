// Function: sub_E595D0
// Address: 0xe595d0
//
_BYTE *__fastcall sub_E595D0(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v5; // rdi
  __m128i *v6; // rdx
  __m128i si128; // xmm0
  __int64 v8; // rax
  _WORD *v9; // rdx

  sub_E993E0();
  v5 = *(_QWORD *)(a1 + 304);
  v6 = *(__m128i **)(v5 + 32);
  if ( *(_QWORD *)(v5 + 24) - (_QWORD)v6 <= 0x11u )
  {
    v5 = sub_CB6200(v5, "\t.cfi_personality ", 0x12u);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F7F9F0);
    v6[1].m128i_i16[0] = 8313;
    *v6 = si128;
    *(_QWORD *)(v5 + 32) += 18LL;
  }
  v8 = sub_CB59D0(v5, a3);
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
  sub_EA12C0(a2, *(_QWORD *)(a1 + 304), *(_QWORD *)(a1 + 312));
  return sub_E4D880(a1);
}
