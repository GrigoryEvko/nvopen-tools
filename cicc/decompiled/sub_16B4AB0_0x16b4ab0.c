// Function: sub_16B4AB0
// Address: 0x16b4ab0
//
__int64 __fastcall sub_16B4AB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rdx
  __int64 v5; // rax
  __m128i *v6; // rdx
  __int64 v7; // rdi
  __m128i si128; // xmm0

  sub_16B2F80(a1, a2, a3, a4);
  v5 = sub_16E8C20(a1, a2, v4);
  v6 = *(__m128i **)(v5 + 24);
  v7 = v5;
  if ( *(_QWORD *)(v5 + 16) - (_QWORD)v6 <= 0x1Du )
    return sub_16E7EE0(v5, "= *cannot print option value*\n", 30);
  si128 = _mm_load_si128((const __m128i *)&xmmword_3F66410);
  qmemcpy(&v6[1], "option value*\n", 14);
  *v6 = si128;
  *(_QWORD *)(v7 + 24) += 30LL;
  return 2602;
}
