// Function: sub_1BE3770
// Address: 0x1be3770
//
__int64 __fastcall sub_1BE3770(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r12
  __int64 v5; // rdx
  __m128i *v6; // rdx
  __m128i si128; // xmm0
  __int64 v8; // rdx

  v4 = a2;
  v5 = *(_QWORD *)(a2 + 24);
  if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v5) <= 2 )
  {
    v4 = sub_16E7EE0(a2, " +\n", 3u);
  }
  else
  {
    *(_BYTE *)(v5 + 2) = 10;
    *(_WORD *)v5 = 11040;
    *(_QWORD *)(a2 + 24) += 3LL;
  }
  sub_16E2CE0(a3, v4);
  v6 = *(__m128i **)(v4 + 24);
  if ( *(_QWORD *)(v4 + 16) - (_QWORD)v6 <= 0x1Bu )
  {
    v4 = sub_16E7EE0(v4, "\"PHI-PREDICATED-INSTRUCTION ", 0x1Cu);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_42CAAA0);
    qmemcpy(&v6[1], "INSTRUCTION ", 12);
    *v6 = si128;
    *(_QWORD *)(v4 + 24) += 28LL;
  }
  sub_1BE27E0(v4, *(_QWORD *)(a1 + 40));
  v8 = *(_QWORD *)(v4 + 24);
  if ( (unsigned __int64)(*(_QWORD *)(v4 + 16) - v8) <= 2 )
    return sub_16E7EE0(v4, "\\l\"", 3u);
  *(_BYTE *)(v8 + 2) = 34;
  *(_WORD *)v8 = 27740;
  *(_QWORD *)(v4 + 24) += 3LL;
  return 27740;
}
