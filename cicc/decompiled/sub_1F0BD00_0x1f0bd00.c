// Function: sub_1F0BD00
// Address: 0x1f0bd00
//
__int64 sub_1F0BD00()
{
  _QWORD *v0; // rax
  __m128i *v1; // rdx
  __int64 v2; // rdi
  __m128i si128; // xmm0
  __m128i *v4; // rdx
  __int64 v5; // rax
  __m128i v6; // xmm0

  v0 = sub_16E8CB0();
  v1 = (__m128i *)v0[3];
  v2 = (__int64)v0;
  if ( v0[2] - (_QWORD)v1 <= 0x3Bu )
  {
    v2 = sub_16E7EE0((__int64)v0, "ScheduleDAG::viewGraph is only available in debug builds on ", 0x3Cu);
    v4 = *(__m128i **)(v2 + 24);
    if ( *(_QWORD *)(v2 + 16) - (_QWORD)v4 > 0x1Cu )
      goto LABEL_3;
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_42F11F0);
    qmemcpy(&v1[3], "g builds on ", 12);
    *v1 = si128;
    v1[1] = _mm_load_si128((const __m128i *)&xmmword_42F1200);
    v1[2] = _mm_load_si128((const __m128i *)&xmmword_42F1210);
    v4 = (__m128i *)(v0[3] + 60LL);
    v5 = v0[2];
    *(_QWORD *)(v2 + 24) = v4;
    if ( (unsigned __int64)(v5 - (_QWORD)v4) > 0x1C )
    {
LABEL_3:
      v6 = _mm_load_si128(xmmword_42EAFE0);
      qmemcpy(&v4[1], "phviz or gv!\n", 13);
      *v4 = v6;
      *(_QWORD *)(v2 + 24) += 29LL;
      return 0x726F207A69766870LL;
    }
  }
  return sub_16E7EE0(v2, "systems with Graphviz or gv!\n", 0x1Du);
}
