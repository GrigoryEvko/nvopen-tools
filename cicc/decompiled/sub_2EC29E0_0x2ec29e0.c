// Function: sub_2EC29E0
// Address: 0x2ec29e0
//
__int64 sub_2EC29E0()
{
  _QWORD *v0; // rax
  __m128i *v1; // rdx
  __int64 v2; // rdi
  __m128i si128; // xmm0
  __m128i *v4; // rdx
  __int64 v5; // rax
  __m128i v6; // xmm0

  v0 = sub_CB72A0();
  v1 = (__m128i *)v0[4];
  v2 = (__int64)v0;
  if ( v0[3] - (_QWORD)v1 <= 0x3Du )
  {
    v2 = sub_CB6200((__int64)v0, "ScheduleDAGMI::viewGraph is only available in debug builds on ", 0x3Eu);
    v4 = *(__m128i **)(v2 + 32);
    if ( *(_QWORD *)(v2 + 24) - (_QWORD)v4 > 0x1Cu )
      goto LABEL_3;
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_42EC320);
    qmemcpy(&v1[3], "bug builds on ", 14);
    *v1 = si128;
    v1[1] = _mm_load_si128((const __m128i *)&xmmword_42EC330);
    v1[2] = _mm_load_si128((const __m128i *)&xmmword_42EAFD0);
    v4 = (__m128i *)(v0[4] + 62LL);
    v5 = v0[3];
    *(_QWORD *)(v2 + 32) = v4;
    if ( (unsigned __int64)(v5 - (_QWORD)v4) > 0x1C )
    {
LABEL_3:
      v6 = _mm_load_si128(xmmword_42EAFE0);
      qmemcpy(&v4[1], "phviz or gv!\n", 13);
      *v4 = v6;
      *(_QWORD *)(v2 + 32) += 29LL;
      return 0x726F207A69766870LL;
    }
  }
  return sub_CB6200(v2, "systems with Graphviz or gv!\n", 0x1Du);
}
