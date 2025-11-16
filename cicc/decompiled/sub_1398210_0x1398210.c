// Function: sub_1398210
// Address: 0x1398210
//
void __fastcall sub_1398210(__int64 a1, __int64 a2)
{
  __int64 v2; // rdi
  __m128i *v3; // rdx
  __m128i si128; // xmm0

  v2 = *(_QWORD *)(a1 + 160);
  if ( v2 )
  {
    sub_1397F00(v2, a2);
  }
  else
  {
    v3 = *(__m128i **)(a2 + 24);
    if ( *(_QWORD *)(a2 + 16) - (_QWORD)v3 <= 0x1Du )
    {
      sub_16E7EE0(a2, "No call graph has been built!\n", 30);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F70830);
      qmemcpy(&v3[1], "s been built!\n", 14);
      *v3 = si128;
      *(_QWORD *)(a2 + 24) += 30LL;
    }
  }
}
