// Function: sub_2E10EC0
// Address: 0x2e10ec0
//
__int64 __fastcall sub_2E10EC0(_QWORD *a1, __int64 a2)
{
  __m128i *v2; // rdx
  __m128i si128; // xmm0

  v2 = *(__m128i **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v2 <= 0x23u )
  {
    sub_CB6200(a2, "********** MACHINEINSTRS **********\n", 0x24u);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_42E9A20);
    v2[2].m128i_i32[0] = 170535466;
    *v2 = si128;
    v2[1] = _mm_load_si128((const __m128i *)&xmmword_42E9A30);
    *(_QWORD *)(a2 + 32) += 36LL;
  }
  return sub_2E823F0(*a1, a2, a1[4]);
}
