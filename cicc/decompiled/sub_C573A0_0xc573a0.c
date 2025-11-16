// Function: sub_C573A0
// Address: 0xc573a0
//
__int64 __fastcall sub_C573A0(__int64 a1, __int64 a2, int a3)
{
  __int64 v3; // rax
  __m128i *v4; // rdx
  __int64 v5; // rdi
  __m128i si128; // xmm0

  sub_C54F20(a1, a2, a3);
  v3 = sub_CB7210(a1);
  v4 = *(__m128i **)(v3 + 32);
  v5 = v3;
  if ( *(_QWORD *)(v3 + 24) - (_QWORD)v4 <= 0x1Du )
    return sub_CB6200(v3, "= *cannot print option value*\n", 30);
  si128 = _mm_load_si128((const __m128i *)&xmmword_3F66410);
  qmemcpy(&v4[1], "option value*\n", 14);
  *v4 = si128;
  *(_QWORD *)(v5 + 32) += 30LL;
  return 2602;
}
