// Function: sub_7D8B10
// Address: 0x7d8b10
//
void __fastcall sub_7D8B10(_QWORD *a1)
{
  __int64 i; // rax
  char v2; // r12
  __int64 v3; // r15
  __m128i *v4; // r14
  __m128i *v5; // r13
  _QWORD *v6; // r12
  __int64 v7; // rax

  for ( i = a1[16]; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v2 = *(_BYTE *)(i + 160);
  v3 = sub_7D7990(v2);
  v4 = (__m128i *)sub_724D80(3);
  v4[8].m128i_i64[0] = (__int64)sub_72C610(v2);
  v4[11] = _mm_loadu_si128((const __m128i *)a1[22]);
  v5 = (__m128i *)sub_724D80(3);
  v5[8].m128i_i64[0] = (__int64)sub_72C610(v2);
  v5[11] = _mm_loadu_si128((const __m128i *)(a1[22] + 16LL));
  v4[7].m128i_i64[1] = (__int64)v5;
  v6 = sub_724D80(10);
  v7 = *(_QWORD *)(*(_QWORD *)(v3 + 160) + 120LL);
  v6[22] = v4;
  v6[23] = v5;
  v6[16] = v7;
  sub_724A80((__int64)a1, 10);
  a1[16] = v3;
  a1[22] = v6;
  a1[23] = v6;
}
