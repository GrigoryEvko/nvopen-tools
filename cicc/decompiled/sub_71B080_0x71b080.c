// Function: sub_71B080
// Address: 0x71b080
//
_QWORD *__fastcall sub_71B080(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 v4; // r14
  __m128i v5; // xmm1
  __m128i v6; // xmm2
  __m128i v7; // xmm3
  __int64 v8; // rax
  __int64 v9; // rax
  __int16 v10; // cx
  __int64 v11; // rax
  __int64 v12; // r15
  _QWORD *v13; // rax
  __int64 v15; // rax
  __int64 v16; // [rsp+8h] [rbp-428h]
  __int64 v17; // [rsp+10h] [rbp-420h]
  _BYTE v18[160]; // [rsp+20h] [rbp-410h] BYREF
  __m128i v19[22]; // [rsp+C0h] [rbp-370h] BYREF
  _QWORD v20[66]; // [rsp+220h] [rbp-210h] BYREF

  v4 = *(_QWORD *)(a1 + 120);
  v17 = qword_4D03C50;
  sub_6E1E00(4u, (__int64)v18, 0, 0);
  memset(v20, 0, 0x1D8u);
  v20[19] = v20;
  v20[3] = *(_QWORD *)&dword_4F063F8;
  if ( dword_4F077BC && qword_4F077A8 <= 0x9F5Fu )
    BYTE2(v20[22]) |= 1u;
  v5 = _mm_loadu_si128(&xmmword_4F06660[1]);
  v6 = _mm_loadu_si128(&xmmword_4F06660[2]);
  v7 = _mm_loadu_si128(&xmmword_4F06660[3]);
  v8 = *a3;
  v19[0].m128i_i64[0] = _mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
  v19[1] = v5;
  v19[2] = v6;
  v19[3] = v7;
  v19[0].m128i_i64[1] = v8;
  v16 = sub_87EF90(7, v19);
  sub_725B90(a2);
  *(_QWORD *)(a2 + 120) = v4;
  *(_BYTE *)(a2 + 136) = 3;
  v9 = *(_QWORD *)(a1 + 8);
  *(_QWORD *)a2 = v16;
  v10 = *(_WORD *)(a2 + 88);
  *(_QWORD *)(a2 + 8) = v9;
  v11 = *(_QWORD *)(a1 + 48);
  v20[0] = v16;
  *(_QWORD *)(a2 + 48) = v11;
  LOBYTE(v11) = *(_BYTE *)(a1 + 88) & 4;
  BYTE1(v11) = 1;
  *(_WORD *)(a2 + 88) = v10 & 0xFEFB | v11;
  *(_BYTE *)(a2 + 172) = *(_BYTE *)(a1 + 172) & 1 | *(_BYTE *)(a2 + 172) & 0xFE;
  *(_BYTE *)(a2 + 169) = *(_BYTE *)(a1 + 169) & 0x80 | *(_BYTE *)(a2 + 169) & 0x7F;
  *(_QWORD *)(a2 + 112) = *(_QWORD *)(a1 + 112);
  *(_QWORD *)(v16 + 88) = a2;
  v12 = sub_73E830(a1);
  if ( (unsigned int)sub_8D32E0(v4) )
  {
    v15 = sub_73DDB0(v12);
    *(_BYTE *)(v15 + 25) &= ~1u;
    v12 = v15;
  }
  if ( (unsigned int)sub_8D3070(v4) )
    *(_BYTE *)(v12 + 25) |= 1u;
  else
    *(_BYTE *)(v12 + 25) |= 2u;
  sub_6E7150((__int64 *)v12, (__int64)v19);
  v13 = (_QWORD *)sub_6E3060(v19);
  sub_6E1C20(v13, 1, &v20[41]);
  v19[0].m128i_i32[0] = 0;
  sub_638AC0((__int64)v20, a3, 0, 1u, v19, 0);
  sub_6E2B30((__int64)v20, (__int64)a3);
  qword_4D03C50 = v17;
  return &qword_4D03C50;
}
