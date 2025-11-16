// Function: sub_3180150
// Address: 0x3180150
//
void __fastcall sub_3180150(__int64 a1, __int64 a2, __int64 a3)
{
  const __m128i *i; // rbx
  __m128i v4; // xmm2
  __int64 v5; // rax
  _QWORD *v6; // rax
  _OWORD v7[2]; // [rsp+0h] [rbp-50h] BYREF
  __int64 v8; // [rsp+20h] [rbp-30h]

  *(_QWORD *)a1 = a1 + 48;
  *(_QWORD *)(a1 + 56) = a1 + 104;
  *(_QWORD *)(a1 + 8) = 1;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 64) = 1;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 104) = 0;
  *(_QWORD *)(a1 + 112) = a3;
  *(_DWORD *)(a1 + 128) = 0;
  *(_QWORD *)(a1 + 136) = 0;
  *(_QWORD *)(a1 + 144) = a1 + 128;
  *(_QWORD *)(a1 + 152) = a1 + 128;
  *(_QWORD *)(a1 + 160) = 0;
  *(_QWORD *)(a1 + 168) = 0;
  *(_QWORD *)(a1 + 176) = 0;
  *(_QWORD *)(a1 + 184) = 0;
  *(_QWORD *)(a1 + 192) = 0;
  *(_BYTE *)(a1 + 204) = 0;
  *(_QWORD *)(a1 + 208) = 0;
  *(_DWORD *)(a1 + 32) = 1065353216;
  *(_DWORD *)(a1 + 88) = 1065353216;
  for ( i = *(const __m128i **)(a2 + 16); i; i = (const __m128i *)i->m128i_i64[0] )
  {
    v4 = _mm_loadu_si128(i + 3);
    v5 = i[4].m128i_i64[0];
    v7[0] = _mm_loadu_si128(i + 2);
    v7[1] = v4;
    v8 = v5;
    v6 = sub_317F9B0(a1, (__int64)v7, 1);
    sub_317E630((__int64)v6, (__int64)i[1].m128i_i64);
  }
  sub_317FCF0((_QWORD *)a1);
}
