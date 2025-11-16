// Function: sub_2F91670
// Address: 0x2f91670
//
__int64 __fastcall sub_2F91670(__int64 a1, __int64 *a2, __int64 a3, char a4)
{
  __int64 v7; // rax
  __m128i v8; // xmm4
  __m128i v9; // xmm3
  __m128i v10; // xmm2
  __m128i v11; // xmm1
  __m128i v12; // xmm0
  __int64 v13; // rdi
  _QWORD *v14; // rax
  __int64 **v15; // rax
  __int64 v16; // rsi

  sub_2F8E8D0(a1, a2);
  *(_QWORD *)(a1 + 584) = a3;
  *(_BYTE *)(a1 + 896) = a4;
  *(_BYTE *)(a1 + 897) = 0;
  *(_QWORD *)a1 = &unk_4A2BD58;
  v7 = a2[6];
  *(_QWORD *)(a1 + 760) = 0;
  *(_QWORD *)(a1 + 592) = v7;
  *(_QWORD *)(a1 + 768) = 0;
  v8 = _mm_loadu_si128(xmmword_3F8F0C0);
  v9 = _mm_loadu_si128(&xmmword_3F8F0C0[1]);
  *(_QWORD *)(a1 + 776) = 0;
  v10 = _mm_loadu_si128(&xmmword_3F8F0C0[2]);
  v11 = _mm_loadu_si128(&xmmword_3F8F0C0[3]);
  *(_QWORD *)(a1 + 784) = 0;
  v12 = _mm_loadu_si128(&xmmword_3F8F0C0[4]);
  *(__m128i *)(a1 + 600) = v8;
  *(_QWORD *)(a1 + 808) = a1 + 824;
  *(_QWORD *)(a1 + 816) = 0x1000000000LL;
  *(__m128i *)(a1 + 616) = v9;
  *(__m128i *)(a1 + 632) = v10;
  *(__m128i *)(a1 + 648) = v11;
  *(__m128i *)(a1 + 664) = v12;
  *(__m128i *)(a1 + 680) = v8;
  *(__m128i *)(a1 + 696) = v9;
  *(__m128i *)(a1 + 712) = v10;
  *(__m128i *)(a1 + 728) = v11;
  *(__m128i *)(a1 + 744) = v12;
  *(_QWORD *)(a1 + 792) = 0;
  *(_QWORD *)(a1 + 800) = 0;
  *(_QWORD *)(a1 + 888) = 0;
  *(_WORD *)(a1 + 898) = 0;
  *(_QWORD *)(a1 + 1200) = a1 + 1216;
  *(_QWORD *)(a1 + 968) = a1 + 984;
  *(_QWORD *)(a1 + 1432) = a1 + 1448;
  *(_QWORD *)(a1 + 1192) = 0xFFFFFFFFLL;
  *(_QWORD *)(a1 + 1424) = 0xFFFFFFFFLL;
  *(_QWORD *)(a1 + 1784) = 0xFFFFFFFFLL;
  *(_QWORD *)(a1 + 1792) = a1 + 1808;
  *(_QWORD *)(a1 + 2208) = 0xFFFFFFFFLL;
  *(_QWORD *)(a1 + 912) = 0;
  *(_QWORD *)(a1 + 920) = 0;
  *(_QWORD *)(a1 + 936) = 0;
  *(_QWORD *)(a1 + 976) = 0x800000000LL;
  *(_QWORD *)(a1 + 1176) = 0;
  *(_DWORD *)(a1 + 1184) = 0;
  *(_QWORD *)(a1 + 1208) = 0x800000000LL;
  *(_QWORD *)(a1 + 1408) = 0;
  *(_DWORD *)(a1 + 1416) = 0;
  *(_QWORD *)(a1 + 1440) = 0x800000000LL;
  *(_QWORD *)(a1 + 1768) = 0;
  *(_DWORD *)(a1 + 1776) = 0;
  *(_QWORD *)(a1 + 1800) = 0x800000000LL;
  *(_QWORD *)(a1 + 2192) = 0;
  *(_DWORD *)(a1 + 2200) = 0;
  *(_BYTE *)(a1 + 2896) = 0;
  v13 = *a2;
  *(_QWORD *)(a1 + 904) = 0;
  *(_DWORD *)(a1 + 928) = 0;
  *(_QWORD *)(a1 + 944) = 0;
  *(_QWORD *)(a1 + 952) = 0;
  *(_DWORD *)(a1 + 960) = 0;
  *(_QWORD *)(a1 + 2904) = 0;
  *(_DWORD *)(a1 + 2912) = 3;
  v14 = (_QWORD *)sub_B2BE50(v13);
  v15 = (__int64 **)sub_BCB120(v14);
  *(_QWORD *)(a1 + 2920) = sub_ACA8A0(v15);
  sub_2F8FF00(a1 + 2928, a1 + 48, a1 + 328);
  *(_QWORD *)(a1 + 3344) = 0;
  *(_QWORD *)(a1 + 3384) = a1 + 3400;
  *(_QWORD *)(a1 + 3352) = 0;
  *(_QWORD *)(a1 + 3360) = 0;
  *(_QWORD *)(a1 + 3376) = 0;
  *(_QWORD *)(a1 + 3392) = 0x600000000LL;
  *(_DWORD *)(a1 + 3448) = 0;
  v16 = a2[2];
  *(_QWORD *)(a1 + 3368) = 0;
  return sub_2FF7BB0(a1 + 600, v16);
}
