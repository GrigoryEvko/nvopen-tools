// Function: sub_1F03E40
// Address: 0x1f03e40
//
__int64 __fastcall sub_1F03E40(__int64 a1, __int64 *a2, __int64 a3, char a4)
{
  __int64 v7; // rax
  __m128i v8; // xmm3
  __m128i v9; // xmm2
  __m128i v10; // xmm1
  __m128i v11; // xmm0
  __int64 v12; // rdi
  _QWORD *v13; // rax
  __int64 **v14; // rax
  __int64 v15; // rax
  __int64 v16; // rsi

  sub_1F010B0(a1, a2);
  *(_QWORD *)(a1 + 616) = a3;
  *(_BYTE *)(a1 + 912) = a4;
  *(_QWORD *)a1 = &unk_49FE610;
  v7 = a2[7];
  *(_QWORD *)(a1 + 776) = 0;
  *(_QWORD *)(a1 + 624) = v7;
  *(_QWORD *)(a1 + 784) = 0;
  v8 = _mm_loadu_si128(xmmword_452E800);
  v9 = _mm_loadu_si128(&xmmword_452E800[1]);
  *(_QWORD *)(a1 + 792) = 0;
  v10 = _mm_loadu_si128(&xmmword_452E800[2]);
  v11 = _mm_loadu_si128(&xmmword_452E800[3]);
  *(_QWORD *)(a1 + 800) = 0;
  *(__m128i *)(a1 + 632) = v8;
  *(__m128i *)(a1 + 648) = v9;
  *(_QWORD *)(a1 + 696) = unk_452E840;
  *(_QWORD *)(a1 + 768) = unk_452E840;
  *(_QWORD *)(a1 + 824) = a1 + 840;
  *(_QWORD *)(a1 + 832) = 0x1000000000LL;
  *(_WORD *)(a1 + 913) = 0;
  *(__m128i *)(a1 + 664) = v10;
  *(__m128i *)(a1 + 680) = v11;
  *(__m128i *)(a1 + 704) = v8;
  *(__m128i *)(a1 + 720) = v9;
  *(__m128i *)(a1 + 736) = v10;
  *(__m128i *)(a1 + 752) = v11;
  *(_QWORD *)(a1 + 808) = 0;
  *(_QWORD *)(a1 + 816) = 0;
  *(_QWORD *)(a1 + 928) = 0;
  *(_QWORD *)(a1 + 936) = 0;
  *(_QWORD *)(a1 + 984) = a1 + 1000;
  *(_QWORD *)(a1 + 1216) = a1 + 1232;
  *(_QWORD *)(a1 + 1448) = a1 + 1464;
  *(_QWORD *)(a1 + 1208) = 0xFFFFFFFFLL;
  *(_QWORD *)(a1 + 1440) = 0xFFFFFFFFLL;
  *(_QWORD *)(a1 + 1672) = 0xFFFFFFFFLL;
  *(_QWORD *)(a1 + 1968) = 0xFFFFFFFFLL;
  *(_QWORD *)(a1 + 992) = 0x800000000LL;
  *(_QWORD *)(a1 + 1224) = 0x800000000LL;
  *(_QWORD *)(a1 + 1456) = 0x800000000LL;
  *(_QWORD *)(a1 + 1688) = 0x800000000LL;
  *(_QWORD *)(a1 + 952) = 0;
  *(_QWORD *)(a1 + 1192) = 0;
  *(_DWORD *)(a1 + 1200) = 0;
  *(_QWORD *)(a1 + 1424) = 0;
  *(_DWORD *)(a1 + 1432) = 0;
  *(_QWORD *)(a1 + 1656) = 0;
  *(_DWORD *)(a1 + 1664) = 0;
  *(_QWORD *)(a1 + 1680) = a1 + 1696;
  *(_QWORD *)(a1 + 1952) = 0;
  *(_DWORD *)(a1 + 1960) = 0;
  v12 = *a2;
  *(_QWORD *)(a1 + 960) = 0;
  *(_QWORD *)(a1 + 968) = 0;
  *(_DWORD *)(a1 + 976) = 0;
  *(_QWORD *)(a1 + 1976) = 0;
  *(_QWORD *)(a1 + 1984) = 0;
  v13 = (_QWORD *)sub_15E0530(v12);
  v14 = (__int64 **)sub_1643270(v13);
  v15 = sub_1599EF0(v14);
  *(_QWORD *)(a1 + 2048) = 0x800000000LL;
  *(_QWORD *)(a1 + 1992) = v15;
  *(_QWORD *)(a1 + 2000) = 0;
  *(_QWORD *)(a1 + 2008) = 0;
  *(_QWORD *)(a1 + 2016) = 0;
  *(_QWORD *)(a1 + 2032) = 0;
  *(_QWORD *)(a1 + 2040) = a1 + 2056;
  *(_QWORD *)(a1 + 2088) = 0;
  *(_DWORD *)(a1 + 2096) = 0;
  v16 = a2[2];
  *(_QWORD *)(a1 + 2024) = 0;
  return sub_1F4B6B0(a1 + 632, v16);
}
