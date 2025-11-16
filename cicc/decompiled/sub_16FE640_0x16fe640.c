// Function: sub_16FE640
// Address: 0x16fe640
//
__int64 __fastcall sub_16FE640(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rcx

  *(_QWORD *)(a1 + 144) = a1 + 160;
  *(_QWORD *)a1 = a4;
  *(_QWORD *)(a1 + 96) = a1 + 112;
  v6 = (a1 + 184) | 4;
  *(_QWORD *)(a1 + 192) = a1 + 184;
  *(_QWORD *)(a1 + 104) = 0x400000000LL;
  *(_QWORD *)(a1 + 200) = a1 + 216;
  *(_QWORD *)(a1 + 208) = 0x400000000LL;
  *(_QWORD *)(a1 + 240) = 0x400000000LL;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_BYTE *)(a1 + 75) = a5;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 88) = 0;
  *(_QWORD *)(a1 + 152) = 0;
  *(_QWORD *)(a1 + 160) = 0;
  *(_QWORD *)(a1 + 168) = 1;
  *(_QWORD *)(a1 + 184) = v6;
  *(_QWORD *)(a1 + 232) = a1 + 248;
  *(_QWORD *)(a1 + 344) = a6;
  return sub_16FE540(a1, a2, a1 + 248, v6, a5, a6, a2, a3, (__int64)"YAML", 4);
}
