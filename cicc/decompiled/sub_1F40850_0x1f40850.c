// Function: sub_1F40850
// Address: 0x1f40850
//
__int64 __fastcall sub_1F40850(__int64 a1, __int64 a2)
{
  int v2; // ecx
  __int64 v4; // rdi
  char v5; // al

  v2 = a1 + 1155;
  v4 = a1 + 1163;
  *(_QWORD *)(v4 - 1163) = &unk_49FEE48;
  *(_QWORD *)(v4 - 1155) = a2;
  *(_QWORD *)(v4 - 1139) = 0;
  *(_QWORD *)(v4 - 1131) = 0;
  *(_QWORD *)(v4 - 1123) = 0;
  *(_DWORD *)(v4 - 1115) = 0;
  *(_QWORD *)(v4 - 8) = 0;
  *(_QWORD *)(v4 + 99) = 0;
  memset((void *)(v4 & 0xFFFFFFFFFFFFFFF8LL), 0, 8LL * ((v2 - ((unsigned int)v4 & 0xFFFFFFF8) + 115) >> 3));
  *(_QWORD *)(a1 + 2307) = 0;
  *(_QWORD *)(a1 + 2414) = 0;
  memset(
    (void *)((a1 + 2315) & 0xFFFFFFFFFFFFFFF8LL),
    0,
    8LL * (((_DWORD)a1 + 2307 - (((_DWORD)a1 + 2315) & 0xFFFFFFF8) + 115) >> 3));
  *(_QWORD *)(a1 + 73900) = 0;
  *(_QWORD *)(a1 + 74007) = 0;
  memset(
    (void *)((a1 + 73908) & 0xFFFFFFFFFFFFFFF8LL),
    0,
    8LL * (((unsigned int)a1 - (((_DWORD)a1 + 73908) & 0xFFFFFFF8) + 74015) >> 3));
  *(_DWORD *)(a1 + 74056) = 0;
  *(_QWORD *)(a1 + 74072) = a1 + 74056;
  *(_QWORD *)(a1 + 74080) = a1 + 74056;
  *(_QWORD *)(a1 + 74064) = 0;
  *(_QWORD *)(a1 + 74088) = 0;
  *(_DWORD *)(a1 + 81512) = 0;
  sub_1F405D0(a1);
  *(_WORD *)(a1 + 16) = 0;
  v5 = byte_4FCB6E0;
  *(_QWORD *)(a1 + 81520) = 0x400000008LL;
  *(_BYTE *)(a1 + 56) = v5;
  *(_QWORD *)(a1 + 81496) = 0x800000012LL;
  *(_QWORD *)(a1 + 81504) = 0x800000004LL;
  *(_QWORD *)(a1 + 81512) = 0x400000000LL;
  *(_QWORD *)(a1 + 97) = 0x400000000LL;
  *(_QWORD *)(a1 + 81528) = 0x400000008LL;
  *(_WORD *)(a1 + 81536) = 0;
  *(_QWORD *)(a1 + 65) = 0x400000000000000LL;
  *(_DWORD *)(a1 + 112) = 0;
  *(_QWORD *)(a1 + 57) = 1;
  *(_QWORD *)(a1 + 73) = 0;
  *(_QWORD *)(a1 + 81) = &loc_1000000;
  *(_QWORD *)(a1 + 89) = 0;
  *(_DWORD *)(a1 + 105) = 0;
  *(_QWORD *)(a1 + 74096) = 0;
  *(_QWORD *)(a1 + 77792) = 0;
  memset(
    (void *)((a1 + 74104) & 0xFFFFFFFFFFFFFFF8LL),
    0,
    8LL * (((unsigned int)a1 - (((_DWORD)a1 + 74104) & 0xFFFFFFF8) + 77800) >> 3));
  sub_1F3E460((_QWORD *)a1, (_DWORD *)(*(_QWORD *)(a1 + 8) + 472LL));
  *(_QWORD *)(a1 + 77800) = 0x1818181818181818LL;
  *(_QWORD *)(a1 + 79640) = 0x1818181818181818LL;
  memset(
    (void *)((a1 + 77808) & 0xFFFFFFFFFFFFFFF8LL),
    0x18u,
    8LL * (((_DWORD)a1 + 77800 - (((_DWORD)a1 + 77808) & 0xFFFFFFF8) + 1848) >> 3));
  *(_QWORD *)(a1 + 78992) = 0x1600000016LL;
  *(_QWORD *)(a1 + 79008) = 0x1300000013LL;
  *(_QWORD *)(a1 + 79016) = 0x1300000013LL;
  *(_QWORD *)(a1 + 79024) = 0x1400000014LL;
  *(_QWORD *)(a1 + 79032) = 0x1400000014LL;
  *(_QWORD *)(a1 + 79040) = 0x1500000015LL;
  *(_QWORD *)(a1 + 79048) = 0x1500000015LL;
  *(_QWORD *)(a1 + 78976) = 0x1100000011LL;
  *(_QWORD *)(a1 + 78984) = 0x1100000011LL;
  *(_QWORD *)(a1 + 79000) = 0x1600000016LL;
  *(_QWORD *)(a1 + 79056) = 0x1200000012LL;
  *(_QWORD *)(a1 + 79064) = 0x1200000012LL;
  *(_QWORD *)(a1 + 79072) = 0x1600000016LL;
  *(_QWORD *)(a1 + 79080) = 0x1600000016LL;
  *(_QWORD *)(a1 + 79088) = 0x1100000011LL;
  *(_QWORD *)(a1 + 79096) = 0x1100000011LL;
  return 0x1100000011LL;
}
