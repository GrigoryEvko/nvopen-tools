// Function: sub_2FED810
// Address: 0x2fed810
//
__int64 __fastcall sub_2FED810(__int64 a1, __int64 a2)
{
  void *v2; // rdx
  int v3; // ecx
  __int64 v5; // rdi
  char v6; // al
  char v7; // al

  v2 = (void *)(a1 + 3400);
  v3 = a1 + 2852;
  v5 = a1 + 2860;
  *(_QWORD *)(v5 - 2852) = a2;
  *(_QWORD *)(v5 - 2836) = 0;
  *(_QWORD *)(v5 - 2828) = 0;
  *(_QWORD *)(v5 - 2860) = &unk_4A2CC60;
  *(_QWORD *)(v5 - 2820) = 0;
  *(_DWORD *)(v5 - 2812) = 0;
  *(_DWORD *)(v5 - 2787) = 0;
  *(_DWORD *)(v5 - 2756) = 0;
  *(_QWORD *)(v5 - 8) = 0;
  *(_QWORD *)(v5 + 532) = 0;
  memset((void *)(v5 & 0xFFFFFFFFFFFFFFF8LL), 0, 8LL * ((v3 - ((unsigned int)v5 & 0xFFFFFFF8) + 548) >> 3));
  memset(v2, 0, 0x890u);
  *(_QWORD *)(a1 + 5866) = 0;
  *(_QWORD *)(a1 + 6406) = 0;
  memset(
    (void *)((a1 + 5874) & 0xFFFFFFFFFFFFFFF8LL),
    0,
    8LL * (((_DWORD)a1 + 5866 - (((_DWORD)a1 + 5874) & 0xFFFFFFF8) + 548) >> 3));
  *(_QWORD *)(a1 + 524896) = 0;
  *(_QWORD *)(a1 + 525162) = 0;
  memset(
    (void *)((a1 + 524904) & 0xFFFFFFFFFFFFFFF8LL),
    0,
    8LL * (((unsigned int)a1 - (((_DWORD)a1 + 524904) & 0xFFFFFFF8) + 525170) >> 3));
  *(_DWORD *)(a1 + 525248) = 0;
  *(_QWORD *)(a1 + 525264) = a1 + 525248;
  *(_QWORD *)(a1 + 525272) = a1 + 525248;
  *(_QWORD *)(a1 + 525256) = 0;
  *(_QWORD *)(a1 + 525280) = 0;
  sub_E42970((_QWORD *)(a1 + 525288), (_DWORD *)(a2 + 512));
  *(_DWORD *)(a1 + 536984) = 0;
  sub_2FEC890(a1);
  *(_DWORD *)(a1 + 104) = 0;
  *(_WORD *)(a1 + 16) = 0;
  v6 = qword_5026E48;
  *(_DWORD *)(a1 + 60) = 0;
  *(_BYTE *)(a1 + 56) = v6;
  *(_QWORD *)(a1 + 536964) = 0x800000012LL;
  *(_QWORD *)(a1 + 536972) = 0x800000004LL;
  *(_QWORD *)(a1 + 536988) = 0x400000008LL;
  *(_QWORD *)(a1 + 536996) = 0x400000008LL;
  v7 = qword_5026AC8;
  *(_QWORD *)(a1 + 64) = 0;
  *(_BYTE *)(a1 + 537006) = v7;
  *(_BYTE *)(a1 + 72) = 4;
  *(_QWORD *)(a1 + 536980) = 4;
  *(_WORD *)(a1 + 537004) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 88) = 0x80000000000080LL;
  *(_DWORD *)(a1 + 96) = 0;
  *(_BYTE *)(a1 + 100) = 0;
  return sub_2FE6690((_QWORD *)(a1 + 534048));
}
