// Function: sub_3702BF0
// Address: 0x3702bf0
//
__int64 __fastcall sub_3702BF0(__int64 a1)
{
  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0x400000000LL;
  *(_BYTE *)(a1 + 36) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 40) = &unk_4A3C600;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_DWORD *)(a1 + 72) = 1;
  sub_37191E0(a1 + 80, a1 + 40);
  *(_QWORD *)(a1 + 208) = a1 + 80;
  *(_BYTE *)(a1 + 154) = 0;
  *(_BYTE *)(a1 + 158) = 0;
  *(_QWORD *)(a1 + 144) = &unk_4A3C998;
  *(_QWORD *)(a1 + 160) = a1 + 176;
  *(_QWORD *)(a1 + 168) = 0x200000000LL;
  *(_QWORD *)(a1 + 200) = 0;
  *(_QWORD *)(a1 + 216) = 0;
  *(_QWORD *)(a1 + 224) = 0;
  *(_QWORD *)(a1 + 232) = 0;
  *(_QWORD *)(a1 + 240) = 0;
  return 0x200000000LL;
}
