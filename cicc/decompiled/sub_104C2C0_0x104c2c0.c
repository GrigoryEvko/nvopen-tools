// Function: sub_104C2C0
// Address: 0x104c2c0
//
unsigned int __fastcall sub_104C2C0(__int64 a1)
{
  __int128 *v1; // rax

  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = &unk_4F8FBD4;
  *(_QWORD *)(a1 + 56) = a1 + 104;
  *(_QWORD *)(a1 + 112) = a1 + 160;
  *(_DWORD *)(a1 + 24) = 2;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)a1 = &unk_49E5C68;
  *(_QWORD *)(a1 + 176) = a1 + 192;
  *(_QWORD *)(a1 + 184) = 0x400000000LL;
  *(_QWORD *)(a1 + 224) = a1 + 240;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 64) = 1;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 104) = 0;
  *(_QWORD *)(a1 + 120) = 1;
  *(_QWORD *)(a1 + 128) = 0;
  *(_QWORD *)(a1 + 136) = 0;
  *(_QWORD *)(a1 + 152) = 0;
  *(_QWORD *)(a1 + 160) = 0;
  *(_BYTE *)(a1 + 168) = 0;
  *(_QWORD *)(a1 + 232) = 0x600000000LL;
  *(_QWORD *)(a1 + 296) = 0;
  *(_QWORD *)(a1 + 304) = 0;
  *(_DWORD *)(a1 + 88) = 1065353216;
  *(_DWORD *)(a1 + 144) = 1065353216;
  *(_BYTE *)(a1 + 312) = 0;
  *(_QWORD *)(a1 + 316) = 0;
  v1 = sub_BC2B00();
  return sub_104C240((__int64)v1);
}
