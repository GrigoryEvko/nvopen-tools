// Function: sub_F110A0
// Address: 0xf110a0
//
unsigned int __fastcall sub_F110A0(__int64 a1)
{
  __int128 *v1; // rax

  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = &unk_4F8AED0;
  *(_QWORD *)(a1 + 56) = a1 + 104;
  *(_QWORD *)(a1 + 112) = a1 + 160;
  *(_DWORD *)(a1 + 24) = 2;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)a1 = &unk_49E4F68;
  *(_QWORD *)(a1 + 176) = a1 + 192;
  *(_QWORD *)(a1 + 184) = 0x10000000000LL;
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
  *(_QWORD *)(a1 + 2240) = 0;
  *(_QWORD *)(a1 + 2248) = 0;
  *(_QWORD *)(a1 + 2256) = 0;
  *(_DWORD *)(a1 + 2264) = 0;
  *(_DWORD *)(a1 + 88) = 1065353216;
  *(_DWORD *)(a1 + 144) = 1065353216;
  *(_QWORD *)(a1 + 2272) = 0;
  *(_QWORD *)(a1 + 2304) = a1 + 2320;
  *(_QWORD *)(a1 + 2280) = 0;
  *(_QWORD *)(a1 + 2288) = 0;
  *(_DWORD *)(a1 + 2296) = 0;
  *(_QWORD *)(a1 + 2312) = 0x1000000000LL;
  v1 = sub_BC2B00();
  return sub_F11020((__int64)v1);
}
