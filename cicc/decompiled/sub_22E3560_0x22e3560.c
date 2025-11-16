// Function: sub_22E3560
// Address: 0x22e3560
//
__int64 __fastcall sub_22E3560(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rdx
  __int64 *v3; // r12
  __int64 result; // rax

  *(_QWORD *)(a1 + 16) = &unk_4FDC04E;
  *(_QWORD *)(a1 + 56) = a1 + 104;
  *(_QWORD *)(a1 + 112) = a1 + 160;
  *(_QWORD *)(a1 + 192) = a1 + 208;
  *(_QWORD *)(a1 + 200) = 0x1000000000LL;
  *(_QWORD *)(a1 + 416) = a1 + 432;
  *(_QWORD *)(a1 + 424) = 0x1000000000LL;
  *(_DWORD *)(a1 + 88) = 1065353216;
  *(_DWORD *)(a1 + 144) = 1065353216;
  *(_QWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 24) = 2;
  *(_QWORD *)(a1 + 32) = 0;
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
  *(_QWORD *)(a1 + 184) = 0;
  *(_DWORD *)(a1 + 560) = 0;
  *(_QWORD *)(a1 + 384) = 1;
  *(_QWORD *)(a1 + 392) = 0;
  *(_QWORD *)(a1 + 400) = 0;
  *(_QWORD *)a1 = &unk_4A0A190;
  *(_DWORD *)(a1 + 408) = 0;
  *(_QWORD *)(a1 + 176) = &unk_4A0A248;
  *(_QWORD *)(a1 + 568) = 0;
  *(_QWORD *)(a1 + 584) = 0;
  *(_QWORD *)(a1 + 592) = 0;
  *(_QWORD *)(a1 + 600) = 0;
  *(_QWORD *)(a1 + 608) = 0;
  *(_QWORD *)(a1 + 616) = 0;
  *(_QWORD *)(a1 + 624) = 0;
  *(_QWORD *)(a1 + 632) = 0;
  *(_QWORD *)(a1 + 640) = 0;
  *(_QWORD *)(a1 + 576) = 8;
  *(_OWORD *)(a1 + 336) = 0;
  *(_OWORD *)(a1 + 352) = 0;
  *(_OWORD *)(a1 + 368) = 0;
  v1 = sub_22077B0(0x40u);
  v2 = *(_QWORD *)(a1 + 576);
  *(_QWORD *)(a1 + 568) = v1;
  v3 = (__int64 *)(v1 + ((4 * v2 - 4) & 0xFFFFFFFFFFFFFFF8LL));
  result = sub_22077B0(0x200u);
  *(_QWORD *)(a1 + 608) = v3;
  *v3 = result;
  *(_QWORD *)(a1 + 640) = v3;
  *(_QWORD *)(a1 + 592) = result;
  *(_QWORD *)(a1 + 600) = result + 512;
  *(_QWORD *)(a1 + 624) = result;
  *(_QWORD *)(a1 + 632) = result + 512;
  *(_QWORD *)(a1 + 584) = result;
  *(_QWORD *)(a1 + 616) = result;
  *(_QWORD *)(a1 + 648) = 0;
  *(_QWORD *)(a1 + 656) = 0;
  return result;
}
