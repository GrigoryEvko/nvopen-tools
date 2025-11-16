// Function: sub_14040F0
// Address: 0x14040f0
//
__int64 __fastcall sub_14040F0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rdx
  __int64 *v3; // r12
  __int64 result; // rax

  *(_QWORD *)(a1 + 16) = &unk_4F992F1;
  *(_QWORD *)(a1 + 80) = a1 + 64;
  *(_QWORD *)(a1 + 88) = a1 + 64;
  *(_QWORD *)(a1 + 128) = a1 + 112;
  *(_QWORD *)(a1 + 136) = a1 + 112;
  *(_QWORD *)(a1 + 184) = a1 + 200;
  *(_QWORD *)(a1 + 192) = 0x1000000000LL;
  *(_QWORD *)(a1 + 416) = a1 + 432;
  *(_QWORD *)(a1 + 424) = 0x1000000000LL;
  *(_QWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 24) = 3;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_DWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_DWORD *)(a1 + 112) = 0;
  *(_QWORD *)(a1 + 120) = 0;
  *(_QWORD *)(a1 + 144) = 0;
  *(_BYTE *)(a1 + 152) = 0;
  *(_QWORD *)(a1 + 168) = 0;
  *(_QWORD *)(a1 + 176) = 0;
  *(_DWORD *)(a1 + 560) = 0;
  *(_QWORD *)(a1 + 328) = 0;
  *(_QWORD *)(a1 + 336) = 0;
  *(_QWORD *)(a1 + 344) = 0;
  *(_QWORD *)(a1 + 352) = 0;
  *(_QWORD *)(a1 + 360) = 0;
  *(_QWORD *)(a1 + 368) = 0;
  *(_QWORD *)(a1 + 376) = 0;
  *(_QWORD *)a1 = &unk_49EADF8;
  *(_QWORD *)(a1 + 384) = 1;
  *(_QWORD *)(a1 + 392) = 0;
  *(_QWORD *)(a1 + 400) = 0;
  *(_DWORD *)(a1 + 408) = 0;
  *(_QWORD *)(a1 + 160) = &unk_49EAEB0;
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
  v1 = sub_22077B0(64);
  v2 = *(_QWORD *)(a1 + 576);
  *(_QWORD *)(a1 + 568) = v1;
  v3 = (__int64 *)(v1 + ((4 * v2 - 4) & 0xFFFFFFFFFFFFFFF8LL));
  result = sub_22077B0(512);
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
