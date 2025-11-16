// Function: sub_1E114F0
// Address: 0x1e114f0
//
__int64 __fastcall sub_1E114F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, __int64 a6)
{
  _QWORD *v7; // rax

  *(_QWORD *)(a1 + 24) = a6 + 168;
  *(_QWORD *)(a1 + 136) = a1 + 152;
  *(_QWORD *)(a1 + 144) = 0x400000000LL;
  *(_QWORD *)(a1 + 184) = a1 + 200;
  *(_QWORD *)(a1 + 232) = a1 + 248;
  *(_QWORD *)(a1 + 240) = 0x800000000LL;
  *(_QWORD *)(a1 + 8) = a3;
  *(_QWORD *)(a1 + 328) = a1 + 320;
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 16) = a4;
  *(_QWORD *)(a1 + 32) = a6;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 88) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 104) = 0;
  *(_QWORD *)(a1 + 112) = 0;
  *(_QWORD *)(a1 + 120) = 0;
  *(_QWORD *)(a1 + 128) = 0;
  *(_QWORD *)(a1 + 192) = 0;
  *(_QWORD *)(a1 + 200) = 0;
  *(_QWORD *)(a1 + 208) = 1;
  *(_QWORD *)(a1 + 224) = 0;
  *(_QWORD *)(a1 + 312) = 0;
  *(_QWORD *)(a1 + 320) = (a1 + 320) | 4;
  *(_WORD *)(a1 + 344) = 0;
  *(_BYTE *)(a1 + 347) = 0;
  *(_QWORD *)(a1 + 352) = 0;
  *(_QWORD *)(a1 + 360) = 0;
  *(_DWORD *)(a1 + 368) = 8;
  v7 = (_QWORD *)malloc(8u);
  if ( !v7 )
  {
    sub_16BD1C0("Allocation failed", 1u);
    v7 = 0;
  }
  *(_QWORD *)(a1 + 352) = v7;
  *v7 = 0;
  *(_QWORD *)(a1 + 360) = 1;
  *(_QWORD *)(a1 + 376) = 0;
  *(_QWORD *)(a1 + 384) = 0;
  *(_QWORD *)(a1 + 392) = 0;
  *(_QWORD *)(a1 + 400) = 0;
  *(_QWORD *)(a1 + 408) = 0;
  *(_QWORD *)(a1 + 416) = 0;
  *(_QWORD *)(a1 + 424) = 0;
  *(_QWORD *)(a1 + 432) = 0;
  *(_QWORD *)(a1 + 440) = 0;
  *(_QWORD *)(a1 + 448) = 0;
  *(_DWORD *)(a1 + 456) = 0;
  *(_QWORD *)(a1 + 464) = 0;
  *(_QWORD *)(a1 + 472) = 0;
  *(_QWORD *)(a1 + 480) = 0;
  *(_DWORD *)(a1 + 488) = 0;
  *(_QWORD *)(a1 + 496) = 0;
  *(_QWORD *)(a1 + 504) = 0;
  *(_QWORD *)(a1 + 512) = 0;
  *(_DWORD *)(a1 + 520) = 0;
  *(_QWORD *)(a1 + 528) = 0;
  *(_QWORD *)(a1 + 536) = 0;
  *(_QWORD *)(a1 + 544) = 0;
  *(_QWORD *)(a1 + 552) = 0;
  *(_QWORD *)(a1 + 560) = 0;
  *(_QWORD *)(a1 + 568) = 0;
  *(_QWORD *)(a1 + 576) = 0;
  *(_QWORD *)(a1 + 584) = 0;
  *(_QWORD *)(a1 + 592) = 0;
  *(_DWORD *)(a1 + 600) = 0;
  *(_QWORD *)(a1 + 608) = a1 + 624;
  *(_DWORD *)(a1 + 336) = a5;
  *(_QWORD *)(a1 + 616) = 0x400000000LL;
  return sub_1E10FF0(a1);
}
