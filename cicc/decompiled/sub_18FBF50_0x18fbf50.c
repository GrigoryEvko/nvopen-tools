// Function: sub_18FBF50
// Address: 0x18fbf50
//
__int64 __fastcall sub_18FBF50(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 v7; // rax

  *(_QWORD *)a1 = a3;
  *(_QWORD *)(a1 + 8) = a4;
  *(_QWORD *)(a1 + 16) = a5;
  *(_QWORD *)(a1 + 24) = a6;
  *(_QWORD *)(a1 + 32) = a2;
  *(_QWORD *)(a1 + 40) = a3;
  *(_QWORD *)(a1 + 48) = a5;
  *(_QWORD *)(a1 + 56) = a6;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = a7;
  v7 = sub_22077B0(640);
  if ( v7 )
  {
    *(_QWORD *)v7 = a7;
    *(_QWORD *)(v7 + 8) = v7 + 24;
    *(_QWORD *)(v7 + 416) = v7 + 448;
    *(_QWORD *)(v7 + 424) = v7 + 448;
    *(_QWORD *)(v7 + 512) = v7 + 528;
    *(_QWORD *)(v7 + 16) = 0x1000000000LL;
    *(_QWORD *)(v7 + 408) = 0;
    *(_QWORD *)(v7 + 432) = 8;
    *(_DWORD *)(v7 + 440) = 0;
    *(_QWORD *)(v7 + 520) = 0x800000000LL;
    *(_DWORD *)(v7 + 600) = 0;
    *(_QWORD *)(v7 + 608) = 0;
    *(_QWORD *)(v7 + 616) = v7 + 600;
    *(_QWORD *)(v7 + 624) = v7 + 600;
    *(_QWORD *)(v7 + 632) = 0;
  }
  *(_QWORD *)(a1 + 80) = v7;
  *(_QWORD *)(a1 + 200) = a1 + 216;
  *(_QWORD *)(a1 + 152) = a1 + 168;
  *(_QWORD *)(a1 + 304) = a1 + 320;
  *(_QWORD *)(a1 + 160) = 0x400000000LL;
  *(_QWORD *)(a1 + 312) = 0x400000000LL;
  *(_QWORD *)(a1 + 352) = a1 + 368;
  *(_QWORD *)(a1 + 88) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 104) = 0;
  *(_DWORD *)(a1 + 112) = 0;
  *(_QWORD *)(a1 + 120) = 0;
  *(_QWORD *)(a1 + 128) = 0;
  *(_QWORD *)(a1 + 136) = 0;
  *(_QWORD *)(a1 + 144) = 0;
  *(_QWORD *)(a1 + 208) = 0;
  *(_QWORD *)(a1 + 216) = 0;
  *(_QWORD *)(a1 + 224) = 1;
  *(_QWORD *)(a1 + 240) = 0;
  *(_QWORD *)(a1 + 248) = 0;
  *(_QWORD *)(a1 + 256) = 0;
  *(_DWORD *)(a1 + 264) = 0;
  *(_QWORD *)(a1 + 272) = 0;
  *(_QWORD *)(a1 + 280) = 0;
  *(_QWORD *)(a1 + 288) = 0;
  *(_QWORD *)(a1 + 296) = 0;
  *(_QWORD *)(a1 + 360) = 0;
  *(_QWORD *)(a1 + 368) = 0;
  *(_QWORD *)(a1 + 376) = 1;
  *(_QWORD *)(a1 + 392) = 0;
  *(_QWORD *)(a1 + 400) = 0;
  *(_QWORD *)(a1 + 408) = 0;
  *(_DWORD *)(a1 + 416) = 0;
  *(_QWORD *)(a1 + 464) = 0x400000000LL;
  *(_QWORD *)(a1 + 424) = 0;
  *(_QWORD *)(a1 + 432) = 0;
  *(_QWORD *)(a1 + 440) = 0;
  *(_QWORD *)(a1 + 448) = 0;
  *(_QWORD *)(a1 + 456) = a1 + 472;
  *(_QWORD *)(a1 + 504) = a1 + 520;
  *(_QWORD *)(a1 + 512) = 0;
  *(_QWORD *)(a1 + 520) = 0;
  *(_QWORD *)(a1 + 528) = 1;
  *(_QWORD *)(a1 + 544) = 0;
  *(_QWORD *)(a1 + 552) = 0;
  *(_QWORD *)(a1 + 560) = 0;
  *(_DWORD *)(a1 + 568) = 0;
  *(_QWORD *)(a1 + 576) = 0;
  *(_DWORD *)(a1 + 592) = 0;
  return a1 + 520;
}
