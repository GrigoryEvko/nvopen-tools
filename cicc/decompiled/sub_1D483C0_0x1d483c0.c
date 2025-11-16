// Function: sub_1D483C0
// Address: 0x1d483c0
//
__int64 __fastcall sub_1D483C0(__int64 a1, __int64 a2, int a3)
{
  _QWORD *v4; // rax
  _QWORD *v5; // rax
  _QWORD *v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // r13
  __int64 v13; // rax
  __int64 *v14; // rdx
  __int64 v15; // rsi
  __int64 v16; // rcx
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax

  *(_QWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 24) = 3;
  *(_QWORD *)(a1 + 16) = &unk_4FC1808;
  *(_QWORD *)(a1 + 80) = a1 + 64;
  *(_QWORD *)(a1 + 88) = a1 + 64;
  *(_QWORD *)(a1 + 128) = a1 + 112;
  *(_QWORD *)(a1 + 136) = a1 + 112;
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
  *(_QWORD *)a1 = &unk_49FB790;
  *(_QWORD *)(a1 + 160) = 0;
  *(_QWORD *)(a1 + 168) = 0;
  *(_DWORD *)(a1 + 176) = 8;
  v4 = (_QWORD *)malloc(8u);
  if ( !v4 )
  {
    sub_16BD1C0("Allocation failed", 1u);
    v4 = 0;
  }
  *(_QWORD *)(a1 + 160) = v4;
  *(_QWORD *)(a1 + 168) = 1;
  *v4 = 0;
  *(_QWORD *)(a1 + 184) = 0;
  *(_QWORD *)(a1 + 192) = 0;
  *(_DWORD *)(a1 + 200) = 8;
  v5 = (_QWORD *)malloc(8u);
  if ( !v5 )
  {
    sub_16BD1C0("Allocation failed", 1u);
    v5 = 0;
  }
  *(_QWORD *)(a1 + 184) = v5;
  *(_QWORD *)(a1 + 192) = 1;
  *v5 = 0;
  *(_QWORD *)(a1 + 208) = 0;
  *(_QWORD *)(a1 + 216) = 0;
  *(_DWORD *)(a1 + 224) = 8;
  v6 = (_QWORD *)malloc(8u);
  if ( !v6 )
  {
    sub_16BD1C0("Allocation failed", 1u);
    v6 = 0;
  }
  *(_QWORD *)(a1 + 208) = v6;
  *v6 = 0;
  *(_QWORD *)(a1 + 216) = 1;
  *(_QWORD *)(a1 + 232) = a2;
  *(_QWORD *)a1 = &unk_49F9B60;
  v7 = sub_22077B0(1008);
  if ( v7 )
  {
    memset((void *)v7, 0, 0x3F0u);
    *(_QWORD *)(v7 + 856) = 4;
    *(_QWORD *)(v7 + 184) = v7 + 200;
    *(_QWORD *)(v7 + 192) = 0x100000000LL;
    *(_QWORD *)(v7 + 400) = v7 + 416;
    *(_QWORD *)(v7 + 408) = 0x800000000LL;
    *(_QWORD *)(v7 + 568) = v7 + 584;
    *(_QWORD *)(v7 + 576) = 0x3200000000LL;
    *(_QWORD *)(v7 + 840) = v7 + 872;
    *(_QWORD *)(v7 + 848) = v7 + 872;
    *(_QWORD *)(v7 + 944) = v7 + 960;
    *(_DWORD *)(v7 + 960) = 0x80000000;
    *(_DWORD *)(v7 + 976) = 1;
    *(_DWORD *)(v7 + 992) = 1;
  }
  *(_QWORD *)(a1 + 248) = v7;
  v8 = sub_22077B0(904);
  v12 = v8;
  if ( v8 )
    sub_1D291B0(v8, a2, a3, v9, v10, v11);
  *(_QWORD *)(a1 + 272) = v12;
  v13 = sub_22077B0(776);
  if ( v13 )
  {
    v14 = *(__int64 **)(a1 + 272);
    *(_QWORD *)(v13 + 8) = 0;
    *(_QWORD *)(v13 + 296) = v13 + 312;
    v15 = *(_QWORD *)(a1 + 248);
    *(_QWORD *)(v13 + 104) = v13 + 120;
    *(_QWORD *)(v13 + 304) = 0xA00000000LL;
    *(_QWORD *)(v13 + 16) = 0;
    *(_QWORD *)(v13 + 24) = 0;
    *(_DWORD *)(v13 + 32) = 0;
    *(_QWORD *)(v13 + 40) = 0;
    *(_QWORD *)(v13 + 48) = 0;
    *(_QWORD *)(v13 + 56) = 0;
    *(_DWORD *)(v13 + 64) = 0;
    *(_QWORD *)(v13 + 72) = 0;
    *(_QWORD *)(v13 + 80) = 0;
    *(_QWORD *)(v13 + 88) = 0;
    *(_DWORD *)(v13 + 96) = 0;
    *(_QWORD *)(v13 + 112) = 0x800000000LL;
    *(_QWORD *)(v13 + 248) = 0;
    *(_QWORD *)(v13 + 256) = 0;
    *(_QWORD *)(v13 + 264) = 0;
    *(_DWORD *)(v13 + 272) = 0;
    *(_QWORD *)(v13 + 280) = 1;
    *(_DWORD *)(v13 + 288) = 0;
    *(_QWORD *)(v13 + 392) = v13 + 408;
    *(_QWORD *)(v13 + 400) = 0x800000000LL;
    v16 = *v14;
    *(_QWORD *)v13 = 0;
    *(_DWORD *)(v13 + 536) = 1;
    *(_QWORD *)(v13 + 544) = v16;
    *(_QWORD *)(v13 + 552) = v14;
    *(_QWORD *)(v13 + 560) = 0;
    *(_QWORD *)(v13 + 568) = 0;
    *(_QWORD *)(v13 + 584) = 0;
    *(_QWORD *)(v13 + 592) = 0;
    *(_QWORD *)(v13 + 600) = 0;
    *(_QWORD *)(v13 + 608) = 0;
    *(_QWORD *)(v13 + 616) = 0;
    *(_QWORD *)(v13 + 624) = 0;
    *(_QWORD *)(v13 + 632) = 0;
    *(_QWORD *)(v13 + 640) = 0;
    *(_QWORD *)(v13 + 648) = 0;
    *(_QWORD *)(v13 + 656) = 0;
    *(_QWORD *)(v13 + 664) = 0;
    *(_QWORD *)(v13 + 672) = 0;
    *(_QWORD *)(v13 + 680) = 0;
    *(_QWORD *)(v13 + 688) = 0;
    *(_QWORD *)(v13 + 696) = 0;
    *(_DWORD *)(v13 + 704) = 0;
    *(_QWORD *)(v13 + 712) = v15;
    *(_QWORD *)(v13 + 728) = 0;
    *(_QWORD *)(v13 + 736) = 0;
    *(_QWORD *)(v13 + 744) = 0;
    *(_DWORD *)(v13 + 752) = 0;
    *(_BYTE *)(v13 + 760) = 0;
  }
  *(_QWORD *)(a1 + 280) = v13;
  *(_DWORD *)(a1 + 304) = a3;
  *(_QWORD *)(a1 + 288) = 0;
  *(_QWORD *)(a1 + 296) = 0;
  *(_QWORD *)(a1 + 336) = 0;
  *(_QWORD *)(a1 + 344) = a1 + 376;
  *(_QWORD *)(a1 + 352) = a1 + 376;
  *(_QWORD *)(a1 + 360) = 4;
  *(_DWORD *)(a1 + 368) = 0;
  *(_QWORD *)(a1 + 408) = 0;
  *(_DWORD *)(a1 + 416) = 0;
  *(_QWORD *)(a1 + 424) = 0;
  *(_QWORD *)(a1 + 432) = 0;
  *(_QWORD *)(a1 + 440) = 0;
  v17 = sub_163A1D0();
  sub_1D8E1B0(v17);
  v18 = sub_163A1D0();
  sub_1376ED0(v18);
  v19 = sub_163A1D0();
  sub_134D8E0(v19);
  v20 = sub_163A1D0();
  return sub_149CBF0(v20);
}
