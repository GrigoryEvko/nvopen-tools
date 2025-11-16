// Function: sub_D241E0
// Address: 0xd241e0
//
_QWORD **__fastcall sub_D241E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r8
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rcx
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // rcx

  *(_QWORD *)a1 = *(_QWORD *)a2;
  *(_QWORD *)(a1 + 8) = *(_QWORD *)(a2 + 8);
  *(_QWORD *)(a1 + 16) = a1 + 32;
  *(_QWORD *)(a1 + 24) = 0x400000000LL;
  v6 = *(unsigned int *)(a2 + 24);
  if ( (_DWORD)v6 )
    sub_D22E90(a1 + 16, (char **)(a2 + 16), a3, a4, v6, a6);
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 64) = a1 + 80;
  if ( *(_DWORD *)(a2 + 72) )
    sub_D234C0(a1 + 64, a2 + 64, a3, a4, v6, a6);
  v7 = *(_QWORD *)(a2 + 80);
  *(_QWORD *)(a2 + 8) = 0;
  *(_QWORD *)a2 = 0;
  *(_QWORD *)(a1 + 80) = v7;
  v8 = *(_QWORD *)(a2 + 88);
  *(_QWORD *)(a2 + 80) = 0;
  *(_DWORD *)(a2 + 24) = 0;
  *(_DWORD *)(a2 + 72) = 0;
  *(_QWORD *)(a1 + 88) = v8;
  *(_QWORD *)(a1 + 104) = 0;
  *(_QWORD *)(a1 + 112) = 0;
  *(_DWORD *)(a1 + 120) = 0;
  LODWORD(v8) = *(_DWORD *)(a2 + 120);
  v9 = *(_QWORD *)(a2 + 104);
  ++*(_QWORD *)(a2 + 96);
  *(_DWORD *)(a1 + 120) = v8;
  *(_QWORD *)(a1 + 104) = v9;
  v10 = *(_QWORD *)(a2 + 112);
  *(_QWORD *)(a2 + 104) = 0;
  *(_QWORD *)(a2 + 112) = 0;
  *(_DWORD *)(a2 + 120) = 0;
  *(_QWORD *)(a1 + 128) = a1 + 144;
  *(_QWORD *)(a1 + 96) = 1;
  *(_QWORD *)(a1 + 112) = v10;
  *(_QWORD *)(a1 + 136) = 0x400000000LL;
  if ( *(_DWORD *)(a2 + 136) )
    sub_D23360(a1 + 128, (char **)(a2 + 128), v10, a4, v6, a6);
  *(_QWORD *)(a1 + 192) = 0;
  *(_QWORD *)(a1 + 184) = 0;
  *(_DWORD *)(a1 + 200) = 0;
  v11 = *(_QWORD *)(a2 + 184);
  *(_QWORD *)(a1 + 176) = 1;
  *(_QWORD *)(a1 + 184) = v11;
  LODWORD(v11) = *(_DWORD *)(a2 + 192);
  ++*(_QWORD *)(a2 + 176);
  *(_DWORD *)(a1 + 192) = v11;
  LODWORD(v11) = *(_DWORD *)(a2 + 196);
  *(_QWORD *)(a2 + 184) = 0;
  *(_DWORD *)(a1 + 196) = v11;
  LODWORD(v11) = *(_DWORD *)(a2 + 200);
  *(_QWORD *)(a2 + 192) = 0;
  *(_DWORD *)(a1 + 200) = v11;
  v12 = *(_QWORD *)(a2 + 208);
  *(_DWORD *)(a2 + 200) = 0;
  *(_QWORD *)(a1 + 208) = v12;
  *(_QWORD *)(a1 + 216) = *(_QWORD *)(a2 + 216);
  *(_QWORD *)(a1 + 224) = a1 + 240;
  *(_QWORD *)(a1 + 232) = 0x400000000LL;
  v13 = *(unsigned int *)(a2 + 232);
  if ( (_DWORD)v13 )
    sub_D22E90(a1 + 224, (char **)(a2 + 224), v10, v13, v6, a6);
  *(_QWORD *)(a1 + 280) = 0;
  *(_QWORD *)(a1 + 272) = a1 + 288;
  v14 = *(unsigned int *)(a2 + 280);
  if ( (_DWORD)v14 )
    sub_D234C0(a1 + 272, a2 + 272, v14, v13, v6, a6);
  v15 = *(_QWORD *)(a2 + 288);
  *(_QWORD *)(a2 + 216) = 0;
  *(_QWORD *)(a2 + 208) = 0;
  *(_QWORD *)(a1 + 288) = v15;
  v16 = *(_QWORD *)(a2 + 296);
  *(_QWORD *)(a2 + 288) = 0;
  *(_DWORD *)(a2 + 232) = 0;
  *(_DWORD *)(a2 + 280) = 0;
  *(_QWORD *)(a1 + 296) = v16;
  *(_QWORD *)(a1 + 312) = 0;
  *(_QWORD *)(a1 + 320) = 0;
  *(_DWORD *)(a1 + 328) = 0;
  v17 = *(_QWORD *)(a2 + 312);
  LODWORD(v16) = *(_DWORD *)(a2 + 328);
  ++*(_QWORD *)(a2 + 304);
  *(_QWORD *)(a1 + 312) = v17;
  v18 = *(_QWORD *)(a2 + 320);
  *(_DWORD *)(a1 + 328) = v16;
  *(_QWORD *)(a1 + 320) = v18;
  *(_QWORD *)(a2 + 312) = 0;
  *(_QWORD *)(a2 + 320) = 0;
  *(_DWORD *)(a2 + 328) = 0;
  *(_QWORD *)(a1 + 352) = a1 + 368;
  *(_QWORD *)(a1 + 400) = a1 + 416;
  *(_QWORD *)(a1 + 360) = 0x400000000LL;
  *(_QWORD *)(a1 + 304) = 1;
  *(_QWORD *)(a1 + 336) = 0;
  *(_QWORD *)(a1 + 344) = 0;
  *(_QWORD *)(a1 + 408) = 0;
  *(_QWORD *)(a1 + 416) = 0;
  *(_QWORD *)(a1 + 424) = 0;
  *(_QWORD *)(a1 + 432) = a1 + 448;
  *(_QWORD *)(a1 + 616) = 0;
  *(_QWORD *)(a1 + 624) = 0;
  *(_DWORD *)(a1 + 632) = 0;
  *(_QWORD *)(a1 + 440) = 0x1000000000LL;
  *(_QWORD *)(a1 + 576) = 0;
  v19 = *(_QWORD *)(a2 + 616);
  LODWORD(v18) = *(_DWORD *)(a2 + 632);
  ++*(_QWORD *)(a2 + 608);
  *(_QWORD *)(a1 + 616) = v19;
  v20 = *(_QWORD *)(a2 + 624);
  *(_DWORD *)(a1 + 632) = v18;
  *(_QWORD *)(a2 + 616) = 0;
  *(_QWORD *)(a2 + 624) = 0;
  *(_DWORD *)(a2 + 632) = 0;
  *(_QWORD *)(a1 + 584) = 0;
  *(_QWORD *)(a1 + 592) = 0;
  *(_DWORD *)(a1 + 600) = 0;
  *(_QWORD *)(a1 + 608) = 1;
  *(_QWORD *)(a1 + 624) = v20;
  *(_QWORD *)(a1 + 640) = a1 + 656;
  *(_QWORD *)(a1 + 648) = 0x400000000LL;
  if ( *(_DWORD *)(a2 + 648) )
    sub_D23200(a1 + 640, (char **)(a2 + 640), a1 + 656, v20, v6, a6);
  return sub_D24110(a1);
}
