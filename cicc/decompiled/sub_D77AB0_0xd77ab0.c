// Function: sub_D77AB0
// Address: 0xd77ab0
//
__int64 __fastcall sub_D77AB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rdx
  __int64 v7; // rax
  int v9; // esi
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rcx
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  int v21; // esi
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rdx
  int v25; // esi
  __int64 v26; // rax
  int v27; // eax
  __int64 v28; // rdx
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rdx
  __int64 v32; // rdx
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rax
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 v40; // rax
  __int64 v41; // rax
  __int64 v42; // rdx
  __int64 result; // rax
  __int64 v44; // rdx

  v6 = a1 + 8;
  v7 = *(_QWORD *)(a2 + 16);
  if ( v7 )
  {
    a4 = a2 + 8;
    v9 = *(_DWORD *)(a2 + 8);
    *(_QWORD *)(a1 + 16) = v7;
    *(_DWORD *)(a1 + 8) = v9;
    *(_QWORD *)(a1 + 24) = *(_QWORD *)(a2 + 24);
    *(_QWORD *)(a1 + 32) = *(_QWORD *)(a2 + 32);
    *(_QWORD *)(v7 + 8) = v6;
    v10 = *(_QWORD *)(a2 + 40);
    *(_QWORD *)(a2 + 16) = 0;
    *(_QWORD *)(a1 + 40) = v10;
    *(_QWORD *)(a2 + 24) = a4;
    *(_QWORD *)(a2 + 32) = a4;
    *(_QWORD *)(a2 + 40) = 0;
  }
  else
  {
    *(_DWORD *)(a1 + 8) = 0;
    *(_QWORD *)(a1 + 16) = 0;
    *(_QWORD *)(a1 + 24) = v6;
    *(_QWORD *)(a1 + 32) = v6;
    *(_QWORD *)(a1 + 40) = 0;
  }
  v11 = *(_QWORD *)(a2 + 48);
  *(_QWORD *)(a2 + 48) = 0;
  *(_QWORD *)(a1 + 48) = v11;
  v12 = *(_QWORD *)(a2 + 56);
  *(_QWORD *)(a2 + 56) = 0;
  *(_QWORD *)(a1 + 56) = v12;
  v13 = *(_QWORD *)(a2 + 64);
  *(_DWORD *)(a2 + 64) = 0;
  *(_QWORD *)(a1 + 64) = v13;
  *(_QWORD *)(a1 + 72) = *(_QWORD *)(a2 + 72);
  *(_QWORD *)(a1 + 80) = *(_QWORD *)(a2 + 80);
  *(_QWORD *)(a1 + 88) = a1 + 104;
  *(_QWORD *)(a1 + 96) = 0x400000000LL;
  if ( *(_DWORD *)(a2 + 96) )
    sub_D763D0(a1 + 88, (char **)(a2 + 88), v6, a4, a5, a6);
  *(_QWORD *)(a1 + 144) = 0;
  *(_QWORD *)(a1 + 136) = a1 + 152;
  v14 = *(unsigned int *)(a2 + 144);
  if ( (_DWORD)v14 )
    sub_D76240(a1 + 136, a2 + 136, v6, v14, a5, a6);
  v15 = *(_QWORD *)(a2 + 152);
  *(_QWORD *)(a2 + 80) = 0;
  v16 = a1 + 216;
  *(_QWORD *)(a2 + 72) = 0;
  *(_QWORD *)(a1 + 152) = v15;
  v17 = *(_QWORD *)(a2 + 160);
  *(_QWORD *)(a2 + 152) = 0;
  *(_QWORD *)(a1 + 160) = v17;
  v18 = *(_QWORD *)(a2 + 168);
  *(_DWORD *)(a2 + 96) = 0;
  *(_DWORD *)(a2 + 144) = 0;
  *(_QWORD *)(a1 + 192) = 0;
  *(_QWORD *)(a1 + 184) = 0;
  *(_DWORD *)(a1 + 200) = 0;
  *(_QWORD *)(a1 + 168) = v18;
  v19 = *(_QWORD *)(a2 + 184);
  *(_QWORD *)(a1 + 176) = 1;
  *(_QWORD *)(a1 + 184) = v19;
  LODWORD(v19) = *(_DWORD *)(a2 + 192);
  ++*(_QWORD *)(a2 + 176);
  *(_DWORD *)(a1 + 192) = v19;
  LODWORD(v19) = *(_DWORD *)(a2 + 196);
  *(_QWORD *)(a2 + 184) = 0;
  *(_DWORD *)(a1 + 196) = v19;
  LODWORD(v19) = *(_DWORD *)(a2 + 200);
  *(_QWORD *)(a2 + 192) = 0;
  *(_DWORD *)(a1 + 200) = v19;
  v20 = *(_QWORD *)(a2 + 224);
  *(_DWORD *)(a2 + 200) = 0;
  if ( v20 )
  {
    v21 = *(_DWORD *)(a2 + 216);
    *(_QWORD *)(a1 + 224) = v20;
    v14 = a2 + 216;
    *(_DWORD *)(a1 + 216) = v21;
    *(_QWORD *)(a1 + 232) = *(_QWORD *)(a2 + 232);
    *(_QWORD *)(a1 + 240) = *(_QWORD *)(a2 + 240);
    *(_QWORD *)(v20 + 8) = v16;
    v22 = *(_QWORD *)(a2 + 248);
    *(_QWORD *)(a2 + 224) = 0;
    *(_QWORD *)(a1 + 248) = v22;
    *(_QWORD *)(a2 + 232) = a2 + 216;
    *(_QWORD *)(a2 + 240) = a2 + 216;
    *(_QWORD *)(a2 + 248) = 0;
  }
  else
  {
    *(_DWORD *)(a1 + 216) = 0;
    *(_QWORD *)(a1 + 224) = 0;
    *(_QWORD *)(a1 + 232) = v16;
    *(_QWORD *)(a1 + 240) = v16;
    *(_QWORD *)(a1 + 248) = 0;
  }
  v23 = *(_QWORD *)(a2 + 272);
  v24 = a1 + 264;
  if ( v23 )
  {
    v25 = *(_DWORD *)(a2 + 264);
    *(_QWORD *)(a1 + 272) = v23;
    v14 = a2 + 264;
    *(_DWORD *)(a1 + 264) = v25;
    *(_QWORD *)(a1 + 280) = *(_QWORD *)(a2 + 280);
    *(_QWORD *)(a1 + 288) = *(_QWORD *)(a2 + 288);
    *(_QWORD *)(v23 + 8) = v24;
    v26 = *(_QWORD *)(a2 + 296);
    *(_QWORD *)(a2 + 272) = 0;
    *(_QWORD *)(a1 + 296) = v26;
    *(_QWORD *)(a2 + 280) = a2 + 264;
    *(_QWORD *)(a2 + 288) = a2 + 264;
    *(_QWORD *)(a2 + 296) = 0;
  }
  else
  {
    *(_DWORD *)(a1 + 264) = 0;
    *(_QWORD *)(a1 + 272) = 0;
    *(_QWORD *)(a1 + 280) = v24;
    *(_QWORD *)(a1 + 288) = v24;
    *(_QWORD *)(a1 + 296) = 0;
  }
  *(_QWORD *)(a1 + 312) = 0;
  *(_QWORD *)(a1 + 320) = 0;
  *(_DWORD *)(a1 + 328) = 0;
  v27 = *(_DWORD *)(a2 + 328);
  v28 = *(_QWORD *)(a2 + 312);
  ++*(_QWORD *)(a2 + 304);
  *(_DWORD *)(a1 + 328) = v27;
  v29 = *(_QWORD *)(a2 + 336);
  *(_QWORD *)(a1 + 312) = v28;
  v30 = *(_QWORD *)(a2 + 320);
  *(_QWORD *)(a2 + 312) = 0;
  *(_QWORD *)(a2 + 320) = 0;
  *(_DWORD *)(a2 + 328) = 0;
  *(_QWORD *)(a1 + 320) = v30;
  *(_QWORD *)(a1 + 336) = v29;
  LODWORD(v29) = *(_DWORD *)(a2 + 344);
  *(_QWORD *)(a1 + 360) = 0;
  *(_QWORD *)(a1 + 368) = 0;
  *(_DWORD *)(a1 + 376) = 0;
  v31 = *(_QWORD *)(a2 + 360);
  *(_DWORD *)(a1 + 344) = v29;
  LODWORD(v29) = *(_DWORD *)(a2 + 376);
  *(_QWORD *)(a1 + 360) = v31;
  v32 = *(_QWORD *)(a2 + 368);
  *(_DWORD *)(a1 + 376) = v29;
  *(_QWORD *)(a1 + 368) = v32;
  ++*(_QWORD *)(a2 + 352);
  *(_QWORD *)(a1 + 304) = 1;
  *(_QWORD *)(a1 + 352) = 1;
  *(_QWORD *)(a2 + 360) = 0;
  *(_QWORD *)(a2 + 368) = 0;
  *(_DWORD *)(a2 + 376) = 0;
  *(_QWORD *)(a1 + 400) = 0;
  *(_QWORD *)(a1 + 392) = 0;
  *(_DWORD *)(a1 + 408) = 0;
  v33 = *(_QWORD *)(a2 + 392);
  *(_QWORD *)(a1 + 384) = 1;
  *(_QWORD *)(a1 + 392) = v33;
  LODWORD(v33) = *(_DWORD *)(a2 + 400);
  ++*(_QWORD *)(a2 + 384);
  *(_DWORD *)(a1 + 400) = v33;
  LODWORD(v33) = *(_DWORD *)(a2 + 404);
  *(_QWORD *)(a2 + 392) = 0;
  *(_DWORD *)(a1 + 404) = v33;
  LODWORD(v33) = *(_DWORD *)(a2 + 408);
  *(_QWORD *)(a2 + 400) = 0;
  *(_DWORD *)(a1 + 408) = v33;
  v34 = *(_QWORD *)(a2 + 416);
  *(_DWORD *)(a2 + 408) = 0;
  *(_QWORD *)(a1 + 416) = v34;
  *(_QWORD *)(a1 + 424) = *(_QWORD *)(a2 + 424);
  *(_QWORD *)(a1 + 432) = a1 + 448;
  *(_QWORD *)(a1 + 440) = 0x400000000LL;
  v35 = *(unsigned int *)(a2 + 440);
  if ( (_DWORD)v35 )
    sub_D763D0(a1 + 432, (char **)(a2 + 432), v35, v14, a5, a6);
  *(_QWORD *)(a1 + 488) = 0;
  *(_QWORD *)(a1 + 480) = a1 + 496;
  if ( *(_DWORD *)(a2 + 488) )
    sub_D76240(a1 + 480, a2 + 480, v35, v14, a5, a6);
  v36 = *(_QWORD *)(a2 + 496);
  *(_QWORD *)(a2 + 424) = 0;
  *(_QWORD *)(a2 + 416) = 0;
  *(_QWORD *)(a1 + 496) = v36;
  v37 = *(_QWORD *)(a2 + 504);
  *(_QWORD *)(a2 + 496) = 0;
  *(_QWORD *)(a1 + 504) = v37;
  v38 = *(_QWORD *)(a2 + 512);
  *(_DWORD *)(a2 + 440) = 0;
  *(_DWORD *)(a2 + 488) = 0;
  *(_QWORD *)(a1 + 512) = v38;
  *(_QWORD *)(a1 + 520) = *(_QWORD *)(a2 + 520);
  v39 = *(_QWORD *)(a2 + 528);
  *(_QWORD *)(a2 + 528) = 0;
  *(_QWORD *)(a1 + 528) = v39;
  v40 = *(_QWORD *)(a2 + 536);
  *(_QWORD *)(a2 + 536) = 0;
  *(_QWORD *)(a1 + 536) = v40;
  v41 = *(_QWORD *)(a2 + 544);
  *(_QWORD *)(a2 + 544) = 0;
  *(_QWORD *)(a1 + 544) = v41;
  *(_QWORD *)(a1 + 560) = 0;
  *(_QWORD *)(a1 + 568) = 0;
  *(_DWORD *)(a1 + 576) = 0;
  v42 = *(_QWORD *)(a2 + 560);
  result = *(unsigned int *)(a2 + 576);
  ++*(_QWORD *)(a2 + 552);
  *(_QWORD *)(a1 + 560) = v42;
  v44 = *(_QWORD *)(a2 + 568);
  *(_QWORD *)(a1 + 552) = 1;
  *(_QWORD *)(a1 + 568) = v44;
  *(_DWORD *)(a1 + 576) = result;
  *(_QWORD *)(a2 + 560) = 0;
  *(_QWORD *)(a2 + 568) = 0;
  *(_DWORD *)(a2 + 576) = 0;
  return result;
}
