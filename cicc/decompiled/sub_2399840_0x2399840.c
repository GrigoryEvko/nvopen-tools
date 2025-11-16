// Function: sub_2399840
// Address: 0x2399840
//
__int64 __fastcall sub_2399840(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rdi
  __int64 v9; // rdx
  int v10; // eax
  __int64 v11; // rdx
  __int64 v12; // rdx
  int v13; // eax
  __int64 v14; // rdx
  __int64 v15; // rdx
  int v16; // eax
  __int64 v17; // rdx
  __int64 v18; // rdx
  int v19; // eax
  __int64 v20; // rdx
  __int64 v21; // rdx
  int v22; // eax
  __int64 v23; // rdx
  __int64 v24; // rdx
  int v25; // eax
  __int64 v26; // rdx
  int v27; // eax
  __int64 v28; // rdx
  __int64 v29; // rdx
  int v30; // eax
  __int64 v31; // rdx
  __int64 v32; // rax
  __int64 v33; // rdx
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rdx
  __int64 v39; // rsi
  __int64 v40; // rax
  __int64 v41; // rdx
  __int64 v42; // rdx
  __int64 v43; // rdx
  __int64 v44; // rdx
  _QWORD *v45; // rax
  char v46; // al
  int v47; // eax
  __int64 v48; // rdx
  __int64 result; // rax
  __int64 v50; // rdx

  v7 = a1 + 320;
  *(_QWORD *)(v7 - 312) = 0;
  *(_QWORD *)(v7 - 304) = 0;
  *(_DWORD *)(v7 - 296) = 0;
  v9 = *(_QWORD *)(a2 + 8);
  v10 = *(_DWORD *)(a2 + 24);
  ++*(_QWORD *)a2;
  *(_QWORD *)(v7 - 312) = v9;
  v11 = *(_QWORD *)(a2 + 16);
  *(_QWORD *)(a2 + 8) = 0;
  *(_QWORD *)(a2 + 16) = 0;
  *(_DWORD *)(a2 + 24) = 0;
  *(_QWORD *)(v7 - 304) = v11;
  *(_QWORD *)(v7 - 280) = 0;
  *(_QWORD *)(v7 - 272) = 0;
  *(_DWORD *)(v7 - 264) = 0;
  v12 = *(_QWORD *)(a2 + 40);
  *(_DWORD *)(v7 - 296) = v10;
  v13 = *(_DWORD *)(a2 + 56);
  *(_QWORD *)(v7 - 280) = v12;
  v14 = *(_QWORD *)(a2 + 48);
  *(_DWORD *)(v7 - 264) = v13;
  *(_QWORD *)(v7 - 272) = v14;
  *(_QWORD *)(v7 - 320) = 1;
  *(_QWORD *)(v7 - 288) = 1;
  ++*(_QWORD *)(a2 + 32);
  *(_QWORD *)(a2 + 40) = 0;
  *(_QWORD *)(a2 + 48) = 0;
  *(_DWORD *)(a2 + 56) = 0;
  *(_QWORD *)(v7 - 248) = 0;
  *(_QWORD *)(v7 - 240) = 0;
  *(_DWORD *)(v7 - 232) = 0;
  v15 = *(_QWORD *)(a2 + 72);
  v16 = *(_DWORD *)(a2 + 88);
  ++*(_QWORD *)(a2 + 64);
  *(_QWORD *)(v7 - 248) = v15;
  v17 = *(_QWORD *)(a2 + 80);
  *(_QWORD *)(a2 + 72) = 0;
  *(_QWORD *)(a2 + 80) = 0;
  *(_DWORD *)(a2 + 88) = 0;
  *(_QWORD *)(v7 - 240) = v17;
  *(_QWORD *)(v7 - 216) = 0;
  *(_QWORD *)(v7 - 208) = 0;
  *(_DWORD *)(v7 - 200) = 0;
  v18 = *(_QWORD *)(a2 + 104);
  *(_DWORD *)(v7 - 232) = v16;
  v19 = *(_DWORD *)(a2 + 120);
  *(_QWORD *)(v7 - 216) = v18;
  v20 = *(_QWORD *)(a2 + 112);
  *(_DWORD *)(v7 - 200) = v19;
  *(_QWORD *)(v7 - 208) = v20;
  *(_QWORD *)(a2 + 104) = 0;
  *(_QWORD *)(a2 + 112) = 0;
  *(_DWORD *)(a2 + 120) = 0;
  *(_QWORD *)(v7 - 256) = 1;
  *(_QWORD *)(v7 - 224) = 1;
  ++*(_QWORD *)(a2 + 96);
  *(_QWORD *)(v7 - 192) = 1;
  *(_QWORD *)(v7 - 184) = 0;
  *(_QWORD *)(v7 - 176) = 0;
  *(_DWORD *)(v7 - 168) = 0;
  v21 = *(_QWORD *)(a2 + 136);
  v22 = *(_DWORD *)(a2 + 152);
  *(_QWORD *)(a2 + 136) = 0;
  *(_QWORD *)(v7 - 184) = v21;
  v23 = *(_QWORD *)(a2 + 144);
  *(_DWORD *)(a2 + 152) = 0;
  *(_QWORD *)(a2 + 144) = 0;
  ++*(_QWORD *)(a2 + 128);
  *(_QWORD *)(v7 - 176) = v23;
  *(_QWORD *)(v7 - 152) = 0;
  *(_QWORD *)(v7 - 144) = 0;
  *(_DWORD *)(v7 - 136) = 0;
  v24 = *(_QWORD *)(a2 + 168);
  *(_DWORD *)(v7 - 168) = v22;
  v25 = *(_DWORD *)(a2 + 184);
  *(_QWORD *)(v7 - 152) = v24;
  v26 = *(_QWORD *)(a2 + 176);
  *(_QWORD *)(a2 + 168) = 0;
  *(_QWORD *)(a2 + 176) = 0;
  *(_DWORD *)(a2 + 184) = 0;
  *(_QWORD *)(v7 - 144) = v26;
  *(_DWORD *)(v7 - 136) = v25;
  ++*(_QWORD *)(a2 + 160);
  *(_QWORD *)(v7 - 160) = 1;
  *(_QWORD *)(v7 - 128) = 1;
  *(_QWORD *)(v7 - 120) = 0;
  *(_QWORD *)(v7 - 112) = 0;
  *(_DWORD *)(v7 - 104) = 0;
  v27 = *(_DWORD *)(a2 + 216);
  v28 = *(_QWORD *)(a2 + 200);
  ++*(_QWORD *)(a2 + 192);
  *(_QWORD *)(a2 + 200) = 0;
  *(_DWORD *)(a2 + 216) = 0;
  *(_QWORD *)(v7 - 120) = v28;
  v29 = *(_QWORD *)(a2 + 208);
  *(_DWORD *)(v7 - 104) = v27;
  *(_QWORD *)(a2 + 208) = 0;
  *(_QWORD *)(v7 - 88) = 0;
  *(_QWORD *)(v7 - 80) = 0;
  *(_DWORD *)(v7 - 72) = 0;
  v30 = *(_DWORD *)(a2 + 248);
  *(_QWORD *)(v7 - 112) = v29;
  v31 = *(_QWORD *)(a2 + 232);
  *(_DWORD *)(v7 - 72) = v30;
  v32 = *(_QWORD *)(a2 + 256);
  *(_QWORD *)(v7 - 88) = v31;
  v33 = *(_QWORD *)(a2 + 240);
  *(_QWORD *)(v7 - 64) = v32;
  v34 = *(_QWORD *)(a2 + 264);
  *(_QWORD *)(v7 - 80) = v33;
  *(_QWORD *)(v7 - 56) = v34;
  v35 = *(_QWORD *)(a2 + 272);
  *(_QWORD *)(a2 + 232) = 0;
  *(_QWORD *)(a2 + 240) = 0;
  *(_DWORD *)(a2 + 248) = 0;
  *(_QWORD *)(v7 - 96) = 1;
  *(_QWORD *)(v7 - 48) = v35;
  ++*(_QWORD *)(a2 + 224);
  v36 = *(_QWORD *)(a2 + 280);
  *(_QWORD *)(v7 - 24) = 0;
  *(_QWORD *)(v7 - 16) = 0;
  *(_DWORD *)(v7 - 8) = 0;
  v37 = *(_QWORD *)(a2 + 296);
  *(_QWORD *)(v7 - 40) = v36;
  LODWORD(v36) = *(_DWORD *)(a2 + 312);
  *(_QWORD *)(v7 - 24) = v37;
  v38 = *(_QWORD *)(a2 + 304);
  ++*(_QWORD *)(a2 + 288);
  v39 = a2 + 320;
  *(_QWORD *)(v7 - 16) = v38;
  *(_DWORD *)(v7 - 8) = v36;
  *(_QWORD *)(v7 - 32) = 1;
  *(_QWORD *)(v39 - 24) = 0;
  *(_QWORD *)(v39 - 16) = 0;
  *(_DWORD *)(v39 - 8) = 0;
  sub_234E5E0(v7, v39, v38, a4, a5, a6);
  *(_QWORD *)(a1 + 416) = &unk_49DDC10;
  *(_QWORD *)(a1 + 424) = *(_QWORD *)(a2 + 424);
  v40 = *(_QWORD *)(a2 + 432);
  *(_QWORD *)(a1 + 448) = 0;
  *(_QWORD *)(a1 + 456) = 0;
  *(_DWORD *)(a1 + 464) = 0;
  *(_QWORD *)(a1 + 432) = v40;
  v41 = *(_QWORD *)(a2 + 448);
  LODWORD(v40) = *(_DWORD *)(a2 + 464);
  ++*(_QWORD *)(a2 + 440);
  *(_QWORD *)(a1 + 448) = v41;
  v42 = *(_QWORD *)(a2 + 456);
  *(_QWORD *)(a2 + 448) = 0;
  *(_QWORD *)(a2 + 456) = 0;
  *(_DWORD *)(a2 + 464) = 0;
  *(_QWORD *)(a1 + 456) = v42;
  *(_QWORD *)(a1 + 480) = 0;
  *(_QWORD *)(a1 + 488) = 0;
  *(_DWORD *)(a1 + 496) = 0;
  v43 = *(_QWORD *)(a2 + 480);
  *(_DWORD *)(a1 + 464) = v40;
  LODWORD(v40) = *(_DWORD *)(a2 + 496);
  *(_QWORD *)(a1 + 480) = v43;
  v44 = *(_QWORD *)(a2 + 488);
  ++*(_QWORD *)(a2 + 472);
  *(_QWORD *)(a1 + 440) = 1;
  *(_QWORD *)(a1 + 472) = 1;
  *(_QWORD *)(a1 + 488) = v44;
  *(_DWORD *)(a1 + 496) = v40;
  v45 = (_QWORD *)(a1 + 520);
  *(_QWORD *)(a2 + 480) = 0;
  *(_QWORD *)(a2 + 488) = 0;
  *(_DWORD *)(a2 + 496) = 0;
  *(_QWORD *)(a1 + 504) = 0;
  *(_QWORD *)(a1 + 512) = 1;
  do
  {
    if ( v45 )
      *v45 = -4096;
    v45 += 11;
  }
  while ( (_QWORD *)(a1 + 872) != v45 );
  sub_23993D0(a1 + 504, a2 + 504);
  sub_C8CF70(a1 + 872, (void *)(a1 + 904), 8, a2 + 904, a2 + 872);
  v46 = *(_BYTE *)(a2 + 968);
  *(_QWORD *)(a1 + 984) = 1;
  *(_BYTE *)(a1 + 968) = v46;
  v47 = *(_DWORD *)(a2 + 976);
  *(_QWORD *)(a1 + 992) = 0;
  *(_QWORD *)(a1 + 1000) = 0;
  *(_DWORD *)(a1 + 1008) = 0;
  v48 = *(_QWORD *)(a2 + 992);
  *(_DWORD *)(a1 + 976) = v47;
  result = *(unsigned int *)(a2 + 1008);
  *(_QWORD *)(a1 + 992) = v48;
  v50 = *(_QWORD *)(a2 + 1000);
  ++*(_QWORD *)(a2 + 984);
  *(_QWORD *)(a1 + 1000) = v50;
  *(_DWORD *)(a1 + 1008) = result;
  *(_QWORD *)(a2 + 992) = 0;
  *(_QWORD *)(a2 + 1000) = 0;
  *(_DWORD *)(a2 + 1008) = 0;
  return result;
}
