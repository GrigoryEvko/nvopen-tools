// Function: sub_D99330
// Address: 0xd99330
//
__int64 __fastcall sub_D99330(__int64 a1, __int64 a2)
{
  __int64 v4; // rdi
  __int64 v5; // rdx
  int v6; // eax
  __int64 v7; // rdx
  __int64 v8; // rdx
  int v9; // eax
  __int64 v10; // rdx
  __int64 v11; // rdx
  int v12; // eax
  __int64 v13; // rdx
  __int64 v14; // rdx
  int v15; // eax
  __int64 v16; // rdx
  __int64 v17; // rdx
  int v18; // eax
  __int64 v19; // rdx
  __int64 v20; // rdx
  int v21; // eax
  __int64 v22; // rdx
  __int64 v23; // rdx
  int v24; // eax
  __int64 v25; // rdx
  int v26; // eax
  __int64 v27; // rdx
  __int64 v28; // rdx
  __int64 v29; // rdx
  __int64 v30; // rdx
  int v31; // eax
  __int64 v32; // rdx
  int v33; // eax
  __int64 v34; // rdx
  __int64 v35; // rdx
  __int64 v36; // rdx
  int v37; // eax
  __int64 v38; // rdx
  int v39; // eax
  __int64 v40; // rdx
  __int64 v41; // rdx
  int v42; // eax
  __int64 v43; // rdx
  __int64 v44; // rdx
  int v45; // eax
  __int64 v46; // rdx
  __int64 v47; // r8
  __int64 v48; // r9
  void *v49; // rdi
  __int64 v50; // rax
  unsigned int v51; // r13d
  unsigned int v52; // r13d
  __int64 v53; // rax
  __int64 v54; // rax
  __int64 v55; // rdx
  __int64 v56; // rdx
  __int64 v57; // rdx
  __int64 v58; // rdx
  __int64 v59; // rax
  __int64 result; // rax
  __int64 v61; // rax
  const void *v62; // rsi
  size_t v63; // rdx
  __int64 v64; // rdx
  size_t v65; // rdx
  int v66; // edx
  int v67; // eax

  v4 = a1 + 224;
  *(_QWORD *)(v4 - 224) = *(_QWORD *)a2;
  *(_QWORD *)(v4 - 216) = *(_QWORD *)(a2 + 8);
  *(_BYTE *)(v4 - 208) = *(_BYTE *)(a2 + 16);
  *(_QWORD *)(v4 - 200) = *(_QWORD *)(a2 + 24);
  *(_QWORD *)(v4 - 192) = *(_QWORD *)(a2 + 32);
  *(_QWORD *)(v4 - 184) = *(_QWORD *)(a2 + 40);
  *(_QWORD *)(v4 - 176) = *(_QWORD *)(a2 + 48);
  *(_QWORD *)(v4 - 168) = *(_QWORD *)(a2 + 56);
  *(_QWORD *)(a2 + 56) = 0;
  *(_QWORD *)(v4 - 160) = 0;
  *(_QWORD *)(v4 - 152) = 0;
  *(_QWORD *)(v4 - 144) = 0;
  *(_DWORD *)(v4 - 136) = 0;
  *(_QWORD *)(v4 - 128) = 0;
  *(_QWORD *)(v4 - 120) = 0;
  *(_QWORD *)(v4 - 112) = 0;
  *(_DWORD *)(v4 - 104) = 0;
  *(_QWORD *)(v4 - 96) = 1;
  *(_QWORD *)(v4 - 88) = 0;
  *(_QWORD *)(v4 - 80) = 0;
  *(_DWORD *)(v4 - 72) = 0;
  v5 = *(_QWORD *)(a2 + 136);
  v6 = *(_DWORD *)(a2 + 152);
  ++*(_QWORD *)(a2 + 128);
  *(_QWORD *)(v4 - 88) = v5;
  v7 = *(_QWORD *)(a2 + 144);
  *(_DWORD *)(v4 - 72) = v6;
  *(_QWORD *)(a2 + 136) = 0;
  *(_QWORD *)(a2 + 144) = 0;
  *(_DWORD *)(a2 + 152) = 0;
  *(_QWORD *)(v4 - 80) = v7;
  *(_QWORD *)(v4 - 64) = 0;
  *(_QWORD *)(v4 - 56) = 0;
  *(_QWORD *)(v4 - 48) = 0;
  *(_DWORD *)(v4 - 40) = 0;
  *(_QWORD *)(v4 - 32) = 0;
  *(_QWORD *)(v4 - 24) = 0;
  *(_QWORD *)(v4 - 16) = 0;
  *(_DWORD *)(v4 - 8) = 0;
  sub_C8CF70(v4, (void *)(a1 + 256), 8, a2 + 256, a2 + 224);
  sub_C8CF70(a1 + 320, (void *)(a1 + 352), 8, a2 + 352, a2 + 320);
  *(_QWORD *)(a1 + 416) = 0;
  *(_QWORD *)(a1 + 424) = a1 + 448;
  *(_BYTE *)(a1 + 444) = 1;
  *(_QWORD *)(a1 + 432) = 8;
  *(_DWORD *)(a1 + 440) = 0;
  sub_C8CF70(a1 + 512, (void *)(a1 + 544), 8, a2 + 544, a2 + 512);
  *(_QWORD *)(a1 + 624) = 0;
  *(_WORD *)(a1 + 608) = 0;
  *(_QWORD *)(a1 + 632) = 0;
  *(_DWORD *)(a1 + 640) = 0;
  v8 = *(_QWORD *)(a2 + 624);
  v9 = *(_DWORD *)(a2 + 640);
  ++*(_QWORD *)(a2 + 616);
  *(_QWORD *)(a1 + 624) = v8;
  v10 = *(_QWORD *)(a2 + 632);
  *(_QWORD *)(a2 + 624) = 0;
  *(_QWORD *)(a2 + 632) = 0;
  *(_DWORD *)(a2 + 640) = 0;
  *(_QWORD *)(a1 + 632) = v10;
  *(_QWORD *)(a1 + 656) = 0;
  *(_QWORD *)(a1 + 664) = 0;
  *(_DWORD *)(a1 + 672) = 0;
  v11 = *(_QWORD *)(a2 + 656);
  *(_DWORD *)(a1 + 640) = v9;
  v12 = *(_DWORD *)(a2 + 672);
  *(_QWORD *)(a1 + 656) = v11;
  v13 = *(_QWORD *)(a2 + 664);
  *(_DWORD *)(a1 + 672) = v12;
  *(_QWORD *)(a1 + 664) = v13;
  ++*(_QWORD *)(a2 + 648);
  *(_QWORD *)(a1 + 616) = 1;
  *(_QWORD *)(a1 + 648) = 1;
  *(_QWORD *)(a2 + 656) = 0;
  *(_QWORD *)(a2 + 664) = 0;
  *(_DWORD *)(a2 + 672) = 0;
  *(_QWORD *)(a1 + 680) = 1;
  *(_QWORD *)(a1 + 688) = 0;
  *(_QWORD *)(a1 + 696) = 0;
  *(_DWORD *)(a1 + 704) = 0;
  v14 = *(_QWORD *)(a2 + 688);
  v15 = *(_DWORD *)(a2 + 704);
  ++*(_QWORD *)(a2 + 680);
  *(_QWORD *)(a1 + 688) = v14;
  v16 = *(_QWORD *)(a2 + 696);
  *(_QWORD *)(a2 + 688) = 0;
  *(_QWORD *)(a2 + 696) = 0;
  *(_DWORD *)(a2 + 704) = 0;
  *(_QWORD *)(a1 + 696) = v16;
  *(_QWORD *)(a1 + 720) = 0;
  *(_QWORD *)(a1 + 728) = 0;
  *(_DWORD *)(a1 + 736) = 0;
  v17 = *(_QWORD *)(a2 + 720);
  *(_DWORD *)(a1 + 704) = v15;
  v18 = *(_DWORD *)(a2 + 736);
  *(_QWORD *)(a1 + 720) = v17;
  v19 = *(_QWORD *)(a2 + 728);
  ++*(_QWORD *)(a2 + 712);
  *(_QWORD *)(a2 + 720) = 0;
  *(_QWORD *)(a2 + 728) = 0;
  *(_DWORD *)(a2 + 736) = 0;
  *(_QWORD *)(a1 + 728) = v19;
  *(_DWORD *)(a1 + 736) = v18;
  *(_QWORD *)(a1 + 712) = 1;
  *(_QWORD *)(a1 + 744) = 1;
  *(_QWORD *)(a1 + 752) = 0;
  *(_QWORD *)(a1 + 760) = 0;
  *(_DWORD *)(a1 + 768) = 0;
  ++*(_QWORD *)(a2 + 744);
  v20 = *(_QWORD *)(a2 + 752);
  v21 = *(_DWORD *)(a2 + 768);
  *(_QWORD *)(a2 + 752) = 0;
  *(_QWORD *)(a1 + 752) = v20;
  v22 = *(_QWORD *)(a2 + 760);
  *(_DWORD *)(a2 + 768) = 0;
  *(_QWORD *)(a2 + 760) = 0;
  *(_QWORD *)(a1 + 760) = v22;
  *(_QWORD *)(a1 + 784) = 0;
  *(_QWORD *)(a1 + 792) = 0;
  *(_DWORD *)(a1 + 800) = 0;
  v23 = *(_QWORD *)(a2 + 784);
  *(_DWORD *)(a1 + 768) = v21;
  v24 = *(_DWORD *)(a2 + 800);
  *(_QWORD *)(a1 + 784) = v23;
  v25 = *(_QWORD *)(a2 + 792);
  ++*(_QWORD *)(a2 + 776);
  *(_QWORD *)(a2 + 784) = 0;
  *(_QWORD *)(a2 + 792) = 0;
  *(_DWORD *)(a2 + 800) = 0;
  *(_DWORD *)(a1 + 800) = v24;
  *(_QWORD *)(a1 + 816) = 0;
  *(_QWORD *)(a1 + 824) = 0;
  *(_DWORD *)(a1 + 832) = 0;
  v26 = *(_DWORD *)(a2 + 832);
  ++*(_QWORD *)(a2 + 808);
  *(_QWORD *)(a1 + 776) = 1;
  *(_QWORD *)(a1 + 792) = v25;
  *(_QWORD *)(a1 + 808) = 1;
  v27 = *(_QWORD *)(a2 + 816);
  *(_QWORD *)(a2 + 816) = 0;
  *(_QWORD *)(a1 + 816) = v27;
  v28 = *(_QWORD *)(a2 + 824);
  *(_DWORD *)(a2 + 832) = 0;
  *(_QWORD *)(a2 + 824) = 0;
  *(_QWORD *)(a1 + 824) = v28;
  *(_QWORD *)(a1 + 848) = 0;
  *(_QWORD *)(a1 + 856) = 0;
  *(_DWORD *)(a1 + 864) = 0;
  v29 = *(_QWORD *)(a2 + 848);
  ++*(_QWORD *)(a2 + 840);
  *(_QWORD *)(a1 + 848) = v29;
  v30 = *(_QWORD *)(a2 + 856);
  *(_QWORD *)(a2 + 848) = 0;
  *(_QWORD *)(a2 + 856) = 0;
  *(_DWORD *)(a1 + 832) = v26;
  v31 = *(_DWORD *)(a2 + 864);
  *(_QWORD *)(a1 + 856) = v30;
  *(_DWORD *)(a2 + 864) = 0;
  *(_QWORD *)(a1 + 880) = 0;
  *(_QWORD *)(a1 + 888) = 0;
  *(_DWORD *)(a1 + 896) = 0;
  v32 = *(_QWORD *)(a2 + 880);
  *(_DWORD *)(a1 + 864) = v31;
  v33 = *(_DWORD *)(a2 + 896);
  ++*(_QWORD *)(a2 + 872);
  *(_QWORD *)(a1 + 840) = 1;
  *(_QWORD *)(a1 + 872) = 1;
  *(_QWORD *)(a1 + 880) = v32;
  v34 = *(_QWORD *)(a2 + 888);
  *(_QWORD *)(a2 + 880) = 0;
  *(_QWORD *)(a2 + 888) = 0;
  *(_DWORD *)(a2 + 896) = 0;
  *(_QWORD *)(a1 + 888) = v34;
  *(_QWORD *)(a1 + 912) = 0;
  *(_QWORD *)(a1 + 920) = 0;
  *(_DWORD *)(a1 + 928) = 0;
  v35 = *(_QWORD *)(a2 + 912);
  ++*(_QWORD *)(a2 + 904);
  *(_QWORD *)(a1 + 912) = v35;
  v36 = *(_QWORD *)(a2 + 920);
  *(_QWORD *)(a2 + 912) = 0;
  *(_QWORD *)(a2 + 920) = 0;
  *(_DWORD *)(a1 + 896) = v33;
  v37 = *(_DWORD *)(a2 + 928);
  *(_QWORD *)(a1 + 920) = v36;
  *(_DWORD *)(a2 + 928) = 0;
  *(_QWORD *)(a1 + 944) = 0;
  *(_QWORD *)(a1 + 952) = 0;
  *(_DWORD *)(a1 + 960) = 0;
  v38 = *(_QWORD *)(a2 + 944);
  *(_DWORD *)(a1 + 928) = v37;
  v39 = *(_DWORD *)(a2 + 960);
  *(_QWORD *)(a1 + 944) = v38;
  v40 = *(_QWORD *)(a2 + 952);
  ++*(_QWORD *)(a2 + 936);
  *(_QWORD *)(a1 + 952) = v40;
  *(_QWORD *)(a1 + 904) = 1;
  *(_QWORD *)(a1 + 936) = 1;
  *(_DWORD *)(a1 + 960) = v39;
  *(_QWORD *)(a2 + 944) = 0;
  *(_QWORD *)(a2 + 952) = 0;
  *(_DWORD *)(a2 + 960) = 0;
  *(_QWORD *)(a1 + 976) = 0;
  *(_QWORD *)(a1 + 984) = 0;
  *(_DWORD *)(a1 + 992) = 0;
  v41 = *(_QWORD *)(a2 + 976);
  v42 = *(_DWORD *)(a2 + 992);
  ++*(_QWORD *)(a2 + 968);
  *(_QWORD *)(a1 + 976) = v41;
  v43 = *(_QWORD *)(a2 + 984);
  *(_QWORD *)(a2 + 976) = 0;
  *(_QWORD *)(a2 + 984) = 0;
  *(_DWORD *)(a2 + 992) = 0;
  *(_QWORD *)(a1 + 984) = v43;
  *(_QWORD *)(a1 + 1008) = 0;
  *(_QWORD *)(a1 + 1016) = 0;
  *(_DWORD *)(a1 + 1024) = 0;
  v44 = *(_QWORD *)(a2 + 1008);
  *(_DWORD *)(a1 + 992) = v42;
  v45 = *(_DWORD *)(a2 + 1024);
  *(_QWORD *)(a1 + 1008) = v44;
  v46 = *(_QWORD *)(a2 + 1016);
  *(_DWORD *)(a1 + 1024) = v45;
  *(_QWORD *)(a1 + 1016) = v46;
  ++*(_QWORD *)(a2 + 1000);
  *(_QWORD *)(a1 + 968) = 1;
  *(_QWORD *)(a1 + 1000) = 1;
  *(_QWORD *)(a2 + 1008) = 0;
  *(_QWORD *)(a2 + 1016) = 0;
  *(_DWORD *)(a2 + 1024) = 0;
  sub_C65750((_QWORD *)(a1 + 1032), (__int64 *)(a2 + 1032));
  sub_C65750((_QWORD *)(a1 + 1048), (__int64 *)(a2 + 1048));
  v49 = (void *)(a1 + 1096);
  *(_QWORD *)(a1 + 1064) = *(_QWORD *)(a2 + 1064);
  v50 = *(_QWORD *)(a2 + 1072);
  *(_QWORD *)(a1 + 1080) = a1 + 1096;
  *(_QWORD *)(a1 + 1072) = v50;
  *(_QWORD *)(a1 + 1088) = 0x400000000LL;
  v51 = *(_DWORD *)(a2 + 1088);
  if ( v51 )
  {
    v47 = a1 + 1080;
    if ( a1 + 1080 != a2 + 1080 )
    {
      v61 = *(_QWORD *)(a2 + 1080);
      v62 = (const void *)(a2 + 1096);
      if ( v61 == a2 + 1096 )
      {
        v63 = 8LL * v51;
        if ( v51 <= 4
          || (sub_C8D5F0(a1 + 1080, (const void *)(a1 + 1096), v51, 8u, v47, v51),
              v49 = *(void **)(a1 + 1080),
              v62 = *(const void **)(a2 + 1080),
              (v63 = 8LL * *(unsigned int *)(a2 + 1088)) != 0) )
        {
          memcpy(v49, v62, v63);
        }
        *(_DWORD *)(a1 + 1088) = v51;
        *(_DWORD *)(a2 + 1088) = 0;
      }
      else
      {
        *(_QWORD *)(a1 + 1080) = v61;
        v67 = *(_DWORD *)(a2 + 1092);
        *(_DWORD *)(a1 + 1088) = v51;
        *(_DWORD *)(a1 + 1092) = v67;
        *(_QWORD *)(a2 + 1080) = v62;
        *(_QWORD *)(a2 + 1088) = 0;
      }
    }
  }
  *(_QWORD *)(a1 + 1136) = 0;
  *(_QWORD *)(a1 + 1128) = a1 + 1144;
  v52 = *(_DWORD *)(a2 + 1136);
  if ( v52 && a1 + 1128 != a2 + 1128 )
  {
    v64 = *(_QWORD *)(a2 + 1128);
    if ( v64 == a2 + 1144 )
    {
      sub_C8D5F0(a1 + 1128, (const void *)(a1 + 1144), v52, 0x10u, v47, v48);
      v65 = 16LL * *(unsigned int *)(a2 + 1136);
      if ( v65 )
        memcpy(*(void **)(a1 + 1128), *(const void **)(a2 + 1128), v65);
      *(_DWORD *)(a1 + 1136) = v52;
    }
    else
    {
      *(_QWORD *)(a1 + 1128) = v64;
      v66 = *(_DWORD *)(a2 + 1140);
      *(_DWORD *)(a1 + 1136) = v52;
      *(_DWORD *)(a1 + 1140) = v66;
      *(_QWORD *)(a2 + 1128) = a2 + 1144;
      *(_DWORD *)(a2 + 1140) = 0;
    }
  }
  v53 = *(_QWORD *)(a2 + 1144);
  *(_QWORD *)(a2 + 1072) = 0;
  *(_QWORD *)(a2 + 1064) = 0;
  *(_QWORD *)(a1 + 1144) = v53;
  v54 = *(_QWORD *)(a2 + 1152);
  *(_QWORD *)(a2 + 1144) = 0;
  *(_DWORD *)(a2 + 1088) = 0;
  *(_DWORD *)(a2 + 1136) = 0;
  *(_QWORD *)(a1 + 1152) = v54;
  *(_QWORD *)(a1 + 1168) = 0;
  *(_QWORD *)(a1 + 1176) = 0;
  *(_DWORD *)(a1 + 1184) = 0;
  v55 = *(_QWORD *)(a2 + 1168);
  LODWORD(v54) = *(_DWORD *)(a2 + 1184);
  ++*(_QWORD *)(a2 + 1160);
  *(_QWORD *)(a1 + 1168) = v55;
  v56 = *(_QWORD *)(a2 + 1176);
  *(_QWORD *)(a2 + 1168) = 0;
  *(_QWORD *)(a2 + 1176) = 0;
  *(_DWORD *)(a2 + 1184) = 0;
  *(_QWORD *)(a1 + 1176) = v56;
  *(_QWORD *)(a1 + 1200) = 0;
  *(_QWORD *)(a1 + 1208) = 0;
  *(_DWORD *)(a1 + 1216) = 0;
  v57 = *(_QWORD *)(a2 + 1200);
  *(_DWORD *)(a1 + 1184) = v54;
  LODWORD(v54) = *(_DWORD *)(a2 + 1216);
  ++*(_QWORD *)(a2 + 1192);
  *(_QWORD *)(a1 + 1160) = 1;
  *(_QWORD *)(a1 + 1192) = 1;
  *(_QWORD *)(a1 + 1200) = v57;
  v58 = *(_QWORD *)(a2 + 1208);
  *(_DWORD *)(a1 + 1216) = v54;
  *(_QWORD *)(a2 + 1200) = 0;
  *(_QWORD *)(a2 + 1208) = 0;
  *(_DWORD *)(a2 + 1216) = 0;
  *(_QWORD *)(a1 + 1232) = a1 + 1256;
  *(_QWORD *)(a1 + 1224) = 0;
  *(_QWORD *)(a1 + 1240) = 16;
  *(_DWORD *)(a1 + 1248) = 0;
  *(_BYTE *)(a1 + 1252) = 1;
  *(_QWORD *)(a1 + 1384) = 0;
  *(_QWORD *)(a1 + 1392) = a1 + 1416;
  *(_QWORD *)(a1 + 1400) = 16;
  *(_DWORD *)(a1 + 1408) = 0;
  *(_BYTE *)(a1 + 1412) = 1;
  v59 = *(_QWORD *)(a2 + 1544);
  *(_QWORD *)(a1 + 1208) = v58;
  *(_QWORD *)(a1 + 1544) = v59;
  result = (unsigned __int8)qword_4F88508;
  *(_QWORD *)(a1 + 1552) = 0;
  *(_BYTE *)(a1 + 1560) = result;
  *(_QWORD *)(a1 + 1564) = 0;
  *(_BYTE *)(a1 + 1572) = 0;
  *(_QWORD *)(a2 + 1544) = 0;
  return result;
}
