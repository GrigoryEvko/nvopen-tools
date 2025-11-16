// Function: sub_C579F0
// Address: 0xc579f0
//
unsigned __int64 __fastcall sub_C579F0(__int64 a1)
{
  __int64 *v1; // r12
  int v2; // edx
  __int64 *v3; // r8
  __int64 v4; // rax
  const char *v5; // rax
  int v6; // edx
  __int64 *v7; // rbx
  __int64 v8; // rax
  const char *v9; // rax
  int v10; // edx
  __int64 *v11; // r15
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  bool v17; // zf
  __int64 v18; // rax
  int v19; // edx
  __int64 *v20; // rax
  __int64 v21; // rdx
  const char **v22; // rsi
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // rax
  const char *v28; // rax
  int v29; // edx
  __int64 *v30; // rbx
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rcx
  __int64 v34; // r8
  __int64 v35; // r9
  __int64 v36; // rax
  const char *v37; // rax
  int v38; // edx
  __int64 *v39; // rbx
  __int64 v40; // rax
  char v41; // al
  const char *v42; // rax
  int v43; // edx
  __int64 *v44; // rax
  __int64 v45; // rdx
  char v46; // al
  int v47; // edx
  __int64 *v48; // rbx
  __int64 v49; // rax
  __int64 v50; // rdx
  __int64 v51; // rcx
  __int64 v52; // r8
  __int64 v53; // r9
  __int64 v54; // rax
  __int64 *v56; // [rsp+10h] [rbp-B0h]
  __int64 *v57; // [rsp+10h] [rbp-B0h]
  __int64 *v58; // [rsp+18h] [rbp-A8h]
  int v59; // [rsp+28h] [rbp-98h] BYREF
  int v60; // [rsp+2Ch] [rbp-94h] BYREF
  int v61; // [rsp+30h] [rbp-90h] BYREF
  int v62; // [rsp+34h] [rbp-8Ch] BYREF
  __int64 v63; // [rsp+38h] [rbp-88h] BYREF
  __int64 v64; // [rsp+40h] [rbp-80h] BYREF
  __int64 *v65; // [rsp+48h] [rbp-78h] BYREF
  const char *v66; // [rsp+50h] [rbp-70h] BYREF
  __int64 v67; // [rsp+58h] [rbp-68h]
  const char *v68; // [rsp+60h] [rbp-60h] BYREF
  __int64 v69; // [rsp+68h] [rbp-58h]
  char v70; // [rsp+80h] [rbp-40h]
  char v71; // [rsp+81h] [rbp-3Fh]

  v1 = (__int64 *)(a1 + 96);
  *(_QWORD *)a1 = off_4979A38;
  *(_QWORD *)(a1 + 16) = off_4979A38;
  *(_QWORD *)(a1 + 32) = off_4979A60;
  *(_QWORD *)(a1 + 48) = off_4979A60;
  *(_QWORD *)(a1 + 72) = a1 + 32;
  *(_QWORD *)(a1 + 88) = a1 + 48;
  *(_QWORD *)(a1 + 96) = "Generic Options";
  *(_QWORD *)(a1 + 80) = a1 + 16;
  *(_QWORD *)(a1 + 112) = byte_3F871B3;
  *(_BYTE *)(a1 + 8) = 0;
  *(_BYTE *)(a1 + 24) = 1;
  *(_BYTE *)(a1 + 40) = 0;
  *(_BYTE *)(a1 + 56) = 1;
  *(_QWORD *)(a1 + 64) = a1;
  *(_QWORD *)(a1 + 104) = 15;
  *(_QWORD *)(a1 + 120) = 0;
  sub_C524B0(a1 + 96);
  v63 = a1;
  v66 = "Display list of available options (--help-list-hidden for more)";
  v67 = 63;
  v59 = 1;
  v60 = 3;
  v65 = (__int64 *)(a1 + 96);
  v69 = 0;
  v68 = (const char *)sub_C52D90();
  *(_QWORD *)(a1 + 128) = &unk_49DC150;
  v2 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)(a1 + 140) &= 0x8000u;
  *(_QWORD *)(a1 + 208) = 0x100000000LL;
  *(_DWORD *)(a1 + 136) = v2;
  *(_WORD *)(a1 + 144) = 0;
  *(_QWORD *)(a1 + 152) = 0;
  *(_QWORD *)(a1 + 160) = 0;
  *(_QWORD *)(a1 + 168) = 0;
  *(_QWORD *)(a1 + 176) = 0;
  *(_QWORD *)(a1 + 184) = 0;
  *(_QWORD *)(a1 + 192) = 0;
  *(_QWORD *)(a1 + 200) = a1 + 216;
  *(_QWORD *)(a1 + 224) = 0;
  *(_QWORD *)(a1 + 232) = a1 + 256;
  *(_QWORD *)(a1 + 240) = 1;
  *(_DWORD *)(a1 + 248) = 0;
  *(_BYTE *)(a1 + 252) = 1;
  v3 = sub_C57470();
  v4 = *(unsigned int *)(a1 + 208);
  if ( v4 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 212) )
  {
    v58 = v3;
    sub_C8D5F0(a1 + 200, a1 + 216, v4 + 1, 8);
    v4 = *(unsigned int *)(a1 + 208);
    v3 = v58;
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 200) + 8 * v4) = v3;
  *(_QWORD *)(a1 + 128) = off_49DC420;
  ++*(_DWORD *)(a1 + 208);
  *(_QWORD *)(a1 + 272) = off_49DC400;
  *(_QWORD *)(a1 + 280) = &unk_49DC1D0;
  *(_QWORD *)(a1 + 312) = nullsub_145;
  *(_QWORD *)(a1 + 304) = sub_C4F650;
  *(_QWORD *)(a1 + 264) = 0;
  sub_C57680(a1 + 128, "help-list", (__int64 *)&v66, &v63, &v59, &v60, &v65, (__int64 *)&v68);
  sub_C53130(a1 + 128);
  v64 = a1 + 16;
  v66 = "Display list of all available options";
  v67 = 37;
  v61 = 1;
  v62 = 3;
  v65 = (__int64 *)(a1 + 96);
  v5 = (const char *)sub_C52D90();
  v69 = 0;
  v68 = v5;
  *(_QWORD *)(a1 + 320) = &unk_49DC150;
  v6 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)(a1 + 332) &= 0x8000u;
  *(_QWORD *)(a1 + 400) = 0x100000000LL;
  *(_DWORD *)(a1 + 328) = v6;
  *(_QWORD *)(a1 + 392) = a1 + 408;
  *(_WORD *)(a1 + 336) = 0;
  *(_QWORD *)(a1 + 344) = 0;
  *(_QWORD *)(a1 + 352) = 0;
  *(_QWORD *)(a1 + 360) = 0;
  *(_QWORD *)(a1 + 368) = 0;
  *(_QWORD *)(a1 + 376) = 0;
  *(_QWORD *)(a1 + 384) = 0;
  *(_QWORD *)(a1 + 416) = 0;
  *(_QWORD *)(a1 + 424) = a1 + 448;
  *(_QWORD *)(a1 + 432) = 1;
  *(_DWORD *)(a1 + 440) = 0;
  *(_BYTE *)(a1 + 444) = 1;
  v7 = sub_C57470();
  v8 = *(unsigned int *)(a1 + 400);
  if ( v8 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 404) )
  {
    sub_C8D5F0(a1 + 392, a1 + 408, v8 + 1, 8);
    v8 = *(unsigned int *)(a1 + 400);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 392) + 8 * v8) = v7;
  *(_QWORD *)(a1 + 320) = off_49DC420;
  ++*(_DWORD *)(a1 + 400);
  *(_QWORD *)(a1 + 472) = &unk_49DC1D0;
  *(_QWORD *)(a1 + 504) = nullsub_145;
  *(_QWORD *)(a1 + 464) = off_49DC400;
  *(_QWORD *)(a1 + 496) = sub_C4F650;
  *(_QWORD *)(a1 + 456) = 0;
  sub_C57680(a1 + 320, "help-list-hidden", (__int64 *)&v66, &v64, &v61, &v62, &v65, (__int64 *)&v68);
  sub_C53130(a1 + 320);
  v9 = (const char *)sub_C52D90();
  v67 = 0;
  v66 = v9;
  *(_QWORD *)(a1 + 512) = &unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)(a1 + 524) &= 0x8000u;
  *(_QWORD *)(a1 + 592) = 0x100000000LL;
  *(_DWORD *)(a1 + 520) = v10;
  *(_WORD *)(a1 + 528) = 0;
  *(_QWORD *)(a1 + 584) = a1 + 600;
  *(_QWORD *)(a1 + 536) = 0;
  *(_QWORD *)(a1 + 544) = 0;
  *(_QWORD *)(a1 + 552) = 0;
  *(_QWORD *)(a1 + 560) = 0;
  *(_QWORD *)(a1 + 568) = 0;
  *(_QWORD *)(a1 + 576) = 0;
  *(_QWORD *)(a1 + 608) = 0;
  *(_QWORD *)(a1 + 616) = a1 + 640;
  *(_QWORD *)(a1 + 624) = 1;
  *(_DWORD *)(a1 + 632) = 0;
  *(_BYTE *)(a1 + 636) = 1;
  v11 = sub_C57470();
  v12 = *(unsigned int *)(a1 + 592);
  if ( v12 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 596) )
  {
    sub_C8D5F0(a1 + 584, a1 + 600, v12 + 1, 8);
    v12 = *(unsigned int *)(a1 + 592);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 584) + 8 * v12) = v11;
  *(_QWORD *)(a1 + 656) = off_49DC400;
  *(_QWORD *)(a1 + 512) = off_49DC4A0;
  ++*(_DWORD *)(a1 + 592);
  *(_QWORD *)(a1 + 648) = 0;
  *(_QWORD *)(a1 + 664) = &unk_49DC1D0;
  *(_QWORD *)(a1 + 696) = nullsub_146;
  *(_QWORD *)(a1 + 688) = sub_C4F660;
  sub_C53080(a1 + 512, (__int64)"help", 4);
  v17 = *(_QWORD *)(a1 + 648) == 0;
  *(_QWORD *)(a1 + 560) = 50;
  *(_QWORD *)(a1 + 552) = "Display available options (--help-hidden for more)";
  if ( v17 )
  {
    *(_QWORD *)(a1 + 648) = a1 + 64;
  }
  else
  {
    v18 = sub_CEADF0(a1 + 512, "help", v13, v14, v15, v16);
    v71 = 1;
    v68 = "cl::location(x) specified more than once!";
    v70 = 3;
    sub_C53280(a1 + 512, (__int64)&v68, 0, 0, v18);
  }
  *(_BYTE *)(a1 + 524) |= 0x18u;
  sub_C57500(a1 + 512, v1);
  sub_C50AB0((__int64 *)&v66, a1 + 512);
  sub_C53130(a1 + 512);
  *(_QWORD *)(a1 + 704) = &unk_49DC150;
  v19 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)(a1 + 716) = *(_DWORD *)(a1 + 716) & 0x8000 | 0x20;
  *(_QWORD *)(a1 + 784) = 0x100000000LL;
  *(_DWORD *)(a1 + 712) = v19;
  *(_QWORD *)(a1 + 776) = a1 + 792;
  *(_WORD *)(a1 + 720) = 0;
  *(_QWORD *)(a1 + 728) = 0;
  *(_QWORD *)(a1 + 736) = 0;
  *(_QWORD *)(a1 + 744) = 0;
  *(_QWORD *)(a1 + 752) = 0;
  *(_QWORD *)(a1 + 760) = 0;
  *(_QWORD *)(a1 + 768) = 0;
  *(_QWORD *)(a1 + 800) = 0;
  *(_QWORD *)(a1 + 808) = a1 + 832;
  *(_QWORD *)(a1 + 816) = 1;
  *(_DWORD *)(a1 + 824) = 0;
  *(_BYTE *)(a1 + 828) = 1;
  v20 = sub_C57470();
  v21 = *(unsigned int *)(a1 + 784);
  if ( v21 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 788) )
  {
    v57 = v20;
    sub_C8D5F0(a1 + 776, a1 + 792, v21 + 1, 8);
    v21 = *(unsigned int *)(a1 + 784);
    v20 = v57;
  }
  v22 = (const char **)&unk_3F7DA1E;
  *(_QWORD *)(*(_QWORD *)(a1 + 776) + 8 * v21) = v20;
  ++*(_DWORD *)(a1 + 784);
  *(_QWORD *)(a1 + 840) = 0;
  *(_QWORD *)(a1 + 704) = &unk_49DC380;
  sub_C53080(a1 + 704, (__int64)&unk_3F7DA1E, 1);
  v17 = *(_QWORD *)(a1 + 840) == 0;
  *(_QWORD *)(a1 + 752) = 16;
  *(_QWORD *)(a1 + 744) = "Alias for --help";
  if ( !v17 )
  {
    v27 = sub_CEADF0(a1 + 704, &unk_3F7DA1E, v23, v24, v25, v26);
    v22 = &v68;
    v71 = 1;
    v68 = "cl::alias must only have one cl::aliasopt(...) specified!";
    v70 = 3;
    sub_C53280(a1 + 704, (__int64)&v68, 0, 0, v27);
  }
  *(_BYTE *)(a1 + 717) |= 0x20u;
  *(_QWORD *)(a1 + 840) = a1 + 512;
  sub_C53EE0(a1 + 704, v22, v23, v24, v25, v26);
  v28 = (const char *)sub_C52D90();
  v67 = 0;
  v66 = v28;
  *(_QWORD *)(a1 + 848) = &unk_49DC150;
  v29 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)(a1 + 860) &= 0x8000u;
  *(_QWORD *)(a1 + 928) = 0x100000000LL;
  *(_DWORD *)(a1 + 856) = v29;
  *(_QWORD *)(a1 + 920) = a1 + 936;
  *(_WORD *)(a1 + 864) = 0;
  *(_QWORD *)(a1 + 872) = 0;
  *(_QWORD *)(a1 + 880) = 0;
  *(_QWORD *)(a1 + 888) = 0;
  *(_QWORD *)(a1 + 896) = 0;
  *(_QWORD *)(a1 + 904) = 0;
  *(_QWORD *)(a1 + 912) = 0;
  *(_QWORD *)(a1 + 944) = 0;
  *(_QWORD *)(a1 + 952) = a1 + 976;
  *(_QWORD *)(a1 + 960) = 1;
  *(_DWORD *)(a1 + 968) = 0;
  *(_BYTE *)(a1 + 972) = 1;
  v30 = sub_C57470();
  v31 = *(unsigned int *)(a1 + 928);
  if ( v31 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 932) )
  {
    sub_C8D5F0(a1 + 920, a1 + 936, v31 + 1, 8);
    v31 = *(unsigned int *)(a1 + 928);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 920) + 8 * v31) = v30;
  *(_QWORD *)(a1 + 992) = off_49DC400;
  *(_QWORD *)(a1 + 848) = off_49DC4A0;
  ++*(_DWORD *)(a1 + 928);
  *(_QWORD *)(a1 + 984) = 0;
  *(_QWORD *)(a1 + 1000) = &unk_49DC1D0;
  *(_QWORD *)(a1 + 1032) = nullsub_146;
  *(_QWORD *)(a1 + 1024) = sub_C4F660;
  sub_C53080(a1 + 848, (__int64)"help-hidden", 11);
  v17 = *(_QWORD *)(a1 + 984) == 0;
  *(_QWORD *)(a1 + 896) = 29;
  *(_QWORD *)(a1 + 888) = "Display all available options";
  if ( v17 )
  {
    *(_QWORD *)(a1 + 984) = a1 + 80;
  }
  else
  {
    v36 = sub_CEADF0(a1 + 848, "help-hidden", v32, v33, v34, v35);
    v71 = 1;
    v68 = "cl::location(x) specified more than once!";
    v70 = 3;
    sub_C53280(a1 + 848, (__int64)&v68, 0, 0, v36);
  }
  *(_BYTE *)(a1 + 860) = *(_BYTE *)(a1 + 860) & 0x87 | 0x38;
  sub_C57500(a1 + 848, v1);
  sub_C50AB0((__int64 *)&v66, a1 + 848);
  sub_C53130(a1 + 848);
  v37 = (const char *)sub_C52D90();
  v69 = 0;
  v68 = v37;
  *(_QWORD *)(a1 + 1040) = &unk_49DC150;
  v38 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)(a1 + 1052) &= 0x8000u;
  *(_QWORD *)(a1 + 1120) = 0x100000000LL;
  *(_DWORD *)(a1 + 1048) = v38;
  *(_WORD *)(a1 + 1056) = 0;
  *(_QWORD *)(a1 + 1064) = 0;
  *(_QWORD *)(a1 + 1072) = 0;
  *(_QWORD *)(a1 + 1080) = 0;
  *(_QWORD *)(a1 + 1088) = 0;
  *(_QWORD *)(a1 + 1096) = 0;
  *(_QWORD *)(a1 + 1104) = 0;
  *(_QWORD *)(a1 + 1112) = a1 + 1128;
  *(_QWORD *)(a1 + 1136) = 0;
  *(_QWORD *)(a1 + 1144) = a1 + 1168;
  *(_QWORD *)(a1 + 1152) = 1;
  *(_DWORD *)(a1 + 1160) = 0;
  *(_BYTE *)(a1 + 1164) = 1;
  v39 = sub_C57470();
  v40 = *(unsigned int *)(a1 + 1120);
  if ( v40 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 1124) )
  {
    sub_C8D5F0(a1 + 1112, a1 + 1128, v40 + 1, 8);
    v40 = *(unsigned int *)(a1 + 1120);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 1112) + 8 * v40) = v39;
  ++*(_DWORD *)(a1 + 1120);
  *(_WORD *)(a1 + 1192) = 0;
  *(_QWORD *)(a1 + 1184) = &unk_49D9748;
  *(_QWORD *)(a1 + 1040) = &unk_49DC090;
  *(_BYTE *)(a1 + 1176) = 0;
  *(_QWORD *)(a1 + 1200) = &unk_49DC1D0;
  *(_QWORD *)(a1 + 1232) = nullsub_23;
  *(_QWORD *)(a1 + 1224) = sub_984030;
  sub_C53080(a1 + 1040, (__int64)"print-options", 13);
  *(_QWORD *)(a1 + 1080) = "Print non-default options after command line parsing";
  v41 = *(_BYTE *)(a1 + 1052);
  *(_WORD *)(a1 + 1192) = 256;
  *(_BYTE *)(a1 + 1176) = 0;
  *(_QWORD *)(a1 + 1088) = 52;
  *(_BYTE *)(a1 + 1052) = v41 & 0x9F | 0x20;
  sub_C578B0(a1 + 1040, v1, (__int64 *)&v68);
  sub_C53130(a1 + 1040);
  v42 = (const char *)sub_C52D90();
  v69 = 0;
  v68 = v42;
  *(_QWORD *)(a1 + 1240) = &unk_49DC150;
  v43 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)(a1 + 1252) &= 0x8000u;
  *(_QWORD *)(a1 + 1320) = 0x100000000LL;
  *(_DWORD *)(a1 + 1248) = v43;
  *(_WORD *)(a1 + 1256) = 0;
  *(_QWORD *)(a1 + 1264) = 0;
  *(_QWORD *)(a1 + 1272) = 0;
  *(_QWORD *)(a1 + 1280) = 0;
  *(_QWORD *)(a1 + 1288) = 0;
  *(_QWORD *)(a1 + 1296) = 0;
  *(_QWORD *)(a1 + 1304) = 0;
  *(_QWORD *)(a1 + 1312) = a1 + 1328;
  *(_QWORD *)(a1 + 1336) = 0;
  *(_QWORD *)(a1 + 1344) = a1 + 1368;
  *(_QWORD *)(a1 + 1352) = 1;
  *(_DWORD *)(a1 + 1360) = 0;
  *(_BYTE *)(a1 + 1364) = 1;
  v44 = sub_C57470();
  v45 = *(unsigned int *)(a1 + 1320);
  if ( v45 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 1324) )
  {
    v56 = v44;
    sub_C8D5F0(a1 + 1312, a1 + 1328, v45 + 1, 8);
    v45 = *(unsigned int *)(a1 + 1320);
    v44 = v56;
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 1312) + 8 * v45) = v44;
  *(_WORD *)(a1 + 1392) = 0;
  ++*(_DWORD *)(a1 + 1320);
  *(_BYTE *)(a1 + 1376) = 0;
  *(_QWORD *)(a1 + 1384) = &unk_49D9748;
  *(_QWORD *)(a1 + 1240) = &unk_49DC090;
  *(_QWORD *)(a1 + 1400) = &unk_49DC1D0;
  *(_QWORD *)(a1 + 1432) = nullsub_23;
  *(_QWORD *)(a1 + 1424) = sub_984030;
  sub_C53080(a1 + 1240, (__int64)"print-all-options", 17);
  *(_WORD *)(a1 + 1392) = 256;
  *(_QWORD *)(a1 + 1280) = "Print all option values after command line parsing";
  v46 = *(_BYTE *)(a1 + 1252);
  *(_QWORD *)(a1 + 1288) = 50;
  *(_BYTE *)(a1 + 1376) = 0;
  *(_BYTE *)(a1 + 1252) = v46 & 0x9F | 0x20;
  sub_C578B0(a1 + 1240, v1, (__int64 *)&v68);
  sub_C53130(a1 + 1240);
  *(_QWORD *)(a1 + 1456) = 0;
  *(_QWORD *)(a1 + 1472) = 0;
  *(_QWORD *)(a1 + 1480) = 0;
  *(_QWORD *)(a1 + 1488) = 0;
  *(_QWORD *)(a1 + 1504) = &unk_49DC150;
  v47 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)(a1 + 1516) &= 0x8000u;
  *(_QWORD *)(a1 + 1584) = 0x100000000LL;
  *(_DWORD *)(a1 + 1512) = v47;
  *(_WORD *)(a1 + 1520) = 0;
  *(_QWORD *)(a1 + 1528) = 0;
  *(_QWORD *)(a1 + 1536) = 0;
  *(_QWORD *)(a1 + 1544) = 0;
  *(_QWORD *)(a1 + 1552) = 0;
  *(_QWORD *)(a1 + 1560) = 0;
  *(_QWORD *)(a1 + 1568) = 0;
  *(_QWORD *)(a1 + 1576) = a1 + 1592;
  *(_QWORD *)(a1 + 1600) = 0;
  *(_QWORD *)(a1 + 1608) = a1 + 1632;
  *(_QWORD *)(a1 + 1616) = 1;
  *(_DWORD *)(a1 + 1624) = 0;
  *(_BYTE *)(a1 + 1628) = 1;
  v48 = sub_C57470();
  v49 = *(unsigned int *)(a1 + 1584);
  if ( v49 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 1588) )
  {
    sub_C8D5F0(a1 + 1576, a1 + 1592, v49 + 1, 8);
    v49 = *(unsigned int *)(a1 + 1584);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 1576) + 8 * v49) = v48;
  *(_QWORD *)(a1 + 1648) = off_49DC400;
  *(_QWORD *)(a1 + 1504) = off_49DC520;
  ++*(_DWORD *)(a1 + 1584);
  *(_QWORD *)(a1 + 1640) = 0;
  *(_QWORD *)(a1 + 1656) = &unk_49DC1D0;
  *(_QWORD *)(a1 + 1688) = nullsub_147;
  *(_QWORD *)(a1 + 1680) = sub_C4F670;
  sub_C53080(a1 + 1504, (__int64)"version", 7);
  v17 = *(_QWORD *)(a1 + 1640) == 0;
  *(_QWORD *)(a1 + 1552) = 35;
  *(_QWORD *)(a1 + 1544) = "Display the version of this program";
  if ( v17 )
  {
    *(_QWORD *)(a1 + 1640) = a1 + 1496;
  }
  else
  {
    v54 = sub_CEADF0(a1 + 1504, "version", v50, v51, v52, v53);
    v71 = 1;
    v68 = "cl::location(x) specified more than once!";
    v70 = 3;
    sub_C53280(a1 + 1504, (__int64)&v68, 0, 0, v54);
  }
  *(_BYTE *)(a1 + 1516) |= 0x18u;
  sub_C57500(a1 + 1504, v1);
  return sub_C53130(a1 + 1504);
}
