// Function: sub_321E420
// Address: 0x321e420
//
__int64 __fastcall sub_321E420(__int64 a1, __int64 a2)
{
  __int64 v4; // r13
  __int64 v5; // rdi
  __int64 (*v6)(void); // rdx
  char v7; // al
  _QWORD *v8; // rax
  unsigned __int64 v9; // rdx
  unsigned int v10; // ecx
  __int64 v11; // rcx
  _DWORD *v12; // r14
  int v13; // eax
  __int64 v14; // r8
  bool v15; // di
  char v16; // si
  int v17; // edx
  bool v18; // al
  __int64 v19; // rax
  unsigned int v20; // r13d
  bool v21; // r15
  int v22; // eax
  bool v23; // al
  __int64 v24; // rsi
  int v25; // edx
  int v26; // ecx
  char v27; // r12
  char v28; // al
  __int64 v29; // rax
  int v30; // eax
  int v31; // edx
  char v32; // al
  __int64 result; // rax
  bool v34; // al
  char v35; // al
  int v36; // eax
  __int64 v37; // rax
  __int64 v38; // rax
  unsigned int v39; // eax
  __int64 v40; // [rsp+8h] [rbp-38h]
  __int64 v41; // [rsp+8h] [rbp-38h]

  v4 = a1 + 528;
  sub_3211CB0(a1, a2);
  *(_QWORD *)(a1 + 528) = 0;
  *(_QWORD *)(a1 + 536) = 0;
  *(_QWORD *)(a1 + 600) = 0;
  *(_QWORD *)a1 = &unk_4A35BF8;
  *(_QWORD *)(a1 + 544) = a1 + 560;
  *(_QWORD *)(a1 + 552) = 0x400000000LL;
  *(_QWORD *)(a1 + 592) = a1 + 608;
  *(_QWORD *)(a1 + 656) = a1 + 672;
  *(_QWORD *)(a1 + 760) = a1 + 776;
  *(_QWORD *)(a1 + 608) = 0;
  *(_QWORD *)(a1 + 616) = 1;
  *(_QWORD *)(a1 + 624) = 0;
  *(_QWORD *)(a1 + 664) = 0;
  *(_QWORD *)(a1 + 672) = 0;
  *(_QWORD *)(a1 + 704) = 0;
  *(_QWORD *)(a1 + 712) = 0;
  *(_QWORD *)(a1 + 720) = 0;
  *(_QWORD *)(a1 + 728) = 0;
  *(_QWORD *)(a1 + 768) = 0x4000000000LL;
  v5 = *(_QWORD *)(a2 + 224);
  *(_QWORD *)(a1 + 632) = 0;
  *(_QWORD *)(a1 + 640) = 0;
  *(_DWORD *)(a1 + 648) = 0;
  *(_QWORD *)(a1 + 680) = 0;
  *(_QWORD *)(a1 + 688) = 0;
  *(_DWORD *)(a1 + 696) = 0;
  *(_QWORD *)(a1 + 736) = 0;
  *(_QWORD *)(a1 + 744) = 0;
  *(_DWORD *)(a1 + 752) = 0;
  v6 = *(__int64 (**)(void))(*(_QWORD *)v5 + 96LL);
  v7 = 0;
  if ( v6 != sub_C13EE0 )
    v7 = v6();
  *(_BYTE *)(a1 + 2784) = v7;
  *(_QWORD *)(a1 + 1296) = 0x400000000LL;
  *(_QWORD *)(a1 + 1288) = a1 + 1304;
  *(_QWORD *)(a1 + 1440) = 0x2000000000LL;
  *(_QWORD *)(a1 + 2480) = 0;
  *(_QWORD *)(a1 + 2488) = 256;
  *(_QWORD *)(a1 + 2752) = 0;
  *(_QWORD *)(a1 + 2760) = 0;
  *(_QWORD *)(a1 + 2768) = 0;
  *(_QWORD *)(a1 + 2776) = 0;
  *(_QWORD *)(a1 + 2792) = 0;
  *(_QWORD *)(a1 + 2800) = 0;
  *(_QWORD *)(a1 + 2808) = 0;
  *(_DWORD *)(a1 + 2816) = 0;
  *(_QWORD *)(a1 + 2968) = 0;
  *(_QWORD *)(a1 + 2976) = 0;
  *(_QWORD *)(a1 + 2984) = 0;
  *(_DWORD *)(a1 + 2992) = 0;
  *(_QWORD *)(a1 + 3000) = 0;
  *(_QWORD *)(a1 + 3008) = 1;
  *(_QWORD *)(a1 + 1432) = a1 + 1448;
  *(_QWORD *)(a1 + 2824) = a1 + 2840;
  *(_QWORD *)(a1 + 2472) = a1 + 2496;
  *(_QWORD *)(a1 + 2832) = 0x1000000000LL;
  v8 = (_QWORD *)(a1 + 3016);
  do
  {
    if ( v8 )
      *v8 = -4096;
    ++v8;
  }
  while ( v8 != (_QWORD *)(a1 + 3048) );
  *(_QWORD *)(a1 + 3048) = 0;
  *(_QWORD *)(a1 + 3056) = 0;
  *(_QWORD *)(a1 + 3064) = 0;
  *(_QWORD *)(a1 + 3072) = 0;
  sub_3244BC0(a1 + 3080, a2, "info_string", 11, v4);
  *(_QWORD *)(a1 + 3576) = 0;
  *(_QWORD *)(a1 + 3640) = a1 + 3656;
  *(_QWORD *)(a1 + 3648) = 0x100000000LL;
  *(_QWORD *)(a1 + 3704) = a1 + 3728;
  *(_QWORD *)(a1 + 3584) = 0;
  *(_QWORD *)(a1 + 3592) = 0;
  *(_DWORD *)(a1 + 3600) = 0;
  *(_QWORD *)(a1 + 3608) = 0;
  *(_QWORD *)(a1 + 3616) = 0;
  *(_QWORD *)(a1 + 3624) = 0;
  *(_DWORD *)(a1 + 3632) = 0;
  *(_DWORD *)(a1 + 3680) = 0;
  *(_DWORD *)(a1 + 3687) = 256;
  *(_QWORD *)(a1 + 3696) = 0;
  *(_QWORD *)(a1 + 3712) = 4;
  *(_DWORD *)(a1 + 3720) = 0;
  *(_BYTE *)(a1 + 3724) = 1;
  *(_DWORD *)(a1 + 3760) = 1;
  sub_3244BC0(a1 + 3776, a2, "skel_string", 11, v4);
  *(_QWORD *)(a1 + 4280) = a1 + 4296;
  *(_QWORD *)(a1 + 4288) = 0x300000000LL;
  *(_QWORD *)(a1 + 4400) = 0x300000000LL;
  *(_QWORD *)(a1 + 4664) = 0x1000000000LL;
  *(_QWORD *)(a1 + 4672) = a1 + 4688;
  *(_QWORD *)(a1 + 4392) = a1 + 4408;
  LOBYTE(v9) = 0;
  *(_QWORD *)(a1 + 4272) = 0;
  *(_QWORD *)(a1 + 4648) = 0;
  *(_QWORD *)(a1 + 4656) = 0;
  *(_QWORD *)(a1 + 4680) = 0;
  *(_BYTE *)(a1 + 4688) = 0;
  *(_QWORD *)(a1 + 4704) = a1 + 4720;
  *(_QWORD *)(a1 + 4712) = 0;
  *(_BYTE *)(a1 + 4720) = 0;
  *(_DWORD *)(a1 + 4736) = 0;
  *(_BYTE *)(a1 + 4756) = 0;
  *(_BYTE *)(a1 + 4776) = 0;
  *(_WORD *)(a1 + 4784) = 256;
  *(_BYTE *)(a1 + 4786) = 0;
  *(_BYTE *)(a1 + 4792) = 0;
  v10 = *(_DWORD *)(*(_QWORD *)(a2 + 200) + 556LL);
  if ( v10 <= 0x1F )
    v9 = (0xD8000222uLL >> v10) & 1;
  *(_BYTE *)(a1 + 4801) = v9;
  *(_QWORD *)(a1 + 4952) = a1 + 4968;
  *(_QWORD *)(a1 + 4904) = a1 + 4920;
  *(_QWORD *)(a1 + 5016) = a1 + 5032;
  *(_QWORD *)(a1 + 4912) = 0x400000000LL;
  *(_QWORD *)(a1 + 5032) = sub_3217EB0;
  *(_QWORD *)(a1 + 4808) = 0;
  *(_QWORD *)(a1 + 4816) = 0;
  *(_QWORD *)(a1 + 4824) = 0;
  *(_DWORD *)(a1 + 4832) = 0;
  *(_QWORD *)(a1 + 4840) = 0;
  *(_QWORD *)(a1 + 4848) = 0;
  *(_QWORD *)(a1 + 4856) = 0;
  *(_DWORD *)(a1 + 4864) = 0;
  *(_BYTE *)(a1 + 4872) = 0;
  *(_QWORD *)(a1 + 4880) = 0;
  *(_QWORD *)(a1 + 4888) = 0;
  *(_QWORD *)(a1 + 4896) = 0;
  *(_QWORD *)(a1 + 4960) = 0;
  *(_QWORD *)(a1 + 4968) = 0;
  *(_QWORD *)(a1 + 4976) = 1;
  *(_QWORD *)(a1 + 4984) = 0;
  *(_QWORD *)(a1 + 4992) = 0;
  *(_QWORD *)(a1 + 5000) = 0;
  *(_DWORD *)(a1 + 5008) = 0;
  *(_QWORD *)(a1 + 5024) = 0;
  *(_QWORD *)(a1 + 5040) = 0;
  *(_QWORD *)(a1 + 5048) = 0;
  *(_QWORD *)(a1 + 5056) = 0;
  *(_QWORD *)(a1 + 5064) = 0;
  *(_QWORD *)(a1 + 5072) = 0;
  *(_QWORD *)(a1 + 5080) = 0;
  *(_QWORD *)(a1 + 5096) = a1 + 5112;
  *(_QWORD *)(a1 + 5104) = 0x100000000LL;
  *(_QWORD *)(a1 + 5352) = 0x100000000LL;
  *(_QWORD *)(a1 + 5152) = a1 + 5168;
  *(_QWORD *)(a1 + 5280) = sub_3217EB0;
  *(_QWORD *)(a1 + 5384) = a1 + 4888;
  *(_QWORD *)(a1 + 5200) = a1 + 5216;
  *(_QWORD *)(a1 + 5344) = a1 + 5360;
  *(_QWORD *)(a1 + 5408) = a1 + 5424;
  *(_QWORD *)(a1 + 5160) = 0x400000000LL;
  *(_QWORD *)(a1 + 5416) = 0x400000000LL;
  *(_QWORD *)(a1 + 5088) = 0;
  *(_QWORD *)(a1 + 5136) = 0;
  *(_QWORD *)(a1 + 5144) = 0;
  *(_QWORD *)(a1 + 5208) = 0;
  *(_QWORD *)(a1 + 5216) = 0;
  *(_QWORD *)(a1 + 5224) = 1;
  *(_QWORD *)(a1 + 5232) = 0;
  *(_QWORD *)(a1 + 5240) = 0;
  *(_QWORD *)(a1 + 5248) = 0;
  *(_DWORD *)(a1 + 5256) = 0;
  *(_QWORD *)(a1 + 5264) = a1 + 5280;
  *(_QWORD *)(a1 + 5272) = 0;
  *(_QWORD *)(a1 + 5288) = 0;
  *(_QWORD *)(a1 + 5296) = 0;
  *(_QWORD *)(a1 + 5304) = 0;
  *(_QWORD *)(a1 + 5312) = 0;
  *(_QWORD *)(a1 + 5320) = 0;
  *(_QWORD *)(a1 + 5328) = 0;
  *(_QWORD *)(a1 + 5336) = 0;
  *(_QWORD *)(a1 + 5392) = 0;
  *(_QWORD *)(a1 + 5400) = 0;
  *(_QWORD *)(a1 + 5456) = a1 + 5472;
  *(_QWORD *)(a1 + 5616) = a1 + 5632;
  *(_QWORD *)(a1 + 5520) = a1 + 5536;
  *(_QWORD *)(a1 + 5664) = a1 + 5680;
  *(_QWORD *)(a1 + 5728) = a1 + 5744;
  *(_QWORD *)(a1 + 5536) = sub_3217840;
  *(_QWORD *)(a1 + 5624) = 0x400000000LL;
  *(_QWORD *)(a1 + 5464) = 0;
  *(_QWORD *)(a1 + 5472) = 0;
  *(_QWORD *)(a1 + 5480) = 1;
  *(_QWORD *)(a1 + 5488) = 0;
  *(_QWORD *)(a1 + 5496) = 0;
  *(_QWORD *)(a1 + 5504) = 0;
  *(_DWORD *)(a1 + 5512) = 0;
  *(_QWORD *)(a1 + 5528) = 0;
  *(_QWORD *)(a1 + 5544) = 0;
  *(_QWORD *)(a1 + 5552) = 0;
  *(_QWORD *)(a1 + 5560) = 0;
  *(_QWORD *)(a1 + 5568) = 0;
  *(_QWORD *)(a1 + 5576) = 0;
  *(_QWORD *)(a1 + 5584) = 0;
  *(_QWORD *)(a1 + 5592) = 0;
  *(_QWORD *)(a1 + 5600) = 0;
  *(_QWORD *)(a1 + 5608) = 0;
  *(_QWORD *)(a1 + 5672) = 0;
  *(_QWORD *)(a1 + 5680) = 0;
  *(_QWORD *)(a1 + 5688) = 1;
  *(_QWORD *)(a1 + 5696) = 0;
  *(_QWORD *)(a1 + 5704) = 0;
  *(_QWORD *)(a1 + 5712) = 0;
  *(_DWORD *)(a1 + 5720) = 0;
  *(_QWORD *)(a1 + 5736) = 0;
  *(_QWORD *)(a1 + 5744) = sub_3217840;
  *(_QWORD *)(a1 + 5824) = a1 + 5840;
  *(_QWORD *)(a1 + 5872) = a1 + 5888;
  *(_QWORD *)(a1 + 5936) = a1 + 5952;
  *(_QWORD *)(a1 + 5832) = 0x400000000LL;
  *(_QWORD *)(a1 + 5952) = sub_3217840;
  *(_QWORD *)(a1 + 5752) = 0;
  *(_QWORD *)(a1 + 5760) = 0;
  *(_QWORD *)(a1 + 5768) = 0;
  *(_QWORD *)(a1 + 5776) = 0;
  *(_QWORD *)(a1 + 5784) = 0;
  *(_QWORD *)(a1 + 5792) = 0;
  *(_QWORD *)(a1 + 5800) = 0;
  *(_QWORD *)(a1 + 5808) = 0;
  *(_QWORD *)(a1 + 5816) = 0;
  *(_QWORD *)(a1 + 5880) = 0;
  *(_QWORD *)(a1 + 5888) = 0;
  *(_QWORD *)(a1 + 5896) = 1;
  *(_QWORD *)(a1 + 5904) = 0;
  *(_QWORD *)(a1 + 5912) = 0;
  *(_QWORD *)(a1 + 5920) = 0;
  *(_DWORD *)(a1 + 5928) = 0;
  *(_QWORD *)(a1 + 5944) = 0;
  *(_QWORD *)(a1 + 5960) = 0;
  *(_QWORD *)(a1 + 5968) = 0;
  *(_QWORD *)(a1 + 5976) = 0;
  *(_QWORD *)(a1 + 5984) = 0;
  *(_QWORD *)(a1 + 5992) = 0;
  *(_QWORD *)(a1 + 6000) = 0;
  *(_QWORD *)(a1 + 6008) = 0;
  *(_QWORD *)(a1 + 6016) = 0;
  *(_QWORD *)(a1 + 6024) = 0;
  *(_QWORD *)(a1 + 6032) = a1 + 6048;
  *(_QWORD *)(a1 + 6040) = 0x400000000LL;
  *(_QWORD *)(a1 + 6080) = a1 + 6096;
  *(_QWORD *)(a1 + 6160) = sub_3217840;
  v11 = *(_QWORD *)(a1 + 8);
  *(_QWORD *)(a1 + 6144) = a1 + 6160;
  *(_QWORD *)(a1 + 6088) = 0;
  *(_QWORD *)(a1 + 6096) = 0;
  *(_QWORD *)(a1 + 6104) = 1;
  *(_QWORD *)(a1 + 6112) = 0;
  *(_QWORD *)(a1 + 6120) = 0;
  *(_QWORD *)(a1 + 6128) = 0;
  *(_DWORD *)(a1 + 6136) = 0;
  *(_QWORD *)(a1 + 6152) = 0;
  *(_QWORD *)(a1 + 6168) = 0;
  *(_QWORD *)(a1 + 6176) = 0;
  *(_QWORD *)(a1 + 6184) = 0;
  *(_QWORD *)(a1 + 6192) = 0;
  *(_QWORD *)(a1 + 6200) = 0;
  *(_QWORD *)(a1 + 6208) = 0;
  *(_QWORD *)(a1 + 6216) = 0;
  *(_DWORD *)(a1 + 6224) = 0;
  *(_QWORD *)(a1 + 6232) = 0;
  *(_QWORD *)(a1 + 6240) = 0;
  *(_QWORD *)(a1 + 6248) = 0;
  *(_DWORD *)(a1 + 6256) = 0;
  *(_QWORD *)(a1 + 6264) = 0;
  *(_QWORD *)(a1 + 6272) = a1 + 6296;
  *(_QWORD *)(a1 + 6280) = 4;
  *(_DWORD *)(a1 + 6288) = 0;
  *(_BYTE *)(a1 + 6292) = 1;
  v12 = *(_DWORD **)(v11 + 200);
  v13 = v12[241];
  v14 = (__int64)(v12 + 128);
  if ( v13 )
  {
    *(_DWORD *)(a1 + 6224) = v13;
    LOBYTE(v9) = v13 == 2;
  }
  else if ( (_BYTE)v9 )
  {
    *(_DWORD *)(a1 + 6224) = 2;
    v13 = 2;
  }
  else
  {
    v36 = v12[139];
    if ( v12[136] == 39 && v12[138] == 3 && (unsigned int)(v36 - 24) <= 1 )
    {
      *(_DWORD *)(a1 + 6224) = 3;
      v13 = 3;
    }
    else if ( v36 == 19 )
    {
      *(_DWORD *)(a1 + 6224) = 4;
      v13 = 4;
    }
    else
    {
      *(_DWORD *)(a1 + 6224) = 1;
      v13 = 1;
    }
  }
  if ( dword_5037068 )
  {
    *(_BYTE *)(a1 + 3687) = dword_5037068 == 1;
  }
  else
  {
    v15 = 1;
    if ( (unsigned int)(v12[136] - 42) > 1 )
      v15 = v13 == 4;
    *(_BYTE *)(a1 + 3687) = v15;
  }
  v16 = byte_5037948;
  if ( !byte_5037948 )
    v16 = v13 == 3;
  *(_BYTE *)(a1 + 3690) = v16;
  *(_BYTE *)(a1 + 3768) = v9;
  v17 = dword_5036788;
  *(_BYTE *)(a1 + 3769) = *(_QWORD *)(*(_QWORD *)(v11 + 200) + 1080LL) != 0;
  if ( v17 )
    v18 = v17 == 1;
  else
    v18 = v13 != 3;
  *(_BYTE *)(a1 + 3686) = v18;
  v19 = *(_QWORD *)(v11 + 200);
  v20 = *(_DWORD *)(v19 + 996);
  if ( v20 )
  {
    if ( (unsigned int)(v12[136] - 42) <= 1 )
    {
      v20 = 2;
      v21 = 0;
      goto LABEL_22;
    }
    goto LABEL_50;
  }
  v39 = (unsigned int)sub_BAA300(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 2488LL));
  v14 = (__int64)(v12 + 128);
  v20 = v39;
  if ( (unsigned int)(v12[136] - 42) <= 1 )
  {
    v20 = 2;
    v21 = 0;
    v19 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 200LL);
    goto LABEL_22;
  }
  if ( v39 )
  {
LABEL_50:
    if ( v20 > 2 )
      goto LABEL_51;
    v21 = 0;
    v19 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 200LL);
LABEL_22:
    if ( (*(_BYTE *)(v19 + 977) & 0x20) != 0 )
      goto LABEL_23;
    goto LABEL_52;
  }
  v20 = 4;
LABEL_51:
  v40 = v14;
  v34 = sub_CC7F40(v14);
  v14 = v40;
  v21 = v34;
  if ( (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 200LL) + 977LL) & 0x20) != 0 )
    goto LABEL_23;
LABEL_52:
  v41 = v14;
  v35 = sub_BAA330(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 2488LL));
  v14 = v41;
  if ( !v35 )
  {
    if ( v12[141] != 8 )
      goto LABEL_25;
LABEL_54:
    if ( v21 )
      goto LABEL_26;
    goto LABEL_25;
  }
LABEL_23:
  v22 = v12[141];
  if ( v22 == 3 || v22 == 8 )
    goto LABEL_54;
LABEL_25:
  v21 = sub_CC7F40(v14);
  if ( v21 )
  {
    if ( v12[141] == 8 )
      sub_C64ED0("XCOFF requires DWARF64 for 64-bit mode!", 1u);
    v21 = 0;
  }
LABEL_26:
  v23 = 0;
  if ( !byte_5036F88 )
    v23 = (unsigned int)(v12[136] - 42) > 1;
  *(_BYTE *)(a1 + 3688) = v23;
  if ( dword_5036D28 )
    *(_BYTE *)(a1 + 3689) = dword_5036D28 == 1;
  else
    *(_BYTE *)(a1 + 3689) = (unsigned int)(v12[136] - 42) <= 1;
  v24 = *(_QWORD *)(a2 + 200);
  v25 = dword_50372C8;
  v26 = *(_DWORD *)(a1 + 6224);
  if ( (*(_DWORD *)(v24 + 564) & 0xFFFFFFFB) == 3 && (v27 = byte_5037868) != 0 )
  {
    *(_BYTE *)(a1 + 3691) = 1;
    if ( v25 )
      goto LABEL_33;
    if ( v20 > 4 )
    {
      *(_DWORD *)(a1 + 3764) = 2 * (*(_DWORD *)(v24 + 564) == 3) + 1;
      if ( v26 != 1 )
      {
        *(_BYTE *)(a1 + 3770) = 1;
        *(_WORD *)(a1 + 3684) = 0;
        *(_BYTE *)(a1 + 3771) = sub_35DDE70(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 200LL) + 856LL);
        goto LABEL_38;
      }
      goto LABEL_86;
    }
  }
  else
  {
    *(_BYTE *)(a1 + 3691) = 0;
    if ( v25 )
    {
LABEL_33:
      *(_DWORD *)(a1 + 3764) = v25;
      if ( v26 == 1 )
      {
        v28 = 1;
        goto LABEL_70;
      }
LABEL_34:
      v28 = 0;
      if ( v20 <= 2 )
      {
        v29 = *(_QWORD *)(a1 + 8);
        *(_BYTE *)(a1 + 3770) = 0;
        *(_WORD *)(a1 + 3684) = 257;
        *(_BYTE *)(a1 + 3771) = sub_35DDE70(*(_QWORD *)(v29 + 200) + 856LL);
        goto LABEL_36;
      }
LABEL_70:
      *(_BYTE *)(a1 + 3684) = v28;
      v38 = *(_QWORD *)(a1 + 8);
      v27 = 1;
      *(_BYTE *)(a1 + 3685) = v20 <= 3;
      *(_BYTE *)(a1 + 3770) = v20 > 4;
      *(_BYTE *)(a1 + 3771) = sub_35DDE70(*(_QWORD *)(v38 + 200) + 856LL);
      if ( v20 > 4 )
        goto LABEL_38;
      goto LABEL_36;
    }
    if ( v20 > 4 )
    {
      *(_DWORD *)(a1 + 3764) = 3;
      if ( v26 != 1 )
      {
        *(_BYTE *)(a1 + 3684) = 0;
        *(_BYTE *)(a1 + 3770) = 1;
        *(_BYTE *)(a1 + 3685) = v20 <= 3;
        goto LABEL_87;
      }
LABEL_86:
      *(_BYTE *)(a1 + 3684) = 1;
      *(_BYTE *)(a1 + 3685) = v20 <= 3;
      *(_BYTE *)(a1 + 3770) = v20 > 4;
LABEL_87:
      v27 = 1;
      *(_BYTE *)(a1 + 3771) = sub_35DDE70(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 200LL) + 856LL);
      goto LABEL_38;
    }
    if ( v26 == 2 )
    {
      if ( *(_DWORD *)(v24 + 564) != 5 )
        v26 = 3;
      *(_DWORD *)(a1 + 3764) = v26;
      goto LABEL_34;
    }
  }
  *(_DWORD *)(a1 + 3764) = 1;
  if ( v26 != 1 )
    goto LABEL_34;
  v37 = *(_QWORD *)(a1 + 8);
  *(_BYTE *)(a1 + 3684) = 1;
  *(_BYTE *)(a1 + 3685) = v20 <= 3;
  *(_BYTE *)(a1 + 3770) = v20 > 4;
  *(_BYTE *)(a1 + 3771) = sub_35DDE70(*(_QWORD *)(v37 + 200) + 856LL);
LABEL_36:
  v27 = byte_5036C48;
  if ( byte_5036C48 )
    v27 = *(_BYTE *)(a1 + 3769) ^ 1;
LABEL_38:
  v30 = dword_50369E8;
  *(_BYTE *)(a1 + 3692) = v27;
  if ( v30 )
  {
    *(_BYTE *)(a1 + 3693) = v30 == 1;
  }
  else
  {
    v31 = *(_DWORD *)(a1 + 6224);
    if ( v31 == 1 )
    {
      v32 = *(_BYTE *)(a1 + 3769) ^ 1;
    }
    else
    {
      v32 = 1;
      if ( v31 == 2 )
        v32 = v12[141] == 5;
    }
    *(_BYTE *)(a1 + 3693) = v32;
  }
  if ( v20 > 4 )
    *(_DWORD *)(a1 + 3760) = dword_5036528;
  *(_WORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 224LL) + 8LL) + 1904LL) = v20;
  result = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 224LL) + 8LL);
  *(_BYTE *)(result + 1906) = v21;
  return result;
}
