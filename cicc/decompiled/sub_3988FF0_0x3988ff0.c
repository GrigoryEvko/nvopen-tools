// Function: sub_3988FF0
// Address: 0x3988ff0
//
__int64 __fastcall sub_3988FF0(__int64 a1, __int64 a2)
{
  __int64 v4; // r14
  char v5; // dl
  __int64 v6; // rdi
  __int64 (*v7)(); // rax
  unsigned int v8; // ecx
  unsigned __int64 v9; // rax
  __int64 v10; // rcx
  _DWORD *v11; // r13
  int v12; // edx
  int v13; // eax
  bool v14; // al
  unsigned int v15; // eax
  int v16; // esi
  bool v17; // di
  __int16 v18; // r8
  char v19; // si
  bool v20; // si
  __int64 v21; // r9
  int v22; // esi
  bool v23; // al
  __int64 result; // rax

  v4 = a1 + 416;
  sub_397F8E0(a1, a2);
  v5 = 0;
  *(_QWORD *)(a1 + 416) = 0;
  *(_QWORD *)(a1 + 424) = 0;
  *(_QWORD *)(a1 + 488) = 0;
  *(_QWORD *)a1 = &unk_4A3FBA0;
  *(_QWORD *)(a1 + 432) = a1 + 448;
  *(_QWORD *)(a1 + 440) = 0x400000000LL;
  *(_QWORD *)(a1 + 480) = a1 + 496;
  *(_QWORD *)(a1 + 664) = a1 + 680;
  *(_QWORD *)(a1 + 496) = 0;
  *(_QWORD *)(a1 + 504) = 1;
  *(_QWORD *)(a1 + 520) = 0;
  *(_QWORD *)(a1 + 552) = 0;
  *(_QWORD *)(a1 + 560) = 0;
  *(_QWORD *)(a1 + 568) = 0;
  *(_QWORD *)(a1 + 576) = 0;
  *(_QWORD *)(a1 + 608) = 0;
  *(_QWORD *)(a1 + 616) = 0;
  *(_QWORD *)(a1 + 624) = 0;
  *(_QWORD *)(a1 + 632) = 0;
  *(_QWORD *)(a1 + 672) = 0x4000000000LL;
  v6 = *(_QWORD *)(a2 + 256);
  *(_QWORD *)(a1 + 528) = 0;
  *(_QWORD *)(a1 + 536) = 0;
  *(_DWORD *)(a1 + 544) = 0;
  *(_QWORD *)(a1 + 584) = 0;
  *(_QWORD *)(a1 + 592) = 0;
  *(_DWORD *)(a1 + 600) = 0;
  *(_QWORD *)(a1 + 640) = 0;
  *(_QWORD *)(a1 + 648) = 0;
  *(_DWORD *)(a1 + 656) = 0;
  v7 = *(__int64 (**)())(*(_QWORD *)v6 + 80LL);
  if ( v7 != sub_168DB50 )
    v5 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v7)(v6, a2, 0);
  *(_QWORD *)(a1 + 1192) = a1 + 1208;
  *(_QWORD *)(a1 + 1200) = 0x400000000LL;
  *(_QWORD *)(a1 + 1336) = a1 + 1352;
  *(_QWORD *)(a1 + 1344) = 0x2000000000LL;
  *(_QWORD *)(a1 + 2376) = a1 + 2392;
  *(_QWORD *)(a1 + 2656) = 0x2000000000LL;
  *(_QWORD *)(a1 + 2384) = 0x10000000000LL;
  *(_QWORD *)(a1 + 3704) = a1 + 3736;
  *(_QWORD *)(a1 + 3712) = a1 + 3736;
  *(_QWORD *)(a1 + 2648) = a1 + 2664;
  *(_BYTE *)(a1 + 3688) = v5;
  *(_QWORD *)(a1 + 3864) = a1 + 3880;
  *(_QWORD *)(a1 + 3696) = 0;
  *(_QWORD *)(a1 + 3720) = 16;
  *(_DWORD *)(a1 + 3728) = 0;
  *(_QWORD *)(a1 + 3872) = 0x1000000000LL;
  *(_QWORD *)(a1 + 4008) = 0;
  *(_QWORD *)(a1 + 4024) = 0;
  *(_QWORD *)(a1 + 4032) = 0;
  sub_39A0020(a1 + 4040, a2, "info_string", 11, v4);
  *(_QWORD *)(a1 + 4432) = 0;
  *(_QWORD *)(a1 + 4464) = a1 + 4480;
  *(_QWORD *)(a1 + 4472) = 0x100000000LL;
  *(_QWORD *)(a1 + 4440) = 0;
  *(_QWORD *)(a1 + 4448) = 0;
  *(_DWORD *)(a1 + 4456) = 0;
  *(_BYTE *)(a1 + 4499) = 0;
  *(_DWORD *)(a1 + 4500) = 16777473;
  sub_39A0020(a1 + 4520, a2, "skel_string", 11, v4);
  *(_QWORD *)(a1 + 4912) = 0;
  *(_QWORD *)(a1 + 4920) = a1 + 4936;
  *(_QWORD *)(a1 + 4928) = 0x300000000LL;
  *(_QWORD *)(a1 + 5040) = 0x300000000LL;
  *(_QWORD *)(a1 + 5296) = a1 + 5312;
  *(_QWORD *)(a1 + 5328) = a1 + 5344;
  *(_QWORD *)(a1 + 5032) = a1 + 5048;
  *(_QWORD *)(a1 + 5264) = 0;
  *(_QWORD *)(a1 + 5272) = 0;
  *(_QWORD *)(a1 + 5280) = 0x1000000000LL;
  *(_QWORD *)(a1 + 5304) = 0;
  *(_BYTE *)(a1 + 5312) = 0;
  *(_QWORD *)(a1 + 5336) = 0;
  *(_BYTE *)(a1 + 5344) = 0;
  *(_QWORD *)(a1 + 5368) = 0;
  *(_BYTE *)(a1 + 5392) = 0;
  *(_WORD *)(a1 + 5400) = 256;
  *(_BYTE *)(a1 + 5402) = 0;
  v8 = *(_DWORD *)(*(_QWORD *)(a2 + 232) + 516LL);
  LOBYTE(v9) = 0;
  if ( v8 <= 0x1E )
    v9 = (0x60000888uLL >> v8) & 1;
  *(_BYTE *)(a1 + 5409) = v9;
  *(_QWORD *)(a1 + 5680) = a1 + 5552;
  *(_QWORD *)(a1 + 5568) = a1 + 5584;
  *(_QWORD *)(a1 + 5616) = a1 + 5632;
  *(_QWORD *)(a1 + 5688) = sub_39841F0;
  *(_QWORD *)(a1 + 5576) = 0x400000000LL;
  *(_QWORD *)(a1 + 5672) = 0x3800000000LL;
  *(_QWORD *)(a1 + 5416) = 0;
  *(_QWORD *)(a1 + 5424) = 0;
  *(_QWORD *)(a1 + 5432) = 0;
  *(_DWORD *)(a1 + 5440) = 0;
  *(_QWORD *)(a1 + 5448) = 0;
  *(_QWORD *)(a1 + 5456) = 0;
  *(_QWORD *)(a1 + 5464) = 0;
  *(_DWORD *)(a1 + 5472) = 0;
  *(_QWORD *)(a1 + 5480) = 0;
  *(_QWORD *)(a1 + 5488) = 0;
  *(_QWORD *)(a1 + 5496) = 0;
  *(_DWORD *)(a1 + 5504) = 0;
  *(_QWORD *)(a1 + 5512) = 0;
  *(_QWORD *)(a1 + 5520) = 0;
  *(_QWORD *)(a1 + 5528) = 0;
  *(_DWORD *)(a1 + 5536) = 0;
  *(_BYTE *)(a1 + 5544) = 0;
  *(_QWORD *)(a1 + 5552) = 0;
  *(_QWORD *)(a1 + 5560) = 0;
  *(_QWORD *)(a1 + 5624) = 0;
  *(_QWORD *)(a1 + 5632) = 0;
  *(_QWORD *)(a1 + 5640) = 1;
  *(_QWORD *)(a1 + 5656) = 0;
  *(_QWORD *)(a1 + 5664) = 0;
  *(_QWORD *)(a1 + 5704) = 0;
  *(_QWORD *)(a1 + 5768) = a1 + 5784;
  *(_QWORD *)(a1 + 5816) = a1 + 5832;
  *(_QWORD *)(a1 + 5880) = a1 + 5752;
  *(_QWORD *)(a1 + 5968) = a1 + 5984;
  *(_QWORD *)(a1 + 6016) = a1 + 6032;
  *(_QWORD *)(a1 + 5776) = 0x400000000LL;
  *(_QWORD *)(a1 + 5872) = 0x3800000000LL;
  *(_QWORD *)(a1 + 5976) = 0x400000000LL;
  *(_QWORD *)(a1 + 5712) = 0;
  *(_QWORD *)(a1 + 5720) = 0;
  *(_QWORD *)(a1 + 5728) = 0;
  *(_QWORD *)(a1 + 5736) = 0;
  *(_QWORD *)(a1 + 5744) = 0;
  *(_QWORD *)(a1 + 5752) = 0;
  *(_QWORD *)(a1 + 5760) = 0;
  *(_QWORD *)(a1 + 5824) = 0;
  *(_QWORD *)(a1 + 5832) = 0;
  *(_QWORD *)(a1 + 5840) = 1;
  *(_QWORD *)(a1 + 5856) = 0;
  *(_QWORD *)(a1 + 5864) = 0;
  *(_QWORD *)(a1 + 5888) = sub_3983DD0;
  *(_QWORD *)(a1 + 5904) = 0;
  *(_QWORD *)(a1 + 5912) = 0;
  *(_QWORD *)(a1 + 5920) = 0;
  *(_QWORD *)(a1 + 5928) = 0;
  *(_QWORD *)(a1 + 5936) = 0;
  *(_QWORD *)(a1 + 5944) = 0;
  *(_QWORD *)(a1 + 5952) = 0;
  *(_QWORD *)(a1 + 5960) = 0;
  *(_DWORD *)(a1 + 6024) = 0;
  *(_DWORD *)(a1 + 6028) = 0;
  *(_QWORD *)(a1 + 6032) = 0;
  *(_QWORD *)(a1 + 6080) = a1 + 5952;
  *(_QWORD *)(a1 + 6168) = a1 + 6184;
  *(_QWORD *)(a1 + 6216) = a1 + 6232;
  *(_QWORD *)(a1 + 6072) = 0x3800000000LL;
  *(_QWORD *)(a1 + 6176) = 0x400000000LL;
  *(_QWORD *)(a1 + 6272) = 0x3800000000LL;
  *(_QWORD *)(a1 + 6280) = a1 + 6152;
  *(_QWORD *)(a1 + 6040) = 1;
  *(_QWORD *)(a1 + 6056) = 0;
  *(_QWORD *)(a1 + 6064) = 0;
  *(_QWORD *)(a1 + 6088) = sub_3983DD0;
  *(_QWORD *)(a1 + 6104) = 0;
  *(_QWORD *)(a1 + 6112) = 0;
  *(_QWORD *)(a1 + 6120) = 0;
  *(_QWORD *)(a1 + 6128) = 0;
  *(_QWORD *)(a1 + 6136) = 0;
  *(_QWORD *)(a1 + 6144) = 0;
  *(_QWORD *)(a1 + 6152) = 0;
  *(_QWORD *)(a1 + 6160) = 0;
  *(_QWORD *)(a1 + 6224) = 0;
  *(_QWORD *)(a1 + 6232) = 0;
  *(_QWORD *)(a1 + 6240) = 1;
  *(_QWORD *)(a1 + 6256) = 0;
  *(_QWORD *)(a1 + 6264) = 0;
  *(_QWORD *)(a1 + 6288) = sub_3983DD0;
  *(_QWORD *)(a1 + 6304) = 0;
  *(_QWORD *)(a1 + 6312) = 0;
  *(_QWORD *)(a1 + 6320) = 0;
  *(_QWORD *)(a1 + 6328) = 0;
  *(_QWORD *)(a1 + 6336) = 0;
  *(_QWORD *)(a1 + 6344) = 0;
  *(_QWORD *)(a1 + 6352) = 0;
  *(_QWORD *)(a1 + 6360) = 0;
  *(_QWORD *)(a1 + 6472) = 0x3800000000LL;
  *(_QWORD *)(a1 + 6480) = a1 + 6352;
  *(_QWORD *)(a1 + 6376) = 0x400000000LL;
  *(_QWORD *)(a1 + 6600) = a1 + 6632;
  *(_QWORD *)(a1 + 6608) = a1 + 6632;
  *(_QWORD *)(a1 + 6672) = a1 + 6704;
  *(_QWORD *)(a1 + 6680) = a1 + 6704;
  *(_QWORD *)(a1 + 6368) = a1 + 6384;
  *(_QWORD *)(a1 + 6416) = a1 + 6432;
  *(_QWORD *)(a1 + 6424) = 0;
  *(_QWORD *)(a1 + 6432) = 0;
  *(_QWORD *)(a1 + 6440) = 1;
  *(_QWORD *)(a1 + 6456) = 0;
  *(_QWORD *)(a1 + 6464) = 0;
  *(_QWORD *)(a1 + 6488) = sub_3983DD0;
  *(_QWORD *)(a1 + 6504) = 0;
  *(_QWORD *)(a1 + 6512) = 0;
  *(_QWORD *)(a1 + 6520) = 0;
  *(_QWORD *)(a1 + 6528) = 0;
  *(_QWORD *)(a1 + 6536) = 0;
  *(_QWORD *)(a1 + 6544) = 0;
  *(_QWORD *)(a1 + 6552) = 0;
  *(_QWORD *)(a1 + 6560) = 0;
  *(_QWORD *)(a1 + 6568) = 0;
  *(_DWORD *)(a1 + 6576) = 0;
  *(_DWORD *)(a1 + 6584) = 0;
  *(_QWORD *)(a1 + 6592) = 0;
  *(_QWORD *)(a1 + 6616) = 4;
  *(_DWORD *)(a1 + 6624) = 0;
  *(_QWORD *)(a1 + 6664) = 0;
  *(_DWORD *)(a1 + 6688) = 4;
  *(_QWORD *)(a1 + 6692) = 0;
  v10 = *(_QWORD *)(a1 + 8);
  v11 = *(_DWORD **)(v10 + 232);
  v12 = v11[207];
  if ( v12 )
  {
    *(_DWORD *)(a1 + 6584) = v12;
    LOBYTE(v9) = v12 == 2;
  }
  else if ( (_BYTE)v9 )
  {
    *(_DWORD *)(a1 + 6584) = 2;
    v12 = 2;
  }
  else if ( v11[126] == 32 && v11[128] == 3 && v11[129] == 27 )
  {
    *(_DWORD *)(a1 + 6584) = 3;
    v12 = 3;
  }
  else
  {
    *(_DWORD *)(a1 + 6584) = 1;
    v12 = 1;
  }
  if ( dword_5056980 )
    *(_BYTE *)(a1 + 4499) = dword_5056980 == 1;
  else
    *(_BYTE *)(a1 + 4499) = (unsigned int)(v11[126] - 34) <= 1;
  *(_BYTE *)(a1 + 4512) = v9;
  v13 = dword_5056300;
  *(_BYTE *)(a1 + 4513) = *(_QWORD *)(*(_QWORD *)(v10 + 232) + 888LL) != 0;
  if ( v13 )
    v14 = v13 == 1;
  else
    v14 = v12 != 3;
  *(_BYTE *)(a1 + 4498) = v14;
  v15 = *(_DWORD *)(*(_QWORD *)(v10 + 232) + 844LL);
  if ( !v15 )
  {
    v15 = (unsigned int)sub_1633B10(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 1688LL));
    v10 = *(_QWORD *)(a1 + 8);
    v12 = *(_DWORD *)(a1 + 6584);
  }
  v16 = v11[126];
  if ( v16 == 34 )
  {
    v18 = 2;
    v17 = 0;
    v15 = 2;
    v19 = 0;
  }
  else
  {
    v17 = v15 != 0 && v16 != 35;
    if ( v17 )
    {
      v18 = v15;
      v17 = v15 > 4;
    }
    else
    {
      v18 = 2;
      v15 = 2;
    }
    v19 = (byte_50568A0 | (v16 == 35)) ^ 1;
  }
  *(_BYTE *)(a1 + 4500) = v19;
  v20 = 0;
  if ( !byte_50567C0 )
    v20 = (unsigned int)(v11[126] - 34) > 1;
  *(_BYTE *)(a1 + 4501) = v20;
  if ( dword_5056560 )
    *(_BYTE *)(a1 + 4502) = dword_5056560 == 1;
  else
    *(_BYTE *)(a1 + 4502) = (unsigned int)(v11[126] - 34) <= 1;
  v21 = *(_QWORD *)(a2 + 232);
  v22 = dword_5056BE0;
  if ( *(_DWORD *)(v21 + 524) == 2 && byte_5057180 )
  {
    *(_BYTE *)(a1 + 4504) = 1;
    if ( !v22 )
LABEL_25:
      v22 = 1;
  }
  else
  {
    *(_BYTE *)(a1 + 4504) = 0;
    if ( v22 )
      goto LABEL_26;
    if ( v15 <= 4 )
    {
      if ( v12 == 2 )
      {
        if ( *(_DWORD *)(v21 + 524) != 3 )
          v12 = 3;
        *(_DWORD *)(a1 + 4508) = v12;
        goto LABEL_27;
      }
      goto LABEL_25;
    }
    v22 = 3;
  }
LABEL_26:
  *(_DWORD *)(a1 + 4508) = v22;
  if ( v12 == 1 )
  {
LABEL_32:
    *(_BYTE *)(a1 + 4496) = 1;
    v23 = 1;
    goto LABEL_29;
  }
LABEL_27:
  if ( v15 <= 2 )
    goto LABEL_32;
  *(_BYTE *)(a1 + 4496) = 0;
  v23 = v15 <= 3;
LABEL_29:
  *(_BYTE *)(a1 + 4497) = v23;
  *(_BYTE *)(a1 + 4514) = v17;
  result = *(_QWORD *)(*(_QWORD *)(v10 + 256) + 8LL);
  *(_WORD *)(result + 1160) = v18;
  return result;
}
