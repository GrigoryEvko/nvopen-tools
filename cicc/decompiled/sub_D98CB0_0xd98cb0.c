// Function: sub_D98CB0
// Address: 0xd98cb0
//
__int64 __fastcall sub_D98CB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // r13
  _QWORD *v12; // rax
  __int64 v13; // rcx
  _QWORD *i; // rdx
  _QWORD *v15; // rax
  __int64 v16; // rdx
  _QWORD *j; // rdx
  _QWORD *v18; // rax
  __int64 v19; // rdx
  _QWORD *k; // rdx
  char v21; // al
  __int64 result; // rax
  bool v23; // dl

  *(_QWORD *)a1 = a2;
  v9 = sub_B2BEC0(a2);
  *(_QWORD *)(a1 + 48) = a6;
  *(_QWORD *)(a1 + 8) = v9;
  *(_QWORD *)(a1 + 24) = a3;
  *(_QWORD *)(a1 + 32) = a4;
  *(_QWORD *)(a1 + 40) = a5;
  v10 = sub_22077B0(32);
  v11 = v10;
  if ( v10 )
    sub_D96A20(v10);
  *(_QWORD *)(a1 + 56) = v11;
  *(_QWORD *)(a1 + 232) = a1 + 256;
  *(_QWORD *)(a1 + 328) = a1 + 352;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_DWORD *)(a1 + 88) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 104) = 0;
  *(_QWORD *)(a1 + 112) = 0;
  *(_DWORD *)(a1 + 120) = 0;
  *(_QWORD *)(a1 + 128) = 0;
  *(_QWORD *)(a1 + 136) = 0;
  *(_QWORD *)(a1 + 144) = 0;
  *(_DWORD *)(a1 + 152) = 0;
  *(_QWORD *)(a1 + 160) = 0;
  *(_QWORD *)(a1 + 168) = 0;
  *(_QWORD *)(a1 + 176) = 0;
  *(_DWORD *)(a1 + 184) = 0;
  *(_QWORD *)(a1 + 192) = 0;
  *(_QWORD *)(a1 + 200) = 0;
  *(_QWORD *)(a1 + 208) = 0;
  *(_DWORD *)(a1 + 216) = 0;
  *(_QWORD *)(a1 + 224) = 0;
  *(_QWORD *)(a1 + 240) = 8;
  *(_DWORD *)(a1 + 248) = 0;
  *(_BYTE *)(a1 + 252) = 1;
  *(_QWORD *)(a1 + 320) = 0;
  *(_QWORD *)(a1 + 336) = 8;
  *(_DWORD *)(a1 + 344) = 0;
  *(_BYTE *)(a1 + 348) = 1;
  *(_QWORD *)(a1 + 416) = 0;
  *(_QWORD *)(a1 + 424) = a1 + 448;
  *(_QWORD *)(a1 + 520) = a1 + 544;
  *(_QWORD *)(a1 + 432) = 8;
  *(_DWORD *)(a1 + 440) = 0;
  *(_BYTE *)(a1 + 444) = 1;
  *(_QWORD *)(a1 + 512) = 0;
  *(_QWORD *)(a1 + 528) = 8;
  *(_DWORD *)(a1 + 536) = 0;
  *(_BYTE *)(a1 + 540) = 1;
  *(_WORD *)(a1 + 608) = 0;
  *(_QWORD *)(a1 + 616) = 0;
  *(_QWORD *)(a1 + 624) = 0;
  *(_QWORD *)(a1 + 632) = 0;
  *(_DWORD *)(a1 + 640) = 0;
  *(_QWORD *)(a1 + 648) = 0;
  *(_QWORD *)(a1 + 656) = 0;
  *(_QWORD *)(a1 + 664) = 0;
  *(_DWORD *)(a1 + 672) = 0;
  *(_QWORD *)(a1 + 680) = 0;
  *(_QWORD *)(a1 + 688) = 0;
  *(_QWORD *)(a1 + 696) = 0;
  *(_DWORD *)(a1 + 704) = 0;
  *(_QWORD *)(a1 + 712) = 0;
  *(_DWORD *)(a1 + 736) = 0;
  *(_QWORD *)(a1 + 720) = 0;
  *(_QWORD *)(a1 + 728) = 0;
  *(_QWORD *)(a1 + 744) = 0;
  *(_QWORD *)(a1 + 752) = 0;
  *(_QWORD *)(a1 + 760) = 0;
  *(_DWORD *)(a1 + 768) = 0;
  *(_QWORD *)(a1 + 776) = 0;
  *(_DWORD *)(a1 + 800) = 128;
  v12 = (_QWORD *)sub_C7D670(7168, 8);
  v13 = *(unsigned int *)(a1 + 800);
  *(_QWORD *)(a1 + 792) = 0;
  *(_QWORD *)(a1 + 784) = v12;
  for ( i = &v12[7 * v13]; i != v12; v12 += 7 )
  {
    if ( v12 )
      *v12 = -4096;
  }
  *(_QWORD *)(a1 + 808) = 0;
  *(_QWORD *)(a1 + 816) = 0;
  *(_QWORD *)(a1 + 824) = 0;
  *(_DWORD *)(a1 + 832) = 0;
  *(_QWORD *)(a1 + 840) = 0;
  *(_DWORD *)(a1 + 864) = 128;
  v15 = (_QWORD *)sub_C7D670(5120, 8);
  v16 = *(unsigned int *)(a1 + 864);
  *(_QWORD *)(a1 + 856) = 0;
  *(_QWORD *)(a1 + 848) = v15;
  for ( j = &v15[5 * v16]; j != v15; v15 += 5 )
  {
    if ( v15 )
      *v15 = -4096;
  }
  *(_QWORD *)(a1 + 872) = 0;
  *(_QWORD *)(a1 + 880) = 0;
  *(_QWORD *)(a1 + 888) = 0;
  *(_DWORD *)(a1 + 896) = 0;
  *(_QWORD *)(a1 + 904) = 0;
  *(_DWORD *)(a1 + 928) = 128;
  v18 = (_QWORD *)sub_C7D670(5120, 8);
  v19 = *(unsigned int *)(a1 + 928);
  *(_QWORD *)(a1 + 920) = 0;
  *(_QWORD *)(a1 + 912) = v18;
  for ( k = &v18[5 * v19]; k != v18; v18 += 5 )
  {
    if ( v18 )
      *v18 = -4096;
  }
  *(_QWORD *)(a1 + 936) = 0;
  *(_QWORD *)(a1 + 944) = 0;
  *(_QWORD *)(a1 + 952) = 0;
  *(_DWORD *)(a1 + 960) = 0;
  *(_QWORD *)(a1 + 968) = 0;
  *(_QWORD *)(a1 + 976) = 0;
  *(_QWORD *)(a1 + 984) = 0;
  *(_DWORD *)(a1 + 992) = 0;
  *(_QWORD *)(a1 + 1000) = 0;
  *(_QWORD *)(a1 + 1008) = 0;
  *(_QWORD *)(a1 + 1016) = 0;
  *(_DWORD *)(a1 + 1024) = 0;
  sub_C656D0(a1 + 1032, 6);
  sub_C656D0(a1 + 1048, 6);
  *(_QWORD *)(a1 + 1064) = 0;
  *(_QWORD *)(a1 + 1080) = a1 + 1096;
  *(_QWORD *)(a1 + 1088) = 0x400000000LL;
  *(_QWORD *)(a1 + 1128) = a1 + 1144;
  *(_QWORD *)(a1 + 1232) = a1 + 1256;
  *(_QWORD *)(a1 + 1392) = a1 + 1416;
  v21 = qword_4F88508;
  *(_QWORD *)(a1 + 1072) = 0;
  *(_BYTE *)(a1 + 1560) = v21;
  *(_QWORD *)(a1 + 1136) = 0;
  *(_QWORD *)(a1 + 1144) = 0;
  *(_QWORD *)(a1 + 1152) = 1;
  *(_QWORD *)(a1 + 1160) = 0;
  *(_QWORD *)(a1 + 1168) = 0;
  *(_QWORD *)(a1 + 1176) = 0;
  *(_DWORD *)(a1 + 1184) = 0;
  *(_QWORD *)(a1 + 1192) = 0;
  *(_QWORD *)(a1 + 1200) = 0;
  *(_QWORD *)(a1 + 1208) = 0;
  *(_DWORD *)(a1 + 1216) = 0;
  *(_QWORD *)(a1 + 1224) = 0;
  *(_QWORD *)(a1 + 1240) = 16;
  *(_DWORD *)(a1 + 1248) = 0;
  *(_BYTE *)(a1 + 1252) = 1;
  *(_QWORD *)(a1 + 1384) = 0;
  *(_QWORD *)(a1 + 1400) = 16;
  *(_DWORD *)(a1 + 1408) = 0;
  *(_BYTE *)(a1 + 1412) = 1;
  *(_QWORD *)(a1 + 1544) = 0;
  *(_QWORD *)(a1 + 1552) = 0;
  *(_QWORD *)(a1 + 1564) = 0;
  *(_BYTE *)(a1 + 1572) = 0;
  result = sub_B6AC80(*(_QWORD *)(a2 + 40), 153);
  v23 = 0;
  if ( result )
    v23 = *(_QWORD *)(result + 16) != 0;
  *(_BYTE *)(a1 + 16) = v23;
  return result;
}
