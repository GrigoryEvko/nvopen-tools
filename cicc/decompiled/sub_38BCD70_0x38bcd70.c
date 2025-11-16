// Function: sub_38BCD70
// Address: 0x38bcd70
//
void __fastcall sub_38BCD70(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 **a5, char a6)
{
  _BYTE *v6; // r12
  __int64 v7; // rax
  __int64 v9; // rdi
  char *v10; // rsi
  __int64 v11; // rdx
  char *(*v12)(); // rax
  _BYTE *v13; // rdi
  _QWORD *v14; // rax
  size_t v15; // rsi
  __int64 v16; // rcx
  __int64 v17; // rdi
  size_t v18; // rdx
  _QWORD *v19; // [rsp+0h] [rbp-40h] BYREF
  size_t n; // [rsp+8h] [rbp-38h]
  _QWORD src[6]; // [rsp+10h] [rbp-30h] BYREF

  v6 = (_BYTE *)(a1 + 960);
  *(_QWORD *)(a1 + 24) = a3;
  *(_QWORD *)(a1 + 112) = a1 + 128;
  *(_QWORD *)(a1 + 168) = a1 + 184;
  *(_QWORD *)(a1 + 216) = a1 + 232;
  *(_QWORD *)(a1 + 272) = a1 + 288;
  *(_QWORD *)(a1 + 64) = a1 + 80;
  *(_QWORD *)(a1 + 320) = a1 + 336;
  *(_QWORD *)(a1 + 16) = a2;
  *(_QWORD *)(a1 + 32) = a4;
  *(_QWORD *)(a1 + 72) = 0x400000000LL;
  *(_QWORD *)(a1 + 176) = 0x400000000LL;
  *(_QWORD *)(a1 + 280) = 0x400000000LL;
  *(_QWORD *)a1 = a5;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 120) = 0;
  *(_QWORD *)(a1 + 128) = 0;
  *(_QWORD *)(a1 + 136) = 1;
  *(_QWORD *)(a1 + 152) = 0;
  *(_QWORD *)(a1 + 160) = 0;
  *(_QWORD *)(a1 + 224) = 0;
  *(_QWORD *)(a1 + 232) = 0;
  *(_QWORD *)(a1 + 240) = 0;
  *(_QWORD *)(a1 + 256) = 0;
  *(_QWORD *)(a1 + 264) = 0;
  *(_QWORD *)(a1 + 328) = 0;
  *(_QWORD *)(a1 + 376) = a1 + 392;
  *(_QWORD *)(a1 + 424) = a1 + 440;
  *(_QWORD *)(a1 + 480) = a1 + 496;
  *(_QWORD *)(a1 + 528) = a1 + 544;
  *(_QWORD *)(a1 + 592) = a1 + 48;
  *(_QWORD *)(a1 + 656) = a1 + 48;
  *(_QWORD *)(a1 + 384) = 0x400000000LL;
  *(_QWORD *)(a1 + 488) = 0x400000000LL;
  *(_QWORD *)(a1 + 584) = 0x1000000000LL;
  *(_QWORD *)(a1 + 648) = 0x1000000000LL;
  *(_QWORD *)(a1 + 336) = 0;
  *(_QWORD *)(a1 + 344) = 0;
  *(_QWORD *)(a1 + 360) = 0;
  *(_QWORD *)(a1 + 368) = 0;
  *(_QWORD *)(a1 + 432) = 0;
  *(_QWORD *)(a1 + 440) = 0;
  *(_QWORD *)(a1 + 448) = 0;
  *(_QWORD *)(a1 + 464) = 0;
  *(_QWORD *)(a1 + 472) = 0;
  *(_QWORD *)(a1 + 536) = 0;
  *(_QWORD *)(a1 + 544) = 0;
  *(_QWORD *)(a1 + 552) = 0;
  *(_QWORD *)(a1 + 568) = 0;
  *(_QWORD *)(a1 + 576) = 0;
  *(_QWORD *)(a1 + 600) = 0;
  *(_QWORD *)(a1 + 608) = 0;
  *(_QWORD *)(a1 + 616) = 0;
  *(_DWORD *)(a1 + 624) = 0;
  *(_QWORD *)(a1 + 632) = 0;
  *(_QWORD *)(a1 + 640) = 0;
  *(_QWORD *)(a1 + 664) = 0;
  *(_QWORD *)(a1 + 672) = 0;
  *(_QWORD *)(a1 + 680) = 0x1000000000LL;
  *(_QWORD *)(a1 + 752) = a1 + 768;
  *(_QWORD *)(a1 + 920) = a1 + 904;
  *(_QWORD *)(a1 + 928) = a1 + 904;
  *(_QWORD *)(a1 + 1000) = a1 + 984;
  *(_QWORD *)(a1 + 1008) = a1 + 984;
  *(_WORD *)(a1 + 1040) = 0;
  *(_QWORD *)(a1 + 696) = 0;
  *(_DWORD *)(a1 + 720) = 0;
  *(_QWORD *)(a1 + 704) = 0;
  *(_QWORD *)(a1 + 712) = 0;
  *(_QWORD *)(a1 + 736) = 0;
  *(_BYTE *)(a1 + 744) = 0;
  *(_QWORD *)(a1 + 760) = 0x8000000000LL;
  *(_DWORD *)(a1 + 904) = 0;
  *(_QWORD *)(a1 + 912) = 0;
  *(_QWORD *)(a1 + 936) = 0;
  *(_QWORD *)(a1 + 944) = a1 + 960;
  *(_QWORD *)(a1 + 952) = 0;
  *(_BYTE *)(a1 + 960) = 0;
  *(_DWORD *)(a1 + 984) = 0;
  *(_QWORD *)(a1 + 992) = 0;
  *(_QWORD *)(a1 + 1016) = 0;
  *(_QWORD *)(a1 + 1024) = 0;
  *(_QWORD *)(a1 + 1032) = 0x10000;
  *(_QWORD *)(a1 + 1044) = 0;
  *(_QWORD *)(a1 + 1052) = 0;
  *(_QWORD *)(a1 + 1060) = 0;
  *(_QWORD *)(a1 + 1068) = 0;
  *(_QWORD *)(a1 + 1080) = 0;
  *(_QWORD *)(a1 + 1088) = 0;
  *(_QWORD *)(a1 + 1096) = 0;
  *(_QWORD *)(a1 + 1104) = 0;
  *(_QWORD *)(a1 + 1112) = 0;
  *(_QWORD *)(a1 + 1224) = a1 + 1208;
  *(_QWORD *)(a1 + 1232) = a1 + 1208;
  *(_QWORD *)(a1 + 1400) = 0x400000000LL;
  *(_QWORD *)(a1 + 1184) = 0x1000000000LL;
  *(_QWORD *)(a1 + 1272) = a1 + 1256;
  *(_QWORD *)(a1 + 1280) = a1 + 1256;
  *(_QWORD *)(a1 + 1360) = 0x1000000000LL;
  *(_QWORD *)(a1 + 1440) = a1 + 1456;
  *(_QWORD *)(a1 + 1120) = 0;
  *(_QWORD *)(a1 + 1128) = 0;
  *(_QWORD *)(a1 + 1136) = 0;
  *(_QWORD *)(a1 + 1144) = 0;
  *(_QWORD *)(a1 + 1152) = 0;
  *(_QWORD *)(a1 + 1160) = 16842756;
  *(_QWORD *)(a1 + 1168) = 0;
  *(_QWORD *)(a1 + 1176) = 0;
  *(_DWORD *)(a1 + 1208) = 0;
  *(_QWORD *)(a1 + 1216) = 0;
  *(_QWORD *)(a1 + 1240) = 0;
  *(_DWORD *)(a1 + 1256) = 0;
  *(_QWORD *)(a1 + 1264) = 0;
  *(_QWORD *)(a1 + 1288) = 0;
  *(_DWORD *)(a1 + 1304) = 0;
  *(_QWORD *)(a1 + 1312) = 0;
  *(_QWORD *)(a1 + 1320) = a1 + 1304;
  *(_QWORD *)(a1 + 1328) = a1 + 1304;
  *(_QWORD *)(a1 + 1336) = 0;
  *(_QWORD *)(a1 + 1344) = 0;
  *(_QWORD *)(a1 + 1352) = 0;
  *(_QWORD *)(a1 + 1376) = 0;
  *(_QWORD *)(a1 + 1384) = 0;
  *(_QWORD *)(a1 + 1392) = a1 + 1408;
  *(_QWORD *)(a1 + 1448) = 0;
  *(_QWORD *)(a1 + 1504) = 0x4000000000LL;
  v7 = qword_50529A0;
  *(_QWORD *)(a1 + 1456) = 0;
  *(_QWORD *)(a1 + 1464) = 0;
  *(_BYTE *)(a1 + 1480) = a6;
  *(_BYTE *)(a1 + 1481) = 0;
  *(_QWORD *)(a1 + 1488) = 0;
  *(_QWORD *)(a1 + 1496) = 0;
  *(_QWORD *)(a1 + 728) = v7;
  if ( a5 && -1431655765 * (unsigned int)(a5[1] - *a5) )
  {
    v9 = **a5;
    v10 = "Unknown buffer";
    v11 = 14;
    v12 = *(char *(**)())(*(_QWORD *)v9 + 16LL);
    if ( v12 == sub_12BCB10
      || (v10 = (char *)((__int64 (__fastcall *)(__int64, char *, __int64))v12)(v9, "Unknown buffer", 14)) != 0 )
    {
      v19 = src;
      sub_38BB9D0((__int64 *)&v19, v10, (__int64)&v10[v11]);
      v13 = *(_BYTE **)(a1 + 944);
      v14 = v13;
      if ( v19 != src )
      {
        v15 = n;
        v16 = src[0];
        if ( v6 == v13 )
        {
          *(_QWORD *)(a1 + 944) = v19;
          *(_QWORD *)(a1 + 952) = v15;
          *(_QWORD *)(a1 + 960) = v16;
        }
        else
        {
          v17 = *(_QWORD *)(a1 + 960);
          *(_QWORD *)(a1 + 944) = v19;
          *(_QWORD *)(a1 + 952) = v15;
          *(_QWORD *)(a1 + 960) = v16;
          if ( v14 )
          {
            v19 = v14;
            src[0] = v17;
            goto LABEL_8;
          }
        }
        v19 = src;
        v14 = src;
LABEL_8:
        n = 0;
        *(_BYTE *)v14 = 0;
        if ( v19 != src )
          j_j___libc_free_0((unsigned __int64)v19);
        return;
      }
      v18 = n;
      if ( n )
      {
        if ( n == 1 )
          *v13 = src[0];
        else
          memcpy(v13, src, n);
        v18 = n;
        v13 = *(_BYTE **)(a1 + 944);
      }
    }
    else
    {
      LOBYTE(src[0]) = 0;
      v13 = *(_BYTE **)(a1 + 944);
      v18 = 0;
      v19 = src;
    }
    *(_QWORD *)(a1 + 952) = v18;
    v13[v18] = 0;
    v14 = v19;
    goto LABEL_8;
  }
}
