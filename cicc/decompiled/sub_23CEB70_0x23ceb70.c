// Function: sub_23CEB70
// Address: 0x23ceb70
//
void __fastcall sub_23CEB70(
        __int64 a1,
        __int64 a2,
        _BYTE *a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        _BYTE *a7,
        __int64 a8,
        _BYTE *a9,
        __int64 a10)
{
  _QWORD *v13; // rdi
  __int16 v14; // dx
  __int16 v15; // ax
  unsigned __int64 v16; // rdx
  __int64 v17; // rax
  int v18; // edx
  _BYTE *v19; // rsi
  __int64 v20; // rdx
  unsigned __int64 v21; // r14
  __int64 *v22; // r12
  __int64 v23; // r15
  __int64 i; // r14

  v13 = (_QWORD *)(a1 + 16);
  *(v13 - 1) = a2;
  *(v13 - 2) = &unk_4A16308;
  sub_AE3F70(v13, a3, a4);
  *(_QWORD *)(a1 + 512) = a1 + 528;
  sub_23CE5F0((__int64 *)(a1 + 512), *(_BYTE **)a5, *(_QWORD *)a5 + *(_QWORD *)(a5 + 8));
  *(_QWORD *)(a1 + 544) = *(_QWORD *)(a5 + 32);
  *(_QWORD *)(a1 + 552) = *(_QWORD *)(a5 + 40);
  *(_QWORD *)(a1 + 560) = *(_QWORD *)(a5 + 48);
  *(_QWORD *)(a1 + 568) = a1 + 584;
  sub_23CE470((__int64 *)(a1 + 568), a7, (__int64)&a7[a8]);
  *(_QWORD *)(a1 + 600) = a1 + 616;
  sub_23CE470((__int64 *)(a1 + 600), a9, (__int64)&a9[a10]);
  *(_BYTE *)(a1 + 688) &= 0xFCu;
  *(_QWORD *)(a1 + 632) = 0x100000000LL;
  *(_QWORD *)(a1 + 640) = 0;
  *(_DWORD *)(a1 + 648) = 2;
  *(_QWORD *)(a1 + 656) = 0;
  *(_QWORD *)(a1 + 664) = 0;
  *(_QWORD *)(a1 + 672) = 0;
  *(_QWORD *)(a1 + 680) = 0;
  *(_BYTE *)(a1 + 848) = 0;
  v14 = *(_WORD *)(a6 + 8);
  *(_QWORD *)(a1 + 856) = *(_QWORD *)a6;
  v15 = v14 & 0x1FFF | *(_WORD *)(a1 + 864) & 0xE000;
  v16 = *(_QWORD *)(a6 + 20) & 0xFFFFFFFF1FFFFFFFLL;
  *(_WORD *)(a1 + 864) = v15;
  *(_QWORD *)(a1 + 868) = *(_QWORD *)(a6 + 12);
  *(_QWORD *)(a1 + 876) = v16 | *(_QWORD *)(a1 + 876) & 0xE0000000LL;
  *(_QWORD *)(a1 + 888) = *(_QWORD *)(a6 + 32);
  v17 = *(_QWORD *)(a6 + 40);
  *(_QWORD *)(a1 + 896) = v17;
  if ( v17 )
  {
    if ( &_pthread_key_create )
      _InterlockedAdd((volatile signed __int32 *)(v17 + 8), 1u);
    else
      ++*(_DWORD *)(v17 + 8);
  }
  *(_WORD *)(a1 + 904) = *(_WORD *)(a6 + 48) & 0x7FFF | *(_WORD *)(a1 + 904) & 0x8000;
  *(_QWORD *)(a1 + 912) = a1 + 928;
  sub_23CE5F0((__int64 *)(a1 + 912), *(_BYTE **)(a6 + 56), *(_QWORD *)(a6 + 56) + *(_QWORD *)(a6 + 64));
  *(_QWORD *)(a1 + 944) = *(_QWORD *)(a6 + 88);
  *(_QWORD *)(a1 + 952) = *(_QWORD *)(a6 + 96);
  *(_QWORD *)(a1 + 960) = *(_QWORD *)(a6 + 104);
  *(_WORD *)(a1 + 968) = *(_WORD *)(a6 + 112);
  *(_WORD *)(a1 + 970) = *(_WORD *)(a6 + 114);
  *(_DWORD *)(a1 + 972) = *(_DWORD *)(a6 + 116);
  v18 = *(_DWORD *)(a6 + 120);
  BYTE1(v18) &= 0x3Fu;
  *(_DWORD *)(a1 + 976) = v18 | *(_DWORD *)(a1 + 976) & 0xC000;
  *(_WORD *)(a1 + 980) = *(_WORD *)(a6 + 124);
  *(_QWORD *)(a1 + 984) = *(_QWORD *)(a6 + 128);
  *(_QWORD *)(a1 + 992) = *(_QWORD *)(a6 + 136);
  *(_QWORD *)(a1 + 1000) = *(_QWORD *)(a6 + 144);
  *(_QWORD *)(a1 + 1008) = a1 + 1024;
  sub_23CE5F0((__int64 *)(a1 + 1008), *(_BYTE **)(a6 + 152), *(_QWORD *)(a6 + 152) + *(_QWORD *)(a6 + 160));
  *(_QWORD *)(a1 + 1040) = a1 + 1056;
  sub_23CE5F0((__int64 *)(a1 + 1040), *(_BYTE **)(a6 + 184), *(_QWORD *)(a6 + 184) + *(_QWORD *)(a6 + 192));
  *(_QWORD *)(a1 + 1072) = a1 + 1088;
  sub_23CE5F0((__int64 *)(a1 + 1072), *(_BYTE **)(a6 + 216), *(_QWORD *)(a6 + 216) + *(_QWORD *)(a6 + 224));
  *(_QWORD *)(a1 + 1104) = a1 + 1120;
  sub_23CE5F0((__int64 *)(a1 + 1104), *(_BYTE **)(a6 + 248), *(_QWORD *)(a6 + 248) + *(_QWORD *)(a6 + 256));
  *(_QWORD *)(a1 + 1136) = a1 + 1152;
  sub_23CE5F0((__int64 *)(a1 + 1136), *(_BYTE **)(a6 + 280), *(_QWORD *)(a6 + 280) + *(_QWORD *)(a6 + 288));
  *(_QWORD *)(a1 + 1168) = a1 + 1184;
  v19 = *(_BYTE **)(a6 + 312);
  sub_23CE5F0((__int64 *)(a1 + 1168), v19, (__int64)&v19[*(_QWORD *)(a6 + 320)]);
  v21 = *(_QWORD *)(a6 + 352) - *(_QWORD *)(a6 + 344);
  *(_QWORD *)(a1 + 1200) = 0;
  *(_QWORD *)(a1 + 1208) = 0;
  *(_QWORD *)(a1 + 1216) = 0;
  if ( v21 )
  {
    if ( v21 > 0x7FFFFFFFFFFFFFE0LL )
      sub_4261EA(a1 + 1168, v19, v20);
    v22 = (__int64 *)sub_22077B0(v21);
  }
  else
  {
    v21 = 0;
    v22 = 0;
  }
  *(_QWORD *)(a1 + 1200) = v22;
  *(_QWORD *)(a1 + 1208) = v22;
  *(_QWORD *)(a1 + 1216) = (char *)v22 + v21;
  v23 = *(_QWORD *)(a6 + 352);
  for ( i = *(_QWORD *)(a6 + 344); v23 != i; v22 += 4 )
  {
    if ( v22 )
    {
      *v22 = (__int64)(v22 + 2);
      sub_23CE5F0(v22, *(_BYTE **)i, *(_QWORD *)i + *(_QWORD *)(i + 8));
    }
    i += 32;
  }
  *(_QWORD *)(a1 + 1208) = v22;
  *(_BYTE *)(a1 + 1224) = *(_BYTE *)(a6 + 368) & 3 | *(_BYTE *)(a1 + 1224) & 0xFC;
  *(_QWORD *)(a1 + 1232) = a1 + 1248;
  sub_23CE5F0((__int64 *)(a1 + 1232), *(_BYTE **)(a6 + 376), *(_QWORD *)(a6 + 376) + *(_QWORD *)(a6 + 384));
}
