// Function: sub_1457DF0
// Address: 0x1457df0
//
__int64 __fastcall sub_1457DF0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 v7; // r13
  unsigned int v8; // eax
  _QWORD *v9; // rax
  __int64 v10; // rdx
  _QWORD *j; // rdx
  _QWORD *v12; // rax
  __int64 v13; // rdx
  _QWORD *k; // rdx
  __int64 v15; // r12
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 result; // rax
  bool v19; // dl
  _QWORD *v20; // rax
  __int64 v21; // rcx
  _QWORD *i; // rdx

  *(_BYTE *)a1 = 1;
  *(_QWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 16) = 0;
  *(_BYTE *)(a1 + 20) = 0;
  *(_QWORD *)(a1 + 24) = a2;
  *(_QWORD *)(a1 + 40) = a3;
  *(_QWORD *)(a1 + 48) = a4;
  *(_QWORD *)(a1 + 56) = a5;
  *(_QWORD *)(a1 + 64) = a6;
  v6 = sub_22077B0(32);
  v7 = v6;
  if ( v6 )
    sub_14562B0(v6);
  *(_QWORD *)(a1 + 72) = v7;
  *(_QWORD *)(a1 + 184) = a1 + 216;
  *(_QWORD *)(a1 + 192) = a1 + 216;
  *(_QWORD *)(a1 + 288) = a1 + 320;
  *(_QWORD *)(a1 + 296) = a1 + 320;
  *(_QWORD *)(a1 + 392) = a1 + 424;
  *(_QWORD *)(a1 + 400) = a1 + 424;
  *(_WORD *)(a1 + 488) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 88) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_DWORD *)(a1 + 104) = 0;
  *(_QWORD *)(a1 + 112) = 0;
  *(_QWORD *)(a1 + 120) = 0;
  *(_QWORD *)(a1 + 128) = 0;
  *(_DWORD *)(a1 + 136) = 0;
  *(_QWORD *)(a1 + 144) = 0;
  *(_QWORD *)(a1 + 152) = 0;
  *(_QWORD *)(a1 + 160) = 0;
  *(_DWORD *)(a1 + 168) = 0;
  *(_QWORD *)(a1 + 176) = 0;
  *(_QWORD *)(a1 + 200) = 8;
  *(_DWORD *)(a1 + 208) = 0;
  *(_QWORD *)(a1 + 280) = 0;
  *(_QWORD *)(a1 + 304) = 8;
  *(_DWORD *)(a1 + 312) = 0;
  *(_QWORD *)(a1 + 384) = 0;
  *(_QWORD *)(a1 + 408) = 8;
  *(_DWORD *)(a1 + 416) = 0;
  *(_QWORD *)(a1 + 496) = 0;
  *(_QWORD *)(a1 + 504) = 0;
  *(_QWORD *)(a1 + 512) = 0;
  *(_DWORD *)(a1 + 520) = 0;
  *(_QWORD *)(a1 + 528) = 0;
  *(_QWORD *)(a1 + 536) = 0;
  *(_QWORD *)(a1 + 544) = 0;
  *(_DWORD *)(a1 + 552) = 0;
  *(_QWORD *)(a1 + 560) = 0;
  *(_QWORD *)(a1 + 568) = 0;
  *(_QWORD *)(a1 + 576) = 0;
  *(_DWORD *)(a1 + 584) = 0;
  *(_QWORD *)(a1 + 592) = 0;
  *(_QWORD *)(a1 + 600) = 0;
  *(_QWORD *)(a1 + 608) = 0;
  *(_DWORD *)(a1 + 616) = 0;
  *(_QWORD *)(a1 + 624) = 0;
  v8 = sub_1454B60(0x56u);
  *(_DWORD *)(a1 + 648) = v8;
  if ( v8 )
  {
    v20 = (_QWORD *)sub_22077B0(56LL * v8);
    v21 = *(unsigned int *)(a1 + 648);
    *(_QWORD *)(a1 + 640) = 0;
    *(_QWORD *)(a1 + 632) = v20;
    for ( i = &v20[7 * v21]; i != v20; v20 += 7 )
    {
      if ( v20 )
        *v20 = -8;
    }
  }
  else
  {
    *(_QWORD *)(a1 + 632) = 0;
    *(_QWORD *)(a1 + 640) = 0;
  }
  *(_QWORD *)(a1 + 656) = 0;
  *(_DWORD *)(a1 + 680) = 128;
  v9 = (_QWORD *)sub_22077B0(5120);
  v10 = *(unsigned int *)(a1 + 680);
  *(_QWORD *)(a1 + 672) = 0;
  *(_QWORD *)(a1 + 664) = v9;
  for ( j = &v9[5 * v10]; j != v9; v9 += 5 )
  {
    if ( v9 )
      *v9 = -8;
  }
  *(_QWORD *)(a1 + 688) = 0;
  *(_QWORD *)(a1 + 696) = 0;
  *(_QWORD *)(a1 + 704) = 0;
  *(_DWORD *)(a1 + 712) = 0;
  *(_QWORD *)(a1 + 720) = 0;
  *(_DWORD *)(a1 + 744) = 128;
  v12 = (_QWORD *)sub_22077B0(5120);
  v13 = *(unsigned int *)(a1 + 744);
  *(_QWORD *)(a1 + 736) = 0;
  *(_QWORD *)(a1 + 728) = v12;
  for ( k = &v12[5 * v13]; k != v12; v12 += 5 )
  {
    if ( v12 )
      *v12 = -8;
  }
  *(_QWORD *)(a1 + 752) = 0;
  *(_QWORD *)(a1 + 760) = 0;
  *(_QWORD *)(a1 + 768) = 0;
  *(_DWORD *)(a1 + 776) = 0;
  *(_QWORD *)(a1 + 784) = 0;
  *(_QWORD *)(a1 + 792) = 0;
  *(_QWORD *)(a1 + 800) = 0;
  *(_DWORD *)(a1 + 808) = 0;
  sub_16BD940(a1 + 816, 6);
  *(_QWORD *)(a1 + 816) = &unk_49EC510;
  sub_16BD940(a1 + 840, 6);
  *(_QWORD *)(a1 + 864) = 0;
  *(_QWORD *)(a1 + 872) = 0;
  *(_QWORD *)(a1 + 936) = 0;
  *(_QWORD *)(a1 + 840) = &unk_49EC570;
  *(_QWORD *)(a1 + 880) = a1 + 896;
  *(_QWORD *)(a1 + 888) = 0x400000000LL;
  *(_QWORD *)(a1 + 928) = a1 + 944;
  *(_QWORD *)(a1 + 944) = 0;
  *(_QWORD *)(a1 + 952) = 1;
  *(_QWORD *)(a1 + 968) = 0;
  *(_QWORD *)(a1 + 1000) = 0;
  v15 = *(_QWORD *)(a2 + 40);
  *(_QWORD *)(a1 + 976) = 0;
  *(_QWORD *)(a1 + 984) = 0;
  *(_DWORD *)(a1 + 992) = 0;
  *(_QWORD *)(a1 + 1008) = 0;
  *(_QWORD *)(a1 + 1016) = 0;
  *(_DWORD *)(a1 + 1024) = 0;
  *(_QWORD *)(a1 + 1032) = 0;
  v16 = sub_15E0FD0(79);
  result = sub_16321A0(v15, v16, v17);
  v19 = 0;
  if ( result )
    v19 = *(_QWORD *)(result + 8) != 0;
  *(_BYTE *)(a1 + 32) = v19;
  return result;
}
