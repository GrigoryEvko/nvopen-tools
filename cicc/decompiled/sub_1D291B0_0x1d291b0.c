// Function: sub_1D291B0
// Address: 0x1d291b0
//
_QWORD *__fastcall sub_1D291B0(__int64 a1, __int64 a2, int a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  int v8; // edx
  _QWORD *result; // rax
  _QWORD *v10; // rdx

  *(_QWORD *)a1 = a2;
  *(_DWORD *)(a1 + 56) = a3;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  v7 = sub_1D29190(a1, 1u, 0, a4, a5, a6);
  *(_WORD *)(a1 + 168) &= 0xF000u;
  *(_QWORD *)(a1 + 128) = v7;
  *(_DWORD *)(a1 + 148) = v8;
  *(_QWORD *)(a1 + 200) = a1 + 192;
  *(_QWORD *)(a1 + 232) = a1 + 248;
  *(_QWORD *)(a1 + 192) = (a1 + 192) | 4;
  *(_WORD *)(a1 + 112) = 1;
  *(_WORD *)(a1 + 114) = 0;
  *(_QWORD *)(a1 + 280) = a1 + 296;
  *(_QWORD *)(a1 + 88) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 104) = 0;
  *(_DWORD *)(a1 + 116) = -1;
  *(_QWORD *)(a1 + 120) = 0;
  *(_QWORD *)(a1 + 136) = 0;
  *(_DWORD *)(a1 + 144) = 0;
  *(_DWORD *)(a1 + 152) = 0;
  *(_QWORD *)(a1 + 160) = 0;
  *(_QWORD *)(a1 + 176) = a1 + 88;
  *(_DWORD *)(a1 + 184) = 0;
  *(_QWORD *)(a1 + 208) = 0;
  *(_QWORD *)(a1 + 216) = 0;
  *(_QWORD *)(a1 + 224) = 0;
  *(_QWORD *)(a1 + 240) = 0x400000000LL;
  *(_QWORD *)(a1 + 288) = 0;
  *(_QWORD *)(a1 + 296) = 0;
  *(_QWORD *)(a1 + 304) = 1;
  sub_16BD940(a1 + 320, 6);
  *(_DWORD *)(a1 + 344) = a3;
  *(_WORD *)(a1 + 656) = 0;
  *(_DWORD *)(a1 + 352) = 0;
  *(_QWORD *)(a1 + 320) = &unk_49F9948;
  *(_QWORD *)(a1 + 376) = a1 + 392;
  *(_QWORD *)(a1 + 424) = a1 + 440;
  *(_QWORD *)(a1 + 464) = a1 + 480;
  *(_QWORD *)(a1 + 472) = 0x800000000LL;
  *(_QWORD *)(a1 + 560) = a1 + 576;
  *(_QWORD *)(a1 + 608) = a1 + 624;
  *(_BYTE *)(a1 + 356) = 0;
  *(_QWORD *)(a1 + 360) = 0;
  *(_QWORD *)(a1 + 368) = 0;
  *(_QWORD *)(a1 + 384) = 0x400000000LL;
  *(_QWORD *)(a1 + 432) = 0;
  *(_QWORD *)(a1 + 440) = 0;
  *(_QWORD *)(a1 + 448) = 1;
  *(_QWORD *)(a1 + 544) = 0;
  *(_QWORD *)(a1 + 552) = 0;
  *(_QWORD *)(a1 + 568) = 0x400000000LL;
  *(_QWORD *)(a1 + 616) = 0;
  *(_QWORD *)(a1 + 624) = 0;
  *(_QWORD *)(a1 + 632) = 1;
  *(_BYTE *)(a1 + 658) = 0;
  *(_QWORD *)(a1 + 664) = 0;
  sub_16BD940(a1 + 672, 6);
  *(_QWORD *)(a1 + 696) = 0;
  *(_QWORD *)(a1 + 704) = 0;
  *(_QWORD *)(a1 + 672) = &unk_49F99A8;
  *(_QWORD *)(a1 + 768) = a1 + 752;
  *(_QWORD *)(a1 + 776) = a1 + 752;
  *(_QWORD *)(a1 + 808) = 0x1000000000LL;
  *(_QWORD *)(a1 + 848) = a1 + 832;
  *(_QWORD *)(a1 + 856) = a1 + 832;
  *(_QWORD *)(a1 + 712) = 0;
  *(_QWORD *)(a1 + 720) = 0;
  *(_QWORD *)(a1 + 728) = 0;
  *(_QWORD *)(a1 + 736) = 0;
  *(_DWORD *)(a1 + 752) = 0;
  *(_QWORD *)(a1 + 760) = 0;
  *(_QWORD *)(a1 + 784) = 0;
  *(_QWORD *)(a1 + 792) = 0;
  *(_QWORD *)(a1 + 800) = 0;
  *(_DWORD *)(a1 + 832) = 0;
  *(_QWORD *)(a1 + 840) = 0;
  *(_QWORD *)(a1 + 864) = 0;
  *(_QWORD *)(a1 + 872) = 0;
  *(_QWORD *)(a1 + 880) = 0;
  *(_QWORD *)(a1 + 888) = 0;
  *(_DWORD *)(a1 + 896) = 0;
  sub_1D172A0(a1, a1 + 88);
  result = (_QWORD *)sub_22077B0(728);
  v10 = result;
  if ( result )
  {
    memset(result, 0, 0x2D8u);
    result[3] = 0x400000000LL;
    result[2] = result + 4;
    result[8] = result + 10;
    result[13] = result + 15;
    result[14] = 0x2000000000LL;
    result[48] = 0x2000000000LL;
    result += 83;
    v10[11] = 1;
    v10[47] = v10 + 49;
    v10[81] = v10 + 83;
    v10[82] = 0x400000000LL;
  }
  *(_QWORD *)(a1 + 648) = v10;
  return result;
}
