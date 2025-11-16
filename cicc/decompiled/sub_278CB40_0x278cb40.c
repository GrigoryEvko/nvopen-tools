// Function: sub_278CB40
// Address: 0x278cb40
//
__int64 sub_278CB40()
{
  char v0; // r13
  char v1; // bl
  __int64 v2; // rax
  __int64 v3; // r12
  _QWORD *v4; // rax
  __int128 *v5; // rax
  int v7; // [rsp+Ch] [rbp-34h]
  __int16 v8; // [rsp+10h] [rbp-30h]

  v0 = qword_4FFBCE8;
  v1 = qword_4FFBC08;
  v2 = sub_22077B0(0x400u);
  v3 = v2;
  if ( v2 )
  {
    *(_QWORD *)(v2 + 8) = 0;
    *(_QWORD *)(v2 + 16) = &unk_4FFB38C;
    *(_QWORD *)(v2 + 56) = v2 + 104;
    *(_QWORD *)(v2 + 112) = v2 + 160;
    *(_QWORD *)v2 = off_4A20CB0;
    LOWORD(v7) = 0;
    BYTE2(v7) = v0;
    HIBYTE(v7) = 1;
    LOBYTE(v8) = v1;
    HIBYTE(v8) = 1;
    *(_DWORD *)(v2 + 184) = v7;
    *(_DWORD *)(v2 + 88) = 1065353216;
    *(_DWORD *)(v2 + 144) = 1065353216;
    *(_DWORD *)(v2 + 24) = 2;
    *(_QWORD *)(v2 + 32) = 0;
    *(_QWORD *)(v2 + 40) = 0;
    *(_QWORD *)(v2 + 48) = 0;
    *(_QWORD *)(v2 + 64) = 1;
    *(_QWORD *)(v2 + 72) = 0;
    *(_QWORD *)(v2 + 80) = 0;
    *(_QWORD *)(v2 + 96) = 0;
    *(_QWORD *)(v2 + 104) = 0;
    *(_QWORD *)(v2 + 120) = 1;
    *(_QWORD *)(v2 + 128) = 0;
    *(_QWORD *)(v2 + 136) = 0;
    *(_QWORD *)(v2 + 152) = 0;
    *(_QWORD *)(v2 + 160) = 0;
    *(_BYTE *)(v2 + 168) = 0;
    *(_QWORD *)(v2 + 176) = &loc_1010000;
    *(_WORD *)(v2 + 188) = v8;
    *(_QWORD *)(v2 + 256) = v2 + 272;
    *(_QWORD *)(v2 + 192) = 0;
    *(_QWORD *)(v2 + 200) = 0;
    *(_QWORD *)(v2 + 208) = 0;
    *(_QWORD *)(v2 + 216) = 0;
    *(_QWORD *)(v2 + 224) = 0;
    *(_QWORD *)(v2 + 232) = 0;
    *(_QWORD *)(v2 + 240) = 0;
    *(_DWORD *)(v2 + 248) = 0;
    *(_QWORD *)(v2 + 264) = 0;
    *(_QWORD *)(v2 + 272) = 0;
    *(_QWORD *)(v2 + 280) = 0;
    *(_QWORD *)(v2 + 288) = 0;
    *(_QWORD *)(v2 + 296) = 0;
    sub_278A360(v2 + 312);
    *(_QWORD *)(v3 + 528) = 0;
    *(_QWORD *)(v3 + 576) = v3 + 592;
    *(_QWORD *)(v3 + 536) = 0;
    *(_QWORD *)(v3 + 544) = 0;
    *(_DWORD *)(v3 + 552) = 0;
    *(_QWORD *)(v3 + 560) = 0;
    *(_QWORD *)(v3 + 568) = 0;
    *(_QWORD *)(v3 + 632) = 0;
    *(_QWORD *)(v3 + 640) = 0;
    *(_QWORD *)(v3 + 648) = 1;
    *(_QWORD *)(v3 + 664) = 0;
    *(_QWORD *)(v3 + 672) = 1;
    *(_QWORD *)(v3 + 584) = 0x400000000LL;
    *(_QWORD *)(v3 + 624) = v3 + 640;
    v4 = (_QWORD *)(v3 + 680);
    do
    {
      if ( v4 )
        *v4 = -4096;
      v4 += 2;
    }
    while ( (_QWORD *)(v3 + 744) != v4 );
    *(_QWORD *)(v3 + 904) = 0;
    *(_QWORD *)(v3 + 744) = v3 + 760;
    *(_QWORD *)(v3 + 824) = v3 + 840;
    *(_QWORD *)(v3 + 752) = 0x400000000LL;
    *(_QWORD *)(v3 + 832) = 0x800000000LL;
    *(_QWORD *)(v3 + 912) = 0;
    *(_QWORD *)(v3 + 920) = 0;
    *(_DWORD *)(v3 + 928) = 0;
    *(_BYTE *)(v3 + 936) = 1;
    *(_QWORD *)(v3 + 944) = v3 + 960;
    *(_QWORD *)(v3 + 952) = 0x400000000LL;
    v5 = sub_BC2B00();
    sub_278C370((__int64)v5);
  }
  return v3;
}
