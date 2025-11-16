// Function: sub_278C790
// Address: 0x278c790
//
__int64 __fastcall sub_278C790(char a1)
{
  char v1; // r14
  char v2; // r13
  __int64 v3; // rax
  __int64 v4; // r12
  _QWORD *v5; // rax
  __int128 *v6; // rax
  _BYTE v8[12]; // [rsp+4h] [rbp-3Ch]
  __int16 v9; // [rsp+10h] [rbp-30h]

  v1 = qword_4FFBCE8;
  v2 = qword_4FFBC08;
  v3 = sub_22077B0(0x400u);
  v4 = v3;
  if ( v3 )
  {
    *(_QWORD *)(v3 + 8) = 0;
    *(_QWORD *)(v3 + 16) = &unk_4FFB38C;
    *(_QWORD *)(v3 + 56) = v3 + 104;
    *(_QWORD *)(v3 + 112) = v3 + 160;
    *(_QWORD *)&v8[1] = 0x10000;
    *(_QWORD *)v3 = off_4A20CB0;
    v8[2] = a1;
    *(_DWORD *)(v3 + 88) = 1065353216;
    *(_DWORD *)(v3 + 144) = 1065353216;
    *(_DWORD *)(v3 + 24) = 2;
    *(_QWORD *)(v3 + 32) = 0;
    *(_QWORD *)(v3 + 40) = 0;
    *(_QWORD *)(v3 + 48) = 0;
    *(_QWORD *)(v3 + 64) = 1;
    *(_QWORD *)(v3 + 72) = 0;
    *(_QWORD *)(v3 + 80) = 0;
    *(_QWORD *)(v3 + 96) = 0;
    *(_QWORD *)(v3 + 104) = 0;
    *(_QWORD *)(v3 + 120) = 1;
    *(_QWORD *)(v3 + 128) = 0;
    *(_QWORD *)(v3 + 136) = 0;
    *(_QWORD *)(v3 + 152) = 0;
    *(_QWORD *)(v3 + 160) = 0;
    *(_BYTE *)(v3 + 168) = 0;
    v8[9] = 0;
    v8[10] = v1;
    v8[11] = 1;
    LOBYTE(v9) = v2;
    HIBYTE(v9) = 1;
    *(_QWORD *)(v3 + 176) = *(_QWORD *)v8;
    *(_QWORD *)(v3 + 192) = 0;
    *(_DWORD *)(v3 + 184) = *(_DWORD *)&v8[8];
    *(_QWORD *)(v3 + 200) = 0;
    *(_WORD *)(v3 + 188) = v9;
    *(_QWORD *)(v3 + 256) = v3 + 272;
    *(_QWORD *)(v3 + 208) = 0;
    *(_QWORD *)(v3 + 216) = 0;
    *(_QWORD *)(v3 + 224) = 0;
    *(_QWORD *)(v3 + 232) = 0;
    *(_QWORD *)(v3 + 240) = 0;
    *(_DWORD *)(v3 + 248) = 0;
    *(_QWORD *)(v3 + 264) = 0;
    *(_QWORD *)(v3 + 272) = 0;
    *(_QWORD *)(v3 + 280) = 0;
    *(_QWORD *)(v3 + 288) = 0;
    *(_QWORD *)(v3 + 296) = 0;
    sub_278A360(v3 + 312);
    *(_QWORD *)(v4 + 528) = 0;
    *(_QWORD *)(v4 + 576) = v4 + 592;
    *(_QWORD *)(v4 + 536) = 0;
    *(_QWORD *)(v4 + 544) = 0;
    *(_DWORD *)(v4 + 552) = 0;
    *(_QWORD *)(v4 + 560) = 0;
    *(_QWORD *)(v4 + 568) = 0;
    *(_QWORD *)(v4 + 632) = 0;
    *(_QWORD *)(v4 + 640) = 0;
    *(_QWORD *)(v4 + 648) = 1;
    *(_QWORD *)(v4 + 664) = 0;
    *(_QWORD *)(v4 + 672) = 1;
    *(_QWORD *)(v4 + 584) = 0x400000000LL;
    *(_QWORD *)(v4 + 624) = v4 + 640;
    v5 = (_QWORD *)(v4 + 680);
    do
    {
      if ( v5 )
        *v5 = -4096;
      v5 += 2;
    }
    while ( (_QWORD *)(v4 + 744) != v5 );
    *(_QWORD *)(v4 + 904) = 0;
    *(_QWORD *)(v4 + 744) = v4 + 760;
    *(_QWORD *)(v4 + 824) = v4 + 840;
    *(_QWORD *)(v4 + 752) = 0x400000000LL;
    *(_QWORD *)(v4 + 832) = 0x800000000LL;
    *(_QWORD *)(v4 + 912) = 0;
    *(_QWORD *)(v4 + 920) = 0;
    *(_DWORD *)(v4 + 928) = 0;
    *(_BYTE *)(v4 + 936) = 1;
    *(_QWORD *)(v4 + 944) = v4 + 960;
    *(_QWORD *)(v4 + 952) = 0x400000000LL;
    v6 = sub_BC2B00();
    sub_278C370((__int64)v6);
  }
  return v4;
}
