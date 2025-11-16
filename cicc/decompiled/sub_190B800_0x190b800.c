// Function: sub_190B800
// Address: 0x190b800
//
__int64 sub_190B800()
{
  __int64 v0; // rax
  __int64 v1; // r12
  _QWORD *v2; // rax
  __int64 v3; // rax

  v0 = sub_22077B0(1032);
  v1 = v0;
  if ( v0 )
  {
    *(_QWORD *)(v0 + 8) = 0;
    *(_QWORD *)(v0 + 16) = &unk_4FAE60C;
    *(_QWORD *)(v0 + 80) = v0 + 64;
    *(_QWORD *)(v0 + 88) = v0 + 64;
    *(_QWORD *)(v0 + 128) = v0 + 112;
    *(_QWORD *)(v0 + 136) = v0 + 112;
    *(_QWORD *)v0 = off_49F33A8;
    *(_WORD *)(v0 + 152) = 0;
    *(_DWORD *)(v0 + 24) = 3;
    *(_QWORD *)(v0 + 32) = 0;
    *(_QWORD *)(v0 + 40) = 0;
    *(_QWORD *)(v0 + 48) = 0;
    *(_DWORD *)(v0 + 64) = 0;
    *(_QWORD *)(v0 + 72) = 0;
    *(_QWORD *)(v0 + 96) = 0;
    *(_DWORD *)(v0 + 112) = 0;
    *(_QWORD *)(v0 + 120) = 0;
    *(_QWORD *)(v0 + 144) = 0;
    *(_BYTE *)(v0 + 154) = 0;
    *(_QWORD *)(v0 + 168) = 0;
    *(_QWORD *)(v0 + 176) = 0;
    *(_QWORD *)(v0 + 208) = 0;
    *(_QWORD *)(v0 + 216) = 0;
    *(_QWORD *)(v0 + 224) = 0;
    *(_DWORD *)(v0 + 232) = 0;
    *(_QWORD *)(v0 + 240) = 0;
    *(_QWORD *)(v0 + 248) = 0;
    *(_QWORD *)(v0 + 256) = 0;
    *(_QWORD *)(v0 + 272) = 0;
    *(_QWORD *)(v0 + 280) = 0;
    *(_QWORD *)(v0 + 288) = 0;
    *(_DWORD *)(v0 + 296) = 0;
    sub_190A6C0(v0 + 312);
    *(_QWORD *)(v1 + 536) = 0;
    *(_QWORD *)(v1 + 584) = v1 + 600;
    *(_QWORD *)(v1 + 544) = 0;
    *(_QWORD *)(v1 + 552) = 0;
    *(_DWORD *)(v1 + 560) = 0;
    *(_QWORD *)(v1 + 568) = 0;
    *(_QWORD *)(v1 + 576) = 0;
    *(_QWORD *)(v1 + 640) = 0;
    *(_QWORD *)(v1 + 648) = 0;
    *(_QWORD *)(v1 + 656) = 1;
    *(_QWORD *)(v1 + 672) = 0;
    *(_QWORD *)(v1 + 680) = 1;
    *(_QWORD *)(v1 + 592) = 0x400000000LL;
    *(_QWORD *)(v1 + 632) = v1 + 648;
    v2 = (_QWORD *)(v1 + 688);
    do
    {
      if ( v2 )
        *v2 = -8;
      v2 += 2;
    }
    while ( (_QWORD *)(v1 + 752) != v2 );
    *(_QWORD *)(v1 + 912) = 0;
    *(_QWORD *)(v1 + 752) = v1 + 768;
    *(_QWORD *)(v1 + 832) = v1 + 848;
    *(_QWORD *)(v1 + 760) = 0x400000000LL;
    *(_QWORD *)(v1 + 840) = 0x800000000LL;
    *(_QWORD *)(v1 + 920) = 0;
    *(_QWORD *)(v1 + 928) = 0;
    *(_DWORD *)(v1 + 936) = 0;
    *(_BYTE *)(v1 + 944) = 1;
    *(_QWORD *)(v1 + 952) = v1 + 968;
    *(_QWORD *)(v1 + 960) = 0x400000000LL;
    v3 = sub_163A1D0();
    sub_190B6E0(v3);
  }
  return v1;
}
