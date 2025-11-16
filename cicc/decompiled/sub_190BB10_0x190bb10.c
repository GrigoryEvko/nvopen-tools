// Function: sub_190BB10
// Address: 0x190bb10
//
__int64 __fastcall sub_190BB10(char a1, char a2)
{
  char v2; // bl
  __int64 v3; // rax
  __int64 v4; // r12
  _QWORD *v5; // rax
  __int64 v6; // rax

  v2 = byte_4FAE880;
  if ( !a2 )
    v2 = 0;
  v3 = sub_22077B0(1032);
  v4 = v3;
  if ( v3 )
  {
    *(_QWORD *)(v3 + 8) = 0;
    *(_QWORD *)(v3 + 16) = &unk_4FAE60C;
    *(_QWORD *)(v3 + 80) = v3 + 64;
    *(_QWORD *)(v3 + 88) = v3 + 64;
    *(_QWORD *)(v3 + 128) = v3 + 112;
    *(_QWORD *)(v3 + 136) = v3 + 112;
    *(_QWORD *)v3 = off_49F33A8;
    *(_DWORD *)(v3 + 24) = 3;
    *(_QWORD *)(v3 + 32) = 0;
    *(_QWORD *)(v3 + 40) = 0;
    *(_QWORD *)(v3 + 48) = 0;
    *(_DWORD *)(v3 + 64) = 0;
    *(_QWORD *)(v3 + 72) = 0;
    *(_QWORD *)(v3 + 96) = 0;
    *(_DWORD *)(v3 + 112) = 0;
    *(_QWORD *)(v3 + 120) = 0;
    *(_QWORD *)(v3 + 144) = 0;
    *(_BYTE *)(v3 + 152) = 0;
    *(_BYTE *)(v3 + 153) = a1;
    *(_BYTE *)(v3 + 154) = v2;
    *(_QWORD *)(v3 + 168) = 0;
    *(_QWORD *)(v3 + 176) = 0;
    *(_QWORD *)(v3 + 208) = 0;
    *(_QWORD *)(v3 + 216) = 0;
    *(_QWORD *)(v3 + 224) = 0;
    *(_DWORD *)(v3 + 232) = 0;
    *(_QWORD *)(v3 + 240) = 0;
    *(_QWORD *)(v3 + 248) = 0;
    *(_QWORD *)(v3 + 256) = 0;
    *(_QWORD *)(v3 + 272) = 0;
    *(_QWORD *)(v3 + 280) = 0;
    *(_QWORD *)(v3 + 288) = 0;
    *(_DWORD *)(v3 + 296) = 0;
    sub_190A6C0(v3 + 312);
    *(_QWORD *)(v4 + 536) = 0;
    *(_QWORD *)(v4 + 584) = v4 + 600;
    *(_QWORD *)(v4 + 544) = 0;
    *(_QWORD *)(v4 + 552) = 0;
    *(_DWORD *)(v4 + 560) = 0;
    *(_QWORD *)(v4 + 568) = 0;
    *(_QWORD *)(v4 + 576) = 0;
    *(_QWORD *)(v4 + 640) = 0;
    *(_QWORD *)(v4 + 648) = 0;
    *(_QWORD *)(v4 + 656) = 1;
    *(_QWORD *)(v4 + 672) = 0;
    *(_QWORD *)(v4 + 680) = 1;
    *(_QWORD *)(v4 + 592) = 0x400000000LL;
    *(_QWORD *)(v4 + 632) = v4 + 648;
    v5 = (_QWORD *)(v4 + 688);
    do
    {
      if ( v5 )
        *v5 = -8;
      v5 += 2;
    }
    while ( (_QWORD *)(v4 + 752) != v5 );
    *(_QWORD *)(v4 + 912) = 0;
    *(_QWORD *)(v4 + 752) = v4 + 768;
    *(_QWORD *)(v4 + 832) = v4 + 848;
    *(_QWORD *)(v4 + 760) = 0x400000000LL;
    *(_QWORD *)(v4 + 840) = 0x800000000LL;
    *(_QWORD *)(v4 + 920) = 0;
    *(_QWORD *)(v4 + 928) = 0;
    *(_DWORD *)(v4 + 936) = 0;
    *(_BYTE *)(v4 + 944) = 1;
    *(_QWORD *)(v4 + 952) = v4 + 968;
    *(_QWORD *)(v4 + 960) = 0x400000000LL;
    v6 = sub_163A1D0();
    sub_190B6E0(v6);
  }
  return v4;
}
