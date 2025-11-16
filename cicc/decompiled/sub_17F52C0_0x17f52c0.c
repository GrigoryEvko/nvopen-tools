// Function: sub_17F52C0
// Address: 0x17f52c0
//
__int64 sub_17F52C0()
{
  __int64 v0; // rax
  __int64 v1; // r12
  __int64 v2; // rdx
  __int64 v3; // rax

  v0 = sub_22077B0(840);
  v1 = v0;
  if ( v0 )
  {
    *(_QWORD *)(v0 + 8) = 0;
    *(_QWORD *)(v0 + 16) = &unk_4FA5EAC;
    *(_QWORD *)(v0 + 80) = v0 + 64;
    *(_QWORD *)(v0 + 88) = v0 + 64;
    *(_QWORD *)(v0 + 128) = v0 + 112;
    *(_QWORD *)(v0 + 136) = v0 + 112;
    *(_QWORD *)v0 = off_49F06F8;
    *(_QWORD *)(v0 + 376) = v0 + 392;
    *(_QWORD *)(v0 + 472) = v0 + 488;
    *(_DWORD *)(v0 + 24) = 5;
    *(_QWORD *)(v0 + 32) = 0;
    *(_QWORD *)(v0 + 40) = 0;
    *(_QWORD *)(v0 + 48) = 0;
    *(_DWORD *)(v0 + 64) = 0;
    *(_QWORD *)(v0 + 72) = 0;
    *(_QWORD *)(v0 + 96) = 0;
    *(_DWORD *)(v0 + 112) = 0;
    *(_QWORD *)(v0 + 120) = 0;
    *(_QWORD *)(v0 + 144) = 0;
    *(_BYTE *)(v0 + 152) = 0;
    *(_QWORD *)(v0 + 384) = 0;
    *(_BYTE *)(v0 + 392) = 0;
    *(_QWORD *)(v0 + 408) = 0;
    *(_QWORD *)(v0 + 416) = 0;
    *(_QWORD *)(v0 + 424) = 0;
    *(_QWORD *)(v0 + 480) = 0x1400000000LL;
    *(_QWORD *)(v0 + 648) = v0 + 664;
    *(_QWORD *)(v0 + 656) = 0x1400000000LL;
    *(_QWORD *)(v0 + 824) = sub_17F4250(0);
    *(_QWORD *)(v1 + 832) = v2;
    v3 = sub_163A1D0();
    sub_17F51D0(v3);
  }
  return v1;
}
