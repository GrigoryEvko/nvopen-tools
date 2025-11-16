// Function: sub_142B040
// Address: 0x142b040
//
__int64 __fastcall sub_142B040(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r12
  __int64 v4; // rax

  v2 = sub_22077B0(856);
  v3 = v2;
  if ( v2 )
  {
    *(_QWORD *)(v2 + 8) = 0;
    *(_QWORD *)(v2 + 16) = &unk_4F99944;
    *(_QWORD *)(v2 + 80) = v2 + 64;
    *(_QWORD *)(v2 + 88) = v2 + 64;
    *(_QWORD *)(v2 + 128) = v2 + 112;
    *(_QWORD *)(v2 + 136) = v2 + 112;
    *(_QWORD *)v2 = off_49EB3D0;
    *(_QWORD *)(v2 + 240) = v2 + 256;
    *(_QWORD *)(v2 + 160) = v2 + 176;
    *(_QWORD *)(v2 + 320) = v2 + 336;
    *(_QWORD *)(v2 + 168) = 0x800000000LL;
    *(_QWORD *)(v2 + 248) = 0x800000000LL;
    *(_QWORD *)(v2 + 328) = 0x800000000LL;
    *(_QWORD *)(v2 + 400) = v2 + 416;
    *(_QWORD *)(v2 + 408) = 0x800000000LL;
    *(_QWORD *)(v2 + 488) = 0x800000000LL;
    *(_QWORD *)(v2 + 480) = v2 + 496;
    *(_QWORD *)(v2 + 568) = v2 + 600;
    *(_QWORD *)(v2 + 576) = v2 + 600;
    *(_DWORD *)(v2 + 24) = 5;
    *(_QWORD *)(v2 + 32) = 0;
    *(_QWORD *)(v2 + 40) = 0;
    *(_QWORD *)(v2 + 48) = 0;
    *(_DWORD *)(v2 + 64) = 0;
    *(_QWORD *)(v2 + 72) = 0;
    *(_QWORD *)(v2 + 96) = 0;
    *(_DWORD *)(v2 + 112) = 0;
    *(_QWORD *)(v2 + 120) = 0;
    *(_QWORD *)(v2 + 144) = 0;
    *(_BYTE *)(v2 + 152) = 0;
    *(_QWORD *)(v2 + 560) = 0;
    *(_QWORD *)(v2 + 584) = 32;
    *(_DWORD *)(v2 + 592) = 0;
    v4 = sub_163A1D0(856, a2);
    sub_142AF50(v4);
  }
  return v3;
}
