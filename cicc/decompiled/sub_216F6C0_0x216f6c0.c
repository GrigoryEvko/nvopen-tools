// Function: sub_216F6C0
// Address: 0x216f6c0
//
__int64 __fastcall sub_216F6C0(unsigned int a1)
{
  __int64 v1; // rax
  __int64 v2; // r12
  __int64 v3; // rax

  v1 = sub_22077B0(184);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 0;
    *(_QWORD *)(v1 + 16) = &unk_4FD2970;
    *(_QWORD *)(v1 + 80) = v1 + 64;
    *(_QWORD *)(v1 + 88) = v1 + 64;
    *(_QWORD *)(v1 + 128) = v1 + 112;
    *(_QWORD *)(v1 + 136) = v1 + 112;
    *(_QWORD *)v1 = off_4A03128;
    *(_QWORD *)(v1 + 160) = 0x4000000400LL;
    *(_DWORD *)(v1 + 24) = 3;
    *(_QWORD *)(v1 + 32) = 0;
    *(_DWORD *)(v1 + 168) = a1 < 0x1E ? 0xFFFF : 0x7FFFFFFF;
    *(_QWORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 48) = 0;
    *(_DWORD *)(v1 + 64) = 0;
    *(_QWORD *)(v1 + 72) = 0;
    *(_QWORD *)(v1 + 96) = 0;
    *(_DWORD *)(v1 + 112) = 0;
    *(_QWORD *)(v1 + 120) = 0;
    *(_QWORD *)(v1 + 144) = 0;
    *(_BYTE *)(v1 + 152) = 0;
    *(_DWORD *)(v1 + 156) = 1024;
    *(_QWORD *)(v1 + 172) = 0xFFFF0000FFFFLL;
    v3 = sub_163A1D0();
    sub_216F4B0(v3);
  }
  return v2;
}
