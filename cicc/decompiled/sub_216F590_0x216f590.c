// Function: sub_216F590
// Address: 0x216f590
//
__int64 sub_216F590()
{
  __int64 v0; // rax
  __int64 v1; // r12
  bool v2; // cf
  __int64 v3; // rax

  v0 = sub_22077B0(184);
  v1 = v0;
  if ( v0 )
  {
    *(_QWORD *)(v0 + 8) = 0;
    v2 = (unsigned int)dword_4FD2A20 < 0x1E;
    *(_QWORD *)(v0 + 16) = &unk_4FD2970;
    *(_QWORD *)(v0 + 80) = v0 + 64;
    *(_QWORD *)(v0 + 88) = v0 + 64;
    *(_QWORD *)(v0 + 128) = v0 + 112;
    *(_QWORD *)(v0 + 136) = v0 + 112;
    *(_QWORD *)v0 = off_4A03128;
    *(_QWORD *)(v0 + 160) = 0x4000000400LL;
    *(_DWORD *)(v0 + 24) = 3;
    *(_QWORD *)(v0 + 32) = 0;
    *(_DWORD *)(v0 + 168) = v2 ? 0xFFFF : 0x7FFFFFFF;
    *(_QWORD *)(v0 + 40) = 0;
    *(_QWORD *)(v0 + 48) = 0;
    *(_DWORD *)(v0 + 64) = 0;
    *(_QWORD *)(v0 + 72) = 0;
    *(_QWORD *)(v0 + 96) = 0;
    *(_DWORD *)(v0 + 112) = 0;
    *(_QWORD *)(v0 + 120) = 0;
    *(_QWORD *)(v0 + 144) = 0;
    *(_BYTE *)(v0 + 152) = 0;
    *(_DWORD *)(v0 + 156) = 1024;
    *(_QWORD *)(v0 + 172) = 0xFFFF0000FFFFLL;
    v3 = sub_163A1D0();
    sub_216F4B0(v3);
  }
  return v1;
}
