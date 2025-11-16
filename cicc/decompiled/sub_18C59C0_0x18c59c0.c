// Function: sub_18C59C0
// Address: 0x18c59c0
//
__int64 sub_18C59C0()
{
  __int64 v0; // rax
  __int64 v1; // r12
  bool v2; // zf

  v0 = sub_22077B0(2376);
  v1 = v0;
  if ( v0 )
  {
    *(_QWORD *)(v0 + 8) = 0;
    v2 = qword_4FADE28 == 0;
    *(_QWORD *)(v0 + 16) = &unk_4FADC8C;
    *(_QWORD *)(v0 + 80) = v0 + 64;
    *(_QWORD *)(v0 + 88) = v0 + 64;
    *(_QWORD *)(v0 + 128) = v0 + 112;
    *(_QWORD *)(v0 + 136) = v0 + 112;
    *(_QWORD *)v0 = off_49F2528;
    *(_QWORD *)(v0 + 160) = v0 + 176;
    *(_QWORD *)(v0 + 168) = 0x1000000000LL;
    *(_QWORD *)(v0 + 312) = v0 + 328;
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
    *(_BYTE *)(v0 + 304) = 0;
    *(_QWORD *)(v0 + 320) = 0x2000000000LL;
    if ( !v2 )
      sub_18C55D0(v0);
  }
  return v1;
}
