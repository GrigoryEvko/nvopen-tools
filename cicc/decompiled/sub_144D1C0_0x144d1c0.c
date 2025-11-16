// Function: sub_144D1C0
// Address: 0x144d1c0
//
_QWORD *sub_144D1C0()
{
  __int64 v0; // rax
  _QWORD *v1; // r12
  __int64 v2; // rdi
  __int64 v3; // rax

  v0 = sub_22077B0(192);
  v1 = (_QWORD *)v0;
  if ( v0 )
  {
    *(_QWORD *)(v0 + 8) = 0;
    *(_QWORD *)(v0 + 16) = &unk_4F9A3A2;
    v2 = v0 + 160;
    *(_QWORD *)(v0 + 80) = v0 + 64;
    *(_QWORD *)(v0 + 88) = v0 + 64;
    *(_QWORD *)(v0 + 128) = v0 + 112;
    *(_QWORD *)(v0 + 136) = v0 + 112;
    *(_QWORD *)v0 = off_49EC088;
    *(_QWORD *)(v0 + 160) = v0 + 176;
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
    *(_BYTE *)(v0 + 152) = 0;
    sub_144C6E0((__int64 *)(v0 + 160), "reg", (__int64)"");
    *v1 = off_49EC138;
    v3 = sub_163A1D0(v2, "reg");
    sub_144D0D0(v3);
  }
  return v1;
}
