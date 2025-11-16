// Function: sub_229CEA0
// Address: 0x229cea0
//
_QWORD *sub_229CEA0()
{
  __int64 v0; // rax
  _QWORD *v1; // r12
  __int128 *v2; // rax

  v0 = sub_22077B0(0xD0u);
  v1 = (_QWORD *)v0;
  if ( v0 )
  {
    *(_QWORD *)(v0 + 8) = 0;
    *(_QWORD *)(v0 + 16) = &unk_4FDB61C;
    *(_QWORD *)(v0 + 56) = v0 + 104;
    *(_QWORD *)(v0 + 112) = v0 + 160;
    *(_QWORD *)v0 = off_4A09030;
    *(_QWORD *)(v0 + 176) = v0 + 192;
    *(_DWORD *)(v0 + 88) = 1065353216;
    *(_DWORD *)(v0 + 144) = 1065353216;
    *(_DWORD *)(v0 + 24) = 2;
    *(_QWORD *)(v0 + 32) = 0;
    *(_QWORD *)(v0 + 40) = 0;
    *(_QWORD *)(v0 + 48) = 0;
    *(_QWORD *)(v0 + 64) = 1;
    *(_QWORD *)(v0 + 72) = 0;
    *(_QWORD *)(v0 + 80) = 0;
    *(_QWORD *)(v0 + 96) = 0;
    *(_QWORD *)(v0 + 104) = 0;
    *(_QWORD *)(v0 + 120) = 1;
    *(_QWORD *)(v0 + 128) = 0;
    *(_QWORD *)(v0 + 136) = 0;
    *(_QWORD *)(v0 + 152) = 0;
    *(_QWORD *)(v0 + 160) = 0;
    *(_BYTE *)(v0 + 168) = 0;
    sub_229AB90((__int64 *)(v0 + 176), "domonly", (__int64)"");
    *v1 = off_4A090E0;
    v2 = sub_BC2B00();
    sub_229BE00((__int64)v2);
  }
  return v1;
}
