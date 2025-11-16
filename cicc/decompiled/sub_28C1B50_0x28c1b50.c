// Function: sub_28C1B50
// Address: 0x28c1b50
//
__int64 sub_28C1B50()
{
  __int64 v0; // rax
  __int64 v1; // r12
  __int128 *v2; // rax

  v0 = sub_22077B0(0x100u);
  v1 = v0;
  if ( v0 )
  {
    *(_QWORD *)(v0 + 8) = 0;
    *(_QWORD *)(v0 + 16) = &unk_50044B4;
    *(_QWORD *)(v0 + 56) = v0 + 104;
    *(_QWORD *)(v0 + 112) = v0 + 160;
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
    *(_QWORD *)v0 = off_4A21840;
    *(_QWORD *)(v0 + 224) = 0;
    *(_QWORD *)(v0 + 232) = 0;
    *(_QWORD *)(v0 + 240) = 0;
    *(_DWORD *)(v0 + 248) = 0;
    *(_DWORD *)(v0 + 88) = 1065353216;
    *(_DWORD *)(v0 + 144) = 1065353216;
    v2 = sub_BC2B00();
    sub_28C1980((__int64)v2);
  }
  return v1;
}
