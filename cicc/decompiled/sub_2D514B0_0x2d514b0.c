// Function: sub_2D514B0
// Address: 0x2d514b0
//
_QWORD *__fastcall sub_2D514B0(__int64 a1)
{
  __int64 v1; // rax
  _QWORD *v2; // r12
  __int128 *v3; // rax

  v1 = sub_22077B0(0x140u);
  v2 = (_QWORD *)v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 0;
    *(_DWORD *)(v1 + 24) = 4;
    *(_QWORD *)(v1 + 16) = &unk_501695C;
    *(_QWORD *)(v1 + 56) = v1 + 104;
    *(_QWORD *)(v1 + 112) = v1 + 160;
    *(_DWORD *)(v1 + 88) = 1065353216;
    *(_DWORD *)(v1 + 144) = 1065353216;
    *(_QWORD *)v1 = &unk_4A26438;
    *(_QWORD *)(v1 + 32) = 0;
    *(_QWORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 48) = 0;
    *(_QWORD *)(v1 + 64) = 1;
    *(_QWORD *)(v1 + 72) = 0;
    *(_QWORD *)(v1 + 80) = 0;
    *(_QWORD *)(v1 + 96) = 0;
    *(_QWORD *)(v1 + 104) = 0;
    *(_QWORD *)(v1 + 120) = 1;
    *(_QWORD *)(v1 + 128) = 0;
    *(_QWORD *)(v1 + 136) = 0;
    *(_QWORD *)(v1 + 152) = 0;
    *(_QWORD *)(v1 + 160) = 0;
    *(_BYTE *)(v1 + 168) = 0;
    *(_QWORD *)(v1 + 176) = a1;
    sub_C7C840(v1 + 184, a1, 1, 35);
    v2[31] = 0;
    v2[33] = 0xA000000000LL;
    v2[36] = 0x9800000000LL;
    v2[32] = 0;
    v2[34] = 0;
    v2[35] = 0;
    v2[37] = 0;
    v2[38] = 0;
    v2[39] = 0x1800000000LL;
    v3 = sub_BC2B00();
    sub_2D50D40((__int64)v3);
  }
  return v2;
}
