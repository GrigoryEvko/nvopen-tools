// Function: sub_2D50DC0
// Address: 0x2d50dc0
//
__int64 sub_2D50DC0()
{
  __int64 v0; // rax
  __int64 v1; // r12
  __int128 *v2; // rax

  v0 = sub_22077B0(0x140u);
  v1 = v0;
  if ( v0 )
  {
    *(_QWORD *)(v0 + 8) = 0;
    *(_DWORD *)(v0 + 24) = 4;
    *(_QWORD *)(v0 + 16) = &unk_501695C;
    *(_QWORD *)(v0 + 56) = v0 + 104;
    *(_QWORD *)(v0 + 112) = v0 + 160;
    *(_DWORD *)(v0 + 88) = 1065353216;
    *(_DWORD *)(v0 + 144) = 1065353216;
    *(_QWORD *)v0 = &unk_4A26438;
    *(_WORD *)(v0 + 224) = 256;
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
    *(_QWORD *)(v0 + 176) = 0;
    *(_BYTE *)(v0 + 216) = 0;
    *(_DWORD *)(v0 + 228) = 1;
    *(_QWORD *)(v0 + 232) = 0;
    *(_QWORD *)(v0 + 240) = 0;
    *(_QWORD *)(v0 + 248) = 0;
    *(_QWORD *)(v0 + 256) = 0;
    *(_QWORD *)(v0 + 264) = 0xA000000000LL;
    *(_QWORD *)(v0 + 288) = 0x9800000000LL;
    *(_QWORD *)(v0 + 272) = 0;
    *(_QWORD *)(v0 + 280) = 0;
    *(_QWORD *)(v0 + 296) = 0;
    *(_QWORD *)(v0 + 304) = 0;
    *(_QWORD *)(v0 + 312) = 0x1800000000LL;
    v2 = sub_BC2B00();
    sub_2D50D40((__int64)v2);
  }
  return v1;
}
