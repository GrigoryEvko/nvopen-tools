// Function: sub_2DD6C10
// Address: 0x2dd6c10
//
__int64 sub_2DD6C10()
{
  __int64 v0; // rax
  __int64 v1; // r12
  int v2; // eax
  __int128 *v3; // rax

  v0 = sub_22077B0(0xC8u);
  v1 = v0;
  if ( v0 )
  {
    *(_QWORD *)(v0 + 8) = 0;
    *(_QWORD *)(v0 + 16) = &unk_501DA30;
    *(_QWORD *)(v0 + 56) = v0 + 104;
    *(_QWORD *)(v0 + 112) = v0 + 160;
    *(_QWORD *)v0 = off_4A27DB8;
    *(_QWORD *)(v0 + 188) = 0x100010100000000LL;
    *(_WORD *)(v0 + 196) = 0;
    v2 = qword_501E008;
    *(_DWORD *)(v1 + 24) = 2;
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
    *(_QWORD *)(v1 + 176) = 0;
    *(_BYTE *)(v1 + 198) = 0;
    *(_DWORD *)(v1 + 184) = v2;
    *(_DWORD *)(v1 + 88) = 1065353216;
    *(_DWORD *)(v1 + 144) = 1065353216;
    v3 = sub_BC2B00();
    sub_2DD6B90((__int64)v3);
  }
  return v1;
}
