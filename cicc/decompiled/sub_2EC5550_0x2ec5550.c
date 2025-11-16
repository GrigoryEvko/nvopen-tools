// Function: sub_2EC5550
// Address: 0x2ec5550
//
_QWORD *sub_2EC5550()
{
  __int64 v0; // rax
  _QWORD *v1; // r12
  __int128 *v2; // rax

  v0 = sub_22077B0(0x118u);
  v1 = (_QWORD *)v0;
  if ( v0 )
  {
    *(_QWORD *)(v0 + 8) = 0;
    *(_QWORD *)(v0 + 16) = &unk_5021074;
    *(_QWORD *)(v0 + 56) = v0 + 104;
    *(_QWORD *)(v0 + 112) = v0 + 160;
    *(_QWORD *)v0 = off_4A29AF0;
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
    *(_QWORD *)(v0 + 176) = 0;
    *(_QWORD *)(v0 + 184) = 0;
    *(_QWORD *)(v0 + 192) = 0;
    sub_2EC5460((_QWORD *)(v0 + 200));
    v1[33] = 0;
    v1[34] = 0;
    v1[25] = &unk_4A29AB0;
    v2 = sub_BC2B00();
    sub_2EC54D0((__int64)v2);
  }
  return v1;
}
