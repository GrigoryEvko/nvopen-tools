// Function: sub_2EAA600
// Address: 0x2eaa600
//
unsigned int __fastcall sub_2EAA600(_QWORD *a1, __int64 *a2)
{
  _QWORD *v2; // rax
  __int64 v3; // rdi
  __int128 *v4; // rax

  a1[1] = 0;
  a1[2] = &unk_50208C0;
  a1[7] = a1 + 13;
  v2 = a1 + 20;
  v3 = (__int64)(a1 + 22);
  *(_QWORD *)(v3 - 64) = v2;
  *(_DWORD *)(v3 - 152) = 4;
  *(_QWORD *)(v3 - 128) = 0;
  *(_QWORD *)(v3 - 144) = 0;
  *(_QWORD *)(v3 - 136) = 0;
  *(_QWORD *)(v3 - 112) = 1;
  *(_QWORD *)(v3 - 104) = 0;
  *(_QWORD *)(v3 - 96) = 0;
  *(_QWORD *)(v3 - 80) = 0;
  *(_QWORD *)(v3 - 72) = 0;
  *(_QWORD *)(v3 - 56) = 1;
  *(_QWORD *)(v3 - 48) = 0;
  *(_QWORD *)(v3 - 40) = 0;
  *(_QWORD *)(v3 - 24) = 0;
  *(_QWORD *)(v3 - 16) = 0;
  *(_BYTE *)(v3 - 8) = 0;
  *(_QWORD *)(v3 - 176) = &unk_4A296D0;
  *(_DWORD *)(v3 - 88) = 1065353216;
  *(_DWORD *)(v3 - 32) = 1065353216;
  sub_2EAA0B0(v3, a2);
  v4 = sub_BC2B00();
  return sub_2EAA580((__int64)v4);
}
