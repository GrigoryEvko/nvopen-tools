// Function: sub_2EB3FB0
// Address: 0x2eb3fb0
//
unsigned int __fastcall sub_2EB3FB0(_QWORD *a1)
{
  _QWORD *v1; // rax
  _QWORD *v2; // rdi
  __int128 *v3; // rax

  a1[1] = 0;
  a1[2] = &unk_50209DC;
  a1[7] = a1 + 13;
  v1 = a1 + 20;
  v2 = a1 + 25;
  *(v2 - 11) = v1;
  *((_DWORD *)v2 - 44) = 2;
  *((_DWORD *)v2 - 28) = 1065353216;
  *(v2 - 25) = &unk_4A298B8;
  *(v2 - 21) = 0;
  *(v2 - 20) = 0;
  *(v2 - 19) = 0;
  *(v2 - 17) = 1;
  *(v2 - 16) = 0;
  *(v2 - 15) = 0;
  *(v2 - 13) = 0;
  *(v2 - 12) = 0;
  *(v2 - 10) = 1;
  *(v2 - 9) = 0;
  *(v2 - 8) = 0;
  *((_DWORD *)v2 - 14) = 1065353216;
  *(v2 - 6) = 0;
  *(v2 - 5) = 0;
  *((_BYTE *)v2 - 32) = 0;
  *(v2 - 3) = 0;
  *(v2 - 2) = 0;
  *(v2 - 1) = 0;
  memset(v2, 0, 0xA0u);
  v3 = sub_BC2B00();
  return sub_2EB3F30((__int64)v3);
}
