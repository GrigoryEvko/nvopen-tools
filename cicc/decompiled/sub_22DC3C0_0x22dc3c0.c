// Function: sub_22DC3C0
// Address: 0x22dc3c0
//
unsigned int __fastcall sub_22DC3C0(_QWORD *a1)
{
  _QWORD *v1; // rax
  _QWORD *v2; // rdi
  __int128 *v3; // rax

  a1[1] = 0;
  a1[2] = &unk_4FDBD0C;
  a1[7] = a1 + 13;
  v1 = a1 + 20;
  v2 = a1 + 22;
  *(v2 - 8) = v1;
  *((_DWORD *)v2 - 38) = 2;
  *(v2 - 16) = 0;
  *(v2 - 18) = 0;
  *(v2 - 17) = 0;
  *(v2 - 14) = 1;
  *(v2 - 13) = 0;
  *(v2 - 12) = 0;
  *(v2 - 10) = 0;
  *(v2 - 9) = 0;
  *(v2 - 7) = 1;
  *(v2 - 6) = 0;
  *(v2 - 5) = 0;
  *(v2 - 3) = 0;
  *(v2 - 2) = 0;
  *((_BYTE *)v2 - 8) = 0;
  *(v2 - 22) = &unk_4A0A0E8;
  *((_DWORD *)v2 - 22) = 1065353216;
  *((_DWORD *)v2 - 8) = 1065353216;
  sub_22DC080(v2);
  v3 = sub_BC2B00();
  return sub_22DC340((__int64)v3);
}
