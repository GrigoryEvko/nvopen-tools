// Function: sub_10278D0
// Address: 0x10278d0
//
unsigned int __fastcall sub_10278D0(__int64 a1)
{
  __int64 v2; // rax
  _QWORD *v3; // rdi
  __int128 *v4; // rax

  *(_QWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 24) = 2;
  *(_QWORD *)(a1 + 16) = &unk_4F8EE48;
  *(_QWORD *)(a1 + 56) = a1 + 104;
  v2 = a1 + 160;
  v3 = (_QWORD *)(a1 + 176);
  *(v3 - 8) = v2;
  *(v3 - 18) = 0;
  *(v3 - 16) = 0;
  *(v3 - 17) = 0;
  *(v3 - 14) = 1;
  *(v3 - 13) = 0;
  *(v3 - 12) = 0;
  *(v3 - 10) = 0;
  *(v3 - 9) = 0;
  *(v3 - 7) = 1;
  *(v3 - 6) = 0;
  *(v3 - 5) = 0;
  *(v3 - 3) = 0;
  *(v3 - 2) = 0;
  *((_BYTE *)v3 - 8) = 0;
  *(v3 - 22) = &unk_49E5748;
  *((_DWORD *)v3 - 22) = 1065353216;
  *((_DWORD *)v3 - 8) = 1065353216;
  sub_FDC0F0(v3);
  *(_BYTE *)(a1 + 184) = 0;
  *(_QWORD *)(a1 + 192) = 0;
  *(_QWORD *)(a1 + 200) = 0;
  *(_QWORD *)(a1 + 208) = 0;
  v4 = sub_BC2B00();
  return sub_1027850((__int64)v4);
}
