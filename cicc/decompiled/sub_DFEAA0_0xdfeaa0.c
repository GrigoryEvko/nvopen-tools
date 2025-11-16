// Function: sub_DFEAA0
// Address: 0xdfeaa0
//
unsigned int __fastcall sub_DFEAA0(__int64 a1)
{
  __int64 v2; // rax
  _QWORD *v3; // rdi
  __int128 *v4; // rax

  *(_QWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 24) = 4;
  *(_QWORD *)(a1 + 16) = &unk_4F89C28;
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
  *(v3 - 22) = &unk_49DEC00;
  *((_DWORD *)v3 - 22) = 1065353216;
  *((_DWORD *)v3 - 8) = 1065353216;
  sub_DFE980(v3);
  *(_BYTE *)(a1 + 216) = 0;
  v4 = sub_BC2B00();
  return sub_DFEA20((__int64)v4);
}
