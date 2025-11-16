// Function: sub_1368F50
// Address: 0x1368f50
//
__int64 __fastcall sub_1368F50(_QWORD *a1, __int64 a2)
{
  _QWORD *v2; // rax
  _QWORD *v3; // rdi
  __int64 v4; // rax

  a1[1] = 0;
  a1[2] = &unk_4F97E48;
  a1[10] = a1 + 8;
  a1[11] = a1 + 8;
  v2 = a1 + 14;
  v3 = a1 + 20;
  *(v3 - 4) = v2;
  *(v3 - 3) = v2;
  *((_DWORD *)v3 - 34) = 3;
  *(v3 - 16) = 0;
  *(v3 - 15) = 0;
  *(v3 - 14) = 0;
  *((_DWORD *)v3 - 24) = 0;
  *(v3 - 11) = 0;
  *(v3 - 8) = 0;
  *((_DWORD *)v3 - 12) = 0;
  *(v3 - 5) = 0;
  *(v3 - 2) = 0;
  *((_BYTE *)v3 - 8) = 0;
  *(v3 - 20) = &unk_49E89A8;
  sub_13689F0(v3);
  v4 = sub_163A1D0(v3, a2);
  return sub_1368E50(v4);
}
