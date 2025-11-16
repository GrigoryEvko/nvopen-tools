// Function: sub_13E6730
// Address: 0x13e6730
//
__int64 __fastcall sub_13E6730(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  _QWORD *v4; // rdi
  __int64 v5; // rax

  *(_QWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 24) = 3;
  *(_QWORD *)(a1 + 16) = &unk_4F99115;
  *(_QWORD *)(a1 + 80) = a1 + 64;
  *(_QWORD *)(a1 + 88) = a1 + 64;
  v3 = a1 + 112;
  v4 = (_QWORD *)(a1 + 160);
  *(v4 - 4) = v3;
  *(v4 - 3) = v3;
  *(v4 - 16) = 0;
  *(v4 - 15) = 0;
  *(v4 - 14) = 0;
  *((_DWORD *)v4 - 24) = 0;
  *(v4 - 11) = 0;
  *(v4 - 8) = 0;
  *((_DWORD *)v4 - 12) = 0;
  *(v4 - 5) = 0;
  *(v4 - 2) = 0;
  *((_BYTE *)v4 - 8) = 0;
  *(v4 - 20) = &unk_49EA7A0;
  sub_13689F0(v4);
  *(_BYTE *)(a1 + 168) = 0;
  *(_QWORD *)(a1 + 176) = 0;
  *(_QWORD *)(a1 + 184) = 0;
  *(_QWORD *)(a1 + 192) = 0;
  v5 = sub_163A1D0(v4, a2);
  return sub_13E6630(v5);
}
