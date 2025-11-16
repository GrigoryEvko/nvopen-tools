// Function: sub_FD5690
// Address: 0xfd5690
//
__int64 __fastcall sub_FD5690(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rbx
  __int64 v7; // rax

  v6 = sub_BC1CD0(a4, &unk_4F81450, a3);
  v7 = sub_BC1CD0(a4, &unk_4F875F0, a3);
  *(_QWORD *)a1 = a3;
  *(_QWORD *)(a1 + 16) = v6 + 8;
  *(_QWORD *)(a1 + 8) = v7 + 8;
  *(_QWORD *)(a1 + 184) = a1 + 208;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_BYTE *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_DWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 88) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 104) = 0;
  *(_QWORD *)(a1 + 112) = 0;
  *(_QWORD *)(a1 + 120) = 0;
  *(_QWORD *)(a1 + 128) = 0;
  *(_DWORD *)(a1 + 136) = 0;
  *(_QWORD *)(a1 + 144) = 0;
  *(_QWORD *)(a1 + 152) = 0;
  *(_QWORD *)(a1 + 160) = 0;
  *(_DWORD *)(a1 + 168) = 0;
  *(_QWORD *)(a1 + 176) = 0;
  *(_QWORD *)(a1 + 192) = 8;
  *(_DWORD *)(a1 + 200) = 0;
  *(_BYTE *)(a1 + 204) = 1;
  sub_CE6510((__int64 *)a1);
  return a1;
}
