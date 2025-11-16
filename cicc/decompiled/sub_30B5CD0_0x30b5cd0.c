// Function: sub_30B5CD0
// Address: 0x30b5cd0
//
__int64 __fastcall sub_30B5CD0(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5)
{
  char v5; // r13
  __int64 v6; // rax

  v5 = qword_502ED48;
  v6 = sub_22D3D20(a4, qword_502E9E0, a3, a5);
  sub_30B5770(*(_QWORD *)(v6 + 8), v5);
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  *(_QWORD *)(a1 + 64) = 2;
  *(_QWORD *)(a1 + 32) = &qword_4F82400;
  *(_DWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 76) = 1;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)a1 = 1;
  return a1;
}
