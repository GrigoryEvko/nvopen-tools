// Function: sub_22AA560
// Address: 0x22aa560
//
__int64 __fastcall sub_22AA560(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r13
  __int64 v7; // rax

  v6 = sub_BC0510(a4, &unk_4FDB698, a3);
  v7 = sub_BC0510(a4, &unk_4FDB6A0, a3);
  sub_22AA2C0(v6 + 8, *a2, v7 + 8, a3 + 312);
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
