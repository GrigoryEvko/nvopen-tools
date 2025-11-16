// Function: sub_2F7AD60
// Address: 0x2f7ad60
//
__int64 __fastcall sub_2F7AD60(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rax
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9

  v5 = sub_BC0510(a4, &unk_5024E68, a3);
  sub_2F7A520(v5 + 8, *a2, a3, v6, v7, v8);
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
