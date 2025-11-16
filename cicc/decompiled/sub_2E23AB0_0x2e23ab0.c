// Function: sub_2E23AB0
// Address: 0x2e23ab0
//
__int64 __fastcall sub_2E23AB0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rax

  v4 = sub_2EB2140(a4, &unk_501EB00);
  sub_2E23880((_QWORD *)(v4 + 8), *a2);
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
