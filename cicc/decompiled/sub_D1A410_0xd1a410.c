// Function: sub_D1A410
// Address: 0xd1a410
//
__int64 __fastcall sub_D1A410(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rax

  v4 = sub_BC1CD0(a4, &unk_4F86B68, a3);
  sub_D19E40(v4 + 8, *a2);
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  *(_QWORD *)(a1 + 64) = 2;
  *(_QWORD *)(a1 + 32) = &unk_4F82400;
  *(_DWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 76) = 1;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)a1 = 1;
  return a1;
}
