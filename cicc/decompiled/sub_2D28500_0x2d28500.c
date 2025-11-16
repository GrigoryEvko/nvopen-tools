// Function: sub_2D28500
// Address: 0x2d28500
//
__int64 __fastcall sub_2D28500(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rax

  v5 = sub_BC1CD0(a4, &unk_50165D8, a3);
  sub_2D27C30((__int64 *)(v5 + 8), *a2, a3);
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
