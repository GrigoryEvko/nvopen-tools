// Function: sub_1095C00
// Address: 0x1095c00
//
__int64 __fastcall sub_1095C00(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rax

  *(_QWORD *)(a2 + 64) = a3;
  sub_2240AE0(a2 + 72, a4);
  v5 = *(_QWORD *)(a2 + 152);
  *(_QWORD *)(a1 + 8) = a3;
  *(_DWORD *)a1 = 1;
  *(_DWORD *)(a1 + 32) = 64;
  *(_QWORD *)(a1 + 16) = v5 - a3;
  *(_QWORD *)(a1 + 24) = 0;
  return a1;
}
