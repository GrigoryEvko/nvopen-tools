// Function: sub_392A760
// Address: 0x392a760
//
__int64 __fastcall sub_392A760(__int64 a1, _QWORD *a2, __int64 a3, unsigned __int64 *a4)
{
  __int64 v5; // rax

  a2[8] = a3;
  sub_2240AE0(a2 + 9, a4);
  v5 = a2[18];
  *(_QWORD *)(a1 + 8) = a3;
  *(_DWORD *)a1 = 1;
  *(_DWORD *)(a1 + 32) = 64;
  *(_QWORD *)(a1 + 16) = v5 - a3;
  *(_QWORD *)(a1 + 24) = 0;
  return a1;
}
