// Function: sub_AF2E80
// Address: 0xaf2e80
//
__int64 __fastcall sub_AF2E80(__int64 a1, int a2, int a3, int a4, char a5, __int64 a6, __int64 a7, __int64 a8)
{
  sub_B971C0(a1, a2, 15, a3, a7, a8, 0, 0);
  *(_DWORD *)(a1 + 20) = a4;
  *(_BYTE *)(a1 + 44) = a5;
  *(_WORD *)(a1 + 2) = 21;
  *(_DWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  *(_DWORD *)(a1 + 4) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_DWORD *)(a1 + 40) = 0;
  return 21;
}
