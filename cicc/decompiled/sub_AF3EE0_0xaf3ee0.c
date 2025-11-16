// Function: sub_AF3EE0
// Address: 0xaf3ee0
//
__int64 __fastcall sub_AF3EE0(__int64 a1, int a2, int a3, int a4, char a5, __int64 a6, __int64 a7, __int64 a8)
{
  char v9; // r12
  char v10; // r8

  v9 = a5 << 7;
  sub_B971C0(a1, a2, 22, a3, a7, a8, 0, 0);
  v10 = *(_BYTE *)(a1 + 1);
  *(_DWORD *)(a1 + 4) = a4;
  *(_WORD *)(a1 + 2) = 30;
  *(_BYTE *)(a1 + 1) = v9 | v10 & 0x7F;
  return 30;
}
