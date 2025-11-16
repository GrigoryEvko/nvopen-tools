// Function: sub_AF3F40
// Address: 0xaf3f40
//
__int64 __fastcall sub_AF3F40(__int64 a1, int a2, int a3, char a4, int a5, int a6)
{
  char v6; // r12
  char v7; // cl

  v6 = a4 << 7;
  sub_B971C0(a1, a2, 23, a3, a5, a6, 0, 0);
  v7 = *(_BYTE *)(a1 + 1);
  *(_WORD *)(a1 + 2) = 47;
  *(_BYTE *)(a1 + 1) = v6 | v7 & 0x7F;
  return 47;
}
