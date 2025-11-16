// Function: sub_AF1680
// Address: 0xaf1680
//
__int64 __fastcall sub_AF1680(__int64 a1, int a2, int a3, int a4, __int16 a5, char a6, __int64 a7, __int64 a8)
{
  char v10; // r12
  char v11; // r9
  __int64 v13; // [rsp-10h] [rbp-30h]

  v10 = a6 << 7;
  sub_B971C0(a1, a2, 6, a3, a7, a8, 0, 0);
  v11 = *(_BYTE *)(a1 + 1);
  *(_DWORD *)(a1 + 4) = a4;
  *(_WORD *)(a1 + 2) = a5;
  *(_BYTE *)(a1 + 1) = v10 | v11 & 0x7F;
  return v13;
}
