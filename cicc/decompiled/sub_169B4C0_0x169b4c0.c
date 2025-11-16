// Function: sub_169B4C0
// Address: 0x169b4c0
//
__int64 __fastcall sub_169B4C0(__int64 a1, char a2)
{
  unsigned int v2; // r13d
  __int64 v3; // rax

  *(_BYTE *)(a1 + 18) = (8 * a2) & 0xF | *(_BYTE *)(a1 + 18) & 0xF0;
  *(_WORD *)(a1 + 16) = **(_WORD **)a1 + 1;
  v2 = sub_1698310(a1);
  v3 = sub_1698470(a1);
  return sub_16A7020(v3, 0, v2);
}
