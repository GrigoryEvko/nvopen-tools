// Function: sub_169B620
// Address: 0x169b620
//
__int64 __fastcall sub_169B620(__int64 a1, char a2)
{
  __int64 v2; // rax
  unsigned int v3; // r13d
  __int64 v4; // rax

  v2 = *(_QWORD *)a1;
  *(_BYTE *)(a1 + 18) = *(_BYTE *)(a1 + 18) & 0xF0 | (8 * a2 + 3) & 0xF;
  *(_WORD *)(a1 + 16) = *(_WORD *)(v2 + 2) - 1;
  v3 = sub_1698310(a1);
  v4 = sub_1698470(a1);
  return sub_16A7020(v4, 0, v3);
}
