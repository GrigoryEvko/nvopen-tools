// Function: sub_12A1BF0
// Address: 0x12a1bf0
//
__int64 __fastcall sub_12A1BF0(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  int v3; // r12d
  unsigned int v4; // eax
  __int64 v6[5]; // [rsp+8h] [rbp-28h] BYREF

  v2 = sub_12A0C10(a1, *(_QWORD *)(a2 + 160));
  v3 = -((*(_BYTE *)(a2 + 168) & 2) == 0);
  v6[0] = 0x10000000CLL;
  v4 = sub_127B390();
  return sub_15A5B30(a1 + 16, (v3 & 0xFFFFFFCE) + 66, v2, v4, 0, v6);
}
