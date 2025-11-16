// Function: sub_C3A020
// Address: 0xc3a020
//
__int64 __fastcall sub_C3A020(__int64 *a1, __int64 a2)
{
  char v2; // r12
  unsigned int v3; // r12d
  _QWORD v5[8]; // [rsp+0h] [rbp-40h] BYREF

  v2 = *(_BYTE *)(*a1 + 24);
  sub_C37380(v5, *a1);
  v3 = sub_C39C10((__int64)a1, a2, (__int64)v5, v2 ^ 1u);
  sub_C338F0((__int64)v5);
  return v3;
}
