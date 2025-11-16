// Function: sub_104DD10
// Address: 0x104dd10
//
void __fastcall sub_104DD10(__int64 *a1, __int64 a2)
{
  __int64 v2; // rdi
  _QWORD v3[4]; // [rsp+0h] [rbp-20h] BYREF

  v3[1] = a1;
  v2 = *a1;
  v3[0] = &unk_49E5D10;
  sub_A68C30(v2, a2, (__int64)v3, 0, 0);
  v3[0] = &unk_49E5D10;
  nullsub_35();
}
