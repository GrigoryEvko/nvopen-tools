// Function: sub_1B2BDA0
// Address: 0x1b2bda0
//
void __fastcall sub_1B2BDA0(__int64 a1, __int64 a2)
{
  __int64 v2; // rdi
  _QWORD v3[4]; // [rsp+0h] [rbp-20h] BYREF

  v3[1] = a1;
  v2 = *(_QWORD *)(a1 + 16);
  v3[0] = &unk_49F6828;
  sub_1559E80(v2, a2, (__int64)v3, 0, 0);
  v3[0] = &unk_49F6828;
  nullsub_544();
}
