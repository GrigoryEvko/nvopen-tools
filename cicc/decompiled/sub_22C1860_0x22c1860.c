// Function: sub_22C1860
// Address: 0x22c1860
//
void __fastcall sub_22C1860(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rax
  _QWORD v7[10]; // [rsp+0h] [rbp-50h] BYREF

  v6 = sub_22C1580(a1);
  if ( v6 )
  {
    v7[0] = off_4A09E60;
    v7[1] = v6;
    v7[2] = a3;
    sub_A68C30(a2, a4, (__int64)v7, 0, 0);
    v7[0] = off_4A09E60;
    nullsub_35();
  }
}
