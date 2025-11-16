// Function: sub_23AE670
// Address: 0x23ae670
//
__int64 __fastcall sub_23AE670(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rdi
  _QWORD v5[2]; // [rsp+0h] [rbp-50h] BYREF
  _QWORD v6[4]; // [rsp+10h] [rbp-40h] BYREF
  char v7; // [rsp+30h] [rbp-20h]
  _QWORD v8[2]; // [rsp+38h] [rbp-18h] BYREF
  _QWORD *v9; // [rsp+48h] [rbp-8h] BYREF

  v3 = *(_QWORD *)(a1 + 40);
  v6[0] = "*** IR Pass {0} invalidated ***\n";
  v6[2] = &v9;
  v5[0] = a2;
  v5[1] = a3;
  v8[0] = &unk_49DB108;
  v8[1] = v5;
  v6[1] = 32;
  v6[3] = 1;
  v7 = 1;
  v9 = v8;
  return sub_CB6840(v3, (__int64)v6);
}
