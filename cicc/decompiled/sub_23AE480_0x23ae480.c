// Function: sub_23AE480
// Address: 0x23ae480
//
__int64 __fastcall sub_23AE480(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rdi
  _QWORD v6[2]; // [rsp+0h] [rbp-70h] BYREF
  _QWORD v7[4]; // [rsp+10h] [rbp-60h] BYREF
  char v8; // [rsp+30h] [rbp-40h]
  _QWORD v9[2]; // [rsp+38h] [rbp-38h] BYREF
  _QWORD v10[2]; // [rsp+48h] [rbp-28h] BYREF
  _QWORD v11[3]; // [rsp+58h] [rbp-18h] BYREF

  v4 = *(_QWORD *)(a1 + 40);
  v7[0] = "*** IR Pass {0} on {1} ignored ***\n";
  v7[2] = v11;
  v6[0] = a2;
  v6[1] = a3;
  v9[0] = &unk_49E6618;
  v7[1] = 35;
  v7[3] = 2;
  v10[0] = &unk_49DB108;
  v10[1] = v6;
  v11[0] = v10;
  v8 = 1;
  v9[1] = a4;
  v11[1] = v9;
  return sub_CB6840(v4, (__int64)v7);
}
