// Function: sub_2E092F0
// Address: 0x2e092f0
//
__int64 __fastcall sub_2E092F0(__int64 *a1, __int64 a2)
{
  __int64 v2; // rax
  _QWORD v4[4]; // [rsp+0h] [rbp-40h] BYREF
  char v5; // [rsp+20h] [rbp-20h]
  _QWORD v6[2]; // [rsp+28h] [rbp-18h] BYREF
  _QWORD *v7; // [rsp+38h] [rbp-8h] BYREF

  v2 = *a1;
  v4[0] = "{0}";
  v4[2] = &v7;
  v6[1] = v2;
  v4[1] = 3;
  v4[3] = 1;
  v5 = 1;
  v6[0] = &unk_4A283E0;
  v7 = v6;
  return sub_CB6840(a2, (__int64)v4);
}
