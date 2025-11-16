// Function: sub_BA8D60
// Address: 0xba8d60
//
_BYTE *__fastcall sub_BA8D60(__int64 a1, __int64 a2, unsigned __int64 a3, __int64 a4)
{
  __int64 v5; // [rsp+8h] [rbp-38h] BYREF
  _QWORD v6[2]; // [rsp+10h] [rbp-30h] BYREF
  _QWORD v7[4]; // [rsp+20h] [rbp-20h] BYREF

  v6[0] = a2;
  v7[1] = &v5;
  v6[1] = a3;
  v5 = a4;
  v7[0] = a1;
  v7[2] = v6;
  return sub_BA8D20(a1, a2, a3, a4, sub_BA8310, (__int64)v7);
}
