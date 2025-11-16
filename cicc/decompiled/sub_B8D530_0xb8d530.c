// Function: sub_B8D530
// Address: 0xb8d530
//
__int64 __fastcall sub_B8D530(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  _QWORD v8[6]; // [rsp+0h] [rbp-30h] BYREF

  v8[0] = sub_B9B140(a1, a2, a3);
  v8[1] = sub_B9B140(a1, a4, a5);
  return sub_B9C770(a1, v8, 2, 0, 1);
}
