// Function: sub_14C9BB0
// Address: 0x14c9bb0
//
__int64 __fastcall sub_14C9BB0(__int64 a1, _DWORD *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // [rsp+0h] [rbp-20h] BYREF
  __int64 v8; // [rsp+8h] [rbp-18h] BYREF
  _QWORD v9[2]; // [rsp+10h] [rbp-10h] BYREF

  v8 = a4;
  v7 = a5;
  v9[0] = &v8;
  v9[1] = &v7;
  return sub_14C9920(
           a1,
           a2,
           a3,
           a6,
           (unsigned __int8 (__fastcall *)(__int64, __int64, __int64, __int64, __int64))sub_14C8370,
           (__int64)v9);
}
