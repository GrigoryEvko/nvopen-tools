// Function: sub_CF9770
// Address: 0xcf9770
//
__int64 __fastcall sub_CF9770(__int64 a1, __int64 a2, _DWORD *a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 v8; // [rsp+0h] [rbp-30h] BYREF
  __int64 v9; // [rsp+8h] [rbp-28h] BYREF
  _QWORD v10[3]; // [rsp+10h] [rbp-20h] BYREF

  v9 = a5;
  v8 = a6;
  v10[0] = &v9;
  v10[1] = &v8;
  sub_CF9460(
    a1,
    a2,
    a3,
    a4,
    a7,
    (__int64)v10,
    (unsigned __int8 (__fastcall *)(__int64, __int64, unsigned __int64, __int64, __int64, __int64, __int64, __int64, __int64))sub_CF8CC0,
    (__int64)v10);
  return a1;
}
