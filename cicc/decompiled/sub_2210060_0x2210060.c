// Function: sub_2210060
// Address: 0x2210060
//
__int64 __fastcall sub_2210060(
        __int64 a1,
        __int64 a2,
        unsigned __int16 *a3,
        unsigned __int16 *a4,
        unsigned __int16 **a5,
        __int64 a6,
        __int64 a7,
        _QWORD *a8)
{
  __int64 result; // rax
  __int64 v10; // rcx
  unsigned __int16 *v11[2]; // [rsp+0h] [rbp-28h] BYREF
  _QWORD v12[3]; // [rsp+10h] [rbp-18h] BYREF

  v11[0] = a3;
  v11[1] = a4;
  v12[0] = a6;
  v12[1] = a7;
  result = sub_220FFA0(v11, (__int64)v12, 0x10FFFFu, 0);
  v10 = v12[0];
  *a5 = v11[0];
  *a8 = v10;
  return result;
}
