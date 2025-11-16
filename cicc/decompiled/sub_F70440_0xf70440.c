// Function: sub_F70440
// Address: 0xf70440
//
__int64 __fastcall sub_F70440(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r10
  _QWORD v6[2]; // [rsp+0h] [rbp-40h] BYREF
  _BYTE v7[32]; // [rsp+10h] [rbp-30h] BYREF
  __int16 v8; // [rsp+30h] [rbp-10h]

  v4 = *(_QWORD *)(a3 + 8);
  v6[0] = a4;
  v6[1] = a3;
  v8 = 257;
  return ((__int64 (__fastcall *)(__int64, __int64, __int64, _QWORD *, __int64, _BYTE *))sub_1061220)(
           a1,
           389,
           v4,
           v6,
           2,
           v7);
}
