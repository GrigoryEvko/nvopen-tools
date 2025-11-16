// Function: sub_17F2180
// Address: 0x17f2180
//
__int64 __fastcall sub_17F2180(__int64 a1, __int64 *a2)
{
  char v3; // r8
  __int64 result; // rax
  __int64 v5; // r9
  __int64 v6; // rsi
  __int64 v7; // [rsp+0h] [rbp-20h] BYREF
  __int64 v8[3]; // [rsp+8h] [rbp-18h] BYREF

  v3 = sub_1636800(a1, a2);
  result = 0;
  if ( !v3 )
  {
    v5 = *(_QWORD *)(a1 + 168);
    v6 = *(_QWORD *)(a1 + 160);
    v7 = a1;
    v8[0] = a1;
    return sub_17EEF60(
             (__int64)a2,
             v6,
             v5,
             (__int64 (__fastcall *)(__int64, __int64 *))sub_17E2420,
             (__int64)&v7,
             v5,
             (__int64 (__fastcall *)(__int64, __int64 *))sub_17E23E0,
             (__int64)v8);
  }
  return result;
}
