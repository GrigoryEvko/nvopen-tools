// Function: sub_1100BD0
// Address: 0x1100bd0
//
unsigned __int8 *__fastcall sub_1100BD0(__m128i *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 *a6)
{
  unsigned __int8 *v6; // r12
  __int64 v8; // rdx
  __int64 v9; // rsi
  __int64 v10; // r12
  _BYTE v11[32]; // [rsp+0h] [rbp-50h] BYREF
  __int16 v12; // [rsp+20h] [rbp-30h]

  v6 = sub_11005E0(a1, (unsigned __int8 *)a2, a3, a4, a5, a6);
  if ( v6 || !(unsigned __int8)sub_9AC470(*(_QWORD *)(a2 - 32), a1 + 6, 0) )
    return v6;
  v8 = *(_QWORD *)(a2 + 8);
  v9 = *(_QWORD *)(a2 - 32);
  v12 = 257;
  v10 = sub_B51D30(43, v9, v8, (__int64)v11, 0, 0);
  sub_B448D0(v10, 1);
  return (unsigned __int8 *)v10;
}
