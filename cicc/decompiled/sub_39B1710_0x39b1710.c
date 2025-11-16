// Function: sub_39B1710
// Address: 0x39b1710
//
__int64 __fastcall sub_39B1710(__int64 a1, __int64 a2, __int64 a3, __int64 a4, char a5, __int64 a6, unsigned int a7)
{
  __int64 v8; // r11
  __int64 v9; // rsi
  __int64 (__fastcall *v10)(__int64, __int64, __int64, __int64, __int64); // r9
  __int64 v12; // [rsp+0h] [rbp-20h] BYREF
  __int64 v13; // [rsp+8h] [rbp-18h]
  int v14; // [rsp+10h] [rbp-10h]
  __int64 v15; // [rsp+18h] [rbp-8h]

  v8 = *(_QWORD *)(a1 + 24);
  v9 = *(_QWORD *)(a1 + 8);
  v12 = a3;
  v15 = a6;
  v13 = a4;
  LOBYTE(v14) = a5;
  v10 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)v8 + 744LL);
  if ( v10 == sub_1F3D840 )
    return -((*(unsigned __int8 (__fastcall **)(__int64, __int64, __int64 *, __int64, _QWORD, _QWORD, __int64, __int64, int, __int64))(*(_QWORD *)v8 + 736LL))(
               v8,
               v9,
               &v12,
               a2,
               a7,
               0,
               v12,
               v13,
               v14,
               v15)
           ^ 1);
  else
    return ((__int64 (__fastcall *)(__int64, __int64, __int64 *, __int64, _QWORD, __int64 (__fastcall *)(__int64, __int64, __int64, __int64, __int64), __int64, __int64, int, __int64))v10)(
             v8,
             v9,
             &v12,
             a2,
             a7,
             v10,
             v12,
             v13,
             v14,
             v15);
}
