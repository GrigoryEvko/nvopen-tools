// Function: sub_1D23D40
// Address: 0x1d23d40
//
__int64 __fastcall sub_1D23D40(__int64 a1, __int64 a2, __int64 *a3, int a4)
{
  __int64 v6; // rsi
  __int64 result; // rax
  unsigned __int8 *v8; // rsi
  __int64 v9; // r12
  _QWORD v10[5]; // [rsp+8h] [rbp-28h] BYREF

  v6 = *a3;
  v10[0] = v6;
  if ( v6 )
    sub_1623A60((__int64)v10, v6, 2);
  result = sub_145CBF0(*(__int64 **)(a1 + 648), 24, 16);
  v8 = (unsigned __int8 *)v10[0];
  *(_QWORD *)result = a2;
  v9 = result;
  *(_QWORD *)(result + 8) = v8;
  if ( v8 )
  {
    sub_1623210((__int64)v10, v8, result + 8);
    *(_DWORD *)(v9 + 16) = a4;
    return v9;
  }
  else
  {
    *(_DWORD *)(result + 16) = a4;
  }
  return result;
}
