// Function: sub_1D17920
// Address: 0x1d17920
//
_QWORD *__fastcall sub_1D17920(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  _QWORD *v5; // rax
  _QWORD *v6; // r12
  unsigned int v7; // eax
  __int64 v8; // rsi
  __int64 *v9; // r13
  __int64 v11; // rsi
  unsigned __int8 *v12; // rsi
  __int64 v13[5]; // [rsp+8h] [rbp-28h] BYREF

  v5 = sub_16BDDE0(a1 + 320, a2, a4);
  v6 = v5;
  if ( !v5 )
    return v6;
  if ( (unsigned int)*((unsigned __int16 *)v5 + 12) - 10 <= 1 )
  {
    v11 = v5[9];
    if ( *(_QWORD *)a3 == v11 )
      return v6;
    v9 = v5 + 9;
    v13[0] = 0;
    if ( v5 + 9 == v13 )
      return v6;
    if ( !v11 )
    {
      v5[9] = 0;
      return v6;
    }
    goto LABEL_13;
  }
  v7 = *(_DWORD *)(a3 + 8);
  if ( !v7 || v7 >= *((_DWORD *)v6 + 16) )
    return v6;
  v8 = *(_QWORD *)a3;
  v9 = v6 + 9;
  v13[0] = v8;
  if ( !v8 )
  {
    if ( v9 == v13 )
      return v6;
    v11 = v6[9];
    if ( !v11 )
      return v6;
LABEL_13:
    sub_161E7C0((__int64)v9, v11);
    goto LABEL_14;
  }
  sub_1623A60((__int64)v13, v8, 2);
  if ( v9 == v13 )
  {
    if ( v13[0] )
      sub_161E7C0((__int64)(v6 + 9), v13[0]);
    return v6;
  }
  v11 = v6[9];
  if ( v11 )
    goto LABEL_13;
LABEL_14:
  v12 = (unsigned __int8 *)v13[0];
  v6[9] = v13[0];
  if ( !v12 )
    return v6;
  sub_1623210((__int64)v13, v12, (__int64)v9);
  return v6;
}
