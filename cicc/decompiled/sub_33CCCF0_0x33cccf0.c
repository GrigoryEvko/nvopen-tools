// Function: sub_33CCCF0
// Address: 0x33cccf0
//
_QWORD *__fastcall sub_33CCCF0(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  _QWORD *v5; // rax
  _QWORD *v6; // r12
  unsigned int v7; // eax
  __int64 v8; // rsi
  __int64 *v9; // r13
  __int64 v11; // rsi
  unsigned __int8 *v12; // rsi
  __int64 v13[5]; // [rsp+8h] [rbp-28h] BYREF

  v5 = sub_C65B40(a1 + 520, a2, a4, (__int64)off_4A367D0);
  v6 = v5;
  if ( !v5 )
    return v6;
  if ( (unsigned int)(*((_DWORD *)v5 + 6) - 11) <= 1 )
  {
    v11 = v5[10];
    if ( *(_QWORD *)a3 == v11 )
      return v6;
    v9 = v5 + 10;
    v13[0] = 0;
    if ( v5 + 10 == v13 )
      return v6;
    if ( !v11 )
    {
      v5[10] = 0;
      return v6;
    }
    goto LABEL_13;
  }
  v7 = *(_DWORD *)(a3 + 8);
  if ( !v7 || v7 >= *((_DWORD *)v6 + 18) )
    return v6;
  v8 = *(_QWORD *)a3;
  v9 = v6 + 10;
  v13[0] = v8;
  if ( !v8 )
  {
    if ( v9 == v13 )
      return v6;
    v11 = v6[10];
    if ( !v11 )
      return v6;
LABEL_13:
    sub_B91220((__int64)v9, v11);
    goto LABEL_14;
  }
  sub_B96E90((__int64)v13, v8, 1);
  if ( v9 == v13 )
  {
    if ( v13[0] )
      sub_B91220((__int64)(v6 + 10), v13[0]);
    return v6;
  }
  v11 = v6[10];
  if ( v11 )
    goto LABEL_13;
LABEL_14:
  v12 = (unsigned __int8 *)v13[0];
  v6[10] = v13[0];
  if ( !v12 )
    return v6;
  sub_B976B0((__int64)v13, v12, (__int64)v9);
  return v6;
}
