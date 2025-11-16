// Function: sub_17050D0
// Address: 0x17050d0
//
__int64 __fastcall sub_17050D0(__int64 *a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rsi
  __int64 v4; // rsi
  unsigned __int8 *v5; // rsi
  _QWORD v6[2]; // [rsp+8h] [rbp-18h] BYREF

  a1[1] = *(_QWORD *)(a2 + 40);
  result = a2 + 24;
  a1[2] = a2 + 24;
  v3 = *(_QWORD *)(a2 + 48);
  v6[0] = v3;
  if ( v3 )
  {
    result = sub_1623A60((__int64)v6, v3, 2);
    v4 = *a1;
    if ( !*a1 )
      goto LABEL_4;
  }
  else
  {
    v4 = *a1;
    if ( !*a1 )
      return result;
  }
  result = sub_161E7C0((__int64)a1, v4);
LABEL_4:
  v5 = (unsigned __int8 *)v6[0];
  *a1 = v6[0];
  if ( v5 )
    return sub_1623210((__int64)v6, v5, (__int64)a1);
  return result;
}
