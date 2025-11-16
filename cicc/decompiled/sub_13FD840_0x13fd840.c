// Function: sub_13FD840
// Address: 0x13fd840
//
_QWORD *__fastcall sub_13FD840(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rsi
  __int64 v4; // [rsp+0h] [rbp-20h] BYREF
  _QWORD v5[3]; // [rsp+8h] [rbp-18h] BYREF

  sub_13FD560(&v4, a2);
  v2 = v4;
  *a1 = v4;
  if ( v2 )
    sub_1623A60(a1, v2, 2);
  if ( v5[0] )
    sub_161E7C0(v5);
  if ( v4 )
    sub_161E7C0(&v4);
  return a1;
}
