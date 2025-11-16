// Function: sub_1441DA0
// Address: 0x1441da0
//
char __fastcall sub_1441DA0(_QWORD *a1, __int64 a2)
{
  char v2; // r13
  unsigned __int64 v4; // rax
  int v5; // edx

  if ( !a2 )
    return 0;
  v2 = sub_1560180(a2 + 112, 7);
  if ( v2 )
    return v2;
  if ( !(unsigned __int8)sub_1441AE0(a1) )
    return 0;
  v4 = sub_15E44B0(a2);
  if ( !v5 )
    return v2;
  return sub_1441D60((__int64)a1, v4);
}
