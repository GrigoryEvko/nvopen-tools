// Function: sub_2647370
// Address: 0x2647370
//
__int64 __fastcall sub_2647370(__int64 *a1, __int64 *a2)
{
  __int64 v2; // rbx
  __int64 i; // r12
  __int64 v5; // [rsp+0h] [rbp-50h]
  __int64 v6; // [rsp+10h] [rbp-40h] BYREF
  __int64 v7; // [rsp+18h] [rbp-38h]

  sub_1039B70(&v6, *a1, 0);
  v2 = v7;
  v5 = v6;
  sub_1039B70(&v6, *a2, 0);
  for ( i = v7; ; i += 8 )
  {
    sub_1039B70(&v6, *a1, 1);
    if ( v7 == v2 )
      break;
    v2 += 8;
    sub_1039B70(&v6, *a2, 1);
    if ( v7 == i )
      break;
  }
  return v5;
}
