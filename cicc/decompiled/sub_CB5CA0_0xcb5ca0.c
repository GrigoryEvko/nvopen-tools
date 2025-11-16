// Function: sub_CB5CA0
// Address: 0xcb5ca0
//
__int64 __fastcall sub_CB5CA0(__int64 *a1)
{
  __int64 v1; // rax
  __int64 v2; // r13
  __int64 v3; // rax

  v1 = (*(__int64 (__fastcall **)(__int64 *))(*a1 + 88))(a1);
  if ( v1 )
  {
    v2 = v1;
    if ( a1[4] != a1[2] )
      sub_CB5AE0(a1);
    v3 = sub_2207820(v2);
    return sub_CB5980((__int64)a1, v3, v2, 1);
  }
  else
  {
    if ( a1[4] != a1[2] )
      sub_CB5AE0(a1);
    return sub_CB5980((__int64)a1, 0, 0, 0);
  }
}
