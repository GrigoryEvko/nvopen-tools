// Function: sub_16E7D60
// Address: 0x16e7d60
//
__int64 __fastcall sub_16E7D60(__int64 *a1)
{
  __int64 v1; // rax
  __int64 v2; // r13
  __int64 v3; // rax

  v1 = (*(__int64 (__fastcall **)(__int64 *))(*a1 + 72))(a1);
  if ( v1 )
  {
    v2 = v1;
    if ( a1[3] != a1[1] )
      sub_16E7BA0(a1);
    v3 = sub_2207820(v2);
    return sub_16E7A40((__int64)a1, v3, v2, 1);
  }
  else
  {
    if ( a1[3] != a1[1] )
      sub_16E7BA0(a1);
    return sub_16E7A40((__int64)a1, 0, 0, 0);
  }
}
