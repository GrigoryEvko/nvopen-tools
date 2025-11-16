// Function: sub_F705A0
// Address: 0xf705a0
//
__int64 __fastcall sub_F705A0(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 v4; // rax
  _QWORD *v5; // rbx

  v4 = sub_D95540(a1);
  v5 = sub_DA2C50((__int64)a3, v4, 0, 0);
  if ( sub_DAEB70((__int64)a3, a1, a2) )
    return sub_DDD5B0(a3, a2, 39, a1, (__int64)v5);
  else
    return 0;
}
