// Function: sub_DCF3A0
// Address: 0xdcf3a0
//
__int64 __fastcall sub_DCF3A0(__int64 *a1, char *a2, int a3)
{
  __int64 *v3; // rax
  __int64 *v5; // rax
  __int64 *v6; // rax

  if ( a3 == 1 )
  {
    v6 = sub_DB9E00((__int64)a1, (__int64)a2);
    return sub_D97BB0(v6, (__int64)a1, 0);
  }
  else if ( a3 == 2 )
  {
    v3 = sub_DB9E00((__int64)a1, (__int64)a2);
    return sub_DCF230(v3, a2, a1, 0);
  }
  else
  {
    if ( a3 )
      BUG();
    v5 = sub_DB9E00((__int64)a1, (__int64)a2);
    return sub_DCF0D0(v5, (__int64)a2, a1, 0);
  }
}
