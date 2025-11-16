// Function: sub_14DBA30
// Address: 0x14dba30
//
__int64 __fastcall sub_14DBA30(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v3; // rax
  __int64 v4; // r12
  __int64 v6; // [rsp+0h] [rbp-60h] BYREF
  __int64 v7; // [rsp+8h] [rbp-58h]
  __int64 v8; // [rsp+10h] [rbp-50h] BYREF
  char v9; // [rsp+50h] [rbp-10h] BYREF

  v3 = &v8;
  v6 = 0;
  v7 = 1;
  do
  {
    *v3 = -8;
    v3 += 2;
  }
  while ( v3 != (__int64 *)&v9 );
  v4 = sub_14DB6D0(a1, a2, a3, (__int64)&v6);
  if ( (v7 & 1) == 0 )
    j___libc_free_0(v8);
  return v4;
}
