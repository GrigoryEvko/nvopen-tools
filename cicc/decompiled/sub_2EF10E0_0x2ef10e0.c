// Function: sub_2EF10E0
// Address: 0x2ef10e0
//
void __fastcall sub_2EF10E0(__int64 *a1, void **a2)
{
  __int64 v2; // r12
  __int64 *v3; // r13
  char *v4; // [rsp+0h] [rbp-30h] BYREF
  __int64 v5; // [rsp+10h] [rbp-20h] BYREF

  v2 = *a1;
  v3 = *(__int64 **)(*a1 + 32);
  sub_CA0F50((__int64 *)&v4, a2);
  sub_2EEFF60(v2, v4, v3);
  if ( v4 != (char *)&v5 )
    j_j___libc_free_0((unsigned __int64)v4);
}
