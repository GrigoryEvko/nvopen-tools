// Function: sub_2EF0910
// Address: 0x2ef0910
//
void __fastcall sub_2EF0910(__int64 a1, void **a2, __int64 a3)
{
  char *v4; // [rsp+0h] [rbp-30h] BYREF
  __int64 v5; // [rsp+10h] [rbp-20h] BYREF

  sub_CA0F50((__int64 *)&v4, a2);
  sub_2EF06E0(a1, v4, a3);
  if ( v4 != (char *)&v5 )
    j_j___libc_free_0((unsigned __int64)v4);
}
