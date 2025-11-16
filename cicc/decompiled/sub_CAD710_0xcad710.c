// Function: sub_CAD710
// Address: 0xcad710
//
__int64 __fastcall sub_CAD710(unsigned __int64 **a1, int a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned int v5; // r13d
  __int64 v6; // rcx
  __int64 v7; // r8
  const char *v9; // [rsp+0h] [rbp-90h] BYREF
  char v10; // [rsp+20h] [rbp-70h]
  char v11; // [rsp+21h] [rbp-6Fh]
  _DWORD v12[6]; // [rsp+30h] [rbp-60h] BYREF
  __int64 *v13; // [rsp+48h] [rbp-48h]
  __int64 v14; // [rsp+58h] [rbp-38h] BYREF

  v5 = 1;
  sub_CAD680((__int64)v12, a1, a3, a4, a5);
  if ( v12[0] != a2 )
  {
    v5 = 0;
    v11 = 1;
    v9 = "Unexpected token";
    v10 = 3;
    sub_CA8C70((__int64 **)a1, (__int64)&v9, (__int64)v12, v6, v7);
  }
  if ( v13 != &v14 )
    j_j___libc_free_0(v13, v14 + 1);
  return v5;
}
