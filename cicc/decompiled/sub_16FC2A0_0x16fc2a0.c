// Function: sub_16FC2A0
// Address: 0x16fc2a0
//
__int64 __fastcall sub_16FC2A0(unsigned __int64 **a1, int a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned int v5; // r13d
  __int64 v6; // rcx
  __int64 v7; // r8
  const char *v9; // [rsp+0h] [rbp-80h] BYREF
  char v10; // [rsp+10h] [rbp-70h]
  char v11; // [rsp+11h] [rbp-6Fh]
  _DWORD v12[6]; // [rsp+20h] [rbp-60h] BYREF
  __int64 *v13; // [rsp+38h] [rbp-48h]
  __int64 v14; // [rsp+48h] [rbp-38h] BYREF

  v5 = 1;
  sub_16FC210((__int64)v12, a1, a3, a4, a5);
  if ( v12[0] != a2 )
  {
    v5 = 0;
    v11 = 1;
    v9 = "Unexpected token";
    v10 = 3;
    sub_16F82E0((__int64 **)a1, (__int64)&v9, (__int64)v12, v6, v7);
  }
  if ( v13 != &v14 )
    j_j___libc_free_0(v13, v14 + 1);
  return v5;
}
