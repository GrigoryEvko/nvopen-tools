// Function: sub_2DB9A90
// Address: 0x2db9a90
//
void __fastcall sub_2DB9A90(__int64 *a1)
{
  __int64 *v1; // rdi
  __int64 *v2; // [rsp+0h] [rbp-B0h] BYREF
  __int64 v3; // [rsp+8h] [rbp-A8h]
  __int64 v4; // [rsp+10h] [rbp-A0h] BYREF
  unsigned __int64 v5[2]; // [rsp+20h] [rbp-90h] BYREF
  _BYTE v6[16]; // [rsp+30h] [rbp-80h] BYREF
  const char *v7; // [rsp+40h] [rbp-70h] BYREF
  char v8; // [rsp+60h] [rbp-50h]
  char v9; // [rsp+61h] [rbp-4Fh]
  _BYTE v10[32]; // [rsp+70h] [rbp-40h] BYREF
  __int16 v11; // [rsp+90h] [rbp-20h]

  v11 = 257;
  v9 = 1;
  v7 = "EdgeBundles";
  v8 = 3;
  v5[0] = (unsigned __int64)v6;
  v5[1] = 0;
  v6[0] = 0;
  sub_2DB9490((__int64)&v2, a1, (void **)&v7, 0, (__int64)v10, (__int64)v5);
  if ( (_BYTE *)v5[0] != v6 )
    j_j___libc_free_0(v5[0]);
  v1 = v2;
  if ( v3 )
  {
    sub_C67930(v2, v3, 0, 0);
    v1 = v2;
  }
  if ( v1 != &v4 )
    j_j___libc_free_0((unsigned __int64)v1);
}
