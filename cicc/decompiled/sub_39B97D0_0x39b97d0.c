// Function: sub_39B97D0
// Address: 0x39b97d0
//
__int64 __fastcall sub_39B97D0(__int64 a1)
{
  unsigned __int64 v2[2]; // [rsp+0h] [rbp-70h] BYREF
  _BYTE v3[16]; // [rsp+10h] [rbp-60h] BYREF
  void *v4; // [rsp+20h] [rbp-50h] BYREF
  __int64 v5; // [rsp+28h] [rbp-48h]
  __int64 v6; // [rsp+30h] [rbp-40h]
  __int64 v7; // [rsp+38h] [rbp-38h]
  int v8; // [rsp+40h] [rbp-30h]
  unsigned __int64 *v9; // [rsp+48h] [rbp-28h]

  v9 = v2;
  v2[0] = (unsigned __int64)v3;
  v2[1] = 0;
  v3[0] = 0;
  v8 = 1;
  v7 = 0;
  v6 = 0;
  v5 = 0;
  v4 = &unk_49EFBE0;
  sub_39DF6B0(&v4);
  if ( v7 != v5 )
    sub_16E7BA0((__int64 *)&v4);
  sub_2241490((unsigned __int64 *)(a1 + 240), (char *)*v9, v9[1]);
  sub_16E7BC0((__int64 *)&v4);
  if ( (_BYTE *)v2[0] != v3 )
    j_j___libc_free_0(v2[0]);
  return 0;
}
