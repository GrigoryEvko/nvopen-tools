// Function: sub_3888960
// Address: 0x3888960
//
__int64 *__fastcall sub_3888960(__int64 *a1, __int64 a2)
{
  unsigned __int64 *v2; // rax
  unsigned __int64 v4[2]; // [rsp+0h] [rbp-70h] BYREF
  _BYTE v5[16]; // [rsp+10h] [rbp-60h] BYREF
  void *v6; // [rsp+20h] [rbp-50h] BYREF
  __int64 v7; // [rsp+28h] [rbp-48h]
  __int64 v8; // [rsp+30h] [rbp-40h]
  __int64 v9; // [rsp+38h] [rbp-38h]
  int v10; // [rsp+40h] [rbp-30h]
  unsigned __int64 *v11; // [rsp+48h] [rbp-28h]

  v11 = v4;
  v4[0] = (unsigned __int64)v5;
  v4[1] = 0;
  v6 = &unk_49EFBE0;
  v5[0] = 0;
  v10 = 1;
  v9 = 0;
  v8 = 0;
  v7 = 0;
  sub_154E060(a2, (__int64)&v6, 0, 0);
  if ( v9 != v7 )
    sub_16E7BA0((__int64 *)&v6);
  v2 = v11;
  *a1 = (__int64)(a1 + 2);
  sub_3887850(a1, (_BYTE *)*v2, *v2 + v2[1]);
  sub_16E7BC0((__int64 *)&v6);
  if ( (_BYTE *)v4[0] != v5 )
    j_j___libc_free_0(v4[0]);
  return a1;
}
