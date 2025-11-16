// Function: sub_147A000
// Address: 0x147a000
//
__int64 __fastcall sub_147A000(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r12
  __int64 v6; // [rsp+0h] [rbp-70h] BYREF
  __int64 v7; // [rsp+8h] [rbp-68h]
  __int64 v8; // [rsp+10h] [rbp-60h]
  __int64 v9; // [rsp+18h] [rbp-58h]
  int v10; // [rsp+20h] [rbp-50h]
  __int64 v11; // [rsp+28h] [rbp-48h]
  __int16 v12; // [rsp+30h] [rbp-40h]

  v6 = a1;
  v11 = a2;
  v7 = 0;
  v8 = 0;
  v9 = 0;
  v10 = 0;
  v12 = 0;
  v4 = sub_148BD30(&v6, a3);
  if ( (_BYTE)v12 )
    v4 = sub_1456E90(a1);
  j___libc_free_0(v8);
  if ( v4 != sub_1456E90(a1) )
  {
    v11 = a2;
    v6 = a1;
    v7 = 0;
    v8 = 0;
    v9 = 0;
    v10 = 0;
    v12 = 0;
    sub_1479A80(&v6, a3);
    if ( (_BYTE)v12 )
      sub_1456E90(a1);
    j___libc_free_0(v8);
  }
  return v4;
}
