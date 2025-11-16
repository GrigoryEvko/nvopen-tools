// Function: sub_1475E30
// Address: 0x1475e30
//
__int64 __fastcall sub_1475E30(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, unsigned int a6)
{
  __int64 v10; // r15
  __int64 v11; // [rsp+8h] [rbp-E8h]
  __int64 v12; // [rsp+10h] [rbp-E0h]
  __int64 v13; // [rsp+18h] [rbp-D8h]
  __int64 v14; // [rsp+20h] [rbp-D0h] BYREF
  unsigned int v15; // [rsp+28h] [rbp-C8h]
  __int64 v16; // [rsp+30h] [rbp-C0h]
  unsigned int v17; // [rsp+38h] [rbp-B8h]
  __int64 v18; // [rsp+40h] [rbp-B0h] BYREF
  unsigned int v19; // [rsp+48h] [rbp-A8h]
  __int64 v20; // [rsp+50h] [rbp-A0h]
  unsigned int v21; // [rsp+58h] [rbp-98h]
  __int64 v22; // [rsp+60h] [rbp-90h] BYREF
  __int64 v23; // [rsp+68h] [rbp-88h] BYREF
  unsigned int v24; // [rsp+70h] [rbp-80h]
  __int64 v25; // [rsp+78h] [rbp-78h] BYREF
  unsigned int v26; // [rsp+80h] [rbp-70h]
  __int64 v27; // [rsp+90h] [rbp-60h] BYREF
  __int64 v28; // [rsp+98h] [rbp-58h] BYREF
  unsigned int v29; // [rsp+A0h] [rbp-50h]
  __int64 v30; // [rsp+A8h] [rbp-48h] BYREF
  unsigned int v31; // [rsp+B0h] [rbp-40h]

  sub_1454BA0((__int64)&v22, a6, a3);
  if ( v22 )
  {
    sub_1454BA0((__int64)&v27, a6, a4);
    if ( v27 && v22 == v27 )
    {
      v11 = sub_145CF40(a2, (__int64)&v23);
      v12 = sub_145CF40(a2, (__int64)&v28);
      v10 = sub_145CF40(a2, (__int64)&v25);
      v13 = sub_145CF40(a2, (__int64)&v30);
      sub_1475920((__int64)&v14, a2, v11, v12, a5, a6);
      sub_1475920((__int64)&v18, a2, v10, v13, a5, a6);
      sub_158C3A0(a1, &v14, &v18);
      if ( v21 > 0x40 && v20 )
        j_j___libc_free_0_0(v20);
      if ( v19 > 0x40 && v18 )
        j_j___libc_free_0_0(v18);
      if ( v17 > 0x40 && v16 )
        j_j___libc_free_0_0(v16);
      if ( v15 > 0x40 && v14 )
        j_j___libc_free_0_0(v14);
    }
    else
    {
      sub_15897D0(a1, a6, 1);
    }
    if ( v31 > 0x40 && v30 )
      j_j___libc_free_0_0(v30);
    if ( v29 > 0x40 && v28 )
      j_j___libc_free_0_0(v28);
  }
  else
  {
    sub_15897D0(a1, a6, 1);
  }
  if ( v26 > 0x40 && v25 )
    j_j___libc_free_0_0(v25);
  if ( v24 > 0x40 && v23 )
    j_j___libc_free_0_0(v23);
  return a1;
}
