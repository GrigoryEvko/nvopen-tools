// Function: sub_33E0180
// Address: 0x33e0180
//
__int64 __fastcall sub_33E0180(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int64 *v7; // r12
  unsigned int v10; // ebx
  unsigned __int64 v11; // r13
  unsigned __int64 v12; // r13
  unsigned __int64 v13; // [rsp+10h] [rbp-90h] BYREF
  unsigned int v14; // [rsp+18h] [rbp-88h]
  unsigned __int64 v15; // [rsp+20h] [rbp-80h] BYREF
  unsigned int v16; // [rsp+28h] [rbp-78h]
  unsigned __int64 v17; // [rsp+30h] [rbp-70h] BYREF
  unsigned int v18; // [rsp+38h] [rbp-68h]
  unsigned __int64 v19; // [rsp+40h] [rbp-60h]
  unsigned int v20; // [rsp+48h] [rbp-58h]
  unsigned __int64 v21; // [rsp+50h] [rbp-50h] BYREF
  unsigned int v22; // [rsp+58h] [rbp-48h]
  unsigned __int64 v23; // [rsp+60h] [rbp-40h]
  unsigned int v24; // [rsp+68h] [rbp-38h]

  LODWORD(v7) = a2;
  if ( (unsigned __int8)sub_33E0010(a2, a4, a5) || (unsigned __int8)sub_33E0010(a4, a2, a3) )
  {
    LODWORD(v7) = 1;
    return (unsigned int)v7;
  }
  sub_33DD090((__int64)&v21, a1, a4, a5, 0);
  sub_33DD090((__int64)&v17, a1, a2, a3, 0);
  v10 = v18;
  v14 = v18;
  if ( v18 > 0x40 )
  {
    v7 = &v13;
    sub_C43780((__int64)&v13, (const void **)&v17);
    v10 = v14;
    if ( v14 > 0x40 )
    {
      sub_C43BD0(&v13, (__int64 *)&v21);
      v10 = v14;
      v12 = v13;
      v14 = 0;
      v16 = v10;
      v15 = v13;
      if ( !v10 )
        goto LABEL_8;
      if ( v10 > 0x40 )
      {
        LOBYTE(v7) = v10 == (unsigned int)sub_C445E0((__int64)&v15);
        if ( v12 )
        {
          j_j___libc_free_0_0(v12);
          if ( v14 > 0x40 )
          {
            if ( v13 )
              j_j___libc_free_0_0(v13);
          }
        }
        goto LABEL_9;
      }
      goto LABEL_28;
    }
    v11 = v13;
  }
  else
  {
    v11 = v17;
  }
  v12 = v21 | v11;
  v16 = v10;
  v13 = v12;
  v15 = v12;
  v14 = 0;
  if ( !v10 )
  {
LABEL_8:
    LODWORD(v7) = 1;
    goto LABEL_9;
  }
LABEL_28:
  LOBYTE(v7) = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v10) == v12;
LABEL_9:
  if ( v20 > 0x40 && v19 )
    j_j___libc_free_0_0(v19);
  if ( v18 > 0x40 && v17 )
    j_j___libc_free_0_0(v17);
  if ( v24 > 0x40 && v23 )
    j_j___libc_free_0_0(v23);
  if ( v22 > 0x40 && v21 )
    j_j___libc_free_0_0(v21);
  return (unsigned int)v7;
}
