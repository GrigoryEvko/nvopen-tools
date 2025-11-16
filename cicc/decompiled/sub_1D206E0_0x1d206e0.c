// Function: sub_1D206E0
// Address: 0x1d206e0
//
__int64 __fastcall sub_1D206E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned int v7; // ebx
  __int64 v8; // r12
  __int64 v9; // r12
  __int64 v11; // [rsp+10h] [rbp-90h] BYREF
  unsigned int v12; // [rsp+18h] [rbp-88h]
  __int64 v13; // [rsp+20h] [rbp-80h] BYREF
  unsigned int v14; // [rsp+28h] [rbp-78h]
  __int64 v15; // [rsp+30h] [rbp-70h] BYREF
  __int64 v16; // [rsp+38h] [rbp-68h]
  __int64 v17; // [rsp+40h] [rbp-60h]
  __int64 v18; // [rsp+48h] [rbp-58h]
  __int64 v19; // [rsp+50h] [rbp-50h] BYREF
  __int64 v20; // [rsp+58h] [rbp-48h]
  __int64 v21; // [rsp+60h] [rbp-40h]
  __int64 v22; // [rsp+68h] [rbp-38h]

  v15 = 0;
  v16 = 1;
  v17 = 0;
  v18 = 1;
  v19 = 0;
  v20 = 1;
  v21 = 0;
  v22 = 1;
  sub_1D1F820(a1, a2, a3, (unsigned __int64 *)&v15, 0);
  sub_1D1F820(a1, a4, a5, (unsigned __int64 *)&v19, 0);
  LOBYTE(v7) = v16;
  v12 = v16;
  if ( (unsigned int)v16 <= 0x40 )
  {
    v8 = v15;
LABEL_3:
    v9 = v19 | v8;
LABEL_4:
    LOBYTE(a5) = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v7) == v9;
    goto LABEL_5;
  }
  sub_16A4FD0((__int64)&v11, (const void **)&v15);
  LOBYTE(v7) = v12;
  if ( v12 <= 0x40 )
  {
    v8 = v11;
    goto LABEL_3;
  }
  sub_16A89F0(&v11, &v19);
  v7 = v12;
  v9 = v11;
  v12 = 0;
  v14 = v7;
  v13 = v11;
  if ( v7 <= 0x40 )
    goto LABEL_4;
  LOBYTE(a5) = v7 == (unsigned int)sub_16A58F0((__int64)&v13);
  if ( v9 )
  {
    j_j___libc_free_0_0(v9);
    if ( v12 > 0x40 )
    {
      if ( v11 )
        j_j___libc_free_0_0(v11);
    }
  }
LABEL_5:
  if ( (unsigned int)v22 > 0x40 && v21 )
    j_j___libc_free_0_0(v21);
  if ( (unsigned int)v20 > 0x40 && v19 )
    j_j___libc_free_0_0(v19);
  if ( (unsigned int)v18 > 0x40 && v17 )
    j_j___libc_free_0_0(v17);
  if ( (unsigned int)v16 > 0x40 && v15 )
    j_j___libc_free_0_0(v15);
  return (unsigned int)a5;
}
