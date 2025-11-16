// Function: sub_33DCC00
// Address: 0x33dcc00
//
const void **__fastcall sub_33DCC00(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, unsigned int a5)
{
  unsigned int v6; // ebx
  unsigned __int64 v7; // r13
  bool v8; // r12
  unsigned int v9; // ecx
  const void **v10; // rax
  const void **v11; // [rsp+10h] [rbp-70h] BYREF
  unsigned int v12; // [rsp+18h] [rbp-68h]
  const void **v13; // [rsp+20h] [rbp-60h] BYREF
  __int64 v14; // [rsp+28h] [rbp-58h]
  const void **v15; // [rsp+30h] [rbp-50h] BYREF
  unsigned int v16; // [rsp+38h] [rbp-48h]
  const void **v17; // [rsp+40h] [rbp-40h] BYREF
  unsigned int v18; // [rsp+48h] [rbp-38h]
  char v19; // [rsp+50h] [rbp-30h]

  sub_33DC4D0((__int64)&v15, a1, a2, a3, a4, a5);
  if ( !v19 )
  {
LABEL_2:
    LOBYTE(v14) = 0;
    return v13;
  }
  v12 = v16;
  if ( v16 > 0x40 )
    sub_C43780((__int64)&v11, (const void **)&v15);
  else
    v11 = v15;
  sub_C46A40((__int64)&v11, 1);
  v6 = v12;
  v7 = (unsigned __int64)v11;
  v12 = 0;
  LODWORD(v14) = v6;
  v13 = v11;
  if ( v18 <= 0x40 )
    v8 = v17 == v11;
  else
    v8 = sub_C43C50((__int64)&v17, (const void **)&v13);
  if ( v6 > 0x40 )
  {
    if ( v7 )
    {
      j_j___libc_free_0_0(v7);
      if ( v12 > 0x40 )
      {
        if ( v11 )
          j_j___libc_free_0_0((unsigned __int64)v11);
      }
    }
  }
  if ( !v8 )
  {
    if ( v19 )
    {
      v19 = 0;
      if ( v18 > 0x40 && v17 )
        j_j___libc_free_0_0((unsigned __int64)v17);
      if ( v16 > 0x40 && v15 )
        j_j___libc_free_0_0((unsigned __int64)v15);
    }
    goto LABEL_2;
  }
  v9 = v16;
  v10 = v15;
  if ( v16 > 0x40 )
    v10 = (const void **)*v15;
  v13 = v10;
  LOBYTE(v14) = 1;
  if ( v19 )
  {
    v19 = 0;
    if ( v18 > 0x40 && v17 )
    {
      j_j___libc_free_0_0((unsigned __int64)v17);
      v9 = v16;
    }
    if ( v9 > 0x40 && v15 )
      j_j___libc_free_0_0((unsigned __int64)v15);
  }
  return v13;
}
