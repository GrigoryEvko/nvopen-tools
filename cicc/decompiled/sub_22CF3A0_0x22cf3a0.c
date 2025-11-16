// Function: sub_22CF3A0
// Address: 0x22cf3a0
//
const void *__fastcall sub_22CF3A0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v8; // rax
  unsigned __int64 v9; // rax
  int v10; // eax
  __int64 v11; // r12
  unsigned int v13; // r14d
  unsigned __int64 v14; // r13
  bool v15; // r12
  const void *v16; // [rsp+0h] [rbp-80h] BYREF
  unsigned int v17; // [rsp+8h] [rbp-78h]
  const void *v18; // [rsp+10h] [rbp-70h] BYREF
  unsigned int v19; // [rsp+18h] [rbp-68h]
  _BYTE v20[8]; // [rsp+20h] [rbp-60h] BYREF
  const void *v21; // [rsp+28h] [rbp-58h] BYREF
  unsigned int v22; // [rsp+30h] [rbp-50h]
  const void *v23; // [rsp+38h] [rbp-48h] BYREF
  unsigned int v24; // [rsp+40h] [rbp-40h]

  v8 = sub_AA4B30(a3);
  v9 = sub_22C1480(a1, v8);
  sub_22CF010((__int64)v20, v9, a2, a3, a4, a5);
  v10 = v20[0];
  if ( v20[0] == 2 )
    return v21;
  v11 = 0;
  if ( (unsigned __int8)(v20[0] - 4) <= 1u )
  {
    v19 = v22;
    if ( v22 > 0x40 )
      sub_C43780((__int64)&v18, &v21);
    else
      v18 = v21;
    sub_C46A40((__int64)&v18, 1);
    v13 = v19;
    v14 = (unsigned __int64)v18;
    v19 = 0;
    v17 = v13;
    v16 = v18;
    if ( v24 <= 0x40 )
      v15 = v23 == v18;
    else
      v15 = sub_C43C50((__int64)&v23, &v16);
    if ( v13 > 0x40 )
    {
      if ( v14 )
      {
        j_j___libc_free_0_0(v14);
        if ( v19 > 0x40 )
        {
          if ( v18 )
            j_j___libc_free_0_0((unsigned __int64)v18);
        }
      }
    }
    if ( v15 )
    {
      v11 = sub_AD8D80(*(_QWORD *)(a2 + 8), (__int64)&v21);
      v10 = v20[0];
    }
    else
    {
      v10 = v20[0];
      v11 = 0;
    }
  }
  if ( (unsigned int)(v10 - 4) <= 1 )
  {
    if ( v24 > 0x40 && v23 )
      j_j___libc_free_0_0((unsigned __int64)v23);
    if ( v22 > 0x40 && v21 )
      j_j___libc_free_0_0((unsigned __int64)v21);
  }
  return (const void *)v11;
}
