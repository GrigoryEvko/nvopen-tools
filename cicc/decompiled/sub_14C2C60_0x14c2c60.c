// Function: sub_14C2C60
// Address: 0x14c2c60
//
__int64 __fastcall sub_14C2C60(__int64 *a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v10; // eax
  unsigned int v11; // r8d
  __int64 v13; // rdx
  unsigned int v14; // esi
  __int64 v15; // rdx
  int v16; // [rsp+0h] [rbp-80h]
  int v17; // [rsp+4h] [rbp-7Ch]
  __int64 v18; // [rsp+10h] [rbp-70h] BYREF
  unsigned int v19; // [rsp+18h] [rbp-68h]
  __int64 v20; // [rsp+20h] [rbp-60h]
  unsigned int v21; // [rsp+28h] [rbp-58h]
  __int64 v22; // [rsp+30h] [rbp-50h] BYREF
  unsigned int v23; // [rsp+38h] [rbp-48h]
  __int64 v24; // [rsp+40h] [rbp-40h]
  unsigned int v25; // [rsp+48h] [rbp-38h]

  v16 = sub_16431D0(*a1);
  v17 = sub_14C23D0((__int64)a1, a3, 0, a4, a5, a6);
  v10 = v17 + sub_14C23D0((__int64)a2, a3, 0, a4, a5, a6);
  v11 = 2;
  if ( v16 + 1 >= v10 )
  {
    v11 = 1;
    if ( v16 + 1 == v10 )
    {
      sub_14C2530((__int64)&v18, a1, a3, 0, a4, a5, a6, 0);
      sub_14C2530((__int64)&v22, a2, a3, 0, a4, a5, a6, 0);
      if ( v19 > 0x40 )
        v13 = *(_QWORD *)(v18 + 8LL * ((v19 - 1) >> 6));
      else
        v13 = v18;
      if ( (v13 & (1LL << ((unsigned __int8)v19 - 1))) != 0 )
        goto LABEL_25;
      v14 = v23;
      v15 = v22;
      if ( v23 > 0x40 )
        v15 = *(_QWORD *)(v22 + 8LL * ((v23 - 1) >> 6));
      if ( (v15 & (1LL << ((unsigned __int8)v23 - 1))) != 0 )
      {
LABEL_25:
        if ( v25 > 0x40 && v24 )
          j_j___libc_free_0_0(v24);
        if ( v23 > 0x40 && v22 )
          j_j___libc_free_0_0(v22);
        if ( v21 > 0x40 && v20 )
          j_j___libc_free_0_0(v20);
        if ( v19 > 0x40 && v18 )
          j_j___libc_free_0_0(v18);
        return 2;
      }
      else
      {
        if ( v25 > 0x40 && v24 )
        {
          j_j___libc_free_0_0(v24);
          v14 = v23;
        }
        if ( v14 > 0x40 && v22 )
          j_j___libc_free_0_0(v22);
        if ( v21 > 0x40 && v20 )
          j_j___libc_free_0_0(v20);
        if ( v19 > 0x40 && v18 )
          j_j___libc_free_0_0(v18);
        return 1;
      }
    }
  }
  return v11;
}
