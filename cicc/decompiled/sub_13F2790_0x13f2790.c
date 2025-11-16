// Function: sub_13F2790
// Address: 0x13f2790
//
__int64 __fastcall sub_13F2790(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 result; // rax
  unsigned int v10; // r14d
  __int64 v11; // r13
  char v12; // bl
  __int64 v13; // rax
  __int64 v14; // [rsp+8h] [rbp-78h]
  __int64 v15; // [rsp+8h] [rbp-78h]
  __int64 v16; // [rsp+10h] [rbp-70h] BYREF
  unsigned int v17; // [rsp+18h] [rbp-68h]
  __int64 v18; // [rsp+20h] [rbp-60h] BYREF
  unsigned int v19; // [rsp+28h] [rbp-58h]
  int v20; // [rsp+30h] [rbp-50h] BYREF
  __int64 v21; // [rsp+38h] [rbp-48h] BYREF
  unsigned int v22; // [rsp+40h] [rbp-40h]
  __int64 v23; // [rsp+48h] [rbp-38h] BYREF
  unsigned int v24; // [rsp+50h] [rbp-30h]

  if ( *(_BYTE *)(sub_1649C60(a2) + 16) == 53 )
    return 0;
  v6 = sub_157EB90(a3);
  v7 = sub_1632FA0(v6);
  v8 = sub_13E7A30(a1 + 4, *a1, v7, a1[3]);
  sub_13F2700(&v20, v8, a2, a3, a4);
  result = v21;
  if ( v20 != 1 )
  {
    result = 0;
    if ( v20 == 3 )
    {
      v19 = v22;
      if ( v22 > 0x40 )
        sub_16A4FD0(&v18, &v21);
      else
        v18 = v21;
      sub_16A7490(&v18, 1);
      v10 = v19;
      v11 = v18;
      v19 = 0;
      v17 = v10;
      v16 = v18;
      if ( v24 <= 0x40 )
        v12 = v23 == v18;
      else
        v12 = sub_16A5220(&v23, &v16);
      if ( v10 > 0x40 )
      {
        if ( v11 )
        {
          j_j___libc_free_0_0(v11);
          if ( v19 > 0x40 )
          {
            if ( v18 )
              j_j___libc_free_0_0(v18);
          }
        }
      }
      result = 0;
      if ( v12 )
      {
        v13 = sub_16498A0(a2);
        result = sub_159C0E0(v13, &v21);
      }
      if ( v20 == 3 )
      {
        if ( v24 > 0x40 && v23 )
        {
          v14 = result;
          j_j___libc_free_0_0(v23);
          result = v14;
        }
        if ( v22 > 0x40 )
        {
          if ( v21 )
          {
            v15 = result;
            j_j___libc_free_0_0(v21);
            return v15;
          }
        }
      }
    }
  }
  return result;
}
