// Function: sub_13F39C0
// Address: 0x13f39c0
//
__int64 __fastcall sub_13F39C0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rbx
  __int64 result; // rax
  unsigned int v13; // r14d
  __int64 v14; // r12
  char v15; // bl
  __int64 v16; // rax
  __int64 v17; // [rsp+8h] [rbp-88h]
  __int64 v18; // [rsp+8h] [rbp-88h]
  __int64 v19; // [rsp+10h] [rbp-80h] BYREF
  unsigned int v20; // [rsp+18h] [rbp-78h]
  __int64 v21; // [rsp+20h] [rbp-70h] BYREF
  unsigned int v22; // [rsp+28h] [rbp-68h]
  int v23; // [rsp+30h] [rbp-60h] BYREF
  __int64 v24; // [rsp+38h] [rbp-58h] BYREF
  unsigned int v25; // [rsp+40h] [rbp-50h]
  __int64 v26; // [rsp+48h] [rbp-48h] BYREF
  unsigned int v27; // [rsp+50h] [rbp-40h]

  v8 = sub_157EB90(a3);
  v9 = sub_1632FA0(v8);
  v10 = sub_13E7A30(a1 + 4, *a1, v9, a1[3]);
  v23 = 0;
  v11 = v10;
  if ( !(unsigned __int8)sub_13EFC20(v10, a2, a3, a4, &v23, a5) )
  {
    sub_13EFEC0(v11);
    sub_13EFC20(v11, a2, a3, a4, &v23, a5);
  }
  result = v24;
  if ( v23 != 1 )
  {
    result = 0;
    if ( v23 == 3 )
    {
      v22 = v25;
      if ( v25 > 0x40 )
        sub_16A4FD0(&v21, &v24);
      else
        v21 = v24;
      sub_16A7490(&v21, 1);
      v13 = v22;
      v14 = v21;
      v22 = 0;
      v20 = v13;
      v19 = v21;
      if ( v27 <= 0x40 )
        v15 = v26 == v21;
      else
        v15 = sub_16A5220(&v26, &v19);
      if ( v13 > 0x40 )
      {
        if ( v14 )
        {
          j_j___libc_free_0_0(v14);
          if ( v22 > 0x40 )
          {
            if ( v21 )
              j_j___libc_free_0_0(v21);
          }
        }
      }
      result = 0;
      if ( v15 )
      {
        v16 = sub_16498A0(a2);
        result = sub_159C0E0(v16, &v24);
      }
      if ( v23 == 3 )
      {
        if ( v27 > 0x40 && v26 )
        {
          v17 = result;
          j_j___libc_free_0_0(v26);
          result = v17;
        }
        if ( v25 > 0x40 )
        {
          if ( v24 )
          {
            v18 = result;
            j_j___libc_free_0_0(v24);
            return v18;
          }
        }
      }
    }
  }
  return result;
}
