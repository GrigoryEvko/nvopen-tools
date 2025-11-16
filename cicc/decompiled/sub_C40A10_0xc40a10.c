// Function: sub_C40A10
// Address: 0xc40a10
//
__int64 __fastcall sub_C40A10(__int64 a1, _QWORD *a2)
{
  _DWORD *v2; // rax
  _DWORD *v3; // rbx
  __int64 v4; // rax
  _QWORD *v6; // rdi
  unsigned __int8 v7; // [rsp+Fh] [rbp-A1h]
  __int64 v8; // [rsp+10h] [rbp-A0h] BYREF
  unsigned int v9; // [rsp+18h] [rbp-98h]
  _QWORD v10[4]; // [rsp+20h] [rbp-90h] BYREF
  void *v11[4]; // [rsp+40h] [rbp-70h] BYREF
  _DWORD *v12; // [rsp+60h] [rbp-50h] BYREF
  unsigned int v13; // [rsp+68h] [rbp-48h]

  sub_C3E660((__int64)&v12, a1);
  v2 = sub_C33340();
  v3 = v2;
  if ( v2 == dword_3F65580 )
    sub_C3C640(v10, (__int64)v2, &v12);
  else
    sub_C3B160((__int64)v10, dword_3F65580, (__int64 *)&v12);
  if ( v13 > 0x40 && v12 )
    j_j___libc_free_0_0(v12);
  if ( a2 )
  {
    if ( v3 == dword_3F65580 )
    {
      sub_C3C460(v11, (__int64)v3);
      if ( v3 != (_DWORD *)v10[0] )
        goto LABEL_9;
    }
    else
    {
      sub_C37380(v11, (__int64)dword_3F65580);
      if ( v3 != (_DWORD *)v10[0] )
      {
LABEL_9:
        v7 = sub_C408D0((__int64)v10, v11);
        goto LABEL_10;
      }
    }
    v7 = sub_C40A10(v10, v11);
LABEL_10:
    if ( v3 == v11[0] )
      sub_C3E660((__int64)&v8, (__int64)v11);
    else
      sub_C3A850((__int64)&v8, (__int64 *)v11);
    if ( v3 == dword_3F655A0 )
    {
      sub_C3C640(&v12, (__int64)v3, &v8);
      v4 = (__int64)v12;
      if ( v3 != (_DWORD *)*a2 )
      {
LABEL_14:
        if ( v3 != (_DWORD *)v4 )
        {
          sub_C33870((__int64)a2, (__int64)&v12);
          goto LABEL_16;
        }
        sub_C338F0((__int64)a2);
        v6 = a2;
        if ( v3 != v12 )
        {
LABEL_33:
          sub_C338E0((__int64)v6, (__int64)&v12);
          if ( v3 != v12 )
            goto LABEL_17;
          goto LABEL_34;
        }
        goto LABEL_39;
      }
    }
    else
    {
      sub_C3B160((__int64)&v12, dword_3F655A0, &v8);
      v4 = (__int64)v12;
      if ( v3 != (_DWORD *)*a2 )
        goto LABEL_14;
    }
    if ( v3 == (_DWORD *)v4 )
    {
      sub_969EE0((__int64)a2);
      sub_C3C840(a2, &v12);
LABEL_16:
      if ( v3 != v12 )
      {
LABEL_17:
        sub_C338F0((__int64)&v12);
        goto LABEL_18;
      }
LABEL_34:
      sub_969EE0((__int64)&v12);
LABEL_18:
      if ( v9 > 0x40 && v8 )
        j_j___libc_free_0_0(v8);
      if ( v3 == v11[0] )
        sub_969EE0((__int64)v11);
      else
        sub_C338F0((__int64)v11);
      goto LABEL_23;
    }
    sub_969EE0((__int64)a2);
    v6 = a2;
    if ( v3 != v12 )
      goto LABEL_33;
LABEL_39:
    sub_C3C840(v6, &v12);
    goto LABEL_16;
  }
  if ( v3 != (_DWORD *)v10[0] )
  {
    v7 = sub_C408D0((__int64)v10, 0);
    if ( v3 != (_DWORD *)v10[0] )
      goto LABEL_24;
LABEL_31:
    sub_969EE0((__int64)v10);
    return v7;
  }
  v7 = sub_C40A10(v10, 0);
LABEL_23:
  if ( v3 == (_DWORD *)v10[0] )
    goto LABEL_31;
LABEL_24:
  sub_C338F0((__int64)v10);
  return v7;
}
