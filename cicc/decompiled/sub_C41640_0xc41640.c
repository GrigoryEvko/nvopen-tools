// Function: sub_C41640
// Address: 0xc41640
//
__int64 __fastcall sub_C41640(__int64 *a1, _DWORD *a2, char a3, bool *a4)
{
  __int64 v5; // rbx
  _DWORD *v7; // rax
  void *v8; // r15
  unsigned int v10; // eax
  __int64 v11; // rsi
  unsigned int v13; // [rsp+Ch] [rbp-94h]
  _BYTE v14[32]; // [rsp+10h] [rbp-90h] BYREF
  __int64 v15; // [rsp+30h] [rbp-70h] BYREF
  unsigned int v16; // [rsp+38h] [rbp-68h]
  __int64 v17[10]; // [rsp+50h] [rbp-50h] BYREF

  v5 = *a1;
  if ( a2 == (_DWORD *)*a1 )
  {
    *a4 = 0;
    return 0;
  }
  v7 = sub_C33340();
  v8 = v7;
  if ( v7 != (_DWORD *)v5 )
  {
    if ( a2 != v7 )
      return sub_C396A0((__int64)a1, a2, a3, a4);
    v13 = sub_C396A0((__int64)a1, dword_3F65580, a3, a4);
    sub_C3A850((__int64)&v15, a1);
    sub_C3C640(v17, (__int64)a2, &v15);
    if ( a2 == (_DWORD *)*a1 )
    {
      if ( a2 == (_DWORD *)v17[0] )
      {
        sub_969EE0((__int64)a1);
        sub_C3C840(a1, v17);
        goto LABEL_17;
      }
      sub_969EE0((__int64)a1);
    }
    else
    {
      if ( a2 != (_DWORD *)v17[0] )
      {
        sub_C33870((__int64)a1, (__int64)v17);
        goto LABEL_17;
      }
      sub_C338F0((__int64)a1);
    }
    if ( v8 == (void *)v17[0] )
      sub_C3C840(a1, v17);
    else
      sub_C338E0((__int64)a1, (__int64)v17);
LABEL_17:
    sub_91D830(v17);
    if ( v16 > 0x40 )
    {
      if ( v15 )
        j_j___libc_free_0_0(v15);
    }
    return v13;
  }
  v10 = sub_C396A0(a1[1], a2, a3, a4);
  v11 = (__int64)a1;
  v13 = v10;
  if ( v5 == *a1 )
    v11 = a1[1];
  sub_C338E0((__int64)v14, v11);
  sub_C338E0((__int64)v17, (__int64)v14);
  sub_C407B0(&v15, v17, a2);
  sub_C338F0((__int64)v17);
  if ( v5 == *a1 )
  {
    if ( v5 == v15 )
    {
      sub_969EE0((__int64)a1);
      sub_C3C840(a1, &v15);
      goto LABEL_10;
    }
    sub_969EE0((__int64)a1);
  }
  else
  {
    if ( v5 != v15 )
    {
      sub_C33870((__int64)a1, (__int64)&v15);
      goto LABEL_10;
    }
    sub_C338F0((__int64)a1);
  }
  if ( v5 != v15 )
  {
    sub_C338E0((__int64)a1, (__int64)&v15);
    if ( v5 != v15 )
      goto LABEL_11;
LABEL_23:
    sub_969EE0((__int64)&v15);
    goto LABEL_12;
  }
  sub_C3C840(a1, &v15);
LABEL_10:
  if ( v5 == v15 )
    goto LABEL_23;
LABEL_11:
  sub_C338F0((__int64)&v15);
LABEL_12:
  sub_C338F0((__int64)v14);
  return v13;
}
