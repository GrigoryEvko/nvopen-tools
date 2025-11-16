// Function: sub_C41050
// Address: 0xc41050
//
_QWORD *__fastcall sub_C41050(_QWORD *a1, __int64 a2, _DWORD *a3, unsigned __int8 a4)
{
  __int64 *v5; // r13
  _DWORD *v6; // r14
  _QWORD *v7; // rbx
  __int64 *v8; // rsi
  _QWORD *v10; // r13
  _DWORD *v11; // [rsp+8h] [rbp-138h]
  int v14; // [rsp+18h] [rbp-128h]
  void *v16[4]; // [rsp+30h] [rbp-110h] BYREF
  _QWORD *v17[4]; // [rsp+50h] [rbp-F0h] BYREF
  __int64 v18[4]; // [rsp+70h] [rbp-D0h] BYREF
  _QWORD *v19; // [rsp+90h] [rbp-B0h] BYREF
  _QWORD *v20; // [rsp+98h] [rbp-A8h]
  _QWORD v21[4]; // [rsp+B0h] [rbp-90h] BYREF
  _QWORD v22[4]; // [rsp+D0h] [rbp-70h] BYREF
  __int64 v23[10]; // [rsp+F0h] [rbp-50h] BYREF

  v5 = *(__int64 **)(a2 + 8);
  v6 = (_DWORD *)*v5;
  v7 = sub_C33340();
  if ( v6 == (_DWORD *)v7 )
  {
    sub_C41050(v22, v5, a3, a4);
    sub_C3C840(v23, v22);
    sub_C3C840(v16, v23);
    sub_969EE0((__int64)v23);
    sub_969EE0((__int64)v22);
  }
  else
  {
    sub_C3C390(v22, v5, a3, a4);
    sub_C338E0((__int64)v23, (__int64)v22);
    sub_C407B0(v16, v23, v6);
    sub_C338F0((__int64)v23);
    sub_C338F0((__int64)v22);
  }
  v8 = (__int64 *)(*(_QWORD *)(a2 + 8) + 24LL);
  if ( (_QWORD *)*v8 == v7 )
    sub_C3C790(v17, (_QWORD **)v8);
  else
    sub_C33EB0(v17, v8);
  if ( (unsigned int)sub_C3CE50(a2) != 2 )
    goto LABEL_6;
  v14 = -*a3;
  if ( v7 == v17[0] )
    sub_C3C790(v18, v17);
  else
    sub_C33EB0(v18, (__int64 *)v17);
  v11 = (_DWORD *)v18[0];
  if ( v7 == (_QWORD *)v18[0] )
  {
    sub_C40CE0(v22, (__int64)v18, v14, a4);
    sub_C3C840(v23, v22);
    sub_C3C840(&v19, v23);
    sub_969EE0((__int64)v23);
    sub_969EE0((__int64)v22);
  }
  else
  {
    sub_C33EB0(v21, v18);
    sub_C3BDC0((__int64)v22, (__int64)v21, v14, a4);
    sub_C338E0((__int64)v23, (__int64)v22);
    sub_C407B0(&v19, v23, v11);
    sub_C338F0((__int64)v23);
    sub_C338F0((__int64)v22);
    sub_C338F0((__int64)v21);
  }
  if ( v7 == v17[0] )
  {
    if ( v7 == v19 )
    {
      sub_969EE0((__int64)v17);
      sub_C3C840(v17, &v19);
      goto LABEL_22;
    }
    sub_969EE0((__int64)v17);
LABEL_27:
    if ( v7 == v19 )
      sub_C3C840(v17, &v19);
    else
      sub_C338E0((__int64)v17, (__int64)&v19);
    goto LABEL_22;
  }
  if ( v7 == v19 )
  {
    sub_C338F0((__int64)v17);
    goto LABEL_27;
  }
  sub_C33870((__int64)v17, (__int64)&v19);
LABEL_22:
  if ( v7 == v19 )
  {
    if ( v20 )
    {
      v10 = &v20[3 * *(v20 - 1)];
      while ( v20 != v10 )
      {
        v10 -= 3;
        if ( v7 == (_QWORD *)*v10 )
          sub_969EE0((__int64)v10);
        else
          sub_C338F0((__int64)v10);
      }
      j_j_j___libc_free_0_0(v10 - 1);
    }
  }
  else
  {
    sub_C338F0((__int64)&v19);
  }
  sub_91D830(v18);
LABEL_6:
  sub_C3C930(a1, (__int64)dword_3F655A0, v16, v17);
  if ( v7 == v17[0] )
    sub_969EE0((__int64)v17);
  else
    sub_C338F0((__int64)v17);
  if ( v7 == v16[0] )
    sub_969EE0((__int64)v16);
  else
    sub_C338F0((__int64)v16);
  return a1;
}
