// Function: sub_C40CE0
// Address: 0xc40ce0
//
_QWORD *__fastcall sub_C40CE0(_QWORD *a1, __int64 a2, unsigned int a3, unsigned __int8 a4)
{
  __int64 *v5; // rsi
  _QWORD *v6; // rbx
  __int64 *v7; // rsi
  _DWORD *v9; // [rsp+8h] [rbp-138h]
  _DWORD *v10; // [rsp+8h] [rbp-138h]
  __int64 v13[4]; // [rsp+30h] [rbp-110h] BYREF
  void *v14[4]; // [rsp+50h] [rbp-F0h] BYREF
  __int64 v15[4]; // [rsp+70h] [rbp-D0h] BYREF
  _QWORD v16[4]; // [rsp+90h] [rbp-B0h] BYREF
  _QWORD v17[4]; // [rsp+B0h] [rbp-90h] BYREF
  _QWORD v18[4]; // [rsp+D0h] [rbp-70h] BYREF
  __int64 v19[10]; // [rsp+F0h] [rbp-50h] BYREF

  v5 = (__int64 *)(*(_QWORD *)(a2 + 8) + 24LL);
  v6 = sub_C33340();
  if ( (_QWORD *)*v5 == v6 )
    sub_C3C790(v15, (_QWORD **)v5);
  else
    sub_C33EB0(v15, v5);
  v9 = (_DWORD *)v15[0];
  if ( (_QWORD *)v15[0] == v6 )
  {
    sub_C40CE0(v18, v15, a3, a4);
    sub_C3C840(v19, v18);
    sub_C3C840(v16, v19);
    sub_969EE0((__int64)v19);
    sub_969EE0((__int64)v18);
  }
  else
  {
    sub_C33EB0(v17, v15);
    sub_C3BDC0((__int64)v18, (__int64)v17, a3, a4);
    sub_C338E0((__int64)v19, (__int64)v18);
    sub_C407B0(v16, v19, v9);
    sub_C338F0((__int64)v19);
    sub_C338F0((__int64)v18);
    sub_C338F0((__int64)v17);
  }
  v7 = *(__int64 **)(a2 + 8);
  if ( (_QWORD *)*v7 == v6 )
    sub_C3C790(v13, (_QWORD **)v7);
  else
    sub_C33EB0(v13, v7);
  v10 = (_DWORD *)v13[0];
  if ( (_QWORD *)v13[0] == v6 )
  {
    sub_C40CE0(v18, v13, a3, a4);
    sub_C3C840(v19, v18);
    sub_C3C840(v14, v19);
    sub_969EE0((__int64)v19);
    sub_969EE0((__int64)v18);
  }
  else
  {
    sub_C33EB0(v17, v13);
    sub_C3BDC0((__int64)v18, (__int64)v17, a3, a4);
    sub_C338E0((__int64)v19, (__int64)v18);
    sub_C407B0(v14, v19, v10);
    sub_C338F0((__int64)v19);
    sub_C338F0((__int64)v18);
    sub_C338F0((__int64)v17);
  }
  sub_C3C930(a1, (__int64)dword_3F655A0, v14, v16);
  if ( v14[0] == v6 )
    sub_969EE0((__int64)v14);
  else
    sub_C338F0((__int64)v14);
  if ( (_QWORD *)v13[0] == v6 )
    sub_969EE0((__int64)v13);
  else
    sub_C338F0((__int64)v13);
  if ( (_QWORD *)v16[0] == v6 )
    sub_969EE0((__int64)v16);
  else
    sub_C338F0((__int64)v16);
  if ( (_QWORD *)v15[0] == v6 )
    sub_969EE0((__int64)v15);
  else
    sub_C338F0((__int64)v15);
  return a1;
}
