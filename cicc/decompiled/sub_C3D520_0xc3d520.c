// Function: sub_C3D520
// Address: 0xc3d520
//
__int64 __fastcall sub_C3D520(__int64 a1, __int64 *a2, __int64 a3, __int64 *a4, unsigned int a5)
{
  bool v9; // r14
  __int64 *v10; // rsi
  _QWORD *v11; // r14
  __int64 *v12; // rsi
  __int64 *v13; // rsi
  __int64 *v14; // rsi
  unsigned int v15; // r12d
  __int64 *v16; // rsi
  bool v18; // al
  _QWORD v19[4]; // [rsp+20h] [rbp-B0h] BYREF
  _QWORD v20[4]; // [rsp+40h] [rbp-90h] BYREF
  _QWORD v21[4]; // [rsp+60h] [rbp-70h] BYREF
  _QWORD v22[10]; // [rsp+80h] [rbp-50h] BYREF

  if ( (unsigned int)sub_C3CE50((__int64)a2) == 1 )
    goto LABEL_28;
  if ( (unsigned int)sub_C3CE50(a3) == 1 || (unsigned int)sub_C3CE50((__int64)a2) == 3 )
    goto LABEL_26;
  if ( (unsigned int)sub_C3CE50(a3) == 3 )
    goto LABEL_28;
  if ( !(unsigned int)sub_C3CE50((__int64)a2) && !(unsigned int)sub_C3CE50(a3) )
  {
    v9 = sub_C3CE80((__int64)a2);
    if ( v9 != sub_C3CE80(a3) )
    {
      v15 = 1;
      v18 = sub_C3CE80((__int64)a4);
      sub_C3D480((__int64)a4, 0, v18, 0);
      return v15;
    }
  }
  if ( !(unsigned int)sub_C3CE50((__int64)a2) )
  {
LABEL_28:
    v15 = 0;
    sub_C3C9E0(a4, a2);
    return v15;
  }
  if ( !(unsigned int)sub_C3CE50(a3) )
  {
LABEL_26:
    v16 = (__int64 *)a3;
    v15 = 0;
    sub_C3C9E0(a4, v16);
    return v15;
  }
  v10 = (__int64 *)a2[1];
  v11 = sub_C33340();
  if ( (_QWORD *)*v10 == v11 )
    sub_C3C790(v19, (_QWORD **)v10);
  else
    sub_C33EB0(v19, v10);
  v12 = (__int64 *)(a2[1] + 24);
  if ( (_QWORD *)*v12 == v11 )
    sub_C3C790(v20, (_QWORD **)v12);
  else
    sub_C33EB0(v20, v12);
  v13 = *(__int64 **)(a3 + 8);
  if ( (_QWORD *)*v13 == v11 )
    sub_C3C790(v21, (_QWORD **)v13);
  else
    sub_C33EB0(v21, v13);
  v14 = (__int64 *)(*(_QWORD *)(a3 + 8) + 24LL);
  if ( (_QWORD *)*v14 == v11 )
    sub_C3C790(v22, (_QWORD **)v14);
  else
    sub_C33EB0(v22, v14);
  v15 = sub_C3D860(a4, v19, v20, v21, v22, a5);
  if ( (_QWORD *)v22[0] == v11 )
    sub_969EE0((__int64)v22);
  else
    sub_C338F0((__int64)v22);
  if ( (_QWORD *)v21[0] == v11 )
    sub_969EE0((__int64)v21);
  else
    sub_C338F0((__int64)v21);
  if ( (_QWORD *)v20[0] == v11 )
    sub_969EE0((__int64)v20);
  else
    sub_C338F0((__int64)v20);
  if ( (_QWORD *)v19[0] == v11 )
    sub_969EE0((__int64)v19);
  else
    sub_C338F0((__int64)v19);
  return v15;
}
