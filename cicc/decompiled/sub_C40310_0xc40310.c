// Function: sub_C40310
// Address: 0xc40310
//
__int64 __fastcall sub_C40310(__int64 a1)
{
  __int64 *v1; // r12
  _QWORD *v4; // r13
  void **v5; // rdi
  char v6; // al
  void **v7; // rdi
  char v8; // al
  __int64 v9; // rsi
  void **v10; // rdi
  int v11; // eax
  __int64 v12[8]; // [rsp+0h] [rbp-40h] BYREF

  LODWORD(v1) = 0;
  if ( (unsigned int)sub_C3CE50(a1) != 2 )
    return (unsigned int)v1;
  v4 = sub_C33340();
  v5 = *(void ***)(a1 + 8);
  if ( *v5 == v4 )
    v6 = sub_C40310(v5);
  else
    v6 = sub_C33940((__int64)v5);
  if ( v6 )
    return 1;
  v7 = (void **)(*(_QWORD *)(a1 + 8) + 24LL);
  v8 = v4 == *v7 ? sub_C40310(v7) : sub_C33940((__int64)v7);
  if ( v8 )
    return 1;
  v1 = *(__int64 **)(a1 + 8);
  if ( v4 == (_QWORD *)*v1 )
    sub_C3C790(v12, (_QWORD **)v1);
  else
    sub_C33EB0(v12, v1);
  v9 = (__int64)(v1 + 3);
  if ( v4 == (_QWORD *)v12[0] )
    sub_C3D800(v12, v9, 1u);
  else
    sub_C3ADF0((__int64)v12, v9, 1);
  v10 = *(void ***)(a1 + 8);
  if ( v4 == *v10 )
    v11 = sub_C3E510((__int64)v10, (__int64)v12);
  else
    v11 = sub_C37950((__int64)v10, (__int64)v12);
  LOBYTE(v1) = v11 != 1;
  if ( v4 == (_QWORD *)v12[0] )
  {
    sub_969EE0((__int64)v12);
    return (unsigned int)v1;
  }
  sub_C338F0((__int64)v12);
  return (unsigned int)v1;
}
