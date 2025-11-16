// Function: sub_16A0F40
// Address: 0x16a0f40
//
__int64 __fastcall sub_16A0F40(__int64 a1, __int64 a2, double a3, double a4, double a5)
{
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  void *v10; // r12
  void **v11; // rdi
  char v12; // al
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  void **v16; // rdi
  char v17; // al
  __int64 v18; // r12
  _BYTE v19[8]; // [rsp+0h] [rbp-40h] BYREF
  _QWORD v20[7]; // [rsp+8h] [rbp-38h] BYREF

  if ( (unsigned int)sub_169C920(a1) != 2 )
    return 0;
  v10 = sub_16982C0();
  v11 = (void **)(*(_QWORD *)(a1 + 8) + 8LL);
  if ( *v11 == v10 )
    v12 = sub_16A0F40(v11, a2, v7, v8, v9);
  else
    v12 = sub_16984B0((__int64)v11);
  if ( v12 )
    return 1;
  v16 = (void **)(*(_QWORD *)(a1 + 8) + 40LL);
  v17 = v10 == *v16 ? sub_16A0F40(v16, a2, v13, v14, v15) : sub_16984B0((__int64)v16);
  if ( v17 )
    return 1;
  v18 = *(_QWORD *)(a1 + 8);
  sub_169C7A0(v20, (__int64 *)(v18 + 8));
  sub_16A0E20((__int64)v19, v18 + 32, 0, a3, a4, a5);
  LOBYTE(v18) = (unsigned int)sub_14A9E40(v18, (__int64)v19) != 1;
  sub_127D120(v20);
  return (unsigned int)v18;
}
