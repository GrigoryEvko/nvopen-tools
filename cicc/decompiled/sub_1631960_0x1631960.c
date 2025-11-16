// Function: sub_1631960
// Address: 0x1631960
//
__int64 __fastcall sub_1631960(__int64 a1, __int64 a2)
{
  __int64 v4; // r13
  __int64 v5; // rax
  __int64 *v6; // rsi
  __int64 **v7; // rax
  __int64 *v8; // rdi
  __int64 v10; // [rsp+0h] [rbp-80h] BYREF
  __int64 v11; // [rsp+8h] [rbp-78h]
  __int64 v12; // [rsp+10h] [rbp-70h] BYREF
  __int64 *v13; // [rsp+30h] [rbp-50h] BYREF
  __int64 v14; // [rsp+38h] [rbp-48h]
  _BYTE v15[64]; // [rsp+40h] [rbp-40h] BYREF

  if ( !a1 )
    return a2;
  v4 = a1;
  if ( !a2 )
    return v4;
  v5 = *(unsigned int *)(a1 + 8);
  v10 = 0;
  v11 = 1;
  v6 = (__int64 *)(a1 - 8 * v5);
  v7 = (__int64 **)&v12;
  do
    *v7++ = (__int64 *)-4LL;
  while ( v7 != &v13 );
  v14 = 0x400000000LL;
  v13 = (__int64 *)v15;
  sub_1630C80((__int64)&v10, v6, (__int64 *)a1);
  sub_1630C80((__int64)&v10, (__int64 *)(a2 - 8LL * *(unsigned int *)(a2 + 8)), (__int64 *)a2);
  v8 = (__int64 *)(*(_QWORD *)(a1 + 16) & 0xFFFFFFFFFFFFFFF8LL);
  if ( (*(_QWORD *)(a1 + 16) & 4) != 0 )
    v8 = (__int64 *)*v8;
  v4 = sub_1628280(v8, v13, (__int64 *)(unsigned int)v14);
  if ( v13 != (__int64 *)v15 )
    _libc_free((unsigned __int64)v13);
  if ( (v11 & 1) != 0 )
    return v4;
  j___libc_free_0(v12);
  return v4;
}
