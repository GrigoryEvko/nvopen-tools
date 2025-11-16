// Function: sub_281EA60
// Address: 0x281ea60
//
__int64 __fastcall sub_281EA60(__int64 a1, _BYTE *a2)
{
  __int64 v2; // rdx
  unsigned __int64 v3; // rax
  __int64 v4; // r13
  unsigned __int64 v5; // r14
  __int64 **v6; // rax
  signed __int64 v7; // r12
  __int64 **v8; // r15
  __int64 *v9; // r14
  __int64 *v10; // rdx
  __int64 result; // rax
  __int64 v12; // [rsp-50h] [rbp-50h]
  _QWORD v13[9]; // [rsp-48h] [rbp-48h] BYREF

  if ( *(_BYTE *)a1 > 0x15u || *(_BYTE *)a1 == 5 )
    return 0;
  v13[0] = sub_9208B0((__int64)a2, *(_QWORD *)(a1 + 8));
  v13[1] = v2;
  v3 = sub_CA1930(v13);
  if ( !v3 )
    return 0;
  v4 = v3 & ((v3 - 1) | 7);
  if ( v4 )
    return 0;
  if ( *a2 )
    return 0;
  v5 = v3 >> 3;
  if ( v3 > 0x87 )
    return 0;
  if ( v5 == 16 )
    return a1;
  v6 = (__int64 **)sub_BCD420(*(__int64 **)(a1 + 8), 0x10 / v5);
  v7 = 0x10 / v5;
  v8 = v6;
  if ( v5 > 0x10 )
    return sub_AD1300(v6, 0, 0);
  v9 = (__int64 *)sub_22077B0(v7 * 8);
  if ( &v9[v7] != v9 )
  {
    v10 = v9;
    do
      *v10++ = a1;
    while ( &v9[v7] != v10 );
    v4 = (v7 * 8) >> 3;
  }
  result = sub_AD1300(v8, v9, v4);
  if ( v9 )
  {
    v12 = result;
    j_j___libc_free_0((unsigned __int64)v9);
    return v12;
  }
  return result;
}
