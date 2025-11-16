// Function: sub_13D6AF0
// Address: 0x13d6af0
//
__int64 __fastcall sub_13D6AF0(double *a1, __int64 a2)
{
  unsigned __int8 v2; // al
  __int64 v3; // r14
  unsigned int v4; // r12d
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // rax
  __int64 v9; // r13
  __int64 v10; // rsi
  __int64 v11; // rbx
  __int64 v12; // rax
  __int64 v13; // r12
  __int64 v14; // r14
  __int64 v15; // rdi
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // rax
  __int64 v19; // rbx
  double v20; // [rsp+8h] [rbp-78h]
  double v21; // [rsp+8h] [rbp-78h]
  _BYTE v22[32]; // [rsp+10h] [rbp-70h] BYREF
  char v23[8]; // [rsp+30h] [rbp-50h] BYREF
  __int64 v24; // [rsp+38h] [rbp-48h] BYREF
  __int64 v25; // [rsp+40h] [rbp-40h]

  v2 = *(_BYTE *)(a2 + 16);
  if ( v2 == 14 )
  {
    v20 = *a1;
    v3 = sub_1698280(a1);
    sub_169D3F0(v22, v20);
    sub_169E320(&v24, v22, v3);
    sub_1698460(v22);
    sub_16A3360(v23, *(_QWORD *)(a2 + 32), 0, v22);
    v4 = sub_1594120(a2, v23);
    v7 = sub_16982C0(a2, v23, v5, v6);
    if ( v24 != v7 )
      goto LABEL_3;
    v9 = v25;
    if ( !v25 )
      return v4;
    v10 = 32LL * *(_QWORD *)(v25 - 8);
    v11 = v25 + v10;
    if ( v25 != v25 + v10 )
    {
      do
      {
        v11 -= 32;
        sub_127D120((_QWORD *)(v11 + 8));
      }
      while ( v9 != v11 );
    }
LABEL_11:
    j_j_j___libc_free_0_0(v9 - 8);
    return v4;
  }
  if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) != 16 )
    return 0;
  if ( v2 > 0x10u )
    return 0;
  v12 = sub_15A1020(a2);
  v13 = v12;
  if ( !v12 || *(_BYTE *)(v12 + 16) != 14 )
    return 0;
  v21 = *a1;
  v14 = sub_1698280(a2);
  sub_169D3F0(v22, v21);
  sub_169E320(&v24, v22, v14);
  sub_1698460(v22);
  sub_16A3360(v23, *(_QWORD *)(v13 + 32), 0, v22);
  v15 = v13;
  v4 = sub_1594120(v13, v23);
  v18 = sub_16982C0(v15, v23, v16, v17);
  if ( v24 == v18 )
  {
    v9 = v25;
    if ( !v25 )
      return v4;
    v19 = v25 + 32LL * *(_QWORD *)(v25 - 8);
    if ( v25 != v19 )
    {
      do
      {
        v19 -= 32;
        sub_127D120((_QWORD *)(v19 + 8));
      }
      while ( v9 != v19 );
    }
    goto LABEL_11;
  }
LABEL_3:
  sub_1698460(&v24);
  return v4;
}
