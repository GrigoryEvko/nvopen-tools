// Function: sub_14D17B0
// Address: 0x14d17b0
//
__int64 __fastcall sub_14D17B0(_QWORD *a1, __int64 a2, double a3)
{
  char v4; // al
  __int64 v5; // r13
  __int64 v6; // rdi
  __int64 v7; // r12
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // rax
  __int64 v12; // r13
  __int64 v13; // rax
  __int64 v14; // rdi
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // rax
  __int64 v18; // r13
  __int64 v19; // rsi
  __int64 v20; // rbx
  __int64 v21; // rsi
  __int64 v22; // rbx
  __int64 v23; // r13
  float v24; // xmm0_4
  __int64 v25; // rdi
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // rax
  __int64 v29; // rsi
  __int64 v30; // rbx
  _BYTE v31[32]; // [rsp+10h] [rbp-70h] BYREF
  _BYTE v32[8]; // [rsp+30h] [rbp-50h] BYREF
  __int64 v33; // [rsp+38h] [rbp-48h] BYREF
  __int64 v34; // [rsp+40h] [rbp-40h]

  v4 = *((_BYTE *)a1 + 8);
  if ( v4 == 1 )
  {
    v12 = sub_1698280(a1);
    sub_169D3F0(v31, a3);
    sub_169E320(&v33, v31, v12);
    sub_1698460(v31);
    v13 = sub_1698260();
    sub_16A3360(v32, v13, 0, v31);
    v14 = *a1;
    v7 = sub_159CCF0(*a1, v32);
    v17 = sub_16982C0(v14, v32, v15, v16);
    if ( v33 != v17 )
      goto LABEL_4;
    v18 = v34;
    if ( !v34 )
      return v7;
    v21 = 32LL * *(_QWORD *)(v34 - 8);
    v22 = v34 + v21;
    if ( v34 != v34 + v21 )
    {
      do
      {
        v22 -= 32;
        sub_127D120((_QWORD *)(v22 + 8));
      }
      while ( v18 != v22 );
    }
    goto LABEL_11;
  }
  if ( v4 == 2 )
  {
    v23 = sub_1698270(a1, a2);
    v24 = a3;
    sub_169D3B0(v31, v24);
    sub_169E320(&v33, v31, v23);
    sub_1698460(v31);
    v25 = *a1;
    v7 = sub_159CCF0(*a1, v32);
    v28 = sub_16982C0(v25, v32, v26, v27);
    if ( v33 != v28 )
      goto LABEL_4;
    v18 = v34;
    if ( !v34 )
      return v7;
    v29 = 32LL * *(_QWORD *)(v34 - 8);
    v30 = v34 + v29;
    if ( v34 != v34 + v29 )
    {
      do
      {
        v30 -= 32;
        sub_127D120((_QWORD *)(v30 + 8));
      }
      while ( v18 != v30 );
    }
LABEL_11:
    j_j_j___libc_free_0_0(v18 - 8);
    return v7;
  }
  v5 = sub_1698280(a1);
  sub_169D3F0(v31, a3);
  sub_169E320(&v33, v31, v5);
  sub_1698460(v31);
  v6 = *a1;
  v7 = sub_159CCF0(*a1, v32);
  v10 = sub_16982C0(v6, v32, v8, v9);
  if ( v33 == v10 )
  {
    v18 = v34;
    if ( !v34 )
      return v7;
    v19 = 32LL * *(_QWORD *)(v34 - 8);
    v20 = v34 + v19;
    if ( v34 != v34 + v19 )
    {
      do
      {
        v20 -= 32;
        sub_127D120((_QWORD *)(v20 + 8));
      }
      while ( v18 != v20 );
    }
    goto LABEL_11;
  }
LABEL_4:
  sub_1698460(&v33);
  return v7;
}
