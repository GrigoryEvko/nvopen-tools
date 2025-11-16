// Function: sub_2AF9BD0
// Address: 0x2af9bd0
//
void __fastcall sub_2AF9BD0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rbx
  __int64 v8; // r14
  __int64 *v9; // rsi
  int v10; // edx
  __int64 *v11; // rcx
  __int64 v12; // rax
  __int64 *v13; // rbx
  __int64 v14; // r14
  unsigned int v15; // r9d
  __int64 v16; // rdx
  bool v17; // al
  __int64 v18; // rax
  __int64 v19; // rdx
  unsigned int v20; // ebx
  __int64 v21; // rdx
  unsigned __int64 v22; // rax
  unsigned int v23; // r9d
  unsigned __int64 v24; // rcx
  bool v26; // [rsp+13h] [rbp-9Dh]
  unsigned int v27; // [rsp+14h] [rbp-9Ch]
  unsigned __int64 v28; // [rsp+18h] [rbp-98h]
  unsigned int v29; // [rsp+18h] [rbp-98h]
  __int64 v30; // [rsp+20h] [rbp-90h] BYREF
  __int64 v31; // [rsp+28h] [rbp-88h]
  __int64 *v32; // [rsp+30h] [rbp-80h] BYREF
  __int64 v33; // [rsp+38h] [rbp-78h]
  _BYTE v34[112]; // [rsp+40h] [rbp-70h] BYREF

  v7 = a3;
  v8 = (8 * a3) >> 3;
  v32 = (__int64 *)v34;
  v33 = 0x800000000LL;
  if ( (unsigned __int64)(8 * a3) > 0x40 )
  {
    sub_C8D5F0((__int64)&v32, v34, (8 * a3) >> 3, 8u, a5, a6);
    v9 = v32;
    v10 = v33;
    v11 = &v32[(unsigned int)v33];
  }
  else
  {
    v9 = (__int64 *)v34;
    v10 = 0;
    v11 = (__int64 *)v34;
  }
  if ( v7 > 0 )
  {
    v12 = 0;
    do
    {
      v11[v12] = a2[v12];
      ++v12;
    }
    while ( v8 - v12 > 0 );
    v9 = v32;
    v10 = v33;
  }
  LODWORD(v33) = v10 + v8;
  sub_9B8FE0(a1, v9, (unsigned int)(v10 + v8));
  if ( *(_BYTE *)a1 != 61 )
    goto LABEL_8;
  v13 = &a2[v7 - 1];
  v26 = 0;
  v27 = 0;
  v14 = sub_B43CA0(a1) + 312;
  while ( 1 )
  {
    v30 = sub_9208B0(v14, *(_QWORD *)(*v13 + 8));
    v31 = v16;
    v28 = (unsigned __int64)sub_CA1930(&v30) >> 3;
    v17 = sub_CE8520(*v13);
    if ( v17 )
      break;
    v15 = v27 | ((1 << v28) - 1);
    if ( a2 == v13 )
      goto LABEL_17;
LABEL_14:
    --v13;
    v27 = v15 << v28;
  }
  v26 = v17;
  v15 = sub_CE8560(*v13) | v27;
  if ( a2 != v13 )
    goto LABEL_14;
LABEL_17:
  v29 = v15;
  v18 = sub_9208B0(v14, *(_QWORD *)(a1 + 8));
  v31 = v19;
  v30 = v18;
  v20 = (unsigned __int64)sub_CA1930(&v30) >> 3;
  v30 = sub_9208B0(v14, *(_QWORD *)(*a2 + 8));
  v31 = v21;
  v22 = sub_CA1930(&v30);
  v23 = v29;
  v24 = a3 * (unsigned int)(v22 >> 3);
  if ( v24 >= v20 )
  {
    if ( v26 )
      goto LABEL_20;
  }
  else
  {
    if ( v26 )
      v23 = ~(-1 << v24) & v29;
    else
      v23 = (1 << v24) - 1;
LABEL_20:
    sub_CE85E0(a1, v23);
  }
LABEL_8:
  if ( v32 != (__int64 *)v34 )
    _libc_free((unsigned __int64)v32);
}
