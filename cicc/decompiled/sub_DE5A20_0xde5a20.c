// Function: sub_DE5A20
// Address: 0xde5a20
//
_QWORD *__fastcall sub_DE5A20(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  char v6; // bl
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rax
  unsigned __int64 v10; // rax
  char v11; // al
  __int64 v12; // rax
  _QWORD *v13; // rax
  __int64 v14; // rax
  _QWORD *v15; // rax
  _QWORD *v16; // rax
  _QWORD *v17; // rbx
  _QWORD *v18; // r12
  __int64 v19; // [rsp+8h] [rbp-78h]
  unsigned int v20; // [rsp+10h] [rbp-70h]
  char v21; // [rsp+10h] [rbp-70h]
  unsigned __int64 v23; // [rsp+20h] [rbp-60h] BYREF
  unsigned int v24; // [rsp+28h] [rbp-58h]
  _QWORD *v25; // [rsp+30h] [rbp-50h] BYREF
  __int64 v26; // [rsp+38h] [rbp-48h]
  _QWORD *v27; // [rsp+40h] [rbp-40h] BYREF
  _QWORD *v28; // [rsp+48h] [rbp-38h]

  if ( sub_D96A50(a2) )
    return (_QWORD *)sub_D970F0((__int64)a1);
  v6 = 0;
  v7 = sub_D95540(a2);
  v20 = sub_D97050((__int64)a1, v7);
  v25 = (_QWORD *)sub_BCAE30(a3);
  v26 = v8;
  if ( v20 >= (unsigned int)sub_CA1930(&v25) )
    goto LABEL_26;
  v9 = sub_DBB9F0((__int64)a1, a2, 0, 0);
  LODWORD(v26) = *(_DWORD *)(v9 + 8);
  if ( (unsigned int)v26 > 0x40 )
  {
    v19 = v9;
    sub_C43780((__int64)&v25, (const void **)v9);
    v9 = v19;
  }
  else
  {
    v25 = *(_QWORD **)v9;
  }
  LODWORD(v28) = *(_DWORD *)(v9 + 24);
  if ( (unsigned int)v28 > 0x40 )
    sub_C43780((__int64)&v27, (const void **)(v9 + 16));
  else
    v27 = *(_QWORD **)(v9 + 16);
  v24 = v20;
  if ( v20 > 0x40 )
  {
    sub_C43690((__int64)&v23, -1, 1);
  }
  else
  {
    v10 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v20;
    if ( !v20 )
      v10 = 0;
    v23 = v10;
  }
  v11 = sub_AB1B10((__int64)&v25, (__int64)&v23);
  if ( v24 > 0x40 && v23 )
  {
    v21 = v11;
    j_j___libc_free_0_0(v23);
    v11 = v21;
  }
  if ( v11 )
  {
    if ( a4 )
    {
      v12 = sub_D95540(a2);
      v13 = sub_DA2C50((__int64)a1, v12, -1, 1u);
      v6 = sub_DDD5B0(a1, a4, 33, a2, (__int64)v13);
    }
  }
  else
  {
    v6 = 1;
  }
  if ( (unsigned int)v28 > 0x40 && v27 )
    j_j___libc_free_0_0(v27);
  if ( (unsigned int)v26 > 0x40 && v25 )
    j_j___libc_free_0_0(v25);
  if ( v6 )
  {
    v14 = sub_D95540(a2);
    v15 = sub_DA2C50((__int64)a1, v14, 1, 0);
    v16 = sub_DC7ED0(a1, a2, (__int64)v15, 0, 0);
    return sub_DC2B70((__int64)a1, (__int64)v16, a3, 0);
  }
  else
  {
LABEL_26:
    v17 = sub_DA2C50((__int64)a1, a3, 1, 0);
    v27 = sub_DC5760((__int64)a1, a2, a3, 0);
    v25 = &v27;
    v28 = v17;
    v26 = 0x200000002LL;
    v18 = sub_DC7EB0(a1, (__int64)&v25, 0, 0);
    if ( v25 != &v27 )
      _libc_free(v25, &v25);
    return v18;
  }
}
