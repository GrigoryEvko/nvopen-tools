// Function: sub_CB0D90
// Address: 0xcb0d90
//
__int64 __fastcall sub_CB0D90(__int64 a1, __int64 a2)
{
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 **v6; // rax
  __int64 *v7; // r12
  __int64 *v8; // r13
  __int64 *v9; // r14
  __int64 v10; // rdi
  unsigned int v11; // ecx
  __int64 *v12; // r14
  __int64 *v13; // r15
  __int64 v14; // rdi
  __int64 *v15; // rdi
  _QWORD *v16; // rax
  unsigned int v17; // r8d
  __int64 **v19; // r15
  __int64 v20; // r12
  __int64 v21; // rax
  __int64 *v22; // r14
  __int64 *v23; // r15
  __int64 *v24; // r14
  __int64 i; // rdx
  __int64 v26; // rdi
  unsigned int v27; // ecx
  __int64 *v28; // r14
  __int64 v29; // rdi
  _QWORD *v30; // [rsp+8h] [rbp-38h]

  v30 = (_QWORD *)sub_CA8A20(*(_QWORD *)(a1 + 80));
  if ( !(unsigned __int8)sub_CAF190(**(__int64 ****)(a1 + 592), a2, v3, v4, v5) )
  {
    v6 = *(__int64 ***)(a1 + 592);
    v7 = *v6;
    *v6 = 0;
    if ( !v7 )
      goto LABEL_15;
    sub_CB0380(v7[16]);
    v8 = (__int64 *)v7[3];
    v9 = &v8[*((unsigned int *)v7 + 8)];
    while ( v9 != v8 )
    {
      v10 = *v8;
      v11 = (unsigned int)(((__int64)v8 - v7[3]) >> 3) >> 7;
      a2 = 4096LL << v11;
      if ( v11 >= 0x1E )
        a2 = 0x40000000000LL;
      ++v8;
      sub_C7D6A0(v10, a2, 16);
    }
    v12 = (__int64 *)v7[9];
    v13 = &v12[2 * *((unsigned int *)v7 + 20)];
    if ( v12 == v13 )
      goto LABEL_10;
    do
    {
      a2 = v12[1];
      v14 = *v12;
      v12 += 2;
      sub_C7D6A0(v14, a2, 16);
    }
    while ( v13 != v12 );
    goto LABEL_9;
  }
  v19 = *(__int64 ***)(a1 + 592);
  v20 = **v19;
  v21 = sub_22077B0(160);
  v22 = (__int64 *)v21;
  if ( v21 )
  {
    a2 = v20;
    sub_CAFBE0(v21, v20);
  }
  v7 = *v19;
  *v19 = v22;
  if ( v7 )
  {
    sub_CB0380(v7[16]);
    v23 = (__int64 *)v7[3];
    v24 = &v23[*((unsigned int *)v7 + 8)];
    if ( v23 != v24 )
    {
      for ( i = v7[3]; ; i = v7[3] )
      {
        v26 = *v23;
        v27 = (unsigned int)(((__int64)v23 - i) >> 3) >> 7;
        a2 = 4096LL << v27;
        if ( v27 >= 0x1E )
          a2 = 0x40000000000LL;
        ++v23;
        sub_C7D6A0(v26, a2, 16);
        if ( v24 == v23 )
          break;
      }
    }
    v28 = (__int64 *)v7[9];
    v13 = &v28[2 * *((unsigned int *)v7 + 20)];
    if ( v28 == v13 )
      goto LABEL_10;
    do
    {
      a2 = v28[1];
      v29 = *v28;
      v28 += 2;
      sub_C7D6A0(v29, a2, 16);
    }
    while ( v13 != v28 );
LABEL_9:
    v13 = (__int64 *)v7[9];
LABEL_10:
    if ( v13 != v7 + 11 )
      _libc_free(v13, a2);
    v15 = (__int64 *)v7[3];
    if ( v15 != v7 + 5 )
      _libc_free(v15, a2);
    j_j___libc_free_0(v7, 160);
  }
LABEL_15:
  v16 = *(_QWORD **)(a1 + 592);
  if ( v16 && *v16 )
  {
    v17 = 1;
    if ( v30 && *v30 )
      LOBYTE(v17) = v16 != v30;
  }
  else
  {
    v17 = 0;
    if ( v30 )
      LOBYTE(v17) = *v30 != 0;
  }
  return v17;
}
