// Function: sub_2A312E0
// Address: 0x2a312e0
//
unsigned __int64 *__fastcall sub_2A312E0(unsigned __int64 *a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v3; // rbx
  unsigned __int64 v4; // r13
  __int64 v5; // rax
  __int64 v8; // rdx
  __int64 v9; // r15
  bool v10; // cf
  unsigned __int64 v11; // rax
  __int64 v12; // r9
  __int64 v13; // r12
  __int64 v14; // r9
  unsigned int v15; // eax
  unsigned int v16; // eax
  __int64 v17; // r12
  unsigned __int64 v18; // rax
  unsigned int v19; // esi
  __int64 v20; // rsi
  unsigned int v21; // esi
  unsigned int v22; // esi
  unsigned int v23; // eax
  __int64 v24; // rax
  unsigned int v25; // eax
  unsigned int v26; // eax
  const void **v27; // rsi
  __int64 v28; // rdi
  unsigned __int64 i; // r14
  unsigned __int64 v30; // rdi
  unsigned __int64 v32; // r12
  __int64 v33; // rax
  unsigned int v34; // eax
  unsigned __int64 v35; // [rsp+10h] [rbp-50h]
  unsigned __int64 v37; // [rsp+20h] [rbp-40h]
  unsigned __int64 v38; // [rsp+20h] [rbp-40h]
  __int64 v39; // [rsp+20h] [rbp-40h]
  __int64 v40; // [rsp+28h] [rbp-38h]

  v3 = a1[1];
  v4 = *a1;
  v5 = (__int64)(v3 - *a1) >> 5;
  if ( v5 == 0x3FFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v8 = 1;
  if ( v5 )
    v8 = (__int64)(v3 - v4) >> 5;
  v9 = a2;
  v10 = __CFADD__(v8, v5);
  v11 = v8 + v5;
  v12 = a2 - v4;
  if ( v10 )
  {
    v32 = 0x7FFFFFFFFFFFFFE0LL;
  }
  else
  {
    if ( !v11 )
    {
      v35 = 0;
      v13 = 32;
      v40 = 0;
      goto LABEL_7;
    }
    if ( v11 > 0x3FFFFFFFFFFFFFFLL )
      v11 = 0x3FFFFFFFFFFFFFFLL;
    v32 = 32 * v11;
  }
  v33 = sub_22077B0(v32);
  v12 = a2 - v4;
  v40 = v33;
  v35 = v33 + v32;
  v13 = v33 + 32;
LABEL_7:
  v14 = v40 + v12;
  if ( v14 )
  {
    v15 = *(_DWORD *)(a3 + 8);
    *(_DWORD *)(v14 + 8) = v15;
    if ( v15 > 0x40 )
    {
      v39 = v14;
      sub_C43780(v14, (const void **)a3);
      v14 = v39;
      v34 = *(_DWORD *)(a3 + 24);
      *(_DWORD *)(v39 + 24) = v34;
      if ( v34 <= 0x40 )
        goto LABEL_10;
    }
    else
    {
      *(_QWORD *)v14 = *(_QWORD *)a3;
      v16 = *(_DWORD *)(a3 + 24);
      *(_DWORD *)(v14 + 24) = v16;
      if ( v16 <= 0x40 )
      {
LABEL_10:
        *(_QWORD *)(v14 + 16) = *(_QWORD *)(a3 + 16);
        goto LABEL_11;
      }
    }
    sub_C43780(v14 + 16, (const void **)(a3 + 16));
  }
LABEL_11:
  if ( a2 == v4 )
    goto LABEL_22;
  v17 = v40;
  v18 = v4;
  while ( !v17 )
  {
LABEL_15:
    v18 += 32LL;
    v20 = v17 + 32;
    if ( a2 == v18 )
      goto LABEL_21;
LABEL_16:
    v17 = v20;
  }
  v21 = *(_DWORD *)(v18 + 8);
  *(_DWORD *)(v17 + 8) = v21;
  if ( v21 <= 0x40 )
  {
    *(_QWORD *)v17 = *(_QWORD *)v18;
    v19 = *(_DWORD *)(v18 + 24);
    *(_DWORD *)(v17 + 24) = v19;
    if ( v19 > 0x40 )
      goto LABEL_20;
    goto LABEL_14;
  }
  v37 = v18;
  sub_C43780(v17, (const void **)v18);
  v18 = v37;
  v22 = *(_DWORD *)(v37 + 24);
  *(_DWORD *)(v17 + 24) = v22;
  if ( v22 <= 0x40 )
  {
LABEL_14:
    *(_QWORD *)(v17 + 16) = *(_QWORD *)(v18 + 16);
    goto LABEL_15;
  }
LABEL_20:
  v38 = v18;
  sub_C43780(v17 + 16, (const void **)(v18 + 16));
  v20 = v17 + 32;
  v18 = v38 + 32;
  if ( a2 != v38 + 32 )
    goto LABEL_16;
LABEL_21:
  v13 = v17 + 64;
LABEL_22:
  if ( a2 != v3 )
  {
    while ( 1 )
    {
      v25 = *(_DWORD *)(v9 + 8);
      *(_DWORD *)(v13 + 8) = v25;
      if ( v25 <= 0x40 )
      {
        *(_QWORD *)v13 = *(_QWORD *)v9;
        v23 = *(_DWORD *)(v9 + 24);
        *(_DWORD *)(v13 + 24) = v23;
        if ( v23 <= 0x40 )
          goto LABEL_25;
LABEL_28:
        v27 = (const void **)(v9 + 16);
        v28 = v13 + 16;
        v9 += 32;
        v13 += 32;
        sub_C43780(v28, v27);
        if ( v3 == v9 )
          break;
      }
      else
      {
        sub_C43780(v13, (const void **)v9);
        v26 = *(_DWORD *)(v9 + 24);
        *(_DWORD *)(v13 + 24) = v26;
        if ( v26 > 0x40 )
          goto LABEL_28;
LABEL_25:
        v24 = *(_QWORD *)(v9 + 16);
        v9 += 32;
        v13 += 32;
        *(_QWORD *)(v13 - 16) = v24;
        if ( v3 == v9 )
          break;
      }
    }
  }
  for ( i = v4; v3 != i; i += 32LL )
  {
    if ( *(_DWORD *)(i + 24) > 0x40u )
    {
      v30 = *(_QWORD *)(i + 16);
      if ( v30 )
        j_j___libc_free_0_0(v30);
    }
    if ( *(_DWORD *)(i + 8) > 0x40u && *(_QWORD *)i )
      j_j___libc_free_0_0(*(_QWORD *)i);
  }
  if ( v4 )
    j_j___libc_free_0(v4);
  *a1 = v40;
  a1[1] = v13;
  a1[2] = v35;
  return a1;
}
