// Function: sub_1213D80
// Address: 0x1213d80
//
__int64 *__fastcall sub_1213D80(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rbx
  __int64 v5; // r12
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rdx
  __int64 v9; // r14
  bool v10; // cf
  unsigned __int64 v11; // rax
  __int64 v12; // rsi
  __int64 v13; // r15
  __int64 v14; // rax
  unsigned int v15; // esi
  unsigned int v16; // esi
  __int64 v17; // r15
  __int64 v18; // rax
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
  __int64 i; // r13
  __int64 v30; // rdi
  __int64 v31; // rdi
  __int64 v33; // r15
  __int64 v34; // rax
  unsigned int v35; // esi
  __int64 v36; // [rsp+8h] [rbp-58h]
  __int64 v37; // [rsp+8h] [rbp-58h]
  __int64 v38; // [rsp+10h] [rbp-50h]
  __int64 v40; // [rsp+20h] [rbp-40h]
  __int64 v41; // [rsp+20h] [rbp-40h]
  __int64 v42; // [rsp+20h] [rbp-40h]
  __int64 v43; // [rsp+28h] [rbp-38h]

  v4 = a1[1];
  v5 = *a1;
  v6 = 0xAAAAAAAAAAAAAAABLL * ((v4 - *a1) >> 4);
  if ( v6 == 0x2AAAAAAAAAAAAAALL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v7 = 1;
  v9 = a2;
  if ( v6 )
    v7 = 0xAAAAAAAAAAAAAAABLL * ((v4 - v5) >> 4);
  v10 = __CFADD__(v7, v6);
  v11 = v7 - 0x5555555555555555LL * ((v4 - v5) >> 4);
  v12 = a2 - v5;
  if ( v10 )
  {
    v33 = 0x7FFFFFFFFFFFFFE0LL;
  }
  else
  {
    if ( !v11 )
    {
      v38 = 0;
      v13 = 48;
      v43 = 0;
      goto LABEL_7;
    }
    if ( v11 > 0x2AAAAAAAAAAAAAALL )
      v11 = 0x2AAAAAAAAAAAAAALL;
    v33 = 48 * v11;
  }
  v36 = a3;
  v34 = sub_22077B0(v33);
  a3 = v36;
  v43 = v34;
  v38 = v34 + v33;
  v13 = v34 + 48;
LABEL_7:
  v14 = v43 + v12;
  if ( v43 + v12 )
  {
    *(_QWORD *)v14 = *(_QWORD *)a3;
    *(_QWORD *)(v14 + 8) = *(_QWORD *)(a3 + 8);
    v15 = *(_DWORD *)(a3 + 24);
    *(_DWORD *)(v14 + 24) = v15;
    if ( v15 > 0x40 )
    {
      v37 = a3;
      v42 = v14;
      sub_C43780(v14 + 16, (const void **)(a3 + 16));
      a3 = v37;
      v14 = v42;
      v35 = *(_DWORD *)(v37 + 40);
      *(_DWORD *)(v42 + 40) = v35;
      if ( v35 <= 0x40 )
        goto LABEL_10;
    }
    else
    {
      *(_QWORD *)(v14 + 16) = *(_QWORD *)(a3 + 16);
      v16 = *(_DWORD *)(a3 + 40);
      *(_DWORD *)(v14 + 40) = v16;
      if ( v16 <= 0x40 )
      {
LABEL_10:
        *(_QWORD *)(v14 + 32) = *(_QWORD *)(a3 + 32);
        goto LABEL_11;
      }
    }
    sub_C43780(v14 + 32, (const void **)(a3 + 32));
  }
LABEL_11:
  if ( a2 == v5 )
    goto LABEL_22;
  v17 = v43;
  v18 = v5;
  while ( !v17 )
  {
LABEL_15:
    v18 += 48;
    v20 = v17 + 48;
    if ( a2 == v18 )
      goto LABEL_21;
LABEL_16:
    v17 = v20;
  }
  *(_QWORD *)v17 = *(_QWORD *)v18;
  *(_QWORD *)(v17 + 8) = *(_QWORD *)(v18 + 8);
  v21 = *(_DWORD *)(v18 + 24);
  *(_DWORD *)(v17 + 24) = v21;
  if ( v21 <= 0x40 )
  {
    *(_QWORD *)(v17 + 16) = *(_QWORD *)(v18 + 16);
    v19 = *(_DWORD *)(v18 + 40);
    *(_DWORD *)(v17 + 40) = v19;
    if ( v19 > 0x40 )
      goto LABEL_20;
    goto LABEL_14;
  }
  v40 = v18;
  sub_C43780(v17 + 16, (const void **)(v18 + 16));
  v18 = v40;
  v22 = *(_DWORD *)(v40 + 40);
  *(_DWORD *)(v17 + 40) = v22;
  if ( v22 <= 0x40 )
  {
LABEL_14:
    *(_QWORD *)(v17 + 32) = *(_QWORD *)(v18 + 32);
    goto LABEL_15;
  }
LABEL_20:
  v41 = v18;
  sub_C43780(v17 + 32, (const void **)(v18 + 32));
  v20 = v17 + 48;
  v18 = v41 + 48;
  if ( a2 != v41 + 48 )
    goto LABEL_16;
LABEL_21:
  v13 = v17 + 96;
LABEL_22:
  if ( a2 != v4 )
  {
    while ( 1 )
    {
      *(_QWORD *)v13 = *(_QWORD *)v9;
      *(_QWORD *)(v13 + 8) = *(_QWORD *)(v9 + 8);
      v25 = *(_DWORD *)(v9 + 24);
      *(_DWORD *)(v13 + 24) = v25;
      if ( v25 <= 0x40 )
      {
        *(_QWORD *)(v13 + 16) = *(_QWORD *)(v9 + 16);
        v23 = *(_DWORD *)(v9 + 40);
        *(_DWORD *)(v13 + 40) = v23;
        if ( v23 <= 0x40 )
          goto LABEL_25;
LABEL_28:
        v27 = (const void **)(v9 + 32);
        v28 = v13 + 32;
        v9 += 48;
        v13 += 48;
        sub_C43780(v28, v27);
        if ( v4 == v9 )
          break;
      }
      else
      {
        sub_C43780(v13 + 16, (const void **)(v9 + 16));
        v26 = *(_DWORD *)(v9 + 40);
        *(_DWORD *)(v13 + 40) = v26;
        if ( v26 > 0x40 )
          goto LABEL_28;
LABEL_25:
        v24 = *(_QWORD *)(v9 + 32);
        v9 += 48;
        v13 += 48;
        *(_QWORD *)(v13 - 16) = v24;
        if ( v4 == v9 )
          break;
      }
    }
  }
  for ( i = v5; v4 != i; i += 48 )
  {
    if ( *(_DWORD *)(i + 40) > 0x40u )
    {
      v30 = *(_QWORD *)(i + 32);
      if ( v30 )
        j_j___libc_free_0_0(v30);
    }
    if ( *(_DWORD *)(i + 24) > 0x40u )
    {
      v31 = *(_QWORD *)(i + 16);
      if ( v31 )
        j_j___libc_free_0_0(v31);
    }
  }
  if ( v5 )
    j_j___libc_free_0(v5, a1[2] - v5);
  *a1 = v43;
  a1[1] = v13;
  a1[2] = v38;
  return a1;
}
