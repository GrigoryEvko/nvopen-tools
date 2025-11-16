// Function: sub_12142F0
// Address: 0x12142f0
//
__int64 __fastcall sub_12142F0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r13
  __int64 v5; // rsi
  __int64 v6; // rax
  __int64 v7; // rdx
  bool v8; // cf
  unsigned __int64 v9; // rax
  __int64 v10; // r12
  __int64 v11; // rdx
  __int64 v12; // rax
  int v13; // edx
  int v14; // edx
  __int64 v15; // rdx
  __int64 v16; // rdx
  __int64 v17; // r15
  __int64 v18; // r13
  unsigned int v19; // eax
  unsigned int v20; // eax
  unsigned __int64 v21; // r14
  __int64 v22; // rbx
  __int64 v23; // r12
  __int64 v24; // r14
  unsigned int v25; // eax
  unsigned int v26; // eax
  unsigned int v27; // eax
  __int64 v28; // rbx
  unsigned int v29; // eax
  unsigned int v30; // eax
  __int64 v31; // r15
  __int64 v32; // r14
  unsigned __int64 v33; // rcx
  __int64 v34; // rax
  __int64 v35; // r12
  unsigned int v36; // eax
  unsigned int v37; // eax
  unsigned int v38; // eax
  __int64 i; // r13
  __int64 v40; // r14
  __int64 v41; // r12
  __int64 v42; // rdi
  __int64 v43; // rdi
  __int64 v44; // rdi
  __int64 v45; // rdi
  __int64 result; // rax
  __int64 v47; // [rsp+8h] [rbp-68h]
  __int64 v48; // [rsp+10h] [rbp-60h]
  _QWORD *v49; // [rsp+18h] [rbp-58h]
  __int64 v50; // [rsp+20h] [rbp-50h]
  __int64 v51; // [rsp+28h] [rbp-48h]
  __int64 v52; // [rsp+30h] [rbp-40h]
  unsigned __int64 v54; // [rsp+38h] [rbp-38h]

  v3 = a2;
  v5 = 0x1FFFFFFFFFFFFFFLL;
  v49 = (_QWORD *)a1;
  v52 = *(_QWORD *)(a1 + 8);
  v6 = (v52 - *(_QWORD *)a1) >> 6;
  v51 = *(_QWORD *)a1;
  if ( v6 == 0x1FFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v7 = 1;
  if ( v6 )
    v7 = (v52 - *(_QWORD *)a1) >> 6;
  v8 = __CFADD__(v7, v6);
  v9 = v7 + v6;
  v48 = v9;
  v10 = a2 - v51;
  v11 = v8;
  if ( v8 )
  {
    a1 = 0x7FFFFFFFFFFFFFC0LL;
    v48 = 0x1FFFFFFFFFFFFFFLL;
  }
  else
  {
    if ( !v9 )
    {
      v50 = 0;
      goto LABEL_7;
    }
    if ( v9 <= 0x1FFFFFFFFFFFFFFLL )
      v5 = v9;
    v48 = v5;
    v5 <<= 6;
    a1 = v5;
  }
  v50 = sub_22077B0(a1);
LABEL_7:
  v12 = v50 + v10;
  if ( v50 + v10 )
  {
    *(_QWORD *)v12 = *(_QWORD *)a3;
    v13 = *(_DWORD *)(a3 + 16);
    *(_DWORD *)(a3 + 16) = 0;
    *(_DWORD *)(v12 + 16) = v13;
    *(_QWORD *)(v12 + 8) = *(_QWORD *)(a3 + 8);
    v14 = *(_DWORD *)(a3 + 32);
    *(_DWORD *)(a3 + 32) = 0;
    *(_DWORD *)(v12 + 32) = v14;
    *(_QWORD *)(v12 + 24) = *(_QWORD *)(a3 + 24);
    v15 = *(_QWORD *)(a3 + 40);
    *(_QWORD *)(a3 + 40) = 0;
    *(_QWORD *)(v12 + 40) = v15;
    v16 = *(_QWORD *)(a3 + 48);
    *(_QWORD *)(a3 + 48) = 0;
    *(_QWORD *)(v12 + 48) = v16;
    v11 = *(_QWORD *)(a3 + 56);
    *(_QWORD *)(a3 + 56) = 0;
    *(_QWORD *)(v12 + 56) = v11;
  }
  v17 = v50;
  if ( a2 != v51 )
  {
    v47 = v3;
    v18 = v51;
    while ( !v17 )
    {
LABEL_29:
      v18 += 64;
      v17 += 64;
      if ( a2 == v18 )
      {
        v3 = v47;
        goto LABEL_31;
      }
    }
    *(_QWORD *)v17 = *(_QWORD *)v18;
    v19 = *(_DWORD *)(v18 + 16);
    *(_DWORD *)(v17 + 16) = v19;
    if ( v19 > 0x40 )
    {
      v5 = v18 + 8;
      a1 = v17 + 8;
      sub_C43780(v17 + 8, (const void **)(v18 + 8));
    }
    else
    {
      *(_QWORD *)(v17 + 8) = *(_QWORD *)(v18 + 8);
    }
    v20 = *(_DWORD *)(v18 + 32);
    *(_DWORD *)(v17 + 32) = v20;
    if ( v20 > 0x40 )
    {
      v5 = v18 + 24;
      a1 = v17 + 24;
      sub_C43780(v17 + 24, (const void **)(v18 + 24));
    }
    else
    {
      *(_QWORD *)(v17 + 24) = *(_QWORD *)(v18 + 24);
    }
    v21 = *(_QWORD *)(v18 + 48) - *(_QWORD *)(v18 + 40);
    *(_QWORD *)(v17 + 40) = 0;
    *(_QWORD *)(v17 + 48) = 0;
    *(_QWORD *)(v17 + 56) = 0;
    if ( v21 )
    {
      if ( v21 > 0x7FFFFFFFFFFFFFE0LL )
LABEL_83:
        sub_4261EA(a1, v5, v11);
      a1 = v21;
      v22 = sub_22077B0(v21);
    }
    else
    {
      v22 = 0;
    }
    *(_QWORD *)(v17 + 40) = v22;
    *(_QWORD *)(v17 + 48) = v22;
    *(_QWORD *)(v17 + 56) = v22 + v21;
    v23 = *(_QWORD *)(v18 + 48);
    v24 = *(_QWORD *)(v18 + 40);
    if ( v23 == v24 )
    {
LABEL_28:
      *(_QWORD *)(v17 + 48) = v22;
      goto LABEL_29;
    }
    while ( 1 )
    {
      if ( !v22 )
        goto LABEL_23;
      *(_QWORD *)v22 = *(_QWORD *)v24;
      *(_QWORD *)(v22 + 8) = *(_QWORD *)(v24 + 8);
      v26 = *(_DWORD *)(v24 + 24);
      *(_DWORD *)(v22 + 24) = v26;
      if ( v26 > 0x40 )
        break;
      *(_QWORD *)(v22 + 16) = *(_QWORD *)(v24 + 16);
      v25 = *(_DWORD *)(v24 + 40);
      *(_DWORD *)(v22 + 40) = v25;
      if ( v25 > 0x40 )
      {
LABEL_27:
        v5 = v24 + 32;
        a1 = v22 + 32;
        v24 += 48;
        v22 += 48;
        sub_C43780(a1, (const void **)v5);
        if ( v23 == v24 )
          goto LABEL_28;
      }
      else
      {
LABEL_22:
        *(_QWORD *)(v22 + 32) = *(_QWORD *)(v24 + 32);
LABEL_23:
        v24 += 48;
        v22 += 48;
        if ( v23 == v24 )
          goto LABEL_28;
      }
    }
    v5 = v24 + 16;
    a1 = v22 + 16;
    sub_C43780(v22 + 16, (const void **)(v24 + 16));
    v27 = *(_DWORD *)(v24 + 40);
    *(_DWORD *)(v22 + 40) = v27;
    if ( v27 > 0x40 )
      goto LABEL_27;
    goto LABEL_22;
  }
LABEL_31:
  v11 = v52;
  v28 = v17 + 64;
  if ( a2 != v52 )
  {
    while ( 1 )
    {
      *(_QWORD *)v28 = *(_QWORD *)v3;
      v29 = *(_DWORD *)(v3 + 16);
      *(_DWORD *)(v28 + 16) = v29;
      if ( v29 > 0x40 )
      {
        v5 = v3 + 8;
        a1 = v28 + 8;
        sub_C43780(v28 + 8, (const void **)(v3 + 8));
      }
      else
      {
        *(_QWORD *)(v28 + 8) = *(_QWORD *)(v3 + 8);
      }
      v30 = *(_DWORD *)(v3 + 32);
      *(_DWORD *)(v28 + 32) = v30;
      if ( v30 > 0x40 )
      {
        v5 = v3 + 24;
        a1 = v28 + 24;
        sub_C43780(v28 + 24, (const void **)(v3 + 24));
      }
      else
      {
        *(_QWORD *)(v28 + 24) = *(_QWORD *)(v3 + 24);
      }
      v31 = *(_QWORD *)(v3 + 48);
      v32 = *(_QWORD *)(v3 + 40);
      *(_QWORD *)(v28 + 40) = 0;
      *(_QWORD *)(v28 + 48) = 0;
      *(_QWORD *)(v28 + 56) = 0;
      v33 = v31 - v32;
      if ( v31 == v32 )
      {
        v35 = 0;
      }
      else
      {
        if ( v33 > 0x7FFFFFFFFFFFFFE0LL )
          goto LABEL_83;
        a1 = v31 - v32;
        v54 = v31 - v32;
        v34 = sub_22077B0(v31 - v32);
        v31 = *(_QWORD *)(v3 + 48);
        v32 = *(_QWORD *)(v3 + 40);
        v33 = v54;
        v35 = v34;
      }
      *(_QWORD *)(v28 + 40) = v35;
      *(_QWORD *)(v28 + 48) = v35;
      *(_QWORD *)(v28 + 56) = v35 + v33;
      if ( v32 != v31 )
        break;
LABEL_48:
      *(_QWORD *)(v28 + 48) = v35;
      v3 += 64;
      v28 += 64;
      if ( v52 == v3 )
        goto LABEL_49;
    }
    while ( 1 )
    {
      if ( !v35 )
        goto LABEL_43;
      *(_QWORD *)v35 = *(_QWORD *)v32;
      *(_QWORD *)(v35 + 8) = *(_QWORD *)(v32 + 8);
      v37 = *(_DWORD *)(v32 + 24);
      *(_DWORD *)(v35 + 24) = v37;
      if ( v37 > 0x40 )
        break;
      *(_QWORD *)(v35 + 16) = *(_QWORD *)(v32 + 16);
      v36 = *(_DWORD *)(v32 + 40);
      *(_DWORD *)(v35 + 40) = v36;
      if ( v36 > 0x40 )
      {
LABEL_47:
        v5 = v32 + 32;
        a1 = v35 + 32;
        v32 += 48;
        v35 += 48;
        sub_C43780(a1, (const void **)v5);
        if ( v31 == v32 )
          goto LABEL_48;
      }
      else
      {
LABEL_42:
        *(_QWORD *)(v35 + 32) = *(_QWORD *)(v32 + 32);
LABEL_43:
        v32 += 48;
        v35 += 48;
        if ( v31 == v32 )
          goto LABEL_48;
      }
    }
    v5 = v32 + 16;
    a1 = v35 + 16;
    sub_C43780(v35 + 16, (const void **)(v32 + 16));
    v38 = *(_DWORD *)(v32 + 40);
    *(_DWORD *)(v35 + 40) = v38;
    if ( v38 > 0x40 )
      goto LABEL_47;
    goto LABEL_42;
  }
LABEL_49:
  for ( i = v51; i != v52; i += 64 )
  {
    v40 = *(_QWORD *)(i + 48);
    v41 = *(_QWORD *)(i + 40);
    if ( v40 != v41 )
    {
      do
      {
        if ( *(_DWORD *)(v41 + 40) > 0x40u )
        {
          v42 = *(_QWORD *)(v41 + 32);
          if ( v42 )
            j_j___libc_free_0_0(v42);
        }
        if ( *(_DWORD *)(v41 + 24) > 0x40u )
        {
          v43 = *(_QWORD *)(v41 + 16);
          if ( v43 )
            j_j___libc_free_0_0(v43);
        }
        v41 += 48;
      }
      while ( v40 != v41 );
      v41 = *(_QWORD *)(i + 40);
    }
    if ( v41 )
      j_j___libc_free_0(v41, *(_QWORD *)(i + 56) - v41);
    if ( *(_DWORD *)(i + 32) > 0x40u )
    {
      v44 = *(_QWORD *)(i + 24);
      if ( v44 )
        j_j___libc_free_0_0(v44);
    }
    if ( *(_DWORD *)(i + 16) > 0x40u )
    {
      v45 = *(_QWORD *)(i + 8);
      if ( v45 )
        j_j___libc_free_0_0(v45);
    }
  }
  if ( v51 )
    j_j___libc_free_0(v51, v49[2] - v51);
  result = v50 + (v48 << 6);
  *v49 = v50;
  v49[1] = v28;
  v49[2] = result;
  return result;
}
