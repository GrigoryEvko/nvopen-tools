// Function: sub_D8A3E0
// Address: 0xd8a3e0
//
__int64 __fastcall sub_D8A3E0(__int64 *a1, __int64 a2, unsigned int *a3, __int64 a4)
{
  __int64 v5; // rdx
  __int64 v6; // rax
  __int64 v7; // rdi
  __int64 v8; // rax
  __int64 v9; // r13
  __int64 v10; // rsi
  bool v11; // cf
  unsigned __int64 v12; // rax
  __int64 v13; // rbx
  const void **v14; // rsi
  __int64 v15; // rbx
  unsigned int v16; // eax
  unsigned int v17; // eax
  __int64 v18; // r15
  __int64 v19; // r13
  unsigned int v20; // eax
  unsigned int v21; // eax
  unsigned __int64 v22; // r14
  __int64 v23; // rbx
  __int64 v24; // r12
  __int64 v25; // r14
  unsigned int v26; // eax
  unsigned int v27; // eax
  unsigned int v28; // eax
  __int64 v29; // rbx
  unsigned int v30; // eax
  unsigned int v31; // eax
  __int64 v32; // r15
  __int64 v33; // r14
  unsigned __int64 v34; // rcx
  __int64 v35; // rax
  __int64 v36; // r12
  unsigned int v37; // eax
  unsigned int v38; // eax
  unsigned int v39; // eax
  __int64 i; // r13
  __int64 v41; // r14
  __int64 v42; // r12
  __int64 v43; // rdi
  __int64 v44; // rdi
  __int64 v45; // rdi
  __int64 v46; // rdi
  __int64 result; // rax
  __int64 v48; // r8
  __int64 v49; // rax
  unsigned int v50; // eax
  __int64 v51; // [rsp+8h] [rbp-68h]
  __int64 v52; // [rsp+8h] [rbp-68h]
  __int64 v53; // [rsp+8h] [rbp-68h]
  __int64 v54; // [rsp+10h] [rbp-60h]
  __int64 v56; // [rsp+20h] [rbp-50h]
  __int64 v57; // [rsp+28h] [rbp-48h]
  __int64 v58; // [rsp+30h] [rbp-40h]
  unsigned __int64 v60; // [rsp+38h] [rbp-38h]

  v5 = *a1;
  v6 = a1[1];
  v7 = 0x1FFFFFFFFFFFFFFLL;
  v58 = v6;
  v8 = (v6 - v5) >> 6;
  v57 = v5;
  if ( v8 == 0x1FFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v9 = a2;
  v10 = 1;
  if ( v8 )
    v10 = v8;
  v11 = __CFADD__(v10, v8);
  v12 = v10 + v8;
  v54 = v12;
  v13 = a2 - v5;
  v14 = (const void **)v11;
  if ( v11 )
  {
    v48 = 0x7FFFFFFFFFFFFFC0LL;
    v54 = 0x1FFFFFFFFFFFFFFLL;
  }
  else
  {
    if ( !v12 )
    {
      v56 = 0;
      goto LABEL_7;
    }
    if ( v12 <= 0x1FFFFFFFFFFFFFFLL )
      v7 = v12;
    v54 = v7;
    v48 = v7 << 6;
  }
  v7 = v48;
  v52 = a4;
  v49 = sub_22077B0(v48);
  a4 = v52;
  v56 = v49;
LABEL_7:
  v15 = v56 + v13;
  if ( v15 )
  {
    *(_QWORD *)v15 = *a3;
    v16 = *(_DWORD *)(a4 + 8);
    *(_DWORD *)(v15 + 16) = v16;
    if ( v16 > 0x40 )
    {
      v14 = (const void **)a4;
      v7 = v15 + 8;
      v53 = a4;
      sub_C43780(v15 + 8, (const void **)a4);
      a4 = v53;
      v50 = *(_DWORD *)(v53 + 24);
      *(_DWORD *)(v15 + 32) = v50;
      if ( v50 <= 0x40 )
        goto LABEL_10;
    }
    else
    {
      *(_QWORD *)(v15 + 8) = *(_QWORD *)a4;
      v17 = *(_DWORD *)(a4 + 24);
      *(_DWORD *)(v15 + 32) = v17;
      if ( v17 <= 0x40 )
      {
LABEL_10:
        *(_QWORD *)(v15 + 24) = *(_QWORD *)(a4 + 16);
LABEL_11:
        *(_QWORD *)(v15 + 40) = 0;
        *(_QWORD *)(v15 + 48) = 0;
        *(_QWORD *)(v15 + 56) = 0;
        goto LABEL_12;
      }
    }
    v14 = (const void **)(a4 + 16);
    v7 = v15 + 24;
    sub_C43780(v15 + 24, (const void **)(a4 + 16));
    goto LABEL_11;
  }
LABEL_12:
  v18 = v56;
  if ( a2 == v57 )
    goto LABEL_34;
  v51 = v9;
  v19 = v57;
  do
  {
    if ( !v18 )
      goto LABEL_32;
    *(_QWORD *)v18 = *(_QWORD *)v19;
    v20 = *(_DWORD *)(v19 + 16);
    *(_DWORD *)(v18 + 16) = v20;
    if ( v20 > 0x40 )
    {
      v14 = (const void **)(v19 + 8);
      v7 = v18 + 8;
      sub_C43780(v18 + 8, (const void **)(v19 + 8));
    }
    else
    {
      *(_QWORD *)(v18 + 8) = *(_QWORD *)(v19 + 8);
    }
    v21 = *(_DWORD *)(v19 + 32);
    *(_DWORD *)(v18 + 32) = v21;
    if ( v21 > 0x40 )
    {
      v14 = (const void **)(v19 + 24);
      v7 = v18 + 24;
      sub_C43780(v18 + 24, (const void **)(v19 + 24));
    }
    else
    {
      *(_QWORD *)(v18 + 24) = *(_QWORD *)(v19 + 24);
    }
    v22 = *(_QWORD *)(v19 + 48) - *(_QWORD *)(v19 + 40);
    *(_QWORD *)(v18 + 40) = 0;
    *(_QWORD *)(v18 + 48) = 0;
    *(_QWORD *)(v18 + 56) = 0;
    if ( v22 )
    {
      if ( v22 > 0x7FFFFFFFFFFFFFE0LL )
LABEL_88:
        sub_4261EA(v7, v14, v5);
      v7 = v22;
      v23 = sub_22077B0(v22);
    }
    else
    {
      v23 = 0;
    }
    *(_QWORD *)(v18 + 40) = v23;
    *(_QWORD *)(v18 + 48) = v23;
    *(_QWORD *)(v18 + 56) = v23 + v22;
    v24 = *(_QWORD *)(v19 + 48);
    v25 = *(_QWORD *)(v19 + 40);
    if ( v24 != v25 )
    {
      while ( 1 )
      {
        if ( !v23 )
          goto LABEL_26;
        *(_QWORD *)v23 = *(_QWORD *)v25;
        *(_QWORD *)(v23 + 8) = *(_QWORD *)(v25 + 8);
        v27 = *(_DWORD *)(v25 + 24);
        *(_DWORD *)(v23 + 24) = v27;
        if ( v27 > 0x40 )
          break;
        *(_QWORD *)(v23 + 16) = *(_QWORD *)(v25 + 16);
        v26 = *(_DWORD *)(v25 + 40);
        *(_DWORD *)(v23 + 40) = v26;
        if ( v26 > 0x40 )
        {
LABEL_30:
          v14 = (const void **)(v25 + 32);
          v7 = v23 + 32;
          v25 += 48;
          v23 += 48;
          sub_C43780(v7, v14);
          if ( v24 == v25 )
            goto LABEL_31;
        }
        else
        {
LABEL_25:
          *(_QWORD *)(v23 + 32) = *(_QWORD *)(v25 + 32);
LABEL_26:
          v25 += 48;
          v23 += 48;
          if ( v24 == v25 )
            goto LABEL_31;
        }
      }
      v14 = (const void **)(v25 + 16);
      v7 = v23 + 16;
      sub_C43780(v23 + 16, (const void **)(v25 + 16));
      v28 = *(_DWORD *)(v25 + 40);
      *(_DWORD *)(v23 + 40) = v28;
      if ( v28 > 0x40 )
        goto LABEL_30;
      goto LABEL_25;
    }
LABEL_31:
    *(_QWORD *)(v18 + 48) = v23;
LABEL_32:
    v19 += 64;
    v18 += 64;
  }
  while ( a2 != v19 );
  v9 = v51;
LABEL_34:
  v5 = v58;
  v29 = v18 + 64;
  if ( a2 != v58 )
  {
    while ( 1 )
    {
      *(_QWORD *)v29 = *(_QWORD *)v9;
      v30 = *(_DWORD *)(v9 + 16);
      *(_DWORD *)(v29 + 16) = v30;
      if ( v30 > 0x40 )
      {
        v14 = (const void **)(v9 + 8);
        v7 = v29 + 8;
        sub_C43780(v29 + 8, (const void **)(v9 + 8));
      }
      else
      {
        *(_QWORD *)(v29 + 8) = *(_QWORD *)(v9 + 8);
      }
      v31 = *(_DWORD *)(v9 + 32);
      *(_DWORD *)(v29 + 32) = v31;
      if ( v31 > 0x40 )
      {
        v14 = (const void **)(v9 + 24);
        v7 = v29 + 24;
        sub_C43780(v29 + 24, (const void **)(v9 + 24));
      }
      else
      {
        *(_QWORD *)(v29 + 24) = *(_QWORD *)(v9 + 24);
      }
      v32 = *(_QWORD *)(v9 + 48);
      v33 = *(_QWORD *)(v9 + 40);
      *(_QWORD *)(v29 + 40) = 0;
      *(_QWORD *)(v29 + 48) = 0;
      *(_QWORD *)(v29 + 56) = 0;
      v34 = v32 - v33;
      if ( v32 == v33 )
      {
        v36 = 0;
      }
      else
      {
        if ( v34 > 0x7FFFFFFFFFFFFFE0LL )
          goto LABEL_88;
        v7 = v32 - v33;
        v60 = v32 - v33;
        v35 = sub_22077B0(v32 - v33);
        v32 = *(_QWORD *)(v9 + 48);
        v33 = *(_QWORD *)(v9 + 40);
        v34 = v60;
        v36 = v35;
      }
      *(_QWORD *)(v29 + 40) = v36;
      *(_QWORD *)(v29 + 48) = v36;
      *(_QWORD *)(v29 + 56) = v36 + v34;
      if ( v33 != v32 )
        break;
LABEL_51:
      *(_QWORD *)(v29 + 48) = v36;
      v9 += 64;
      v29 += 64;
      if ( v58 == v9 )
        goto LABEL_52;
    }
    while ( 1 )
    {
      if ( !v36 )
        goto LABEL_46;
      *(_QWORD *)v36 = *(_QWORD *)v33;
      *(_QWORD *)(v36 + 8) = *(_QWORD *)(v33 + 8);
      v38 = *(_DWORD *)(v33 + 24);
      *(_DWORD *)(v36 + 24) = v38;
      if ( v38 > 0x40 )
        break;
      *(_QWORD *)(v36 + 16) = *(_QWORD *)(v33 + 16);
      v37 = *(_DWORD *)(v33 + 40);
      *(_DWORD *)(v36 + 40) = v37;
      if ( v37 > 0x40 )
      {
LABEL_50:
        v14 = (const void **)(v33 + 32);
        v7 = v36 + 32;
        v33 += 48;
        v36 += 48;
        sub_C43780(v7, v14);
        if ( v32 == v33 )
          goto LABEL_51;
      }
      else
      {
LABEL_45:
        *(_QWORD *)(v36 + 32) = *(_QWORD *)(v33 + 32);
LABEL_46:
        v33 += 48;
        v36 += 48;
        if ( v32 == v33 )
          goto LABEL_51;
      }
    }
    v14 = (const void **)(v33 + 16);
    v7 = v36 + 16;
    sub_C43780(v36 + 16, (const void **)(v33 + 16));
    v39 = *(_DWORD *)(v33 + 40);
    *(_DWORD *)(v36 + 40) = v39;
    if ( v39 > 0x40 )
      goto LABEL_50;
    goto LABEL_45;
  }
LABEL_52:
  for ( i = v57; i != v58; i += 64 )
  {
    v41 = *(_QWORD *)(i + 48);
    v42 = *(_QWORD *)(i + 40);
    if ( v41 != v42 )
    {
      do
      {
        if ( *(_DWORD *)(v42 + 40) > 0x40u )
        {
          v43 = *(_QWORD *)(v42 + 32);
          if ( v43 )
            j_j___libc_free_0_0(v43);
        }
        if ( *(_DWORD *)(v42 + 24) > 0x40u )
        {
          v44 = *(_QWORD *)(v42 + 16);
          if ( v44 )
            j_j___libc_free_0_0(v44);
        }
        v42 += 48;
      }
      while ( v41 != v42 );
      v42 = *(_QWORD *)(i + 40);
    }
    if ( v42 )
      j_j___libc_free_0(v42, *(_QWORD *)(i + 56) - v42);
    if ( *(_DWORD *)(i + 32) > 0x40u )
    {
      v45 = *(_QWORD *)(i + 24);
      if ( v45 )
        j_j___libc_free_0_0(v45);
    }
    if ( *(_DWORD *)(i + 16) > 0x40u )
    {
      v46 = *(_QWORD *)(i + 8);
      if ( v46 )
        j_j___libc_free_0_0(v46);
    }
  }
  if ( v57 )
    j_j___libc_free_0(v57, a1[2] - v57);
  result = v56 + (v54 << 6);
  *a1 = v56;
  a1[1] = v29;
  a1[2] = result;
  return result;
}
