// Function: sub_9CCC50
// Address: 0x9ccc50
//
__int64 __fastcall sub_9CCC50(__int64 a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 v4; // rcx
  __int64 v5; // rax
  __int64 v6; // rdx
  bool v7; // cf
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rdx
  __int64 v10; // rbx
  __int64 v11; // r14
  __int64 v12; // r15
  unsigned int v13; // eax
  unsigned int v14; // eax
  unsigned __int64 v15; // rbx
  __int64 v16; // r13
  __int64 v17; // r9
  __int64 v18; // rbx
  unsigned int v19; // eax
  unsigned int v20; // eax
  unsigned int v21; // eax
  __int64 v22; // rbx
  unsigned int v23; // eax
  unsigned int v24; // eax
  __int64 v25; // r15
  __int64 v26; // rax
  unsigned __int64 v27; // r14
  __int64 v28; // rax
  __int64 v29; // r13
  __int64 v30; // r14
  unsigned int v31; // edx
  unsigned int v32; // edx
  unsigned int v33; // edx
  __int64 i; // r13
  __int64 v35; // r12
  __int64 v36; // r14
  __int64 v37; // rdi
  __int64 v38; // rdi
  __int64 v39; // rdi
  __int64 v40; // rdi
  __int64 result; // rax
  __int64 v42; // [rsp+0h] [rbp-60h]
  _QWORD *v43; // [rsp+8h] [rbp-58h]
  __int64 v44; // [rsp+10h] [rbp-50h]
  __int64 v45; // [rsp+18h] [rbp-48h]
  __int64 v46; // [rsp+20h] [rbp-40h]
  __int64 v47; // [rsp+20h] [rbp-40h]
  __int64 v48; // [rsp+28h] [rbp-38h]

  v3 = 0x1FFFFFFFFFFFFFFLL;
  v4 = *(_QWORD *)a1;
  v43 = (_QWORD *)a1;
  v48 = *(_QWORD *)(a1 + 8);
  v5 = (v48 - *(_QWORD *)a1) >> 6;
  v45 = *(_QWORD *)a1;
  if ( v5 == 0x1FFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v6 = 1;
  if ( v5 )
    v6 = (v48 - *(_QWORD *)a1) >> 6;
  v7 = __CFADD__(v6, v5);
  v8 = v6 + v5;
  v42 = v8;
  v9 = v7;
  if ( v7 )
  {
    a1 = 0x7FFFFFFFFFFFFFC0LL;
    v42 = 0x1FFFFFFFFFFFFFFLL;
  }
  else
  {
    if ( !v8 )
    {
      v44 = 0;
      goto LABEL_7;
    }
    if ( v8 <= 0x1FFFFFFFFFFFFFFLL )
      v3 = v8;
    v42 = v3;
    v3 <<= 6;
    a1 = v3;
  }
  v44 = sub_22077B0(a1);
LABEL_7:
  v10 = v44 + a2 - v45;
  if ( v10 )
  {
    a1 = v10 + 8;
    v3 = 64;
    *(_OWORD *)v10 = 0;
    *(_OWORD *)(v10 + 16) = 0;
    *(_OWORD *)(v10 + 32) = 0;
    *(_OWORD *)(v10 + 48) = 0;
    sub_AADB10(v10 + 8, 64, 1);
    *(_QWORD *)(v10 + 40) = 0;
    *(_QWORD *)(v10 + 48) = 0;
    *(_QWORD *)(v10 + 56) = 0;
  }
  v11 = v45;
  v12 = v44;
  if ( a2 != v45 )
  {
    while ( !v12 )
    {
LABEL_28:
      v11 += 64;
      v12 += 64;
      if ( a2 == v11 )
        goto LABEL_29;
    }
    *(_QWORD *)v12 = *(_QWORD *)v11;
    v13 = *(_DWORD *)(v11 + 16);
    *(_DWORD *)(v12 + 16) = v13;
    if ( v13 > 0x40 )
    {
      v3 = v11 + 8;
      a1 = v12 + 8;
      sub_C43780(v12 + 8, v11 + 8);
    }
    else
    {
      *(_QWORD *)(v12 + 8) = *(_QWORD *)(v11 + 8);
    }
    v14 = *(_DWORD *)(v11 + 32);
    *(_DWORD *)(v12 + 32) = v14;
    if ( v14 > 0x40 )
    {
      v3 = v11 + 24;
      a1 = v12 + 24;
      sub_C43780(v12 + 24, v11 + 24);
    }
    else
    {
      *(_QWORD *)(v12 + 24) = *(_QWORD *)(v11 + 24);
    }
    v15 = *(_QWORD *)(v11 + 48) - *(_QWORD *)(v11 + 40);
    *(_QWORD *)(v12 + 40) = 0;
    *(_QWORD *)(v12 + 48) = 0;
    *(_QWORD *)(v12 + 56) = 0;
    if ( v15 )
    {
      if ( v15 > 0x7FFFFFFFFFFFFFE0LL )
LABEL_81:
        sub_4261EA(a1, v3, v9, v4);
      a1 = v15;
      v16 = sub_22077B0(v15);
    }
    else
    {
      v16 = 0;
    }
    *(_QWORD *)(v12 + 40) = v16;
    *(_QWORD *)(v12 + 48) = v16;
    *(_QWORD *)(v12 + 56) = v16 + v15;
    v17 = *(_QWORD *)(v11 + 48);
    v18 = *(_QWORD *)(v11 + 40);
    if ( v17 == v18 )
    {
LABEL_27:
      *(_QWORD *)(v12 + 48) = v16;
      goto LABEL_28;
    }
    while ( 1 )
    {
      if ( !v16 )
        goto LABEL_22;
      *(_QWORD *)v16 = *(_QWORD *)v18;
      *(_QWORD *)(v16 + 8) = *(_QWORD *)(v18 + 8);
      v20 = *(_DWORD *)(v18 + 24);
      *(_DWORD *)(v16 + 24) = v20;
      if ( v20 > 0x40 )
        break;
      *(_QWORD *)(v16 + 16) = *(_QWORD *)(v18 + 16);
      v19 = *(_DWORD *)(v18 + 40);
      *(_DWORD *)(v16 + 40) = v19;
      if ( v19 > 0x40 )
      {
LABEL_26:
        v3 = v18 + 32;
        a1 = v16 + 32;
        v47 = v17;
        v18 += 48;
        sub_C43780(v16 + 32, v3);
        v17 = v47;
        v16 += 48;
        if ( v47 == v18 )
          goto LABEL_27;
      }
      else
      {
LABEL_21:
        *(_QWORD *)(v16 + 32) = *(_QWORD *)(v18 + 32);
LABEL_22:
        v18 += 48;
        v16 += 48;
        if ( v17 == v18 )
          goto LABEL_27;
      }
    }
    v3 = v18 + 16;
    a1 = v16 + 16;
    v46 = v17;
    sub_C43780(v16 + 16, v18 + 16);
    v21 = *(_DWORD *)(v18 + 40);
    v17 = v46;
    *(_DWORD *)(v16 + 40) = v21;
    if ( v21 > 0x40 )
      goto LABEL_26;
    goto LABEL_21;
  }
LABEL_29:
  v22 = v12 + 64;
  if ( a2 != v48 )
  {
    while ( 1 )
    {
      *(_QWORD *)v22 = *(_QWORD *)a2;
      v23 = *(_DWORD *)(a2 + 16);
      *(_DWORD *)(v22 + 16) = v23;
      if ( v23 > 0x40 )
      {
        v3 = a2 + 8;
        a1 = v22 + 8;
        sub_C43780(v22 + 8, a2 + 8);
      }
      else
      {
        *(_QWORD *)(v22 + 8) = *(_QWORD *)(a2 + 8);
      }
      v24 = *(_DWORD *)(a2 + 32);
      *(_DWORD *)(v22 + 32) = v24;
      if ( v24 > 0x40 )
      {
        v3 = a2 + 24;
        a1 = v22 + 24;
        sub_C43780(v22 + 24, a2 + 24);
      }
      else
      {
        *(_QWORD *)(v22 + 24) = *(_QWORD *)(a2 + 24);
      }
      v25 = *(_QWORD *)(a2 + 48);
      v26 = *(_QWORD *)(a2 + 40);
      *(_QWORD *)(v22 + 40) = 0;
      *(_QWORD *)(v22 + 48) = 0;
      *(_QWORD *)(v22 + 56) = 0;
      v27 = v25 - v26;
      if ( v25 == v26 )
      {
        v29 = 0;
      }
      else
      {
        if ( v27 > 0x7FFFFFFFFFFFFFE0LL )
          goto LABEL_81;
        a1 = v25 - v26;
        v28 = sub_22077B0(v27);
        v25 = *(_QWORD *)(a2 + 48);
        v29 = v28;
        v26 = *(_QWORD *)(a2 + 40);
      }
      v9 = v29 + v27;
      *(_QWORD *)(v22 + 40) = v29;
      *(_QWORD *)(v22 + 48) = v29;
      *(_QWORD *)(v22 + 56) = v29 + v27;
      if ( v26 != v25 )
        break;
LABEL_46:
      *(_QWORD *)(v22 + 48) = v29;
      a2 += 64;
      v22 += 64;
      if ( v48 == a2 )
        goto LABEL_47;
    }
    v30 = v26;
    while ( 1 )
    {
      if ( !v29 )
        goto LABEL_41;
      *(_QWORD *)v29 = *(_QWORD *)v30;
      *(_QWORD *)(v29 + 8) = *(_QWORD *)(v30 + 8);
      v32 = *(_DWORD *)(v30 + 24);
      *(_DWORD *)(v29 + 24) = v32;
      if ( v32 > 0x40 )
        break;
      *(_QWORD *)(v29 + 16) = *(_QWORD *)(v30 + 16);
      v31 = *(_DWORD *)(v30 + 40);
      *(_DWORD *)(v29 + 40) = v31;
      if ( v31 > 0x40 )
      {
LABEL_45:
        v3 = v30 + 32;
        a1 = v29 + 32;
        v30 += 48;
        v29 += 48;
        sub_C43780(a1, v3);
        if ( v25 == v30 )
          goto LABEL_46;
      }
      else
      {
LABEL_40:
        v9 = *(_QWORD *)(v30 + 32);
        *(_QWORD *)(v29 + 32) = v9;
LABEL_41:
        v30 += 48;
        v29 += 48;
        if ( v25 == v30 )
          goto LABEL_46;
      }
    }
    v3 = v30 + 16;
    a1 = v29 + 16;
    sub_C43780(v29 + 16, v30 + 16);
    v33 = *(_DWORD *)(v30 + 40);
    *(_DWORD *)(v29 + 40) = v33;
    if ( v33 > 0x40 )
      goto LABEL_45;
    goto LABEL_40;
  }
LABEL_47:
  for ( i = v45; i != v48; i += 64 )
  {
    v35 = *(_QWORD *)(i + 48);
    v36 = *(_QWORD *)(i + 40);
    if ( v35 != v36 )
    {
      do
      {
        if ( *(_DWORD *)(v36 + 40) > 0x40u )
        {
          v37 = *(_QWORD *)(v36 + 32);
          if ( v37 )
            j_j___libc_free_0_0(v37);
        }
        if ( *(_DWORD *)(v36 + 24) > 0x40u )
        {
          v38 = *(_QWORD *)(v36 + 16);
          if ( v38 )
            j_j___libc_free_0_0(v38);
        }
        v36 += 48;
      }
      while ( v35 != v36 );
      v36 = *(_QWORD *)(i + 40);
    }
    if ( v36 )
      j_j___libc_free_0(v36, *(_QWORD *)(i + 56) - v36);
    if ( *(_DWORD *)(i + 32) > 0x40u )
    {
      v39 = *(_QWORD *)(i + 24);
      if ( v39 )
        j_j___libc_free_0_0(v39);
    }
    if ( *(_DWORD *)(i + 16) > 0x40u )
    {
      v40 = *(_QWORD *)(i + 8);
      if ( v40 )
        j_j___libc_free_0_0(v40);
    }
  }
  if ( v45 )
    j_j___libc_free_0(v45, v43[2] - v45);
  result = v44 + (v42 << 6);
  *v43 = v44;
  v43[1] = v22;
  v43[2] = result;
  return result;
}
