// Function: sub_2766DC0
// Address: 0x2766dc0
//
__int64 __fastcall sub_2766DC0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // rsi
  __int64 v7; // rdi
  int v8; // r9d
  unsigned int v9; // ecx
  _QWORD *v10; // r15
  _QWORD *v11; // rax
  __int64 v12; // rdx
  __int64 v13; // r14
  __int64 v14; // r12
  unsigned __int64 v15; // rax
  __int64 v16; // rax
  unsigned __int64 v17; // rbx
  unsigned int v18; // eax
  const void **v19; // rsi
  __int64 v20; // rdi
  unsigned int v21; // ecx
  const void *v22; // rax
  const void *v23; // r14
  unsigned __int64 v24; // r15
  __int64 v25; // r13
  unsigned __int64 i; // r12
  unsigned __int64 v27; // rdi
  int v29; // edx
  int v30; // ecx
  int v31; // eax
  int v32; // esi
  __int64 v33; // r8
  unsigned int v34; // edx
  __int64 v35; // rdi
  int v36; // r10d
  _QWORD *v37; // r9
  int v38; // eax
  int v39; // edx
  __int64 v40; // rdi
  int v41; // r9d
  unsigned int v42; // r14d
  _QWORD *v43; // r8
  __int64 v44; // rsi
  __int64 v45; // r12
  signed __int64 v46; // r14
  unsigned __int64 v47; // [rsp+8h] [rbp-98h]
  unsigned int v48; // [rsp+14h] [rbp-8Ch]
  unsigned __int64 v49; // [rsp+18h] [rbp-88h]
  const void *v50; // [rsp+20h] [rbp-80h] BYREF
  unsigned int v51; // [rsp+28h] [rbp-78h]
  const void *v52; // [rsp+30h] [rbp-70h] BYREF
  unsigned int v53; // [rsp+38h] [rbp-68h]
  const void *v54; // [rsp+40h] [rbp-60h] BYREF
  unsigned int v55; // [rsp+48h] [rbp-58h]
  const void *v56; // [rsp+50h] [rbp-50h] BYREF
  unsigned int v57; // [rsp+58h] [rbp-48h]
  const void *v58; // [rsp+60h] [rbp-40h] BYREF
  unsigned int v59; // [rsp+68h] [rbp-38h]

  v6 = *(unsigned int *)(a3 + 24);
  if ( (_DWORD)v6 )
  {
    v7 = *(_QWORD *)(a3 + 8);
    v8 = 1;
    v9 = (v6 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
    v10 = (_QWORD *)(v7 + 32LL * v9);
    v11 = 0;
    v12 = *v10;
    if ( *v10 == a1 )
    {
LABEL_3:
      v13 = v10[2];
      v14 = v10[1];
      v15 = v13 - v14;
      if ( v13 == v14 )
      {
        v47 = 0;
      }
      else
      {
        if ( v15 > 0x7FFFFFFFFFFFFFF8LL )
          sub_4261EA(v7, v6, 0x7FFFFFFFFFFFFFF8LL);
        v16 = sub_22077B0(v15);
        v13 = v10[2];
        v14 = v10[1];
        v47 = v16;
      }
      v17 = v47;
      if ( v14 == v13 )
      {
        v45 = 0;
        v46 = 0;
        goto LABEL_14;
      }
      while ( 1 )
      {
        if ( !v17 )
          goto LABEL_9;
        *(_QWORD *)v17 = *(_QWORD *)v14;
        v18 = *(_DWORD *)(v14 + 16);
        *(_DWORD *)(v17 + 16) = v18;
        if ( v18 <= 0x40 )
        {
          *(_QWORD *)(v17 + 8) = *(_QWORD *)(v14 + 8);
LABEL_9:
          v14 += 24;
          v17 += 24LL;
          if ( v13 == v14 )
            goto LABEL_13;
        }
        else
        {
          v19 = (const void **)(v14 + 8);
          v20 = v17 + 8;
          v14 += 24;
          v17 += 24LL;
          sub_C43780(v20, v19);
          if ( v13 == v14 )
          {
LABEL_13:
            v46 = 0xAAAAAAAAAAAAAAABLL * ((__int64)(v17 - v47) >> 3);
            v45 = v46 >> 2;
            goto LABEL_14;
          }
        }
      }
    }
    while ( v12 != -4096 )
    {
      if ( !v11 && v12 == -8192 )
        v11 = v10;
      v9 = (v6 - 1) & (v8 + v9);
      v10 = (_QWORD *)(v7 + 32LL * v9);
      v12 = *v10;
      if ( *v10 == a1 )
        goto LABEL_3;
      ++v8;
    }
    v29 = *(_DWORD *)(a3 + 16);
    if ( !v11 )
      v11 = v10;
    ++*(_QWORD *)a3;
    v30 = v29 + 1;
    if ( 4 * (v29 + 1) < (unsigned int)(3 * v6) )
    {
      if ( (int)v6 - *(_DWORD *)(a3 + 20) - v30 > (unsigned int)v6 >> 3 )
        goto LABEL_92;
      sub_2765FD0(a3, v6);
      v38 = *(_DWORD *)(a3 + 24);
      if ( v38 )
      {
        v39 = v38 - 1;
        v40 = *(_QWORD *)(a3 + 8);
        v41 = 1;
        v42 = (v38 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
        v43 = 0;
        v30 = *(_DWORD *)(a3 + 16) + 1;
        v11 = (_QWORD *)(v40 + 32LL * v42);
        v44 = *v11;
        if ( *v11 != a1 )
        {
          while ( v44 != -4096 )
          {
            if ( !v43 && v44 == -8192 )
              v43 = v11;
            v42 = v39 & (v41 + v42);
            v11 = (_QWORD *)(v40 + 32LL * v42);
            v44 = *v11;
            if ( *v11 == a1 )
              goto LABEL_92;
            ++v41;
          }
          if ( v43 )
            v11 = v43;
        }
        goto LABEL_92;
      }
LABEL_138:
      ++*(_DWORD *)(a3 + 16);
      BUG();
    }
  }
  else
  {
    ++*(_QWORD *)a3;
  }
  sub_2765FD0(a3, 2 * v6);
  v31 = *(_DWORD *)(a3 + 24);
  if ( !v31 )
    goto LABEL_138;
  v32 = v31 - 1;
  v33 = *(_QWORD *)(a3 + 8);
  v34 = (v31 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
  v30 = *(_DWORD *)(a3 + 16) + 1;
  v11 = (_QWORD *)(v33 + 32LL * v34);
  v35 = *v11;
  if ( *v11 != a1 )
  {
    v36 = 1;
    v37 = 0;
    while ( v35 != -4096 )
    {
      if ( !v37 && v35 == -8192 )
        v37 = v11;
      v34 = v32 & (v36 + v34);
      v11 = (_QWORD *)(v33 + 32LL * v34);
      v35 = *v11;
      if ( *v11 == a1 )
        goto LABEL_92;
      ++v36;
    }
    if ( v37 )
      v11 = v37;
  }
LABEL_92:
  *(_DWORD *)(a3 + 16) = v30;
  if ( *v11 != -4096 )
    --*(_DWORD *)(a3 + 20);
  *v11 = a1;
  v46 = 0;
  v45 = 0;
  v17 = 0;
  v11[1] = 0;
  v11[2] = 0;
  v11[3] = 0;
  v47 = 0;
LABEL_14:
  v21 = *(_DWORD *)(a2 + 8);
  v48 = v21;
  v51 = v21;
  if ( v21 <= 0x40 )
  {
    v22 = *(const void **)a2;
    v53 = v21;
    v50 = v22;
LABEL_16:
    v52 = v50;
    v55 = v48;
LABEL_17:
    v54 = v52;
LABEL_18:
    v58 = v54;
    v57 = v48;
LABEL_19:
    v56 = v58;
    v59 = v48;
LABEL_20:
    v58 = v56;
    goto LABEL_21;
  }
  sub_C43780((__int64)&v50, (const void **)a2);
  v48 = v51;
  v53 = v51;
  if ( v51 <= 0x40 )
    goto LABEL_16;
  sub_C43780((__int64)&v52, &v50);
  v48 = v53;
  v55 = v53;
  if ( v53 <= 0x40 )
    goto LABEL_17;
  sub_C43780((__int64)&v54, &v52);
  v48 = v55;
  v59 = v55;
  if ( v55 <= 0x40 )
    goto LABEL_18;
  sub_C43780((__int64)&v58, &v54);
  v48 = v59;
  v57 = v59;
  if ( v59 <= 0x40 )
    goto LABEL_19;
  sub_C43780((__int64)&v56, &v58);
  if ( v59 > 0x40 && v58 )
    j_j___libc_free_0_0((unsigned __int64)v58);
  v48 = v57;
  v59 = v57;
  if ( v57 <= 0x40 )
    goto LABEL_20;
  sub_C43780((__int64)&v58, &v56);
  v48 = v59;
LABEL_21:
  if ( v45 <= 0 )
  {
    v24 = v47;
LABEL_64:
    if ( v46 != 2 )
    {
      if ( v46 != 3 )
      {
        if ( v46 != 1 )
        {
          v24 = v17;
          if ( v48 <= 0x40 )
            goto LABEL_36;
          goto LABEL_34;
        }
        goto LABEL_101;
      }
      if ( *(_DWORD *)(v24 + 16) <= 0x40u )
      {
        if ( *(const void **)(v24 + 8) == v58 )
          goto LABEL_33;
      }
      else if ( sub_C43C50(v24 + 8, &v58) )
      {
        goto LABEL_33;
      }
      v24 += 24LL;
    }
    if ( *(_DWORD *)(v24 + 16) <= 0x40u )
    {
      if ( *(const void **)(v24 + 8) == v58 )
        goto LABEL_33;
    }
    else if ( sub_C43C50(v24 + 8, &v58) )
    {
      goto LABEL_33;
    }
    v24 += 24LL;
LABEL_101:
    if ( *(_DWORD *)(v24 + 16) <= 0x40u )
    {
      if ( *(const void **)(v24 + 8) != v58 )
        v24 = v17;
    }
    else if ( !sub_C43C50(v24 + 8, &v58) )
    {
      v24 = v17;
    }
LABEL_33:
    if ( v48 <= 0x40 )
      goto LABEL_36;
    goto LABEL_34;
  }
  v23 = v58;
  v24 = v47;
  while ( 1 )
  {
    while ( 1 )
    {
      if ( *(_DWORD *)(v24 + 16) > 0x40u )
      {
        if ( sub_C43C50(v24 + 8, &v58) )
          goto LABEL_33;
      }
      else if ( *(const void **)(v24 + 8) == v23 )
      {
        goto LABEL_33;
      }
      v49 = v24 + 24;
      if ( *(_DWORD *)(v24 + 40) <= 0x40u )
      {
        if ( v23 == *(const void **)(v24 + 32) )
          goto LABEL_57;
      }
      else if ( sub_C43C50(v24 + 32, &v58) )
      {
        goto LABEL_57;
      }
      v49 = v24 + 48;
      if ( *(_DWORD *)(v24 + 64) <= 0x40u )
      {
        if ( v23 == *(const void **)(v24 + 56) )
          goto LABEL_57;
      }
      else if ( sub_C43C50(v24 + 56, &v58) )
      {
        goto LABEL_57;
      }
      v49 = v24 + 72;
      if ( *(_DWORD *)(v24 + 88) <= 0x40u )
        break;
      if ( sub_C43C50(v24 + 80, &v58) )
        goto LABEL_57;
      v24 += 96LL;
      if ( !--v45 )
      {
LABEL_63:
        v46 = 0xAAAAAAAAAAAAAAABLL * ((__int64)(v17 - v24) >> 3);
        goto LABEL_64;
      }
    }
    if ( v23 == *(const void **)(v24 + 80) )
      break;
    v24 += 96LL;
    if ( !--v45 )
      goto LABEL_63;
  }
LABEL_57:
  v24 = v49;
  if ( v48 <= 0x40 )
    goto LABEL_36;
LABEL_34:
  if ( v58 )
    j_j___libc_free_0_0((unsigned __int64)v58);
LABEL_36:
  if ( v57 > 0x40 && v56 )
    j_j___libc_free_0_0((unsigned __int64)v56);
  if ( v55 > 0x40 && v54 )
    j_j___libc_free_0_0((unsigned __int64)v54);
  if ( v53 > 0x40 && v52 )
    j_j___libc_free_0_0((unsigned __int64)v52);
  if ( v51 > 0x40 && v50 )
  {
    j_j___libc_free_0_0((unsigned __int64)v50);
    if ( v17 == v24 )
      goto LABEL_71;
LABEL_47:
    v25 = *(_QWORD *)v24;
    goto LABEL_48;
  }
  if ( v17 != v24 )
    goto LABEL_47;
LABEL_71:
  v25 = 0;
LABEL_48:
  for ( i = v47; v17 != i; i += 24LL )
  {
    if ( *(_DWORD *)(i + 16) > 0x40u )
    {
      v27 = *(_QWORD *)(i + 8);
      if ( v27 )
        j_j___libc_free_0_0(v27);
    }
  }
  if ( v47 )
    j_j___libc_free_0(v47);
  return v25;
}
