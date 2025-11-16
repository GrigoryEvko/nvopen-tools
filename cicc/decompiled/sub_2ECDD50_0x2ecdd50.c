// Function: sub_2ECDD50
// Address: 0x2ecdd50
//
void __fastcall sub_2ECDD50(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v9; // r8
  __int64 v10; // r9
  unsigned __int64 v11; // r13
  unsigned __int64 v12; // rax
  __int64 v13; // rdx
  _DWORD *v14; // rax
  _DWORD *i; // rdx
  unsigned __int64 v16; // rax
  __int64 v17; // rdx
  _DWORD *v18; // rax
  _DWORD *j; // rdx
  unsigned __int64 v20; // r12
  unsigned __int64 v21; // rax
  unsigned __int64 v22; // r14
  unsigned __int64 *v23; // rcx
  unsigned __int64 v24; // r14
  __int64 v25; // r12
  unsigned int v26; // edx
  __int64 v27; // rdi
  __int64 v28; // r11
  __int64 v29; // rsi
  unsigned __int64 v30; // rdi
  __int64 v31; // rdx
  int v32; // ecx
  unsigned int *v33; // rax
  __int64 v34; // r8
  unsigned int v35; // ecx
  __int64 v36; // r12
  __int64 v37; // r13
  __int64 v38; // rdx
  char *v39; // rsi
  __int64 v40; // rdx
  unsigned __int64 v41; // rax
  unsigned __int64 v42; // rcx
  char *v43; // rax
  __int64 v44; // r12
  unsigned int *v45; // rdi
  char *v46; // r12
  unsigned __int64 *v47; // [rsp+0h] [rbp-50h]
  unsigned __int64 v48; // [rsp+8h] [rbp-48h]
  unsigned __int64 v49; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v50; // [rsp+18h] [rbp-38h]

  sub_2EC8570(a1, a2, a3, a4, a5, a6);
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 8) = a3;
  *(_QWORD *)(a1 + 16) = a4;
  if ( !(unsigned __int8)sub_2FF7B70(a3) )
    return;
  v11 = *(unsigned int *)(*(_QWORD *)(a1 + 8) + 48LL);
  v12 = *(unsigned int *)(a1 + 368);
  if ( v11 != v12 )
  {
    if ( v11 >= v12 )
    {
      if ( v11 > *(unsigned int *)(a1 + 372) )
      {
        sub_C8D5F0(a1 + 360, (const void *)(a1 + 376), *(unsigned int *)(*(_QWORD *)(a1 + 8) + 48LL), 4u, v9, v10);
        v12 = *(unsigned int *)(a1 + 368);
      }
      v13 = *(_QWORD *)(a1 + 360);
      v14 = (_DWORD *)(v13 + 4 * v12);
      for ( i = (_DWORD *)(v13 + 4 * v11); i != v14; ++v14 )
      {
        if ( v14 )
          *v14 = 0;
      }
    }
    *(_DWORD *)(a1 + 368) = v11;
  }
  v16 = *(unsigned int *)(a1 + 200);
  if ( v11 != v16 )
  {
    if ( v11 >= v16 )
    {
      if ( v11 > *(unsigned int *)(a1 + 204) )
      {
        sub_C8D5F0(a1 + 192, (const void *)(a1 + 208), v11, 4u, v9, v10);
        v16 = *(unsigned int *)(a1 + 200);
      }
      v17 = *(_QWORD *)(a1 + 192);
      v18 = (_DWORD *)(v17 + 4 * v16);
      for ( j = (_DWORD *)(v17 + 4 * v11); j != v18; ++v18 )
      {
        if ( v18 )
          *v18 = 0;
      }
    }
    *(_DWORD *)(a1 + 200) = v11;
  }
  v50 = v11;
  if ( (unsigned int)v11 > 0x40 )
  {
    sub_C43690((__int64)&v49, 0, 0);
    v20 = *(unsigned int *)(a1 + 448);
    if ( v11 == v20 )
      goto LABEL_53;
  }
  else
  {
    v20 = *(unsigned int *)(a1 + 448);
    v49 = 0;
    if ( v11 == v20 )
      goto LABEL_32;
  }
  v21 = *(_QWORD *)(a1 + 440);
  v22 = v21 + 16 * v20;
  if ( v11 < v20 )
  {
    v44 = v21 + 16 * v11;
    while ( v44 != v22 )
    {
      while ( 1 )
      {
        v22 -= 16LL;
        if ( *(_DWORD *)(v22 + 8) <= 0x40u || !*(_QWORD *)v22 )
          break;
        j_j___libc_free_0_0(*(_QWORD *)v22);
        if ( v44 == v22 )
          goto LABEL_52;
      }
    }
LABEL_52:
    *(_DWORD *)(a1 + 448) = v11;
LABEL_53:
    if ( v50 > 0x40 )
      goto LABEL_54;
    goto LABEL_32;
  }
  v23 = &v49;
  v48 = v11 - v20;
  if ( v11 > *(unsigned int *)(a1 + 452) )
  {
    v45 = (unsigned int *)(a1 + 440);
    if ( v21 > (unsigned __int64)&v49 || v22 <= (unsigned __int64)&v49 )
    {
      sub_AE4800(v45, v11);
      v21 = *(_QWORD *)(a1 + 440);
      v20 = *(unsigned int *)(a1 + 448);
      v23 = &v49;
    }
    else
    {
      v46 = (char *)&v49 - v21;
      sub_AE4800(v45, v11);
      v21 = *(_QWORD *)(a1 + 440);
      v23 = (unsigned __int64 *)&v46[v21];
      v20 = *(unsigned int *)(a1 + 448);
    }
  }
  v24 = v48;
  v25 = v21 + 16 * v20;
  do
  {
    while ( 1 )
    {
      if ( !v25 )
        goto LABEL_27;
      v26 = *((_DWORD *)v23 + 2);
      *(_DWORD *)(v25 + 8) = v26;
      if ( v26 > 0x40 )
        break;
      *(_QWORD *)v25 = *v23;
LABEL_27:
      v25 += 16;
      if ( !--v24 )
        goto LABEL_31;
    }
    v27 = v25;
    v47 = v23;
    v25 += 16;
    sub_C43780(v27, (const void **)v23);
    v23 = v47;
    --v24;
  }
  while ( v24 );
LABEL_31:
  *(_DWORD *)(a1 + 448) += v48;
  if ( v50 <= 0x40 )
    goto LABEL_32;
LABEL_54:
  if ( v49 )
    j_j___libc_free_0_0(v49);
LABEL_32:
  if ( !(_DWORD)v11 )
  {
    v39 = *(char **)(a1 + 344);
    v40 = *(_QWORD *)(a1 + 336);
    v30 = 0;
    v42 = (__int64)&v39[-v40] >> 2;
    goto LABEL_43;
  }
  v28 = 4 * v11;
  v29 = 0;
  LODWORD(v30) = 0;
  do
  {
    *(_DWORD *)(*(_QWORD *)(a1 + 360) + v29) = v30;
    v31 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 32LL) + 8 * v29;
    v32 = *(_DWORD *)(v31 + 8);
    v33 = *(unsigned int **)(v31 + 24);
    v30 = (unsigned int)(v32 + v30);
    if ( v33 && !*(_DWORD *)(v31 + 16) && v32 )
    {
      v34 = (__int64)&v33[v32 - 1 + 1];
      do
      {
        while ( 1 )
        {
          v35 = *v33;
          v36 = 4 * v29 + *(_QWORD *)(a1 + 440);
          v37 = 1LL << *v33;
          v38 = *(_QWORD *)v36;
          if ( *(_DWORD *)(v36 + 8) > 0x40u )
            break;
          ++v33;
          *(_QWORD *)v36 = v37 | v38;
          if ( v33 == (unsigned int *)v34 )
            goto LABEL_41;
        }
        ++v33;
        *(_QWORD *)(v38 + 8LL * (v35 >> 6)) |= v37;
      }
      while ( v33 != (unsigned int *)v34 );
    }
LABEL_41:
    v29 += 4;
  }
  while ( v28 != v29 );
  v39 = *(char **)(a1 + 344);
  v40 = *(_QWORD *)(a1 + 336);
  v41 = (__int64)&v39[-v40] >> 2;
  v42 = v41;
  if ( v30 > v41 )
  {
    sub_1CFD340(a1 + 336, v39, v30 - v41, &dword_4451070);
  }
  else
  {
LABEL_43:
    if ( v42 > v30 )
    {
      v43 = (char *)(v40 + 4 * v30);
      if ( v39 != v43 )
        *(_QWORD *)(a1 + 344) = v43;
    }
  }
}
