// Function: sub_2BA0610
// Address: 0x2ba0610
//
__int64 __fastcall sub_2BA0610(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // r15
  unsigned int v9; // esi
  __int64 v10; // rdx
  int v11; // r10d
  __int64 v12; // r13
  unsigned __int64 v13; // r12
  unsigned int v14; // ecx
  __int64 v15; // rax
  __int64 v16; // r8
  __int64 v17; // rax
  int v19; // eax
  int v20; // edx
  __int64 v21; // rdx
  unsigned __int64 v22; // rcx
  unsigned __int64 v23; // rsi
  int v24; // eax
  __int64 v25; // rdi
  char *v26; // rsi
  __int64 v27; // rcx
  __int64 v28; // rdi
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rax
  __int64 v32; // rdx
  _BYTE *v33; // r15
  unsigned __int64 v34; // r12
  unsigned __int64 v35; // rdi
  int v36; // edx
  int v37; // ecx
  __int64 v38; // rsi
  unsigned int v39; // eax
  __int64 v40; // rdi
  unsigned __int64 v41; // r15
  __int64 v42; // rdi
  int v43; // eax
  int v44; // ecx
  __int64 v45; // rdi
  unsigned int v46; // eax
  __int64 v47; // rsi
  __int64 v48; // [rsp+28h] [rbp-78h]
  _QWORD v49[2]; // [rsp+30h] [rbp-70h] BYREF
  __int64 v50; // [rsp+40h] [rbp-60h]
  __int64 v51; // [rsp+48h] [rbp-58h]
  __int64 v52; // [rsp+50h] [rbp-50h]
  _BYTE *v53; // [rsp+58h] [rbp-48h]
  __int64 v54; // [rsp+60h] [rbp-40h]
  _BYTE v55[56]; // [rsp+68h] [rbp-38h] BYREF

  v8 = *a2;
  v9 = *(_DWORD *)(a1 + 24);
  if ( !v9 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_32;
  }
  v10 = *(_QWORD *)(a1 + 8);
  v11 = 1;
  v12 = 0;
  v13 = (0xBF58476D1CE4E5B9LL * v8) ^ ((0xBF58476D1CE4E5B9LL * v8) >> 31);
  v14 = v13 & (v9 - 1);
  v15 = v10 + 16LL * v14;
  v16 = *(_QWORD *)v15;
  if ( v8 == *(_QWORD *)v15 )
  {
LABEL_3:
    v17 = *(unsigned int *)(v15 + 8);
    return *(_QWORD *)(a1 + 32) + 56 * v17 + 8;
  }
  while ( v16 != -1 )
  {
    if ( !v12 && v16 == -2 )
      v12 = v15;
    a6 = (unsigned int)(v11 + 1);
    v14 = (v9 - 1) & (v11 + v14);
    v15 = v10 + 16LL * v14;
    v16 = *(_QWORD *)v15;
    if ( v8 == *(_QWORD *)v15 )
      goto LABEL_3;
    ++v11;
  }
  if ( !v12 )
    v12 = v15;
  v19 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v20 = v19 + 1;
  if ( 4 * (v19 + 1) >= 3 * v9 )
  {
LABEL_32:
    sub_9E25D0(a1, 2 * v9);
    v36 = *(_DWORD *)(a1 + 24);
    if ( v36 )
    {
      v37 = v36 - 1;
      v38 = *(_QWORD *)(a1 + 8);
      v20 = *(_DWORD *)(a1 + 16) + 1;
      v39 = v37 & (((0xBF58476D1CE4E5B9LL * v8) >> 31) ^ (484763065 * v8));
      v12 = v38 + 16LL * v39;
      v40 = *(_QWORD *)v12;
      if ( v8 == *(_QWORD *)v12 )
        goto LABEL_15;
      a6 = 1;
      v16 = 0;
      while ( v40 != -1 )
      {
        if ( !v16 && v40 == -2 )
          v16 = v12;
        v39 = v37 & (a6 + v39);
        v12 = v38 + 16LL * v39;
        v40 = *(_QWORD *)v12;
        if ( v8 == *(_QWORD *)v12 )
          goto LABEL_15;
        a6 = (unsigned int)(a6 + 1);
      }
LABEL_36:
      if ( v16 )
        v12 = v16;
      goto LABEL_15;
    }
LABEL_56:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
  if ( v9 - *(_DWORD *)(a1 + 20) - v20 <= v9 >> 3 )
  {
    sub_9E25D0(a1, v9);
    v43 = *(_DWORD *)(a1 + 24);
    if ( v43 )
    {
      v44 = v43 - 1;
      v45 = *(_QWORD *)(a1 + 8);
      v46 = v44 & v13;
      a6 = 1;
      v16 = 0;
      v20 = *(_DWORD *)(a1 + 16) + 1;
      v12 = v45 + 16LL * (v44 & (unsigned int)v13);
      v47 = *(_QWORD *)v12;
      if ( v8 == *(_QWORD *)v12 )
        goto LABEL_15;
      while ( v47 != -1 )
      {
        if ( !v16 && v47 == -2 )
          v16 = v12;
        v46 = v44 & (a6 + v46);
        v12 = v45 + 16LL * v46;
        v47 = *(_QWORD *)v12;
        if ( v8 == *(_QWORD *)v12 )
          goto LABEL_15;
        a6 = (unsigned int)(a6 + 1);
      }
      goto LABEL_36;
    }
    goto LABEL_56;
  }
LABEL_15:
  *(_DWORD *)(a1 + 16) = v20;
  if ( *(_QWORD *)v12 != -1 )
    --*(_DWORD *)(a1 + 20);
  *(_QWORD *)v12 = v8;
  *(_DWORD *)(v12 + 8) = 0;
  v21 = *(unsigned int *)(a1 + 40);
  v22 = *(unsigned int *)(a1 + 44);
  v23 = v21 + 1;
  v49[0] = *a2;
  v24 = v21;
  v48 = 0;
  v49[1] = 1;
  v50 = 0;
  v51 = 0;
  v52 = 0;
  v53 = v55;
  v54 = 0;
  if ( v21 + 1 > v22 )
  {
    v41 = *(_QWORD *)(a1 + 32);
    v42 = a1 + 32;
    if ( v41 > (unsigned __int64)v49 || (unsigned __int64)v49 >= v41 + 56 * v21 )
    {
      sub_2B55770(v42, v23, v21, v22, v16, a6);
      v21 = *(unsigned int *)(a1 + 40);
      v25 = *(_QWORD *)(a1 + 32);
      v26 = (char *)v49;
      v24 = *(_DWORD *)(a1 + 40);
    }
    else
    {
      sub_2B55770(v42, v23, v21, v22, v16, a6);
      v25 = *(_QWORD *)(a1 + 32);
      v21 = *(unsigned int *)(a1 + 40);
      v26 = (char *)v49 + v25 - v41;
      v24 = *(_DWORD *)(a1 + 40);
    }
  }
  else
  {
    v25 = *(_QWORD *)(a1 + 32);
    v26 = (char *)v49;
  }
  v27 = 7 * v21;
  v28 = v25 + 56 * v21;
  if ( v28 )
  {
    v29 = *(_QWORD *)v26;
    *(_QWORD *)(v28 + 24) = 0;
    *(_QWORD *)(v28 + 16) = 0;
    *(_DWORD *)(v28 + 32) = 0;
    *(_QWORD *)v28 = v29;
    *(_QWORD *)(v28 + 8) = 1;
    v30 = *((_QWORD *)v26 + 2);
    ++*((_QWORD *)v26 + 1);
    v31 = *(_QWORD *)(v28 + 16);
    *(_QWORD *)(v28 + 16) = v30;
    LODWORD(v30) = *((_DWORD *)v26 + 6);
    *((_QWORD *)v26 + 2) = v31;
    LODWORD(v31) = *(_DWORD *)(v28 + 24);
    *(_DWORD *)(v28 + 24) = v30;
    LODWORD(v30) = *((_DWORD *)v26 + 7);
    *((_DWORD *)v26 + 6) = v31;
    LODWORD(v31) = *(_DWORD *)(v28 + 28);
    *(_DWORD *)(v28 + 28) = v30;
    v32 = *((unsigned int *)v26 + 8);
    *((_DWORD *)v26 + 7) = v31;
    LODWORD(v31) = *(_DWORD *)(v28 + 32);
    *(_DWORD *)(v28 + 32) = v32;
    *((_DWORD *)v26 + 8) = v31;
    *(_QWORD *)(v28 + 40) = v28 + 56;
    *(_QWORD *)(v28 + 48) = 0;
    if ( *((_DWORD *)v26 + 12) )
      sub_2B553F0(v28 + 40, (__int64)(v26 + 40), v32, v27, v16, a6);
    v24 = *(_DWORD *)(a1 + 40);
  }
  v33 = v53;
  *(_DWORD *)(a1 + 40) = v24 + 1;
  v34 = (unsigned __int64)&v33[72 * (unsigned int)v54];
  if ( v33 != (_BYTE *)v34 )
  {
    do
    {
      v34 -= 72LL;
      v35 = *(_QWORD *)(v34 + 8);
      if ( v35 != v34 + 24 )
        _libc_free(v35);
    }
    while ( v33 != (_BYTE *)v34 );
    v34 = (unsigned __int64)v53;
  }
  if ( (_BYTE *)v34 != v55 )
    _libc_free(v34);
  sub_C7D6A0(v50, 16LL * (unsigned int)v52, 8);
  sub_C7D6A0(0, 0, 8);
  v17 = (unsigned int)(*(_DWORD *)(a1 + 40) - 1);
  *(_DWORD *)(v12 + 8) = v17;
  return *(_QWORD *)(a1 + 32) + 56 * v17 + 8;
}
