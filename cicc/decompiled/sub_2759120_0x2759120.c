// Function: sub_2759120
// Address: 0x2759120
//
__int64 __fastcall sub_2759120(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // r13
  unsigned int v9; // esi
  __int64 v10; // rcx
  int v11; // r10d
  __int64 v12; // r14
  unsigned int v13; // edx
  __int64 v14; // rax
  __int64 v15; // r8
  __int64 v16; // rax
  int v18; // eax
  int v19; // edx
  __int64 v20; // rax
  unsigned __int64 v21; // rdx
  __int64 v22; // rax
  unsigned __int64 v23; // rsi
  int v24; // ecx
  __int64 v25; // rdi
  char *v26; // rdx
  __int64 v27; // rax
  __int64 v28; // rcx
  __int64 v29; // rsi
  __int64 v30; // rcx
  __int64 v31; // rcx
  __int64 v32; // rdx
  _BYTE *v33; // r13
  _BYTE *v34; // r12
  unsigned __int64 v35; // r15
  unsigned __int64 v36; // rdi
  _QWORD *v37; // r13
  _QWORD *v38; // r12
  unsigned __int64 v39; // r15
  unsigned __int64 v40; // rdi
  int v41; // eax
  unsigned int v42; // eax
  __int64 v43; // rdi
  int v44; // esi
  __int64 v45; // rcx
  unsigned __int64 v46; // r12
  __int64 v47; // rdi
  int v48; // eax
  int v49; // esi
  __int64 v50; // rdi
  __int64 v51; // r15
  int v52; // eax
  __int64 v53; // rcx
  __int64 v54; // [rsp+20h] [rbp-90h]
  __int64 v55; // [rsp+28h] [rbp-88h]
  _QWORD *v56; // [rsp+30h] [rbp-80h]
  __int64 v57; // [rsp+38h] [rbp-78h]
  _QWORD v58[2]; // [rsp+40h] [rbp-70h] BYREF
  __int64 v59; // [rsp+50h] [rbp-60h]
  __int64 v60; // [rsp+58h] [rbp-58h]
  __int64 v61; // [rsp+60h] [rbp-50h]
  _BYTE *v62; // [rsp+68h] [rbp-48h]
  __int64 v63; // [rsp+70h] [rbp-40h]
  _BYTE v64[56]; // [rsp+78h] [rbp-38h] BYREF

  v8 = *a2;
  v9 = *(_DWORD *)(a1 + 24);
  if ( !v9 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_39;
  }
  v10 = *(_QWORD *)(a1 + 8);
  v11 = 1;
  v12 = 0;
  v13 = (v9 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
  v14 = v10 + 16LL * v13;
  v15 = *(_QWORD *)v14;
  if ( v8 == *(_QWORD *)v14 )
  {
LABEL_3:
    v16 = *(unsigned int *)(v14 + 8);
    return *(_QWORD *)(a1 + 32) + 56 * v16 + 8;
  }
  while ( v15 != -4096 )
  {
    if ( !v12 && v15 == -8192 )
      v12 = v14;
    a6 = (unsigned int)(v11 + 1);
    v13 = (v9 - 1) & (v11 + v13);
    v14 = v10 + 16LL * v13;
    v15 = *(_QWORD *)v14;
    if ( v8 == *(_QWORD *)v14 )
      goto LABEL_3;
    ++v11;
  }
  if ( !v12 )
    v12 = v14;
  v18 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v19 = v18 + 1;
  if ( 4 * (v18 + 1) >= 3 * v9 )
  {
LABEL_39:
    sub_B23080(a1, 2 * v9);
    v41 = *(_DWORD *)(a1 + 24);
    if ( v41 )
    {
      v15 = (unsigned int)(v41 - 1);
      a6 = *(_QWORD *)(a1 + 8);
      v42 = v15 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v19 = *(_DWORD *)(a1 + 16) + 1;
      v12 = a6 + 16LL * v42;
      v43 = *(_QWORD *)v12;
      if ( v8 != *(_QWORD *)v12 )
      {
        v44 = 1;
        v45 = 0;
        while ( v43 != -4096 )
        {
          if ( !v45 && v43 == -8192 )
            v45 = v12;
          v42 = v15 & (v44 + v42);
          v12 = a6 + 16LL * v42;
          v43 = *(_QWORD *)v12;
          if ( v8 == *(_QWORD *)v12 )
            goto LABEL_15;
          ++v44;
        }
        if ( v45 )
          v12 = v45;
      }
      goto LABEL_15;
    }
    goto LABEL_66;
  }
  if ( v9 - *(_DWORD *)(a1 + 20) - v19 <= v9 >> 3 )
  {
    sub_B23080(a1, v9);
    v48 = *(_DWORD *)(a1 + 24);
    if ( v48 )
    {
      v49 = v48 - 1;
      v50 = *(_QWORD *)(a1 + 8);
      v15 = 0;
      LODWORD(v51) = (v48 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v19 = *(_DWORD *)(a1 + 16) + 1;
      v52 = 1;
      v12 = v50 + 16LL * (unsigned int)v51;
      v53 = *(_QWORD *)v12;
      if ( v8 != *(_QWORD *)v12 )
      {
        while ( v53 != -4096 )
        {
          if ( !v15 && v53 == -8192 )
            v15 = v12;
          a6 = (unsigned int)(v52 + 1);
          v51 = v49 & (unsigned int)(v51 + v52);
          v12 = v50 + 16 * v51;
          v53 = *(_QWORD *)v12;
          if ( v8 == *(_QWORD *)v12 )
            goto LABEL_15;
          ++v52;
        }
        if ( v15 )
          v12 = v15;
      }
      goto LABEL_15;
    }
LABEL_66:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
LABEL_15:
  *(_DWORD *)(a1 + 16) = v19;
  if ( *(_QWORD *)v12 != -4096 )
    --*(_DWORD *)(a1 + 20);
  *(_QWORD *)v12 = v8;
  *(_DWORD *)(v12 + 8) = 0;
  v20 = *a2;
  v21 = *(unsigned int *)(a1 + 44);
  v56 = v58;
  v58[0] = v20;
  v62 = v64;
  v22 = *(unsigned int *)(a1 + 40);
  v23 = v22 + 1;
  v24 = v22;
  v54 = 0;
  v55 = 0;
  v57 = 0;
  v58[1] = 1;
  v59 = 0;
  v60 = 0;
  v61 = 0;
  v63 = 0;
  if ( v22 + 1 > v21 )
  {
    v46 = *(_QWORD *)(a1 + 32);
    v47 = a1 + 32;
    if ( v46 > (unsigned __int64)v58 || (v21 = 7 * v22, (unsigned __int64)v58 >= v46 + 56 * v22) )
    {
      sub_2758F00(v47, v23, v21, v22, v15, a6);
      v22 = *(unsigned int *)(a1 + 40);
      v25 = *(_QWORD *)(a1 + 32);
      v26 = (char *)v58;
      v24 = *(_DWORD *)(a1 + 40);
    }
    else
    {
      sub_2758F00(v47, v23, v21, v22, v15, a6);
      v25 = *(_QWORD *)(a1 + 32);
      v22 = *(unsigned int *)(a1 + 40);
      v24 = *(_DWORD *)(a1 + 40);
      v26 = (char *)v58 + v25 - v46;
    }
  }
  else
  {
    v25 = *(_QWORD *)(a1 + 32);
    v26 = (char *)v58;
  }
  v27 = v25 + 56 * v22;
  if ( v27 )
  {
    v28 = *(_QWORD *)v26;
    *(_QWORD *)(v27 + 24) = 0;
    *(_QWORD *)(v27 + 16) = 0;
    *(_DWORD *)(v27 + 32) = 0;
    *(_QWORD *)v27 = v28;
    *(_QWORD *)(v27 + 8) = 1;
    v29 = *((_QWORD *)v26 + 2);
    ++*((_QWORD *)v26 + 1);
    v30 = *(_QWORD *)(v27 + 16);
    *(_QWORD *)(v27 + 16) = v29;
    LODWORD(v29) = *((_DWORD *)v26 + 6);
    *((_QWORD *)v26 + 2) = v30;
    LODWORD(v30) = *(_DWORD *)(v27 + 24);
    *(_DWORD *)(v27 + 24) = v29;
    LODWORD(v29) = *((_DWORD *)v26 + 7);
    *((_DWORD *)v26 + 6) = v30;
    LODWORD(v30) = *(_DWORD *)(v27 + 28);
    *(_DWORD *)(v27 + 28) = v29;
    LODWORD(v29) = *((_DWORD *)v26 + 8);
    *((_DWORD *)v26 + 7) = v30;
    LODWORD(v30) = *(_DWORD *)(v27 + 32);
    *(_DWORD *)(v27 + 32) = v29;
    *((_DWORD *)v26 + 8) = v30;
    *(_QWORD *)(v27 + 40) = v27 + 56;
    *(_QWORD *)(v27 + 48) = 0;
    v31 = *((unsigned int *)v26 + 12);
    if ( (_DWORD)v31 )
      sub_27589D0(v27 + 40, (__int64)(v26 + 40), (__int64)v26, v31, v15, a6);
    v24 = *(_DWORD *)(a1 + 40);
  }
  v32 = (unsigned int)v63;
  v33 = v62;
  *(_DWORD *)(a1 + 40) = v24 + 1;
  v34 = &v33[56 * v32];
  if ( v33 != v34 )
  {
    do
    {
      v35 = *((_QWORD *)v34 - 4);
      v34 -= 56;
      while ( v35 )
      {
        sub_2754510(*(_QWORD *)(v35 + 24));
        v36 = v35;
        v35 = *(_QWORD *)(v35 + 16);
        j_j___libc_free_0(v36);
      }
    }
    while ( v33 != v34 );
    v34 = v62;
  }
  if ( v34 != v64 )
    _libc_free((unsigned __int64)v34);
  sub_C7D6A0(v59, 16LL * (unsigned int)v61, 8);
  v37 = v56;
  v38 = &v56[7 * (unsigned int)v57];
  if ( v56 != v38 )
  {
    do
    {
      v39 = *(v38 - 4);
      v38 -= 7;
      while ( v39 )
      {
        sub_2754510(*(_QWORD *)(v39 + 24));
        v40 = v39;
        v39 = *(_QWORD *)(v39 + 16);
        j_j___libc_free_0(v40);
      }
    }
    while ( v37 != v38 );
    v38 = v56;
  }
  if ( v38 != v58 )
    _libc_free((unsigned __int64)v38);
  sub_C7D6A0(0, 16LL * (unsigned int)v55, 8);
  v16 = (unsigned int)(*(_DWORD *)(a1 + 40) - 1);
  *(_DWORD *)(v12 + 8) = v16;
  return *(_QWORD *)(a1 + 32) + 56 * v16 + 8;
}
