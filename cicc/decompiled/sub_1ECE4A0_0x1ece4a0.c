// Function: sub_1ECE4A0
// Address: 0x1ece4a0
//
__int64 __fastcall sub_1ECE4A0(__int64 a1, __int64 a2, _QWORD *a3)
{
  __int64 v3; // r13
  __int64 i; // rax
  unsigned int v5; // r14d
  __int64 v6; // rbx
  const void **v7; // r12
  __int64 v8; // rbx
  unsigned int *v9; // rax
  unsigned int *v10; // rbx
  __int64 v11; // rdx
  unsigned int *v12; // r14
  unsigned int v13; // edi
  __int64 v14; // rax
  __int64 v15; // r8
  __int64 v16; // rcx
  __int64 v17; // rdx
  int v18; // r12d
  size_t v19; // r15
  char *v20; // rax
  char *v21; // rdi
  unsigned int v22; // eax
  unsigned int j; // edx
  __int64 v24; // rcx
  float *v25; // rax
  float *v26; // rdx
  float v27; // xmm0_4
  _BYTE *v28; // rdi
  float *v29; // rdx
  float *v30; // rax
  float *v31; // rbx
  __int64 v32; // rbx
  __int64 v33; // r12
  __int64 v34; // rax
  __int64 v35; // rcx
  __int64 v36; // rdx
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 v39; // rdx
  _BOOL8 v40; // rdi
  unsigned int v42; // edi
  __int64 v43; // r8
  __int64 v44; // rcx
  __int64 v45; // rdx
  __int64 v46; // r12
  unsigned int v47; // eax
  int v48; // xmm0_4
  __int64 v49; // rdx
  float *v50; // rax
  float *v51; // rcx
  float *v52; // rdx
  float v53; // xmm0_4
  __int64 v54; // rbx
  unsigned int *v56; // [rsp+18h] [rbp-68h]
  unsigned int v57; // [rsp+2Ch] [rbp-54h]
  __int64 v59; // [rsp+38h] [rbp-48h]
  __int64 v60; // [rsp+38h] [rbp-48h]
  unsigned int v61; // [rsp+38h] [rbp-48h]
  __int64 v62; // [rsp+38h] [rbp-48h]
  unsigned int v63; // [rsp+40h] [rbp-40h]
  void *dest; // [rsp+48h] [rbp-38h] BYREF

  v3 = a1 + 8;
  *(_DWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = a1 + 8;
  *(_QWORD *)(a1 + 32) = a1 + 8;
  *(_QWORD *)(a1 + 40) = 0;
  for ( i = a3[1]; *a3 != i; i = a3[1] )
  {
    v5 = *(_DWORD *)(i - 4);
    a3[1] = i - 4;
    v6 = 88LL * v5;
    v7 = *(const void ***)(*(_QWORD *)(a2 + 160) + v6);
    v63 = *(_DWORD *)v7;
    sub_1ECC890(&dest, *(unsigned int *)v7);
    if ( 4LL * v63 )
    {
      memmove(dest, v7[1], 4LL * v63);
      v8 = *(_QWORD *)(a2 + 160) + v6;
      v9 = *(unsigned int **)(v8 + 64);
      v56 = *(unsigned int **)(v8 + 72);
      if ( v9 == v56 )
        goto LABEL_27;
    }
    else
    {
      v54 = *(_QWORD *)(a2 + 160) + v6;
      v9 = *(unsigned int **)(v54 + 64);
      v56 = *(unsigned int **)(v54 + 72);
      if ( v56 == v9 )
      {
        v28 = dest;
        v29 = (float *)dest;
LABEL_70:
        v31 = v29;
LABEL_32:
        v32 = ((char *)v31 - v28) >> 2;
        goto LABEL_33;
      }
    }
    v57 = v5;
    v10 = v9;
    do
    {
      v11 = *(_QWORD *)(a2 + 208) + 48LL * *v10;
      v12 = *(unsigned int **)v11;
      v13 = *(_DWORD *)(v11 + 20);
      v14 = *(_QWORD *)(a1 + 16);
      if ( v57 == v13 )
      {
        v42 = *(_DWORD *)(v11 + 24);
        v43 = v3;
        if ( v14 )
        {
          do
          {
            while ( 1 )
            {
              v44 = *(_QWORD *)(v14 + 16);
              v45 = *(_QWORD *)(v14 + 24);
              if ( v42 <= *(_DWORD *)(v14 + 32) )
                break;
              v14 = *(_QWORD *)(v14 + 24);
              if ( !v45 )
                goto LABEL_55;
            }
            v43 = v14;
            v14 = *(_QWORD *)(v14 + 16);
          }
          while ( v44 );
LABEL_55:
          if ( v3 != v43 && v42 < *(_DWORD *)(v43 + 32) )
            v43 = v3;
        }
        v46 = *v12;
        v61 = *(_DWORD *)(v43 + 36);
        v21 = (char *)sub_2207820(4 * v46);
        if ( v21 && v46 )
          v21 = (char *)memset(v21, 0, 4 * v46);
        if ( *v12 )
        {
          v47 = 0;
          do
          {
            v48 = *(_DWORD *)(*((_QWORD *)v12 + 1) + 4 * (v61 + (unsigned __int64)(v12[1] * v47)));
            v49 = v47++;
            *(_DWORD *)&v21[4 * v49] = v48;
          }
          while ( *v12 > v47 );
        }
        v50 = (float *)dest;
        v51 = (float *)((char *)dest + 4 * v63);
        if ( dest != v51 )
        {
          v52 = (float *)v21;
          do
          {
            v53 = *v50++ + *v52++;
            *(v50 - 1) = v53;
          }
          while ( v51 != v50 );
        }
      }
      else
      {
        v15 = v3;
        if ( v14 )
        {
          do
          {
            while ( 1 )
            {
              v16 = *(_QWORD *)(v14 + 16);
              v17 = *(_QWORD *)(v14 + 24);
              if ( v13 <= *(_DWORD *)(v14 + 32) )
                break;
              v14 = *(_QWORD *)(v14 + 24);
              if ( !v17 )
                goto LABEL_11;
            }
            v15 = v14;
            v14 = *(_QWORD *)(v14 + 16);
          }
          while ( v16 );
LABEL_11:
          if ( v3 != v15 && v13 < *(_DWORD *)(v15 + 32) )
            v15 = v3;
        }
        v18 = *(_DWORD *)(v15 + 36);
        v19 = 4LL * v12[1];
        v59 = v12[1];
        v20 = (char *)sub_2207820(v19);
        v21 = v20;
        if ( v20 && v59 )
          v21 = (char *)memset(v20, 0, v19);
        v22 = v12[1];
        if ( v22 )
        {
          for ( j = 0; j < v22; ++j )
          {
            v24 = j;
            *(_DWORD *)&v21[4 * v24] = *(_DWORD *)(*((_QWORD *)v12 + 1) + 4 * (v24 + v18 * v22));
            v22 = v12[1];
          }
        }
        v25 = (float *)dest;
        if ( 4LL * v63 )
        {
          v26 = (float *)v21;
          do
          {
            v27 = *v25 + *v26++;
            *v25++ = v27;
          }
          while ( v26 != (float *)&v21[4 * v63] );
        }
      }
      if ( v21 )
        j_j___libc_free_0_0(v21);
      ++v10;
    }
    while ( v56 != v10 );
    v5 = v57;
LABEL_27:
    v28 = dest;
    v29 = (float *)((char *)dest + 4 * v63);
    if ( dest == v29 )
      goto LABEL_70;
    v30 = (float *)((char *)dest + 4);
    v31 = (float *)dest;
    if ( (char *)dest + 4 != (char *)v29 )
    {
      do
      {
        if ( *v31 > *v30 )
          v31 = v30;
        ++v30;
      }
      while ( v29 != v30 );
      goto LABEL_32;
    }
    LODWORD(v32) = 0;
LABEL_33:
    v33 = v3;
    v34 = *(_QWORD *)(a1 + 16);
    if ( !v34 )
      goto LABEL_40;
    do
    {
      while ( 1 )
      {
        v35 = *(_QWORD *)(v34 + 16);
        v36 = *(_QWORD *)(v34 + 24);
        if ( v5 <= *(_DWORD *)(v34 + 32) )
          break;
        v34 = *(_QWORD *)(v34 + 24);
        if ( !v36 )
          goto LABEL_38;
      }
      v33 = v34;
      v34 = *(_QWORD *)(v34 + 16);
    }
    while ( v35 );
LABEL_38:
    if ( v3 == v33 || v5 < *(_DWORD *)(v33 + 32) )
    {
LABEL_40:
      v60 = v33;
      v37 = sub_22077B0(40);
      *(_DWORD *)(v37 + 32) = v5;
      v33 = v37;
      *(_DWORD *)(v37 + 36) = 0;
      v38 = sub_609E00((_QWORD *)a1, v60, (unsigned int *)(v37 + 32));
      if ( v39 )
      {
        v40 = v3 == v39 || v38 || v5 < *(_DWORD *)(v39 + 32);
        sub_220F040(v40, v33, v39, v3);
        ++*(_QWORD *)(a1 + 40);
      }
      else
      {
        v62 = v38;
        j_j___libc_free_0(v33, 40);
        v33 = v62;
      }
      v28 = dest;
    }
    *(_DWORD *)(v33 + 36) = v32;
    if ( v28 )
      j_j___libc_free_0_0(v28);
  }
  return a1;
}
