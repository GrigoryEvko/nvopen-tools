// Function: sub_1ED2780
// Address: 0x1ed2780
//
__int64 __fastcall sub_1ED2780(_QWORD *a1, unsigned int a2)
{
  __int64 *v2; // rax
  __int64 v3; // rbx
  unsigned int *v4; // rax
  unsigned int *v5; // rdx
  __int64 v6; // r13
  __int64 v7; // rax
  int v8; // esi
  unsigned int v9; // r12d
  int *v10; // r14
  __int64 v11; // r13
  unsigned int v12; // eax
  unsigned int v13; // r15d
  unsigned int v14; // esi
  int v15; // r10d
  __int64 v16; // r11
  int v17; // r9d
  __int64 v18; // r8
  float *v19; // rsi
  float *v20; // rcx
  float *v21; // rdx
  float v22; // xmm1_4
  __int64 k; // rax
  float v24; // xmm0_4
  __int64 v25; // rax
  __int64 v26; // rax
  unsigned int *v27; // rdx
  unsigned int *v28; // rsi
  __int64 v29; // r13
  __int64 v30; // rax
  int v31; // edi
  __int64 v32; // rbx
  int v33; // eax
  unsigned int m; // ecx
  unsigned int v35; // edx
  int v36; // xmm0_4
  __int64 v37; // rax
  size_t v38; // rdx
  float *v39; // rax
  float *v40; // rdx
  float *v41; // rcx
  float v42; // xmm0_4
  void *v43; // rdi
  __int64 result; // rax
  int *v45; // r14
  int v46; // eax
  unsigned int v47; // eax
  unsigned int v48; // ecx
  unsigned int i; // edx
  int v50; // xmm0_4
  __int64 v51; // rax
  __int64 v52; // rax
  void *v53; // rax
  unsigned int v54; // eax
  unsigned int v55; // eax
  unsigned int v56; // ecx
  unsigned int j; // edx
  int v58; // xmm0_4
  __int64 v59; // rax
  __int64 v60; // rax
  void *v61; // rax
  float *v62; // rax
  float *v63; // rdx
  __int64 v64; // rcx
  float *v65; // rsi
  float v66; // xmm0_4
  __int64 v67; // rdi
  __int64 v68; // rdi
  __int64 v69; // rdi
  __int64 v70; // rdi
  __int64 v71; // rdi
  __int64 v72; // rdi
  unsigned int v73; // [rsp+8h] [rbp-88h]
  unsigned int v74; // [rsp+Ch] [rbp-84h]
  unsigned int v75; // [rsp+10h] [rbp-80h]
  unsigned int v76; // [rsp+14h] [rbp-7Ch]
  int v77; // [rsp+18h] [rbp-78h]
  __int64 v79; // [rsp+20h] [rbp-70h]
  unsigned int v80; // [rsp+20h] [rbp-70h]
  const void *v82; // [rsp+30h] [rbp-60h] BYREF
  __int64 v83; // [rsp+38h] [rbp-58h] BYREF
  unsigned __int64 v84; // [rsp+40h] [rbp-50h]
  void *src; // [rsp+48h] [rbp-48h] BYREF
  unsigned __int64 v86; // [rsp+50h] [rbp-40h] BYREF
  void *dest[7]; // [rsp+58h] [rbp-38h] BYREF

  v2 = (__int64 *)(a1[20] + 88LL * a2);
  v3 = *v2;
  v4 = (unsigned int *)v2[8];
  v74 = *v4;
  v79 = a1[26];
  v73 = v4[1];
  v5 = (unsigned int *)(v79 + 48LL * *v4);
  v6 = 48LL * v73;
  v7 = v6 + v79;
  v8 = *(_DWORD *)(v6 + v79 + 20);
  v76 = v5[5];
  v77 = v8;
  if ( a2 == v76 )
  {
    v9 = *(_DWORD *)(v6 + v79 + 20);
    v80 = v5[6];
    if ( a2 == v8 )
      v9 = *(_DWORD *)(v7 + 24);
    v45 = *(int **)v5;
    v46 = **(_DWORD **)v5;
    LODWORD(v86) = *(_DWORD *)(*(_QWORD *)v5 + 4LL);
    HIDWORD(v86) = v46;
    sub_1ECC890(dest, (unsigned int)(v46 * v86));
    if ( *v45 )
    {
      v47 = v45[1];
      v48 = 0;
      do
      {
        for ( i = 0; v47 > i; v47 = v45[1] )
        {
          v50 = *(_DWORD *)(*((_QWORD *)v45 + 1) + 4 * (i + (unsigned __int64)(v48 * v47)));
          v51 = HIDWORD(v86) * i++;
          *((_DWORD *)dest[0] + v48 + v51) = v50;
        }
        ++v48;
      }
      while ( *v45 > v48 );
    }
    v10 = (int *)sub_22077B0(40);
    if ( v10 )
    {
      v52 = v86;
      v86 = 0;
      *(_QWORD *)v10 = v52;
      v53 = dest[0];
      dest[0] = 0;
      *((_QWORD *)v10 + 1) = v53;
      sub_1ECBD10((unsigned int *)v10 + 4, (unsigned int *)v10);
    }
    if ( dest[0] )
      j_j___libc_free_0_0(dest[0]);
    v11 = *(_QWORD *)(v6 + a1[26]);
    if ( a2 != v8 )
      goto LABEL_5;
  }
  else
  {
    v9 = *(_DWORD *)(v6 + v79 + 20);
    if ( a2 == v8 )
      v9 = *(_DWORD *)(v7 + 24);
    v10 = *(int **)v5;
    v11 = *(_QWORD *)v7;
    v80 = v5[5];
    if ( a2 != v8 )
      goto LABEL_5;
  }
  v54 = *(_DWORD *)v11;
  LODWORD(v86) = *(_DWORD *)(v11 + 4);
  HIDWORD(v86) = v54;
  sub_1ECC890(dest, v54 * (unsigned int)v86);
  if ( *(_DWORD *)v11 )
  {
    v55 = *(_DWORD *)(v11 + 4);
    v56 = 0;
    do
    {
      for ( j = 0; v55 > j; v55 = *(_DWORD *)(v11 + 4) )
      {
        v58 = *(_DWORD *)(*(_QWORD *)(v11 + 8) + 4 * (j + (unsigned __int64)(v56 * v55)));
        v59 = HIDWORD(v86) * j++;
        *((_DWORD *)dest[0] + v56 + v59) = v58;
      }
      ++v56;
    }
    while ( *(_DWORD *)v11 > v56 );
  }
  v11 = sub_22077B0(40);
  if ( v11 )
  {
    v60 = v86;
    v86 = 0;
    *(_QWORD *)v11 = v60;
    v61 = dest[0];
    dest[0] = 0;
    *(_QWORD *)(v11 + 8) = v61;
    sub_1ECBD10((unsigned int *)(v11 + 16), (unsigned int *)v11);
  }
  if ( dest[0] )
    j_j___libc_free_0_0(dest[0]);
LABEL_5:
  v12 = *v10;
  v13 = *(_DWORD *)v3;
  v14 = *v10;
  HIDWORD(v82) = *(_DWORD *)v11;
  LODWORD(v82) = v12;
  v75 = v12;
  sub_1ECC890(&v83, HIDWORD(v82) * v14);
  v15 = 0;
  v16 = HIDWORD(v82);
  v17 = HIDWORD(v82);
  if ( v75 )
  {
    do
    {
      v18 = 0;
      if ( v17 )
      {
        do
        {
          v19 = (float *)(*((_QWORD *)v10 + 1) + 4LL * (unsigned int)(v10[1] * v15));
          v20 = (float *)(*(_QWORD *)(v11 + 8) + 4LL * (unsigned int)(*(_DWORD *)(v11 + 4) * v18));
          v21 = *(float **)(v3 + 8);
          v22 = (float)(*v19 + *v20) + *v21;
          if ( v13 > 1 )
          {
            for ( k = 1; k != v13; ++k )
            {
              v24 = (float)(v19[k] + v20[k]) + v21[k];
              v22 = fminf(v24, v22);
            }
          }
          v25 = v18 + (unsigned int)(HIDWORD(v82) * v15);
          ++v18;
          *(float *)(v83 + 4 * v25) = v22;
        }
        while ( v16 != v18 );
      }
      ++v15;
    }
    while ( v15 != v75 );
  }
  if ( a2 == v76 )
  {
    v70 = *((_QWORD *)v10 + 4);
    if ( v70 )
      j_j___libc_free_0_0(v70);
    v71 = *((_QWORD *)v10 + 3);
    if ( v71 )
      j_j___libc_free_0_0(v71);
    v72 = *((_QWORD *)v10 + 1);
    if ( v72 )
      j_j___libc_free_0_0(v72);
    j_j___libc_free_0(v10, 40);
  }
  if ( a2 == v77 )
  {
    v67 = *(_QWORD *)(v11 + 32);
    if ( v67 )
      j_j___libc_free_0_0(v67);
    v68 = *(_QWORD *)(v11 + 24);
    if ( v68 )
      j_j___libc_free_0_0(v68);
    v69 = *(_QWORD *)(v11 + 8);
    if ( v69 )
      j_j___libc_free_0_0(v69);
    j_j___libc_free_0(v11, 40);
  }
  v26 = a1[20] + 88LL * v80;
  v27 = *(unsigned int **)(v26 + 64);
  v28 = *(unsigned int **)(v26 + 72);
  if ( v27 == v28 )
    goto LABEL_60;
  while ( 1 )
  {
    v29 = *v27;
    v30 = a1[26] + 48 * v29;
    v31 = *(_DWORD *)(v30 + 20);
    if ( v9 == v31 || v9 == *(_DWORD *)(v30 + 24) )
      break;
    if ( v28 == ++v27 )
      goto LABEL_60;
  }
  if ( (_DWORD)v29 == -1 )
  {
LABEL_60:
    sub_1ECC910((__int64)&v86, &v82);
    sub_1ED1D50(a1, v80, v9, (__int64 *)&v86);
    v43 = dest[0];
    if ( dest[0] )
LABEL_31:
      j_j___libc_free_0_0(v43);
  }
  else
  {
    v32 = *(_QWORD *)v30;
    if ( v80 == v31 )
    {
      sub_1ECC910((__int64)&v86, &v82);
      v62 = *(float **)(v32 + 8);
      v63 = (float *)dest[0];
      v64 = (unsigned int)(HIDWORD(v86) * v86);
      v65 = &v62[v64];
      if ( v64 * 4 )
      {
        do
        {
          v66 = *v63 + *v62++;
          *v63++ = v66;
        }
        while ( v62 != v65 );
      }
      sub_1ED1B40((__int64)a1, v29, (__int64 *)&v86);
      v43 = dest[0];
      if ( dest[0] )
        goto LABEL_31;
    }
    else
    {
      v84 = __PAIR64__((unsigned int)v82, HIDWORD(v82));
      sub_1ECC890(&src, (unsigned int)((_DWORD)v82 * HIDWORD(v82)));
      v33 = HIDWORD(v82);
      for ( m = 0; (unsigned int)v82 > m; ++m )
      {
        v35 = 0;
        if ( v33 )
        {
          do
          {
            v36 = *(_DWORD *)(v83 + 4 * (v35 + (unsigned __int64)(m * v33)));
            v37 = HIDWORD(v84) * v35++;
            *((_DWORD *)src + m + v37) = v36;
            v33 = HIDWORD(v82);
          }
          while ( HIDWORD(v82) > v35 );
        }
      }
      v86 = v84;
      sub_1ECC890(dest, (unsigned int)(v84 * HIDWORD(v84)));
      v38 = 4LL * (unsigned int)(HIDWORD(v86) * v86);
      if ( v38 )
      {
        memmove(dest[0], src, v38);
        v39 = (float *)dest[0];
        v40 = *(float **)(v32 + 8);
        v41 = (float *)((char *)dest[0] + 4 * (unsigned int)(HIDWORD(v86) * v86));
        if ( dest[0] != v41 )
        {
          do
          {
            v42 = *v39++ + *v40++;
            *(v39 - 1) = v42;
          }
          while ( v41 != v39 );
        }
      }
      sub_1ED1B40((__int64)a1, v29, (__int64 *)&v86);
      if ( dest[0] )
        j_j___libc_free_0_0(dest[0]);
      v43 = src;
      if ( src )
        goto LABEL_31;
    }
  }
  sub_1ECDA00(a1, v74, v80);
  result = sub_1ECDA00(a1, v73, v9);
  if ( v83 )
    return j_j___libc_free_0_0(v83);
  return result;
}
