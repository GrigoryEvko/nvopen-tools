// Function: sub_35C0840
// Address: 0x35c0840
//
void __fastcall sub_35C0840(_QWORD *a1, unsigned int a2)
{
  __int64 *v2; // rax
  __int64 v3; // rbx
  unsigned int *v4; // rax
  size_t v5; // rdx
  size_t v6; // rax
  int *v7; // r13
  unsigned __int64 v8; // r12
  unsigned int v9; // r15d
  unsigned int v10; // r14d
  int v11; // r9d
  unsigned int v12; // r15d
  __int64 k; // r8
  float *v14; // rsi
  float *v15; // rcx
  float *v16; // rdx
  float v17; // xmm1_4
  __int64 m; // rax
  float v19; // xmm0_4
  __int64 v20; // rax
  unsigned int *v21; // rdx
  unsigned int *v22; // rsi
  __int64 v23; // r13
  __int64 v24; // rax
  int v25; // edi
  __int64 v26; // rbx
  _DWORD *v27; // rax
  _DWORD *v28; // r12
  unsigned int v29; // r8d
  __int64 v30; // rsi
  unsigned int v31; // ecx
  char *v32; // rax
  int v33; // xmm0_4
  __int64 v34; // rdx
  float *v35; // rax
  float *v36; // r14
  size_t v37; // rdx
  size_t v38; // r15
  float *v39; // rax
  float *v40; // rdx
  float v41; // xmm0_4
  unsigned __int64 v42; // rdi
  int *v43; // r13
  int v44; // r14d
  int v45; // r15d
  void *v46; // rax
  unsigned __int64 v47; // r12
  unsigned int v48; // eax
  unsigned int v49; // esi
  unsigned int v50; // ecx
  unsigned int i; // edx
  __int64 v52; // r8
  int v53; // xmm0_4
  __int64 v54; // rax
  __int64 v55; // rax
  unsigned int v56; // r15d
  void *v57; // rax
  unsigned __int64 v58; // r14
  unsigned int v59; // eax
  unsigned int v60; // esi
  unsigned int v61; // ecx
  unsigned int j; // edx
  __int64 v63; // r8
  int v64; // xmm0_4
  __int64 v65; // rax
  unsigned __int64 v66; // rdi
  float *v67; // rax
  float *v68; // r12
  size_t v69; // rax
  size_t v70; // r14
  float *v71; // rax
  float *ii; // rdx
  float v73; // xmm0_4
  unsigned __int64 v74; // rdi
  unsigned __int64 v75; // rdi
  unsigned __int64 v76; // rdi
  unsigned __int64 v77; // rdi
  unsigned __int64 v78; // rdi
  unsigned __int64 v79; // rdi
  void *v80; // rdi
  size_t v81; // rdx
  size_t v82; // rdx
  size_t v83; // rax
  size_t v84; // r14
  unsigned int v85; // [rsp+4h] [rbp-8Ch]
  unsigned int v86; // [rsp+8h] [rbp-88h]
  int v87; // [rsp+Ch] [rbp-84h]
  int v88; // [rsp+10h] [rbp-80h]
  __int64 v90; // [rsp+18h] [rbp-78h]
  unsigned int v91; // [rsp+20h] [rbp-70h]
  unsigned int v92; // [rsp+24h] [rbp-6Ch]
  __int64 n; // [rsp+30h] [rbp-60h]
  unsigned int v95; // [rsp+38h] [rbp-58h]
  __int64 v96; // [rsp+38h] [rbp-58h]
  __int64 v97; // [rsp+38h] [rbp-58h]
  __int64 s; // [rsp+40h] [rbp-50h]
  char *sa; // [rsp+40h] [rbp-50h]
  size_t v100; // [rsp+48h] [rbp-48h]
  unsigned int v101; // [rsp+48h] [rbp-48h]
  int v102; // [rsp+48h] [rbp-48h]
  unsigned __int64 v103; // [rsp+50h] [rbp-40h] BYREF
  float *v104; // [rsp+58h] [rbp-38h]

  v2 = (__int64 *)(a1[20] + 96LL * a2);
  v3 = *v2;
  v4 = (unsigned int *)v2[9];
  v86 = *v4;
  v100 = a1[26];
  v85 = v4[1];
  v5 = v100 + 48LL * *v4;
  s = 48LL * v85;
  v87 = *(_DWORD *)(v5 + 20);
  v6 = s + v100;
  v88 = *(_DWORD *)(s + v100 + 20);
  if ( a2 != v87 )
  {
    if ( a2 == v88 )
      v92 = *(_DWORD *)(v6 + 24);
    else
      v92 = *(_DWORD *)(s + v100 + 20);
    v7 = *(int **)v5;
    v8 = *(_QWORD *)v6;
    v91 = *(_DWORD *)(v5 + 20);
    if ( a2 != v88 )
      goto LABEL_5;
    goto LABEL_62;
  }
  v91 = *(_DWORD *)(v5 + 24);
  if ( a2 == *(_DWORD *)(s + v100 + 20) )
    v92 = *(_DWORD *)(v6 + 24);
  else
    v92 = *(_DWORD *)(s + v100 + 20);
  v43 = *(int **)v5;
  v44 = **(_DWORD **)v5;
  v45 = *(_DWORD *)(*(_QWORD *)v5 + 4LL);
  v96 = (unsigned int)(v45 * v44);
  v46 = (void *)sub_2207820(4 * v96);
  v47 = (unsigned __int64)v46;
  if ( v46 && v45 * v44 )
    memset(v46, 0, 4 * v96);
  if ( *v43 )
  {
    v48 = v43[1];
    v49 = 0;
    do
    {
      v50 = 0;
      for ( i = 0; v48 > i; v48 = v43[1] )
      {
        v52 = i++;
        v53 = *(_DWORD *)(*((_QWORD *)v43 + 1) + 4 * (v52 + v49 * v48));
        v54 = v50;
        v50 += v44;
        *(_DWORD *)(v47 + 4 * (v49 + v54)) = v53;
      }
      ++v49;
    }
    while ( *v43 > v49 );
  }
  v55 = sub_22077B0(0x28u);
  v7 = (int *)v55;
  if ( v55 )
  {
    *(_DWORD *)v55 = v45;
    *(_DWORD *)(v55 + 4) = v44;
    *(_QWORD *)(v55 + 8) = v47;
    sub_35B9650((unsigned int *)(v55 + 16), (unsigned int *)v55);
  }
  else if ( v47 )
  {
    j_j___libc_free_0_0(v47);
  }
  v8 = *(_QWORD *)(a1[26] + 48LL * v85);
  if ( a2 == v88 )
  {
LABEL_62:
    v56 = *(_DWORD *)v8;
    v102 = *(_DWORD *)(v8 + 4);
    v97 = (unsigned int)(*(_DWORD *)v8 * v102);
    v57 = (void *)sub_2207820(4 * v97);
    v58 = (unsigned __int64)v57;
    if ( v57 && v97 )
      memset(v57, 0, 4 * v97);
    if ( *(_DWORD *)v8 )
    {
      v59 = *(_DWORD *)(v8 + 4);
      v60 = 0;
      do
      {
        v61 = 0;
        for ( j = 0; v59 > j; v59 = *(_DWORD *)(v8 + 4) )
        {
          v63 = j++;
          v64 = *(_DWORD *)(*(_QWORD *)(v8 + 8) + 4 * (v63 + v60 * v59));
          v65 = v61;
          v61 += v56;
          *(_DWORD *)(v58 + 4 * (v60 + v65)) = v64;
        }
        ++v60;
      }
      while ( *(_DWORD *)v8 > v60 );
    }
    v8 = sub_22077B0(0x28u);
    if ( v8 )
    {
      *(_DWORD *)(v8 + 4) = v56;
      *(_QWORD *)(v8 + 8) = v58;
      *(_DWORD *)v8 = v102;
      sub_35B9650((unsigned int *)(v8 + 16), (unsigned int *)v8);
    }
    else if ( v58 )
    {
      j_j___libc_free_0_0(v58);
    }
  }
LABEL_5:
  v9 = *v7;
  v10 = *(_DWORD *)v3;
  v101 = *(_DWORD *)v8;
  v90 = (unsigned int)(*v7 * *(_DWORD *)v8);
  n = 4 * v90;
  sa = (char *)sub_2207820(4 * v90);
  if ( sa && v90 )
    memset(sa, 0, n);
  v11 = 0;
  if ( v9 )
  {
    v95 = v9;
    v12 = 0;
    do
    {
      if ( v101 )
      {
        for ( k = 0; k != v101; *(float *)&sa[4 * v12 + 4 * k++] = v17 )
        {
          v14 = (float *)(*((_QWORD *)v7 + 1) + 4LL * (unsigned int)(v7[1] * v11));
          v15 = (float *)(*(_QWORD *)(v8 + 8) + 4LL * (unsigned int)(*(_DWORD *)(v8 + 4) * k));
          v16 = *(float **)(v3 + 8);
          v17 = (float)(*v14 + *v15) + *v16;
          if ( v10 > 1 )
          {
            for ( m = 1; m != v10; ++m )
            {
              v19 = (float)(v14[m] + v15[m]) + v16[m];
              v17 = fminf(v19, v17);
            }
          }
        }
      }
      ++v11;
      v12 += v101;
    }
    while ( v11 != v95 );
    v9 = v95;
  }
  if ( a2 == v87 )
  {
    v74 = *((_QWORD *)v7 + 4);
    if ( v74 )
      j_j___libc_free_0_0(v74);
    v75 = *((_QWORD *)v7 + 3);
    if ( v75 )
      j_j___libc_free_0_0(v75);
    v76 = *((_QWORD *)v7 + 1);
    if ( v76 )
      j_j___libc_free_0_0(v76);
    j_j___libc_free_0((unsigned __int64)v7);
    if ( a2 != v88 )
      goto LABEL_20;
    goto LABEL_91;
  }
  if ( a2 == v88 )
  {
LABEL_91:
    v77 = *(_QWORD *)(v8 + 32);
    if ( v77 )
      j_j___libc_free_0_0(v77);
    v78 = *(_QWORD *)(v8 + 24);
    if ( v78 )
      j_j___libc_free_0_0(v78);
    v79 = *(_QWORD *)(v8 + 8);
    if ( v79 )
      j_j___libc_free_0_0(v79);
    j_j___libc_free_0(v8);
  }
LABEL_20:
  v20 = a1[20] + 96LL * v91;
  v21 = *(unsigned int **)(v20 + 72);
  v22 = *(unsigned int **)(v20 + 80);
  if ( v21 == v22 )
    goto LABEL_98;
  while ( 1 )
  {
    v23 = *v21;
    v24 = a1[26] + 48 * v23;
    v25 = *(_DWORD *)(v24 + 20);
    if ( v25 == v92 || *(_DWORD *)(v24 + 24) == v92 )
      break;
    if ( v22 == ++v21 )
      goto LABEL_98;
  }
  if ( (_DWORD)v23 == -1 )
  {
LABEL_98:
    v103 = __PAIR64__(v101, v9);
    v80 = (void *)sub_2207820(n);
    if ( v80 && v90 )
      v80 = memset(v80, 0, n);
    v104 = (float *)v80;
    v81 = 4LL * (unsigned int)(HIDWORD(v103) * v103);
    if ( v81 )
      memcpy(v80, sa, v81);
    sub_35BFD50(a1, v91, v92, (__int64 *)&v103);
    v66 = (unsigned __int64)v104;
    if ( v104 )
      goto LABEL_73;
    goto LABEL_45;
  }
  v26 = *(_QWORD *)v24;
  if ( v25 != v91 )
  {
    v27 = (_DWORD *)sub_2207820(n);
    v28 = v27;
    if ( v27 && v90 )
      memset(v27, 0, n);
    v29 = 0;
    v30 = 0;
    if ( v9 )
    {
      do
      {
        if ( v101 )
        {
          v31 = 0;
          v32 = &sa[4 * v29];
          do
          {
            v33 = *(_DWORD *)v32;
            v34 = v31;
            v32 += 4;
            v31 += v9;
            v28[v30 + v34] = v33;
          }
          while ( &sa[4 * v101 + 4 * (unsigned __int64)v29] != v32 );
        }
        ++v30;
        v29 += v101;
      }
      while ( v9 != v30 );
    }
    v103 = __PAIR64__(v9, v101);
    v35 = (float *)sub_2207820(n);
    v36 = v35;
    if ( v35 )
    {
      if ( v90 )
        memset(v35, 0, n);
      v104 = v36;
      v37 = 4LL * (unsigned int)(HIDWORD(v103) * v103);
      v38 = v37;
      if ( v37 )
      {
        memcpy(v36, v28, v37);
        v39 = *(float **)(v26 + 8);
LABEL_40:
        v40 = (float *)((char *)v39 + v38);
        do
        {
          v41 = *v36 + *v39++;
          *v36++ = v41;
        }
        while ( v39 != v40 );
        sub_35BFB40((__int64)a1, v23, (__int64 *)&v103);
        v42 = (unsigned __int64)v104;
        if ( !v104 )
        {
LABEL_72:
          v66 = (unsigned __int64)v28;
LABEL_73:
          j_j___libc_free_0_0(v66);
          goto LABEL_45;
        }
        goto LABEL_43;
      }
    }
    else
    {
      v104 = 0;
      v82 = 4LL * (unsigned int)(v103 * HIDWORD(v103));
      v38 = v82;
      if ( v82 )
      {
        memcpy(0, v28, v82);
        v39 = *(float **)(v26 + 8);
        goto LABEL_40;
      }
    }
    sub_35BFB40((__int64)a1, v23, (__int64 *)&v103);
    v42 = (unsigned __int64)v104;
    if ( !v104 )
    {
LABEL_44:
      if ( !v28 )
        goto LABEL_45;
      goto LABEL_72;
    }
LABEL_43:
    j_j___libc_free_0_0(v42);
    goto LABEL_44;
  }
  v103 = __PAIR64__(v101, v9);
  v67 = (float *)sub_2207820(n);
  v68 = v67;
  if ( !v67 )
  {
    v104 = 0;
    v83 = 4LL * (unsigned int)(HIDWORD(v103) * v103);
    v84 = v83;
    if ( !v83 )
      goto LABEL_80;
    memcpy(0, sa, v83);
    ii = *(float **)(v26 + 8);
    v71 = (float *)v84;
    goto LABEL_79;
  }
  if ( v90 )
    memset(v67, 0, n);
  v104 = v68;
  v69 = 4LL * (unsigned int)(HIDWORD(v103) * v103);
  v70 = v69;
  if ( v69 )
  {
    memcpy(v68, sa, v69);
    v71 = (float *)((char *)v68 + v70);
    for ( ii = *(float **)(v26 + 8); v68 != v71; *(v68 - 1) = v73 )
LABEL_79:
      v73 = *v68++ + *ii++;
  }
LABEL_80:
  sub_35BFB40((__int64)a1, v23, (__int64 *)&v103);
  v66 = (unsigned __int64)v104;
  if ( v104 )
    goto LABEL_73;
LABEL_45:
  sub_35BAE80(a1, v86, v91);
  sub_35BAE80(a1, v85, v92);
  if ( sa )
    j_j___libc_free_0_0((unsigned __int64)sa);
}
