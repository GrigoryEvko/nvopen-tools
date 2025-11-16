// Function: sub_823A00
// Address: 0x823a00
//
void __fastcall sub_823A00(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 *a6)
{
  __int64 v7; // rbx
  __int64 v8; // rsi
  int v9; // edi
  __int64 i; // rdx
  __int64 *v11; // rax
  __int64 v12; // rcx
  __int64 *v13; // r13
  __int64 v14; // rbx
  __int64 v15; // r15
  __int64 v16; // r8
  _QWORD *v17; // rax
  __int64 v18; // r14
  __int64 v19; // rdi
  _QWORD *v20; // rax
  __int64 v21; // r8
  _QWORD *v22; // r9
  _QWORD *v23; // rdx
  _QWORD *v24; // rsi
  __int64 *v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 v30; // rax
  __int64 v31; // r14
  unsigned int v32; // edx
  __int64 v33; // rdi
  __int64 v34; // r8
  int v35; // eax
  __int64 v36; // rcx
  unsigned int v37; // r15d
  unsigned int v38; // ebx
  _QWORD *v39; // rax
  _QWORD *v40; // rsi
  _QWORD *v41; // rcx
  __int64 v42; // r8
  __int64 *v43; // rcx
  __int64 v44; // rdi
  unsigned int k; // edx
  _QWORD *v46; // rax
  __int64 *v47; // rax
  __int64 v48; // rdx
  __int64 *v49; // rdi
  int v50; // eax
  __int64 v51; // rcx
  _QWORD *v52; // rax
  _QWORD *v53; // rcx
  __int64 *v54; // rcx
  __int64 v55; // rdi
  unsigned int j; // edx
  _QWORD *v57; // rax
  __int64 v58; // rdx
  __int64 v59; // rcx
  _QWORD *v60; // r13
  _QWORD *v61; // rax
  unsigned int v62; // [rsp-44h] [rbp-44h]
  unsigned int v63; // [rsp-44h] [rbp-44h]
  _QWORD *v64; // [rsp-40h] [rbp-40h]
  __int64 v65; // [rsp-40h] [rbp-40h]
  int v66; // [rsp-40h] [rbp-40h]
  int v67; // [rsp-40h] [rbp-40h]

  if ( !a1 )
    return;
  v7 = 1;
  if ( a2 )
    v7 = a2;
  if ( qword_4F195D0 )
  {
    v8 = *(_QWORD *)qword_4F195D0;
    v9 = *(_DWORD *)(qword_4F195D0 + 8);
  }
  else
  {
    v60 = (_QWORD *)sub_822B10(16, a2, a3, a4, a5, (__int64)a6);
    if ( v60 )
    {
      v61 = (_QWORD *)sub_822B10(32, a2, v58, v59, a5, (__int64)a6);
      v8 = (__int64)v61;
      if ( v61 )
        *v61 = 0;
      v61[2] = 0;
      v9 = 1;
      *v60 = v61;
      v60[1] = 1;
    }
    else
    {
      v8 = MEMORY[0];
      v9 = MEMORY[8];
    }
    qword_4F195D0 = (__int64)v60;
  }
  for ( i = (unsigned int)v7 & v9; ; i = v9 & (unsigned int)(i + 1) )
  {
    v11 = (__int64 *)(v8 + 16LL * (unsigned int)i);
    v12 = *v11;
    if ( v7 == *v11 )
      break;
    if ( !v12 )
      goto LABEL_24;
  }
  v13 = (__int64 *)v11[1];
  if ( v13 )
    goto LABEL_11;
LABEL_24:
  v25 = (__int64 *)sub_822B10(24, v8, i, v12, a5, (__int64)a6);
  v13 = v25;
  if ( v25 )
  {
    *v25 = 0;
    v25[1] = 0;
    v25[2] = 0;
    v30 = sub_822B10(8, v8, v26, v27, v28, v29);
    v13[1] = 1;
    *v13 = v30;
  }
  v31 = qword_4F195D0;
  v8 = *(unsigned int *)(qword_4F195D0 + 8);
  v12 = *(_QWORD *)qword_4F195D0;
  v32 = v7 & *(_DWORD *)(qword_4F195D0 + 8);
  v33 = 16LL * v32;
  a6 = (__int64 *)(*(_QWORD *)qword_4F195D0 + v33);
  v34 = *a6;
  if ( *a6 )
  {
    do
    {
      v32 = v8 & (v32 + 1);
      v47 = (__int64 *)(v12 + 16LL * v32);
    }
    while ( *v47 );
    v48 = a6[1];
    *v47 = v34;
    v47[1] = v48;
    *a6 = 0;
    v49 = (__int64 *)(*(_QWORD *)v31 + v33);
    *v49 = v7;
    v49[1] = (__int64)v13;
    i = *(unsigned int *)(v31 + 8);
    v50 = *(_DWORD *)(v31 + 12) + 1;
    *(_DWORD *)(v31 + 12) = v50;
    if ( 2 * v50 <= (unsigned int)i )
      goto LABEL_11;
    v51 = (unsigned int)(2 * i);
    v63 = i;
    v37 = i + 1;
    v38 = v51 + 1;
    v67 = v51 + 2;
    v52 = (_QWORD *)sub_822B10(16LL * (unsigned int)(v51 + 2), v8, i, v51, v34, (__int64)a6);
    v40 = v52;
    if ( v67 )
    {
      v53 = &v52[2 * v38 + 2];
      do
      {
        if ( v52 )
          *v52 = 0;
        v52 += 2;
      }
      while ( v53 != v52 );
    }
    v42 = *(_QWORD *)v31;
    if ( v37 )
    {
      v54 = *(__int64 **)v31;
      do
      {
        v55 = *v54;
        if ( *v54 )
        {
          for ( j = v55 & v38; ; j = v38 & (j + 1) )
          {
            v57 = &v40[2 * j];
            if ( !*v57 )
              break;
          }
          *v57 = v55;
          v57[1] = v54[1];
        }
        v54 += 2;
      }
      while ( (__int64 *)(v42 + 16LL * v63 + 16) != v54 );
    }
    goto LABEL_63;
  }
  *a6 = v7;
  a6[1] = (__int64)v13;
  i = *(unsigned int *)(v31 + 8);
  v35 = *(_DWORD *)(v31 + 12) + 1;
  *(_DWORD *)(v31 + 12) = v35;
  if ( 2 * v35 > (unsigned int)i )
  {
    v36 = (unsigned int)(2 * i);
    v62 = i;
    v37 = i + 1;
    v38 = v36 + 1;
    v66 = v36 + 2;
    v39 = (_QWORD *)sub_822B10(16LL * (unsigned int)(v36 + 2), v8, i, v36, 0, (__int64)a6);
    v40 = v39;
    if ( v66 )
    {
      v41 = &v39[2 * v38 + 2];
      do
      {
        if ( v39 )
          *v39 = 0;
        v39 += 2;
      }
      while ( v41 != v39 );
    }
    v42 = *(_QWORD *)v31;
    if ( v37 )
    {
      v43 = *(__int64 **)v31;
      do
      {
        v44 = *v43;
        if ( *v43 )
        {
          for ( k = v44 & v38; ; k = v38 & (k + 1) )
          {
            v46 = &v40[2 * k];
            if ( !*v46 )
              break;
          }
          *v46 = v44;
          v46[1] = v43[1];
        }
        v43 += 2;
      }
      while ( (__int64 *)(v42 + 16LL * v62 + 16) != v43 );
    }
LABEL_63:
    *(_QWORD *)v31 = v40;
    *(_DWORD *)(v31 + 8) = v38;
    v8 = 16LL * v37;
    sub_822B90(v42, v8);
  }
LABEL_11:
  v14 = v13[2];
  v15 = v13[1];
  v16 = *v13;
  if ( v14 == v15 )
  {
    if ( v14 <= 1 )
    {
      v19 = 16;
      v18 = 2;
    }
    else
    {
      v18 = v14 + (v14 >> 1) + 1;
      v19 = 8 * v18;
    }
    v64 = (_QWORD *)*v13;
    v20 = (_QWORD *)sub_822B10(v19, v8, i, v12, v16, (__int64)a6);
    v21 = (__int64)v64;
    v22 = v20;
    if ( v14 > 0 )
    {
      v23 = v64;
      v24 = &v20[v14];
      do
      {
        if ( v20 )
          *v20 = *v23;
        ++v20;
        ++v23;
      }
      while ( v20 != v24 );
    }
    v65 = (__int64)v22;
    sub_822B90(v21, 8 * v15);
    v13[1] = v18;
    *v13 = v65;
    v16 = v65;
  }
  v17 = (_QWORD *)(v16 + 8 * v14);
  if ( v17 )
    *v17 = a1;
  v13[2] = v14 + 1;
}
