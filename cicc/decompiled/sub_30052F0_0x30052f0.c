// Function: sub_30052F0
// Address: 0x30052f0
//
void __fastcall sub_30052F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  __int64 v8; // rbx
  _QWORD *v9; // rax
  unsigned __int64 v10; // r13
  _BYTE *v11; // rdx
  _QWORD *i; // rdx
  unsigned __int64 *v13; // r14
  unsigned __int64 v14; // rcx
  unsigned __int64 *v15; // r13
  _BYTE *v16; // rcx
  unsigned __int64 v17; // rbx
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // rsi
  __int64 v20; // rcx
  __int64 v21; // r15
  __int64 v22; // rdx
  unsigned __int64 v23; // r10
  __int64 v24; // rbx
  _QWORD *v25; // rdx
  _QWORD *v26; // rbx
  unsigned __int64 v27; // r13
  int v28; // ebx
  unsigned __int64 v29; // rdx
  _BYTE *v30; // r15
  __int64 v31; // rdx
  _BYTE *v32; // rsi
  _BYTE *v33; // r13
  unsigned __int64 *v34; // rdx
  unsigned __int64 v35; // r12
  unsigned __int64 v36; // rdi
  unsigned __int64 *v37; // rdx
  unsigned __int64 *j; // r13
  unsigned __int64 v39; // r15
  unsigned __int64 v40; // rdi
  _BYTE *v41; // rbx
  unsigned __int64 v42; // r12
  unsigned __int64 v43; // rdi
  _BYTE *v44; // rdx
  _BYTE *v45; // rbx
  unsigned __int64 v46; // r10
  unsigned __int64 v47; // rdi
  unsigned __int64 *v48; // rbx
  unsigned __int64 v49; // r13
  unsigned __int64 v50; // rdi
  unsigned __int64 *v51; // r15
  unsigned __int64 *v52; // r13
  unsigned __int64 v53; // rdx
  unsigned __int64 v54; // r8
  unsigned __int64 v55; // rdi
  unsigned __int64 *v56; // r15
  unsigned __int64 *v57; // r13
  unsigned __int64 v58; // rdx
  unsigned __int64 v59; // r8
  unsigned __int64 v60; // rdi
  unsigned __int64 *v61; // r15
  unsigned __int64 v62; // rdi
  unsigned __int64 v63; // [rsp+8h] [rbp-98h]
  _BYTE *v64; // [rsp+20h] [rbp-80h]
  _BYTE *v65; // [rsp+20h] [rbp-80h]
  unsigned __int64 v66; // [rsp+20h] [rbp-80h]
  unsigned int v67; // [rsp+28h] [rbp-78h]
  int v68; // [rsp+28h] [rbp-78h]
  unsigned int v69; // [rsp+28h] [rbp-78h]
  unsigned __int64 v70; // [rsp+28h] [rbp-78h]
  unsigned __int64 v71; // [rsp+28h] [rbp-78h]
  unsigned __int64 v72; // [rsp+28h] [rbp-78h]
  unsigned __int64 v73; // [rsp+28h] [rbp-78h]
  _BYTE *v74; // [rsp+30h] [rbp-70h] BYREF
  __int64 v75; // [rsp+38h] [rbp-68h]
  _BYTE v76[96]; // [rsp+40h] [rbp-60h] BYREF

  v7 = *(_QWORD *)(a1 + 104);
  v75 = 0x600000000LL;
  *(_DWORD *)(a1 + 120) = *(_DWORD *)(v7 + 120);
  v8 = (__int64)(*(_QWORD *)(v7 + 104) - *(_QWORD *)(v7 + 96)) >> 3;
  v9 = v76;
  v10 = (unsigned int)(v8 + 1);
  v74 = v76;
  if ( (_DWORD)v8 == -1 )
  {
    v13 = *(unsigned __int64 **)(a1 + 24);
    v14 = *(unsigned int *)(a1 + 32);
    v15 = &v13[v14];
    if ( v15 != v13 )
      goto LABEL_16;
    goto LABEL_32;
  }
  v11 = v76;
  if ( v10 > 6 )
  {
    sub_239B9C0((__int64)&v74, v10, (__int64)v76, a4, a5, a6);
    v11 = v74;
    v9 = &v74[8 * (unsigned int)v75];
  }
  for ( i = &v11[8 * v10]; i != v9; ++v9 )
  {
    if ( v9 )
      *v9 = 0;
  }
  v13 = *(unsigned __int64 **)(a1 + 24);
  v14 = *(unsigned int *)(a1 + 32);
  LODWORD(v75) = v8 + 1;
  v15 = &v13[v14];
  if ( v13 == v15 )
  {
    if ( v74 != v76 )
      goto LABEL_79;
LABEL_32:
    v27 = (unsigned int)v75;
    v28 = v75;
    if ( (unsigned int)v75 <= v14 )
    {
      v37 = v13;
      if ( (_DWORD)v75 )
      {
        v56 = (unsigned __int64 *)v76;
        v57 = &v13[(unsigned int)v75];
        do
        {
          v58 = *v56;
          *v56 = 0;
          v59 = *v13;
          *v13 = v58;
          if ( v59 )
          {
            v60 = *(_QWORD *)(v59 + 24);
            if ( v60 != v59 + 40 )
            {
              v72 = v59;
              _libc_free(v60);
              v59 = v72;
            }
            j_j___libc_free_0(v59);
          }
          ++v13;
          ++v56;
        }
        while ( v13 != v57 );
        v37 = *(unsigned __int64 **)(a1 + 24);
        v14 = *(unsigned int *)(a1 + 32);
      }
      for ( j = &v37[v14]; v13 != j; --j )
      {
        v39 = *(j - 1);
        if ( v39 )
        {
          v40 = *(_QWORD *)(v39 + 24);
          if ( v40 != v39 + 40 )
            _libc_free(v40);
          j_j___libc_free_0(v39);
        }
      }
      *(_DWORD *)(a1 + 32) = v28;
      v41 = v74;
      v33 = &v74[8 * (unsigned int)v75];
      if ( v74 == v33 )
        goto LABEL_48;
      do
      {
        v42 = *((_QWORD *)v33 - 1);
        v33 -= 8;
        if ( v42 )
        {
          v43 = *(_QWORD *)(v42 + 24);
          if ( v43 != v42 + 40 )
            _libc_free(v43);
          j_j___libc_free_0(v42);
        }
      }
      while ( v41 != v33 );
    }
    else
    {
      v29 = *(unsigned int *)(a1 + 36);
      if ( (unsigned int)v75 > v29 )
      {
        v61 = &v13[v14];
        while ( v61 != v13 )
        {
          while ( 1 )
          {
            a5 = *--v61;
            if ( !a5 )
              break;
            v62 = *(_QWORD *)(a5 + 24);
            if ( v62 != a5 + 40 )
            {
              v73 = a5;
              _libc_free(v62);
              a5 = v73;
            }
            j_j___libc_free_0(a5);
            if ( v61 == v13 )
              goto LABEL_104;
          }
        }
LABEL_104:
        *(_DWORD *)(a1 + 32) = 0;
        sub_239B9C0(a1 + 24, v27, v29, v14, a5, a6);
        v13 = *(unsigned __int64 **)(a1 + 24);
        v27 = (unsigned int)v75;
        v14 = 0;
      }
      else if ( v14 )
      {
        v14 *= 8LL;
        v51 = (unsigned __int64 *)v76;
        v52 = (unsigned __int64 *)((char *)v13 + v14);
        do
        {
          v53 = *v51;
          *v51 = 0;
          v54 = *v13;
          *v13 = v53;
          if ( v54 )
          {
            v55 = *(_QWORD *)(v54 + 24);
            if ( v55 != v54 + 40 )
            {
              v66 = v14;
              v70 = v54;
              _libc_free(v55);
              v14 = v66;
              v54 = v70;
            }
            v71 = v14;
            j_j___libc_free_0(v54);
            v14 = v71;
          }
          ++v13;
          ++v51;
        }
        while ( v13 != v52 );
        v27 = (unsigned int)v75;
        v13 = (unsigned __int64 *)(v14 + *(_QWORD *)(a1 + 24));
      }
      v30 = v74;
      v31 = 8 * v27;
      v32 = &v74[8 * v27];
      v33 = &v74[v14];
      if ( &v74[v14] != v32 )
      {
        v34 = (unsigned __int64 *)((char *)v13 + v31 - v14);
        do
        {
          if ( v13 )
          {
            *v13 = *(_QWORD *)v33;
            *(_QWORD *)v33 = 0;
          }
          ++v13;
          v33 += 8;
        }
        while ( v13 != v34 );
        v30 = v74;
        v33 = &v74[8 * (unsigned int)v75];
      }
      *(_DWORD *)(a1 + 32) = v28;
      if ( v30 == v33 )
        goto LABEL_48;
      do
      {
        v35 = *((_QWORD *)v33 - 1);
        v33 -= 8;
        if ( v35 )
        {
          v36 = *(_QWORD *)(v35 + 24);
          if ( v36 != v35 + 40 )
            _libc_free(v36);
          j_j___libc_free_0(v35);
        }
      }
      while ( v33 != v30 );
    }
    v33 = v74;
LABEL_48:
    if ( v33 != v76 )
      _libc_free((unsigned __int64)v33);
    return;
  }
  do
  {
LABEL_16:
    v19 = *v13;
    if ( !*v13 )
      goto LABEL_15;
    if ( *(_QWORD *)v19 )
    {
      v20 = (unsigned int)(*(_DWORD *)(*(_QWORD *)v19 + 24LL) + 1);
      v21 = 8 * v20;
    }
    else
    {
      v21 = 0;
      LODWORD(v20) = 0;
    }
    v22 = (unsigned int)v75;
    if ( (unsigned int)v75 > (unsigned int)v20 || (v23 = (unsigned int)(v20 + 1), a5 = v23, v23 == (unsigned int)v75) )
    {
      v16 = v74;
      goto LABEL_11;
    }
    v24 = 8 * v23;
    if ( v23 < (unsigned int)v75 )
    {
      v16 = v74;
      v44 = &v74[8 * (unsigned int)v75];
      v45 = &v74[v24];
      if ( v44 == v45 )
        goto LABEL_29;
      do
      {
        v46 = *((_QWORD *)v44 - 1);
        v44 -= 8;
        if ( v46 )
        {
          v47 = *(_QWORD *)(v46 + 24);
          if ( v47 != v46 + 40 )
          {
            v63 = v46;
            v64 = v44;
            v68 = a5;
            _libc_free(v47);
            v46 = v63;
            v44 = v64;
            LODWORD(a5) = v68;
          }
          v65 = v44;
          v69 = a5;
          j_j___libc_free_0(v46);
          v44 = v65;
          a5 = v69;
        }
      }
      while ( v45 != v44 );
    }
    else
    {
      if ( v23 > HIDWORD(v75) )
      {
        v67 = v20 + 1;
        sub_239B9C0((__int64)&v74, v23, (unsigned int)v75, HIDWORD(v75), v23, a6);
        v22 = (unsigned int)v75;
        a5 = v67;
      }
      v16 = v74;
      v25 = &v74[8 * v22];
      v26 = &v74[v24];
      if ( v25 == v26 )
        goto LABEL_29;
      do
      {
        if ( v25 )
          *v25 = 0;
        ++v25;
      }
      while ( v26 != v25 );
    }
    v16 = v74;
LABEL_29:
    LODWORD(v75) = a5;
    v19 = *v13;
LABEL_11:
    *v13 = 0;
    v17 = *(_QWORD *)&v16[v21];
    *(_QWORD *)&v16[v21] = v19;
    if ( v17 )
    {
      v18 = *(_QWORD *)(v17 + 24);
      if ( v18 != v17 + 40 )
        _libc_free(v18);
      j_j___libc_free_0(v17);
    }
LABEL_15:
    ++v13;
  }
  while ( v15 != v13 );
  v13 = *(unsigned __int64 **)(a1 + 24);
  v14 = *(unsigned int *)(a1 + 32);
  if ( v74 == v76 )
    goto LABEL_32;
  v48 = &v13[v14];
  if ( v48 != v13 )
  {
    do
    {
      v49 = *--v48;
      if ( v49 )
      {
        v50 = *(_QWORD *)(v49 + 24);
        if ( v50 != v49 + 40 )
          _libc_free(v50);
        j_j___libc_free_0(v49);
      }
    }
    while ( v48 != v13 );
    v13 = *(unsigned __int64 **)(a1 + 24);
  }
LABEL_79:
  if ( v13 != (unsigned __int64 *)(a1 + 40) )
    _libc_free((unsigned __int64)v13);
  *(_QWORD *)(a1 + 24) = v74;
  *(_QWORD *)(a1 + 32) = v75;
}
