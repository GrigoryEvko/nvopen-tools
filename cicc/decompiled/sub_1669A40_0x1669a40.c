// Function: sub_1669A40
// Address: 0x1669a40
//
void __fastcall sub_1669A40(_BYTE *a1)
{
  unsigned __int64 *v1; // rax
  unsigned __int64 *v2; // rsi
  unsigned __int64 *v3; // r12
  unsigned __int64 *v5; // rdx
  unsigned __int64 *v6; // r15
  unsigned __int64 v7; // r13
  unsigned __int64 *v8; // rax
  __int64 v9; // r9
  __int64 v10; // rax
  unsigned __int64 *v11; // rdi
  unsigned __int64 v12; // r15
  unsigned __int64 *v13; // rax
  unsigned __int64 *v14; // rcx
  _BYTE *v15; // r8
  _BYTE *v16; // rdx
  __int64 v17; // rax
  __int64 *v18; // rax
  _BYTE *v19; // r8
  __int64 v20; // r9
  __int64 v21; // rax
  __int64 v22; // rax
  unsigned __int64 *v23; // rax
  char v24; // dl
  __int64 v25; // rdx
  __int64 v26; // rsi
  unsigned int v27; // eax
  __int64 *v28; // rcx
  __int64 v29; // r9
  unsigned __int64 *v30; // rax
  __int64 v31; // rdx
  unsigned __int64 *v32; // rsi
  unsigned __int64 *v33; // rcx
  unsigned int v34; // eax
  unsigned __int64 v35; // rdi
  unsigned __int64 *v36; // rsi
  unsigned __int64 *v37; // rcx
  int v38; // ecx
  unsigned __int64 *v39; // rsi
  unsigned __int64 *v40; // rcx
  __int64 v41; // r13
  __int64 *v42; // r12
  __int64 v43; // r14
  _BYTE *v44; // rax
  __int64 v45; // rax
  __int64 *i; // r13
  __int64 v47; // rdi
  _BYTE *v48; // rax
  __int64 v49; // rdi
  __int64 v50; // rsi
  unsigned __int64 v51; // rdx
  int v52; // r10d
  unsigned __int64 *v53; // rdx
  unsigned __int64 *v54; // [rsp+0h] [rbp-1A0h]
  __int64 v55; // [rsp+0h] [rbp-1A0h]
  unsigned __int64 *v56; // [rsp+8h] [rbp-198h]
  _BYTE *v57; // [rsp+8h] [rbp-198h]
  _BYTE *v58; // [rsp+8h] [rbp-198h]
  __int64 v59; // [rsp+8h] [rbp-198h]
  _BYTE *v60; // [rsp+8h] [rbp-198h]
  unsigned __int64 v61; // [rsp+18h] [rbp-188h] BYREF
  const char *v62; // [rsp+20h] [rbp-180h] BYREF
  char v63; // [rsp+30h] [rbp-170h]
  char v64; // [rsp+31h] [rbp-16Fh]
  _BYTE *v65; // [rsp+40h] [rbp-160h] BYREF
  __int64 v66; // [rsp+48h] [rbp-158h]
  _BYTE v67[64]; // [rsp+50h] [rbp-150h] BYREF
  __int64 v68; // [rsp+90h] [rbp-110h] BYREF
  unsigned __int64 *v69; // [rsp+98h] [rbp-108h]
  unsigned __int64 *v70; // [rsp+A0h] [rbp-100h]
  __int64 v71; // [rsp+A8h] [rbp-F8h]
  int v72; // [rsp+B0h] [rbp-F0h]
  _BYTE v73[72]; // [rsp+B8h] [rbp-E8h] BYREF
  __int64 s; // [rsp+100h] [rbp-A0h] BYREF
  unsigned __int64 *v75; // [rsp+108h] [rbp-98h]
  unsigned __int64 *v76; // [rsp+110h] [rbp-90h]
  _BYTE v77[12]; // [rsp+118h] [rbp-88h]
  _BYTE v78[120]; // [rsp+128h] [rbp-78h] BYREF

  v1 = (unsigned __int64 *)v73;
  v2 = (unsigned __int64 *)*((_QWORD *)a1 + 100);
  v3 = (unsigned __int64 *)*((_QWORD *)a1 + 99);
  v68 = 0;
  v69 = (unsigned __int64 *)v73;
  v70 = (unsigned __int64 *)v73;
  v71 = 8;
  v72 = 0;
  s = 0;
  v75 = (unsigned __int64 *)v78;
  v76 = (unsigned __int64 *)v78;
  *(_QWORD *)v77 = 8;
  *(_DWORD *)&v77[8] = 0;
  v54 = v2;
  if ( v3 == v2 )
    return;
  v5 = (unsigned __int64 *)v73;
  while ( 1 )
  {
    v7 = *v3;
    if ( v5 != v1 )
      break;
    v6 = &v1[HIDWORD(v71)];
    if ( v6 == v1 )
    {
      v53 = v1;
    }
    else
    {
      do
      {
        if ( v7 == *v1 )
          break;
        ++v1;
      }
      while ( v6 != v1 );
      v53 = v6;
    }
LABEL_17:
    while ( v53 != v1 )
    {
      if ( *v1 < 0xFFFFFFFFFFFFFFFELL )
        goto LABEL_6;
      ++v1;
    }
    if ( v1 == v6 )
      goto LABEL_19;
LABEL_7:
    v3 += 2;
    if ( v54 == v3 )
    {
LABEL_67:
      if ( v76 != v75 )
        _libc_free((unsigned __int64)v76);
      v35 = (unsigned __int64)v70;
      if ( v70 != v69 )
        goto LABEL_119;
      return;
    }
LABEL_8:
    v5 = v70;
    v1 = v69;
  }
  v6 = &v5[(unsigned int)v71];
  v1 = (unsigned __int64 *)sub_16CC9F0(&v68, *v3);
  if ( v7 == *v1 )
  {
    if ( v70 == v69 )
      v53 = &v70[HIDWORD(v71)];
    else
      v53 = &v70[(unsigned int)v71];
    goto LABEL_17;
  }
  if ( v70 == v69 )
  {
    v1 = &v70[HIDWORD(v71)];
    v53 = v1;
    goto LABEL_17;
  }
  v1 = &v70[(unsigned int)v71];
LABEL_6:
  if ( v1 != v6 )
    goto LABEL_7;
LABEL_19:
  v8 = v75;
  if ( v76 != v75 )
    goto LABEL_20;
  v39 = &v75[*(unsigned int *)&v77[4]];
  if ( v75 == v39 )
  {
LABEL_123:
    if ( *(_DWORD *)&v77[4] >= *(_DWORD *)v77 )
    {
LABEL_20:
      sub_16CCBA0(&s, v7);
    }
    else
    {
      ++*(_DWORD *)&v77[4];
      *v39 = v7;
      ++s;
    }
  }
  else
  {
    v40 = 0;
    while ( v7 != *v8 )
    {
      if ( *v8 == -2 )
        v40 = v8;
      if ( v39 == ++v8 )
      {
        if ( !v40 )
          goto LABEL_123;
        *v40 = v7;
        --*(_DWORD *)&v77[8];
        ++s;
        break;
      }
    }
  }
  v9 = v3[1];
  while ( 1 )
  {
LABEL_22:
    v10 = sub_164EE90(v9);
    v11 = v76;
    v12 = v10;
    v13 = v75;
    if ( v76 == v75 )
    {
      v14 = &v76[*(unsigned int *)&v77[4]];
      if ( v76 == v14 )
      {
        v51 = (unsigned __int64)v76;
      }
      else
      {
        do
        {
          if ( v12 == *v13 )
            break;
          ++v13;
        }
        while ( v14 != v13 );
        v51 = (unsigned __int64)&v76[*(unsigned int *)&v77[4]];
      }
    }
    else
    {
      v56 = &v76[*(unsigned int *)v77];
      v13 = (unsigned __int64 *)sub_16CC9F0(&s, v12);
      v14 = v56;
      if ( v12 == *v13 )
      {
        v11 = v76;
        v51 = (unsigned __int64)(v76 == v75 ? &v76[*(unsigned int *)&v77[4]] : &v76[*(unsigned int *)v77]);
      }
      else
      {
        v11 = v76;
        if ( v76 != v75 )
        {
          v13 = &v76[*(unsigned int *)v77];
          goto LABEL_26;
        }
        v13 = &v76[*(unsigned int *)&v77[4]];
        v51 = (unsigned __int64)v13;
      }
    }
    while ( (unsigned __int64 *)v51 != v13 && *v13 >= 0xFFFFFFFFFFFFFFFELL )
      ++v13;
LABEL_26:
    if ( v14 != v13 )
      break;
    v23 = v69;
    if ( v70 != v69 )
      goto LABEL_39;
    v32 = &v69[HIDWORD(v71)];
    if ( v69 != v32 )
    {
      v33 = 0;
      while ( v12 != *v23 )
      {
        if ( *v23 == -2 )
          v33 = v23;
        if ( v32 == ++v23 )
        {
          if ( !v33 )
            goto LABEL_87;
          *v33 = v12;
          --v72;
          ++v68;
          goto LABEL_40;
        }
      }
      goto LABEL_61;
    }
LABEL_87:
    if ( HIDWORD(v71) < (unsigned int)v71 )
    {
      ++HIDWORD(v71);
      *v32 = v12;
      ++v68;
    }
    else
    {
LABEL_39:
      sub_16CCBA0(&v68, v12);
      if ( !v24 )
        goto LABEL_83;
    }
LABEL_40:
    v25 = *((unsigned int *)a1 + 196);
    if ( !(_DWORD)v25 )
      goto LABEL_83;
    v26 = *((_QWORD *)a1 + 96);
    v27 = (v25 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
    v28 = (__int64 *)(v26 + 16LL * v27);
    v29 = *v28;
    if ( v12 != *v28 )
    {
      v38 = 1;
      while ( v29 != -8 )
      {
        v52 = v38 + 1;
        v27 = (v25 - 1) & (v38 + v27);
        v28 = (__int64 *)(v26 + 16LL * v27);
        v29 = *v28;
        if ( v12 == *v28 )
          goto LABEL_42;
        v38 = v52;
      }
LABEL_83:
      v11 = v76;
LABEL_61:
      ++s;
      if ( v11 != v75 )
      {
        v34 = 4 * (*(_DWORD *)&v77[4] - *(_DWORD *)&v77[8]);
        if ( v34 < 0x20 )
          v34 = 32;
        if ( *(_DWORD *)v77 > v34 )
        {
          sub_16CC920(&s);
          goto LABEL_7;
        }
        memset(v11, -1, 8LL * *(unsigned int *)v77);
      }
      *(_QWORD *)&v77[4] = 0;
      v3 += 2;
      if ( v54 != v3 )
        goto LABEL_8;
      goto LABEL_67;
    }
LABEL_42:
    v11 = v76;
    v30 = v75;
    if ( v28 == (__int64 *)(v26 + 16 * v25) )
      goto LABEL_61;
    v31 = *((_QWORD *)a1 + 99) + 16LL * *((unsigned int *)v28 + 2);
    if ( *((_QWORD *)a1 + 100) == v31 )
      goto LABEL_61;
    v9 = *(_QWORD *)(v31 + 8);
    if ( v76 != v75 )
    {
LABEL_45:
      v59 = *(_QWORD *)(v31 + 8);
      sub_16CCBA0(&s, v12);
      v9 = v59;
      continue;
    }
    v36 = &v75[*(unsigned int *)&v77[4]];
    if ( v36 == v75 )
    {
LABEL_89:
      if ( *(_DWORD *)&v77[4] >= *(_DWORD *)v77 )
        goto LABEL_45;
      ++*(_DWORD *)&v77[4];
      *v36 = v12;
      ++s;
    }
    else
    {
      v37 = 0;
      while ( v12 != *v30 )
      {
        if ( *v30 == -2 )
          v37 = v30;
        if ( v36 == ++v30 )
        {
          if ( !v37 )
            goto LABEL_89;
          *v37 = v12;
          --*(_DWORD *)&v77[8];
          ++s;
          goto LABEL_22;
        }
      }
    }
  }
  v15 = v67;
  v61 = v12;
  v65 = v67;
  v16 = v67;
  v66 = 0x800000000LL;
  v17 = 0;
  while ( 1 )
  {
    v57 = v15;
    *(_QWORD *)&v16[8 * v17] = v61;
    LODWORD(v66) = v66 + 1;
    v18 = (__int64 *)sub_1668F90((__int64)(a1 + 760), &v61);
    v19 = v57;
    v20 = *v18;
    if ( v61 != *v18 )
    {
      v21 = (unsigned int)v66;
      if ( (unsigned int)v66 >= HIDWORD(v66) )
      {
        v55 = v20;
        sub_16CD150(&v65, v57, 0, 8);
        v21 = (unsigned int)v66;
        v20 = v55;
        v19 = v57;
      }
      *(_QWORD *)&v65[8 * v21] = v20;
      LODWORD(v66) = v66 + 1;
    }
    v58 = v19;
    v22 = sub_164EE90(v20);
    v15 = v58;
    v61 = v22;
    if ( v22 == v12 )
      break;
    v17 = (unsigned int)v66;
    if ( (unsigned int)v66 >= HIDWORD(v66) )
    {
      sub_16CD150(&v65, v58, 0, 8);
      v17 = (unsigned int)v66;
      v15 = v58;
    }
    v16 = v65;
  }
  v41 = *(_QWORD *)a1;
  v64 = 1;
  v62 = "EH pads can't handle each other's exceptions";
  v42 = (__int64 *)v65;
  v63 = 3;
  v43 = (unsigned int)v66;
  if ( v41 )
  {
    sub_16E2CE0(&v62, v41);
    v44 = *(_BYTE **)(v41 + 24);
    v15 = v58;
    if ( (unsigned __int64)v44 >= *(_QWORD *)(v41 + 16) )
    {
      sub_16E7DE0(v41, 10);
      v45 = *(_QWORD *)a1;
      v15 = v58;
    }
    else
    {
      *(_QWORD *)(v41 + 24) = v44 + 1;
      *v44 = 10;
      v45 = *(_QWORD *)a1;
    }
    a1[72] = 1;
    if ( v45 )
    {
      for ( i = &v42[v43]; i != v42; ++v42 )
      {
        v49 = *v42;
        if ( *v42 )
        {
          v50 = *(_QWORD *)a1;
          v60 = v15;
          if ( *(_BYTE *)(v49 + 16) > 0x17u )
            sub_155BD40(v49, v50, (__int64)(a1 + 16), 0);
          else
            sub_1553920((__int64 *)v49, v50, 1, (__int64)(a1 + 16));
          v47 = *(_QWORD *)a1;
          v15 = v60;
          v48 = *(_BYTE **)(*(_QWORD *)a1 + 24LL);
          if ( (unsigned __int64)v48 >= *(_QWORD *)(*(_QWORD *)a1 + 16LL) )
          {
            sub_16E7DE0(v47, 10);
            v15 = v60;
          }
          else
          {
            *(_QWORD *)(v47 + 24) = v48 + 1;
            *v48 = 10;
          }
        }
      }
    }
  }
  else
  {
    a1[72] = 1;
  }
  if ( v65 != v15 )
    _libc_free((unsigned __int64)v65);
  if ( v76 != v75 )
    _libc_free((unsigned __int64)v76);
  v35 = (unsigned __int64)v70;
  if ( v70 == v69 )
    return;
LABEL_119:
  _libc_free(v35);
}
