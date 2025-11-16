// Function: sub_3256960
// Address: 0x3256960
//
void __fastcall sub_3256960(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, _QWORD *a6)
{
  unsigned __int8 v7; // al
  _BYTE *v8; // r12
  unsigned __int64 v9; // rdx
  __int64 v10; // rax
  _BYTE *v11; // rbx
  __int64 v12; // r15
  unsigned __int8 v13; // cl
  _QWORD *v14; // rsi
  __int64 v15; // rax
  unsigned __int64 v16; // rsi
  __int64 v17; // rdi
  _BYTE *v18; // r8
  __int64 v19; // rdi
  __int64 v20; // rcx
  int v21; // ecx
  int v22; // ecx
  __m128i *v23; // rax
  const __m128i *v24; // rcx
  __m128i *v25; // rbx
  __int64 v26; // rdx
  __int64 v27; // rbx
  unsigned __int8 v28; // al
  __int64 v29; // r14
  unsigned __int8 **v30; // rdx
  unsigned __int8 *v31; // rax
  size_t v32; // rdx
  char *v33; // r8
  unsigned __int8 *v34; // r15
  unsigned __int8 *v35; // r12
  unsigned __int8 v36; // al
  __int64 v37; // rdi
  unsigned int v38; // ecx
  char **v39; // r12
  __int64 v40; // r15
  int v41; // eax
  char **v42; // r10
  int v43; // r11d
  unsigned int i; // r9d
  char *v45; // rcx
  bool v46; // al
  int v47; // eax
  int v48; // eax
  __int64 *v49; // r12
  unsigned __int8 v50; // al
  __int64 v51; // rdx
  int v52; // r15d
  unsigned int v53; // eax
  int v54; // edx
  _BYTE *v55; // rax
  unsigned __int8 v56; // al
  __int64 v57; // r15
  int v58; // eax
  int v59; // r11d
  int v60; // ecx
  unsigned int j; // r10d
  bool v62; // al
  const void *v63; // rsi
  unsigned int v64; // r10d
  int v65; // eax
  __int64 v66; // r14
  __int64 v67; // rsi
  __int64 v68; // rcx
  __int64 v69; // rax
  unsigned __int64 v70; // rdx
  _BYTE *v71; // rdi
  __int64 v72; // rcx
  __int64 v73; // rax
  int v74; // r12d
  int v75; // eax
  __int64 v76; // r15
  int v77; // eax
  int v78; // r11d
  unsigned int v79; // r9d
  char *v80; // rcx
  bool v81; // al
  unsigned int v82; // r9d
  int v83; // eax
  unsigned int v84; // r9d
  size_t v85; // [rsp+8h] [rbp-1D8h]
  _QWORD *v86; // [rsp+8h] [rbp-1D8h]
  size_t v87; // [rsp+8h] [rbp-1D8h]
  char *v88; // [rsp+10h] [rbp-1D0h]
  size_t v89; // [rsp+10h] [rbp-1D0h]
  char *v90; // [rsp+10h] [rbp-1D0h]
  char *v91; // [rsp+18h] [rbp-1C8h]
  char *v92; // [rsp+18h] [rbp-1C8h]
  int v93; // [rsp+20h] [rbp-1C0h]
  char *v94; // [rsp+20h] [rbp-1C0h]
  int v95; // [rsp+20h] [rbp-1C0h]
  __int64 v96; // [rsp+28h] [rbp-1B8h]
  _QWORD *v101; // [rsp+58h] [rbp-188h]
  int v102; // [rsp+68h] [rbp-178h]
  int v103; // [rsp+68h] [rbp-178h]
  char **v104; // [rsp+68h] [rbp-178h]
  int v105; // [rsp+68h] [rbp-178h]
  int v106; // [rsp+68h] [rbp-178h]
  int v107; // [rsp+68h] [rbp-178h]
  char **v108; // [rsp+68h] [rbp-178h]
  void *s1e; // [rsp+70h] [rbp-170h]
  char *s1; // [rsp+70h] [rbp-170h]
  unsigned int s1a; // [rsp+70h] [rbp-170h]
  char *s1f; // [rsp+70h] [rbp-170h]
  unsigned int s1b; // [rsp+70h] [rbp-170h]
  char *s1c; // [rsp+70h] [rbp-170h]
  unsigned int s1d; // [rsp+70h] [rbp-170h]
  int n; // [rsp+78h] [rbp-168h]
  size_t na; // [rsp+78h] [rbp-168h]
  int nb; // [rsp+78h] [rbp-168h]
  size_t nf; // [rsp+78h] [rbp-168h]
  int nc; // [rsp+78h] [rbp-168h]
  size_t nd; // [rsp+78h] [rbp-168h]
  int ne; // [rsp+78h] [rbp-168h]
  unsigned __int64 v123; // [rsp+88h] [rbp-158h] BYREF
  _BYTE *v124; // [rsp+90h] [rbp-150h] BYREF
  __int64 v125; // [rsp+98h] [rbp-148h]
  _BYTE v126[128]; // [rsp+A0h] [rbp-140h] BYREF
  _BYTE *v127; // [rsp+120h] [rbp-C0h] BYREF
  __int64 v128; // [rsp+128h] [rbp-B8h]
  _BYTE v129[176]; // [rsp+130h] [rbp-B0h] BYREF

  v124 = v126;
  v101 = a6;
  v125 = 0x800000000LL;
  if ( !a6 )
  {
    v12 = 0;
    LODWORD(v9) = 0;
    v22 = 0;
    v127 = v129;
    HIDWORD(v128) = 8;
    goto LABEL_27;
  }
  v7 = *((_BYTE *)a6 - 16);
  if ( (v7 & 2) != 0 )
  {
    if ( *((_DWORD *)a6 - 6) != 2 )
    {
LABEL_4:
      v8 = v126;
      v9 = 0;
      v10 = 0;
      v11 = v126;
      goto LABEL_5;
    }
    v26 = *(a6 - 4);
  }
  else
  {
    if ( ((*((_WORD *)a6 - 8) >> 6) & 0xF) != 2 )
      goto LABEL_4;
    v26 = (__int64)&a6[-((v7 >> 2) & 0xF) - 2];
  }
  v27 = *(_QWORD *)(v26 + 8);
  if ( !v27 )
    goto LABEL_4;
  v96 = a1 + 8;
  while ( 2 )
  {
    v28 = *(_BYTE *)(v27 - 16);
    v29 = v27 - 16;
    if ( (v28 & 2) != 0 )
      v30 = *(unsigned __int8 ***)(v27 - 32);
    else
      v30 = (unsigned __int8 **)(v29 - 8LL * ((v28 >> 2) & 0xF));
    v31 = sub_AF34D0(*v30);
    v32 = 0;
    v33 = (char *)byte_3F871B3;
    v34 = v31;
    if ( !v31 )
      goto LABEL_42;
    v35 = v31 - 16;
    v36 = *(v31 - 16);
    if ( (v36 & 2) != 0 )
    {
      v37 = *(_QWORD *)(*((_QWORD *)v34 - 4) + 24LL);
      if ( !v37 )
        goto LABEL_107;
    }
    else
    {
      v37 = *(_QWORD *)&v35[-8 * ((v36 >> 2) & 0xF) + 24];
      if ( !v37 )
        goto LABEL_93;
    }
    v33 = (char *)sub_B91420(v37);
    if ( !v32 )
    {
      v36 = *(v34 - 16);
      if ( (v36 & 2) != 0 )
      {
LABEL_107:
        v33 = *(char **)(*((_QWORD *)v34 - 4) + 16LL);
        if ( !v33 )
          goto LABEL_108;
        goto LABEL_94;
      }
LABEL_93:
      v33 = *(char **)&v35[-8 * ((v36 >> 2) & 0xF) + 16];
      if ( !v33 )
      {
LABEL_108:
        v32 = 0;
        goto LABEL_42;
      }
LABEL_94:
      v33 = (char *)sub_B91420((__int64)v33);
    }
LABEL_42:
    v38 = *(_DWORD *)(a1 + 32);
    if ( !v38 )
    {
      ++*(_QWORD *)(a1 + 8);
LABEL_44:
      v39 = 0;
      s1 = v33;
      na = v32;
      sub_A2B260(v96, 2 * v38);
      v32 = na;
      v33 = s1;
      v103 = *(_DWORD *)(a1 + 32);
      if ( !v103 )
        goto LABEL_52;
      v40 = *(_QWORD *)(a1 + 16);
      v41 = sub_C94890(s1, na);
      v42 = 0;
      v32 = na;
      v33 = s1;
      v43 = 1;
      nb = v103 - 1;
      for ( i = (v103 - 1) & v41; ; i = nb & v84 )
      {
        v39 = (char **)(v40 + 24LL * i);
        v45 = *v39;
        if ( *v39 == (char *)-1LL )
          goto LABEL_130;
        v46 = v33 + 2 == 0;
        if ( v45 != (char *)-2LL )
        {
          if ( v39[1] != (char *)v32 )
            goto LABEL_135;
          v93 = v43;
          v104 = v42;
          s1a = i;
          if ( !v32 )
            goto LABEL_52;
          v85 = v32;
          v88 = *v39;
          v91 = v33;
          v47 = memcmp(v33, v45, v32);
          v33 = v91;
          v45 = v88;
          v32 = v85;
          i = s1a;
          v42 = v104;
          v43 = v93;
          v46 = v47 == 0;
        }
        if ( v46 )
          goto LABEL_52;
        if ( !v42 && v45 == (char *)-2LL )
          v42 = v39;
LABEL_135:
        v84 = v43 + i;
        ++v43;
      }
    }
    v105 = *(_DWORD *)(a1 + 32);
    v57 = *(_QWORD *)(a1 + 16);
    s1f = v33;
    v39 = 0;
    nf = v32;
    v58 = sub_C94890(v33, v32);
    v32 = nf;
    v59 = 1;
    v33 = s1f;
    v60 = v105 - 1;
    for ( j = (v105 - 1) & v58; ; j = v60 & v64 )
    {
      a6 = (_QWORD *)(v57 + 24LL * j);
      v62 = v33 + 1 == 0;
      v63 = (const void *)*a6;
      if ( *a6 != -1 )
      {
        v62 = v33 + 2 == 0;
        if ( v63 != (const void *)-2LL )
        {
          if ( v32 != a6[1] )
            goto LABEL_78;
          v106 = v59;
          s1b = j;
          nc = v60;
          if ( !v32 )
            goto LABEL_85;
          v86 = (_QWORD *)(v57 + 24LL * j);
          v89 = v32;
          v94 = v33;
          v65 = memcmp(v33, v63, v32);
          v33 = v94;
          v32 = v89;
          a6 = v86;
          v60 = nc;
          j = s1b;
          v62 = v65 == 0;
          v59 = v106;
        }
      }
      if ( v62 )
      {
LABEL_85:
        v49 = a6 + 2;
        if ( !a6[2] )
          goto LABEL_56;
        goto LABEL_57;
      }
      if ( v63 == (const void *)-1LL )
        break;
LABEL_78:
      if ( v63 == (const void *)-2LL && !v39 )
        v39 = (char **)a6;
      v64 = v59 + j;
      ++v59;
    }
    v75 = *(_DWORD *)(a1 + 24);
    v38 = *(_DWORD *)(a1 + 32);
    if ( !v39 )
      v39 = (char **)a6;
    ++*(_QWORD *)(a1 + 8);
    v48 = v75 + 1;
    if ( 4 * v48 >= 3 * v38 )
      goto LABEL_44;
    if ( v38 - (v48 + *(_DWORD *)(a1 + 28)) > v38 >> 3 )
      goto LABEL_53;
    v39 = 0;
    s1c = v33;
    nd = v32;
    sub_A2B260(v96, v38);
    v32 = nd;
    v33 = s1c;
    v107 = *(_DWORD *)(a1 + 32);
    if ( !v107 )
      goto LABEL_52;
    v76 = *(_QWORD *)(a1 + 16);
    v77 = sub_C94890(s1c, nd);
    v42 = 0;
    v32 = nd;
    v33 = s1c;
    v78 = 1;
    ne = v107 - 1;
    v79 = (v107 - 1) & v77;
    while ( 2 )
    {
      v39 = (char **)(v76 + 24LL * v79);
      v80 = *v39;
      if ( *v39 != (char *)-1LL )
      {
        v81 = v33 + 2 == 0;
        if ( v80 != (char *)-2LL )
        {
          if ( (char *)v32 != v39[1] )
          {
LABEL_119:
            if ( v42 || v80 != (char *)-2LL )
              v39 = v42;
            v82 = v78 + v79;
            v42 = v39;
            ++v78;
            v79 = ne & v82;
            continue;
          }
          v95 = v78;
          v108 = v42;
          s1d = v79;
          if ( !v32 )
            goto LABEL_52;
          v87 = v32;
          v90 = *v39;
          v92 = v33;
          v83 = memcmp(v33, v80, v32);
          v33 = v92;
          v80 = v90;
          v32 = v87;
          v79 = s1d;
          v42 = v108;
          v78 = v95;
          v81 = v83 == 0;
        }
        if ( v81 )
          goto LABEL_52;
        if ( v80 == (char *)-1LL )
          goto LABEL_126;
        goto LABEL_119;
      }
      break;
    }
LABEL_130:
    if ( v33 == (char *)-1LL )
      goto LABEL_52;
LABEL_126:
    if ( v42 )
      v39 = v42;
LABEL_52:
    v48 = *(_DWORD *)(a1 + 24) + 1;
LABEL_53:
    *(_DWORD *)(a1 + 24) = v48;
    if ( *v39 != (char *)-1LL )
      --*(_DWORD *)(a1 + 28);
    *v39 = v33;
    v49 = (__int64 *)(v39 + 2);
    *(v49 - 1) = v32;
    *v49 = 0;
LABEL_56:
    *v49 = sub_B2F650((__int64)v33, v32);
LABEL_57:
    v50 = *(_BYTE *)(v27 - 16);
    if ( (v50 & 2) != 0 )
      v51 = *(_QWORD *)(v27 - 32);
    else
      v51 = v29 - 8LL * ((v50 >> 2) & 0xF);
    v52 = 0;
    if ( **(_BYTE **)v51 == 20 )
    {
      v53 = *(_DWORD *)(*(_QWORD *)v51 + 4LL);
      v52 = (unsigned __int16)v53 >> 3;
      if ( (v53 & 0x10000000) == 0 )
        v52 = (unsigned __int16)(v53 >> 3);
    }
    v54 = v125;
    if ( HIDWORD(v125) <= (unsigned int)v125 )
    {
      v67 = sub_C8D7D0((__int64)&v124, (__int64)v126, 0, 0x10u, (unsigned __int64 *)&v127, (__int64)a6);
      v68 = 16LL * (unsigned int)v125;
      v69 = v68 + v67;
      if ( v68 + v67 )
      {
        *(_DWORD *)v69 = v52;
        *(_QWORD *)(v69 + 8) = *v49;
        v68 = 16LL * (unsigned int)v125;
      }
      v70 = (unsigned __int64)v124;
      v71 = &v124[v68];
      if ( v124 != &v124[v68] )
      {
        v72 = v67 + v68;
        v73 = v67;
        do
        {
          if ( v73 )
          {
            *(_DWORD *)v73 = *(_DWORD *)v70;
            *(_QWORD *)(v73 + 8) = *(_QWORD *)(v70 + 8);
          }
          v73 += 16;
          v70 += 16LL;
        }
        while ( v73 != v72 );
        v71 = v124;
      }
      v74 = (int)v127;
      if ( v71 != v126 )
        _libc_free((unsigned __int64)v71);
      LODWORD(v125) = v125 + 1;
      v124 = (_BYTE *)v67;
      HIDWORD(v125) = v74;
    }
    else
    {
      v55 = &v124[16 * (unsigned int)v125];
      if ( v55 )
      {
        *(_DWORD *)v55 = v52;
        *((_QWORD *)v55 + 1) = *v49;
        v54 = v125;
      }
      LODWORD(v125) = v54 + 1;
    }
    v56 = *(_BYTE *)(v27 - 16);
    if ( (v56 & 2) != 0 )
    {
      if ( *(_DWORD *)(v27 - 24) != 2 )
        goto LABEL_68;
      v66 = *(_QWORD *)(v27 - 32);
LABEL_89:
      v27 = *(_QWORD *)(v66 + 8);
      if ( !v27 )
        goto LABEL_68;
      continue;
    }
    break;
  }
  if ( ((*(_WORD *)(v27 - 16) >> 6) & 0xF) == 2 )
  {
    v66 = v29 - 8LL * ((v56 >> 2) & 0xF);
    goto LABEL_89;
  }
LABEL_68:
  v8 = v124;
  v10 = 16LL * (unsigned int)v125;
  v11 = &v124[v10];
  v9 = v10 >> 4;
LABEL_5:
  v12 = 0;
  if ( LOBYTE(qword_4F813A8[8]) && !a4 )
  {
    v13 = *((_BYTE *)v101 - 16);
    v14 = (v13 & 2) != 0 ? (_QWORD *)*(v101 - 4) : &v101[-((v13 >> 2) & 0xF) - 2];
    v12 = 0;
    if ( *(_BYTE *)*v14 == 20 )
      v12 = *(unsigned int *)(*v14 + 4LL);
  }
  v127 = v129;
  v128 = 0x800000000LL;
  if ( (unsigned __int64)v10 <= 0x80 )
  {
    v23 = (__m128i *)v129;
    v22 = 0;
  }
  else
  {
    n = v9;
    v15 = sub_C8D7D0((__int64)&v127, (__int64)v129, v9, 0x10u, &v123, (__int64)a6);
    v16 = (unsigned __int64)v127;
    LODWORD(v9) = n;
    v17 = 16LL * (unsigned int)v128;
    v18 = &v127[v17];
    if ( v127 != &v127[v17] )
    {
      v19 = v15 + v17;
      v20 = v15;
      do
      {
        if ( v20 )
        {
          *(_DWORD *)v20 = *(_DWORD *)v16;
          *(_QWORD *)(v20 + 8) = *(_QWORD *)(v16 + 8);
        }
        v20 += 16;
        v16 += 16LL;
      }
      while ( v20 != v19 );
      v18 = v127;
    }
    v21 = v123;
    if ( v18 != v129 )
    {
      v102 = v123;
      s1e = (void *)v15;
      _libc_free((unsigned __int64)v18);
      v21 = v102;
      v15 = (__int64)s1e;
      LODWORD(v9) = n;
    }
    HIDWORD(v128) = v21;
    v22 = v128;
    v127 = (_BYTE *)v15;
    v23 = (__m128i *)(16LL * (unsigned int)v128 + v15);
  }
  if ( v8 != v11 )
  {
    v24 = (const __m128i *)(v11 - 16);
    v25 = (__m128i *)((char *)v23 + v11 - v8);
    do
    {
      if ( v23 )
        *v23 = _mm_loadu_si128(v24);
      ++v23;
      --v24;
    }
    while ( v23 != v25 );
    v22 = v128;
  }
LABEL_27:
  LODWORD(v128) = v22 + v9;
  (*(void (__fastcall **)(_QWORD, __int64, __int64, __int64, __int64, __int64, _BYTE **, _QWORD))(**(_QWORD **)(*(_QWORD *)a1 + 224LL)
                                                                                                + 1232LL))(
    *(_QWORD *)(*(_QWORD *)a1 + 224LL),
    a2,
    a3,
    a4,
    a5,
    v12,
    &v127,
    *(_QWORD *)(*(_QWORD *)a1 + 280LL));
  if ( v127 != v129 )
    _libc_free((unsigned __int64)v127);
  if ( v124 != v126 )
    _libc_free((unsigned __int64)v124);
}
