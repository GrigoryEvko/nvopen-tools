// Function: sub_FA2310
// Address: 0xfa2310
//
__int64 __fastcall sub_FA2310(__int64 a1, __int64 a2, __int64 a3, unsigned int **a4)
{
  __int64 *v4; // rdx
  __int64 v6; // rax
  unsigned __int64 v7; // rax
  int v8; // edx
  __int64 v9; // rsi
  __int64 v10; // rax
  void **p_base; // r15
  __int64 v12; // r14
  __int64 v13; // rbx
  __int64 v14; // r11
  __m128i *v15; // rax
  __int64 v16; // rdx
  __m128i *v17; // rax
  unsigned __int64 v18; // rax
  __int64 v19; // r14
  __int64 v20; // r12
  unsigned int i; // r15d
  __int64 v22; // rax
  __int64 *v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r9
  __int64 v26; // r8
  _QWORD *v27; // rax
  _QWORD *v28; // rax
  __int64 v29; // r12
  __int64 v30; // rsi
  unsigned int *v31; // r15
  __int64 v32; // r14
  __int64 v33; // rdx
  __int64 v34; // r9
  __int64 *v35; // r8
  __int64 v36; // r10
  __int64 v37; // rcx
  unsigned __int64 v38; // rdx
  __int64 *v39; // rax
  __int64 *v40; // r14
  __int64 v41; // rdx
  __int64 v42; // r9
  __m128i *v43; // rdi
  unsigned int v44; // r12d
  __int64 v46; // r12
  __int64 *v47; // rcx
  __int64 *v48; // rax
  _BYTE *v49; // rdi
  __m128i *v50; // rax
  void **v51; // r14
  __m128i *v52; // r8
  __int64 v53; // rax
  _BYTE *v54; // rdi
  char *v55; // rax
  char *v56; // rdx
  __int64 v57; // rcx
  __int64 v58; // r9
  __int64 v59; // rdx
  __int64 v60; // r15
  __int64 v61; // r8
  __int64 v62; // r12
  _QWORD *v63; // rax
  __int64 *v64; // rdx
  __int64 v65; // rdx
  size_t v66; // rsi
  _QWORD *v67; // r8
  int v68; // r10d
  unsigned int v69; // ecx
  unsigned __int64 v70; // rax
  __int64 *v71; // rax
  __int64 v72; // r15
  __int64 v73; // r12
  __int64 v74; // rdi
  __int64 v75; // rax
  _QWORD *v76; // rax
  _QWORD *v77; // rax
  _QWORD *v78; // rdx
  _QWORD *v79; // rax
  char v80; // cl
  __int64 **v81; // r15
  __int64 *v82; // r12
  __int64 *v83; // rax
  __int64 v84; // rdx
  __int64 v85; // rdx
  __int64 *v86; // rdx
  const __m128i *v87; // rcx
  __int64 v88; // rdx
  __int64 v89; // rax
  __int64 v90; // rdi
  __int64 v91; // rax
  __int64 v92; // r12
  __int64 v93; // rax
  __int64 *v94; // r12
  __int64 v95; // rdx
  __int64 v97; // [rsp+10h] [rbp-240h]
  __int64 v99; // [rsp+18h] [rbp-238h]
  __int64 *v100; // [rsp+18h] [rbp-238h]
  __int64 v101; // [rsp+20h] [rbp-230h]
  __int64 *v102; // [rsp+20h] [rbp-230h]
  __int64 v103; // [rsp+28h] [rbp-228h]
  int v104; // [rsp+28h] [rbp-228h]
  __int64 *v105; // [rsp+28h] [rbp-228h]
  __int64 v106; // [rsp+28h] [rbp-228h]
  const __m128i *v107; // [rsp+28h] [rbp-228h]
  _QWORD v108[2]; // [rsp+30h] [rbp-220h] BYREF
  void *base; // [rsp+40h] [rbp-210h] BYREF
  __m128i *v110; // [rsp+48h] [rbp-208h]
  __int64 v111; // [rsp+50h] [rbp-200h]
  __m128i *v112; // [rsp+60h] [rbp-1F0h] BYREF
  __m128i *v113; // [rsp+68h] [rbp-1E8h]
  __int64 v114; // [rsp+70h] [rbp-1E0h]
  __int64 v115; // [rsp+80h] [rbp-1D0h] BYREF
  __int64 v116; // [rsp+88h] [rbp-1C8h]
  __int64 v117; // [rsp+90h] [rbp-1C0h]
  __int64 v118; // [rsp+A0h] [rbp-1B0h] BYREF
  char *v119; // [rsp+A8h] [rbp-1A8h]
  char v120; // [rsp+B8h] [rbp-198h] BYREF
  char v121; // [rsp+D8h] [rbp-178h]
  char v122; // [rsp+E0h] [rbp-170h]
  __int64 v123; // [rsp+F0h] [rbp-160h] BYREF
  __int64 v124; // [rsp+F8h] [rbp-158h]
  __int64 v125; // [rsp+100h] [rbp-150h] BYREF
  unsigned int v126; // [rsp+108h] [rbp-148h]
  char v127; // [rsp+10Ch] [rbp-144h]
  char v128; // [rsp+110h] [rbp-140h] BYREF
  __int64 *v129; // [rsp+180h] [rbp-D0h] BYREF
  __int64 v130; // [rsp+188h] [rbp-C8h]
  __int64 v131; // [rsp+190h] [rbp-C0h] BYREF
  int v132; // [rsp+198h] [rbp-B8h]
  unsigned __int8 v133; // [rsp+19Ch] [rbp-B4h]
  __int16 v134; // [rsp+1A0h] [rbp-B0h] BYREF

  v4 = (__int64 *)(a3 + 48);
  v6 = *v4;
  base = 0;
  v7 = v6 & 0xFFFFFFFFFFFFFFF8LL;
  v110 = 0;
  v111 = 0;
  if ( (__int64 *)v7 == v4 )
  {
    v9 = 0;
  }
  else
  {
    if ( !v7 )
      BUG();
    v8 = *(unsigned __int8 *)(v7 - 24);
    v9 = 0;
    v10 = v7 - 24;
    if ( (unsigned int)(v8 - 30) < 0xB )
      v9 = v10;
  }
  p_base = &base;
  v12 = sub_F90890(a1, v9, (__int64)&base);
  sub_F8E8E0((__m128i **)&base, v12);
  v112 = 0;
  v113 = 0;
  v114 = 0;
  v103 = sub_F90890(a1, a2, (__int64)&v112);
  sub_F8E8E0(&v112, v103);
  v13 = *(_QWORD *)(a2 + 40);
  v14 = v103;
  if ( v12 != v13 )
  {
    v15 = (__m128i *)base;
    v16 = 0;
    if ( base == v110 )
    {
LABEL_12:
      v17 = v112;
      if ( v112 == v113 )
        goto LABEL_65;
      while ( v17->m128i_i64[0] != v16 )
      {
        if ( v113 == ++v17 )
          goto LABEL_65;
      }
      v101 = v17->m128i_i64[1];
      if ( !v101 )
LABEL_65:
        v101 = v103;
      v123 = 0;
      v124 = (__int64)&v128;
      v125 = 2;
      v126 = 0;
      v127 = 1;
      v18 = *(_QWORD *)(v13 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v18 != v13 + 48 )
      {
        if ( !v18 )
          BUG();
        v19 = v18 - 24;
        if ( (unsigned int)*(unsigned __int8 *)(v18 - 24) - 30 <= 0xA )
        {
          v104 = sub_B46E30(v19);
          if ( v104 )
          {
            v20 = v101;
            for ( i = 0; v104 != i; ++i )
            {
              v22 = sub_B46EC0(v19, i);
              v26 = v22;
              if ( v22 == v20 )
              {
                v20 = 0;
                continue;
              }
              if ( v22 != v101 )
              {
                if ( !v127 )
                  goto LABEL_53;
                v27 = (_QWORD *)v124;
                v23 = (__int64 *)(v124 + 8LL * HIDWORD(v125));
                if ( (__int64 *)v124 != v23 )
                {
                  while ( v26 != *v27 )
                  {
                    if ( v23 == ++v27 )
                      goto LABEL_54;
                  }
                  goto LABEL_29;
                }
LABEL_54:
                if ( HIDWORD(v125) < (unsigned int)v125 )
                {
                  ++HIDWORD(v125);
                  *v23 = v26;
                  ++v123;
                }
                else
                {
LABEL_53:
                  v97 = v26;
                  sub_C8CC70((__int64)&v123, v26, (__int64)v23, v24, v26, v25);
                  v26 = v97;
                }
              }
LABEL_29:
              sub_AA5980(v26, v13, 0);
            }
          }
        }
      }
      v134 = 257;
      v28 = sub_BD2C40(72, 1u);
      v29 = (__int64)v28;
      if ( v28 )
        sub_B4C8F0((__int64)v28, v101, 1u, 0, 0);
      v30 = v29;
      (*(void (__fastcall **)(unsigned int *, __int64, __int64 **, unsigned int *, unsigned int *))(*(_QWORD *)a4[11]
                                                                                                  + 16LL))(
        a4[11],
        v29,
        &v129,
        a4[7],
        a4[8]);
      v31 = *a4;
      v32 = (__int64)&(*a4)[4 * *((unsigned int *)a4 + 2)];
      if ( *a4 != (unsigned int *)v32 )
      {
        do
        {
          v33 = *((_QWORD *)v31 + 1);
          v30 = *v31;
          v31 += 4;
          sub_B99FD0(v29, v30, v33);
        }
        while ( (unsigned int *)v32 != v31 );
      }
      sub_F91380((char *)a2);
      if ( *(_QWORD *)(a1 + 8) )
      {
        v35 = &v131;
        v36 = 0;
        v37 = 0;
        v38 = HIDWORD(v125) - v126;
        v129 = &v131;
        v130 = 0x200000000LL;
        if ( v38 > 2 )
        {
          sub_C8D5F0((__int64)&v129, &v131, v38, 0x10u, (__int64)&v131, v34);
          v36 = (unsigned int)v130;
          v35 = &v131;
          v37 = (unsigned int)v130;
        }
        v39 = (__int64 *)v124;
        if ( v127 )
          v40 = (__int64 *)(v124 + 8LL * HIDWORD(v125));
        else
          v40 = (__int64 *)(v124 + 8LL * (unsigned int)v125);
        if ( (__int64 *)v124 != v40 )
        {
          while ( 1 )
          {
            v41 = *v39;
            v42 = (__int64)v39;
            if ( (unsigned __int64)*v39 < 0xFFFFFFFFFFFFFFFELL )
              break;
            if ( v40 == ++v39 )
              goto LABEL_43;
          }
          if ( v40 != v39 )
          {
            do
            {
              v46 = v41 | 4;
              if ( v37 + 1 > (unsigned __int64)HIDWORD(v130) )
              {
                v99 = v42;
                v102 = v35;
                sub_C8D5F0((__int64)&v129, v35, v37 + 1, 0x10u, (__int64)v35, v42);
                v37 = (unsigned int)v130;
                v42 = v99;
                v35 = v102;
              }
              v47 = &v129[2 * v37];
              *v47 = v13;
              v47[1] = v46;
              v37 = (unsigned int)(v130 + 1);
              v48 = (__int64 *)(v42 + 8);
              LODWORD(v130) = v130 + 1;
              if ( (__int64 *)(v42 + 8) == v40 )
                break;
              while ( 1 )
              {
                v41 = *v48;
                v42 = (__int64)v48;
                if ( (unsigned __int64)*v48 < 0xFFFFFFFFFFFFFFFELL )
                  break;
                if ( v40 == ++v48 )
                  goto LABEL_62;
              }
            }
            while ( v40 != v48 );
LABEL_62:
            v36 = (unsigned int)v37;
          }
        }
LABEL_43:
        v30 = (__int64)v129;
        v105 = v35;
        sub_FFB3D0(*(_QWORD *)(a1 + 8), v129, v36);
        if ( v129 != v105 )
          _libc_free(v129, v30);
      }
      if ( v127 )
        goto LABEL_46;
      v49 = (_BYTE *)v124;
LABEL_70:
      _libc_free(v49, v30);
LABEL_46:
      v43 = v112;
      v44 = 1;
      goto LABEL_47;
    }
    while ( 1 )
    {
      while ( v13 != v15->m128i_i64[1] )
      {
        if ( v110 == ++v15 )
          goto LABEL_12;
      }
      if ( v16 )
        goto LABEL_66;
      v16 = v15->m128i_i64[0];
      if ( v110 == ++v15 )
        goto LABEL_12;
    }
  }
  v50 = v110;
  v51 = (void **)&v112;
  v52 = (__m128i *)base;
  v43 = v112;
  if ( (char *)v110 - (_BYTE *)base > (unsigned __int64)((char *)v113 - (char *)v112) )
  {
    v51 = &base;
    v52 = v112;
    v50 = v113;
    p_base = (void **)&v112;
  }
  if ( v52 == v50 )
    goto LABEL_67;
  v53 = (char *)v50 - (char *)v52;
  if ( v53 != 16 )
  {
    if ( v53 <= 16 )
    {
      v56 = (char *)v51[1];
      v54 = *v51;
    }
    else
    {
      qsort(v52, v53 >> 4, 0x10u, (__compar_fn_t)sub_F8E3B0);
      v56 = (char *)v51[1];
      v54 = *v51;
      v14 = v103;
    }
LABEL_93:
    v65 = v56 - v54;
    v66 = v65 >> 4;
    if ( v65 > 16 )
    {
      v106 = v14;
      qsort(v54, v66, 0x10u, (__compar_fn_t)sub_F8E3B0);
      v54 = *v51;
      v14 = v106;
      v66 = ((_BYTE *)v51[1] - (_BYTE *)*v51) >> 4;
    }
    v67 = *p_base;
    v68 = v66;
    if ( (_DWORD)v66 && (unsigned int)(((_BYTE *)p_base[1] - (_BYTE *)*p_base) >> 4) )
    {
      v30 = 0;
      v69 = 0;
      do
      {
        v70 = *(_QWORD *)&v54[16 * (unsigned int)v30];
        if ( v67[2 * v69] == v70 )
          goto LABEL_79;
        if ( v67[2 * v69] < v70 )
          ++v69;
        else
          v30 = (unsigned int)(v30 + 1);
      }
      while ( (unsigned int)(((_BYTE *)p_base[1] - (_BYTE *)*p_base) >> 4) != v69 && (_DWORD)v30 != v68 );
    }
LABEL_66:
    v43 = v112;
LABEL_67:
    v44 = 0;
    goto LABEL_47;
  }
  v54 = *v51;
  v30 = (__int64)v51[1];
  if ( *v51 == (void *)v30 )
    goto LABEL_66;
  v55 = (char *)*v51;
  while ( v52->m128i_i64[0] != *(_QWORD *)v55 )
  {
    v55 += 16;
    v56 = v55;
    if ( (char *)v30 == v55 )
      goto LABEL_93;
  }
LABEL_79:
  if ( *(_BYTE *)a2 != 31 )
  {
    v121 = 0;
    v122 = 0;
    v118 = a2;
    sub_B540B0(&v118);
    v129 = 0;
    v130 = (__int64)&v134;
    v131 = 16;
    v133 = 1;
    v132 = 0;
    v59 = ((char *)v110 - (_BYTE *)base) >> 4;
    if ( (_DWORD)v59 )
    {
      v60 = 0;
      v61 = 1;
      v30 = *(_QWORD *)base;
      v62 = 16LL * (unsigned int)(v59 - 1);
LABEL_82:
      v63 = (_QWORD *)v130;
      v64 = (__int64 *)(v130 + 8LL * HIDWORD(v131));
      if ( (__int64 *)v130 == v64 )
      {
LABEL_89:
        if ( HIDWORD(v131) < (unsigned int)v131 )
        {
          ++HIDWORD(v131);
          *v64 = v30;
          v61 = v133;
          v129 = (__int64 *)((char *)v129 + 1);
          goto LABEL_86;
        }
        goto LABEL_88;
      }
      while ( v30 != *v63 )
      {
        if ( v64 == ++v63 )
          goto LABEL_89;
      }
LABEL_86:
      while ( v62 != v60 )
      {
        v60 += 16;
        v30 = *(_QWORD *)((char *)base + v60);
        if ( (_BYTE)v61 )
          goto LABEL_82;
LABEL_88:
        sub_C8CC70((__int64)&v129, v30, (__int64)v64, v57, v61, v58);
        v61 = v133;
      }
    }
    v71 = &v125;
    v123 = 0;
    v124 = 1;
    do
    {
      *v71 = -4096;
      v71 += 2;
    }
    while ( v71 != (__int64 *)&v129 );
    v72 = v118;
    v73 = ((*(_DWORD *)(v118 + 4) & 0x7FFFFFFu) >> 1) - 1;
LABEL_109:
    v74 = *(_QWORD *)(a1 + 8);
    while ( v73 )
    {
      --v73;
      v75 = 32;
      if ( v73 != 4294967294LL )
        v75 = 32LL * (unsigned int)(2 * v73 + 3);
      v115 = *(_QWORD *)(*(_QWORD *)(v72 - 8) + v75);
      if ( v74 )
      {
        v76 = sub_FA20A0((__int64)&v123, &v115);
        ++*(_DWORD *)v76;
      }
      v30 = *(_QWORD *)(*(_QWORD *)(v72 - 8) + 32LL * (unsigned int)(2 * v73 + 2));
      if ( v133 )
      {
        v77 = (_QWORD *)v130;
        v78 = (_QWORD *)(v130 + 8LL * HIDWORD(v131));
        if ( (_QWORD *)v130 == v78 )
          goto LABEL_109;
        while ( v30 != *v77 )
        {
          if ( v78 == ++v77 )
            goto LABEL_109;
        }
      }
      else if ( !sub_C8CA60((__int64)&v129, v30) )
      {
        goto LABEL_109;
      }
      sub_AA5980(v115, v13, 0);
      v30 = v72;
      sub_B541A0((__int64)&v118, v72, v73);
      v74 = *(_QWORD *)(a1 + 8);
      if ( v74 )
      {
        v30 = (__int64)&v115;
        v79 = sub_FA20A0((__int64)&v123, &v115);
        --*(_DWORD *)v79;
        v74 = *(_QWORD *)(a1 + 8);
      }
    }
    if ( !v74 )
    {
LABEL_145:
      if ( (v124 & 1) == 0 )
      {
        v30 = 16LL * v126;
        sub_C7D6A0(v125, v30, 8);
      }
      if ( !v133 )
        _libc_free(v130, v30);
      if ( v122 )
      {
        v92 = v118;
        v93 = sub_B53F50((__int64)&v118);
        v30 = 2;
        sub_B99FD0(v92, 2u, v93);
      }
      if ( !v121 )
        goto LABEL_46;
      v49 = v119;
      if ( v119 == &v120 )
        goto LABEL_46;
      goto LABEL_70;
    }
    v115 = 0;
    v116 = 0;
    v117 = 0;
    v80 = v124 & 1;
    if ( (unsigned int)v124 >> 1 )
    {
      if ( v80 )
      {
        v81 = &v129;
        v82 = &v125;
      }
      else
      {
        v84 = v126;
        v83 = (__int64 *)v125;
        v82 = (__int64 *)v125;
        v81 = (__int64 **)(v125 + 16LL * v126);
        if ( v81 == (__int64 **)v125 )
          goto LABEL_133;
      }
      do
      {
        if ( *v82 != -4096 && *v82 != -8192 )
          break;
        v82 += 2;
      }
      while ( v82 != (__int64 *)v81 );
    }
    else
    {
      if ( v80 )
      {
        v94 = &v125;
        v95 = 16;
      }
      else
      {
        v94 = (__int64 *)v125;
        v95 = 2LL * v126;
      }
      v82 = &v94[v95];
      v81 = (__int64 **)v82;
    }
    if ( v80 )
    {
      v83 = &v125;
      v85 = 16;
      goto LABEL_134;
    }
    v83 = (__int64 *)v125;
    v84 = v126;
LABEL_133:
    v85 = 2 * v84;
LABEL_134:
    v86 = &v83[v85];
    v87 = (const __m128i *)v108;
    if ( v86 == v82 )
    {
      v88 = 0;
      v30 = 0;
    }
    else
    {
      do
      {
        if ( !*((_DWORD *)v82 + 2) )
        {
          v100 = v86;
          v107 = v87;
          v89 = *v82 | 4;
          v108[0] = v13;
          v108[1] = v89;
          sub_F9E360((__int64)&v115, v87);
          v86 = v100;
          v87 = v107;
        }
        for ( v82 += 2; v81 != (__int64 **)v82; v82 += 2 )
        {
          if ( *v82 != -4096 && *v82 != -8192 )
            break;
        }
      }
      while ( v82 != v86 );
      v30 = v115;
      v74 = *(_QWORD *)(a1 + 8);
      v88 = (v116 - v115) >> 4;
    }
    sub_FFB3D0(v74, v30, v88);
    if ( v115 )
    {
      v30 = v117 - v115;
      j_j___libc_free_0(v115, v117 - v115);
    }
    goto LABEL_145;
  }
  sub_F902B0((__int64 *)a4, v14);
  sub_AA5980(v112->m128i_i64[1], v13, 0);
  sub_F91380((char *)a2);
  v90 = *(_QWORD *)(a1 + 8);
  if ( !v90 )
    goto LABEL_46;
  v44 = 1;
  v91 = v112->m128i_i64[1];
  v129 = (__int64 *)v13;
  v130 = v91 | 4;
  sub_FFB3D0(v90, &v129, 1);
  v43 = v112;
LABEL_47:
  if ( v43 )
    j_j___libc_free_0(v43, v114 - (_QWORD)v43);
  if ( base )
    j_j___libc_free_0(base, v111 - (_QWORD)base);
  return v44;
}
