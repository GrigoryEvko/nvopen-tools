// Function: sub_37CFFC0
// Address: 0x37cffc0
//
void __fastcall sub_37CFFC0(__int64 a1, unsigned int a2, __int64 a3, unsigned __int64 a4, __int64 a5)
{
  __int64 v5; // rax
  __int64 v6; // rcx
  __int64 v7; // r12
  unsigned int v9; // esi
  __int64 v10; // r9
  int v11; // edx
  __int64 v12; // rcx
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rsi
  __int64 v16; // rax
  char v17; // di
  __int64 v18; // rax
  __int64 v19; // rcx
  __int64 v20; // r13
  unsigned int *k; // rax
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rdi
  unsigned int v25; // ecx
  int *v26; // r15
  int v27; // esi
  __int64 v28; // r8
  __int64 v29; // rdx
  __int64 v30; // r8
  __int64 v31; // rcx
  __int64 v32; // rdx
  __int64 v33; // rax
  unsigned __int64 v34; // rbx
  unsigned __int64 v35; // rdx
  int v36; // esi
  unsigned int v37; // eax
  int v38; // edx
  unsigned int v39; // edi
  _DWORD *v40; // r13
  __int64 v41; // rdi
  __int64 v42; // rax
  int *v43; // r14
  int v44; // edx
  _DWORD *v45; // rdi
  _DWORD *v46; // rdx
  int v47; // esi
  int v48; // eax
  __int32 v49; // r11d
  unsigned __int64 v50; // rax
  int v51; // ebx
  __m128i *v52; // rdx
  __int64 v53; // rax
  const __m128i *v54; // rbx
  const __m128i *v55; // rcx
  const __m128i *v56; // r13
  __int32 v57; // ebx
  const __m128i *v58; // r12
  __m128i *v59; // r14
  __int8 v60; // al
  __m128i v61; // xmm0
  unsigned __int64 v62; // rdi
  unsigned __int64 v63; // r14
  int *v64; // rbx
  unsigned int v65; // edx
  int v66; // eax
  int *v67; // rax
  unsigned __int64 v68; // rax
  _QWORD *v69; // r15
  unsigned int v70; // esi
  _QWORD *v71; // r12
  __int64 v72; // rbx
  __int64 v73; // rax
  bool v74; // al
  __int64 v75; // rdi
  __int64 v76; // rdx
  __int64 v77; // rdx
  int *v78; // rdi
  int *v79; // rax
  int v80; // ebx
  _DWORD *v81; // r10
  unsigned int v82; // r10d
  char v83; // r14
  __int64 v84; // r13
  __int64 v85; // rbx
  unsigned int *j; // rax
  __int64 v87; // rdx
  unsigned int v88; // esi
  __int64 v89; // rdi
  unsigned int v90; // ecx
  int *v91; // rax
  int v92; // r8d
  int v93; // eax
  int i; // r9d
  int v95; // r10d
  unsigned int *v96; // rax
  unsigned int *v97; // rbx
  unsigned int v98; // edx
  unsigned int *v99; // r15
  __int64 v100; // r12
  int *v101; // rax
  __int64 v102; // rcx
  __int64 v103; // r8
  unsigned int *v104; // rax
  unsigned int v105; // r10d
  int v106; // r10d
  int *v107; // [rsp+10h] [rbp-180h]
  int *v109; // [rsp+20h] [rbp-170h]
  __int64 v110; // [rsp+28h] [rbp-168h]
  __int64 v111; // [rsp+38h] [rbp-158h]
  __int64 v112; // [rsp+48h] [rbp-148h]
  unsigned int v113; // [rsp+48h] [rbp-148h]
  int v114; // [rsp+48h] [rbp-148h]
  __int64 v115; // [rsp+50h] [rbp-140h]
  __int64 v116; // [rsp+50h] [rbp-140h]
  unsigned __int64 v117; // [rsp+50h] [rbp-140h]
  unsigned __int64 v118; // [rsp+50h] [rbp-140h]
  char v119; // [rsp+5Bh] [rbp-135h]
  unsigned __int64 v121; // [rsp+60h] [rbp-130h]
  _BYTE *v122; // [rsp+68h] [rbp-128h]
  char v123; // [rsp+70h] [rbp-120h]
  __int64 v124; // [rsp+70h] [rbp-120h]
  __int64 v125; // [rsp+78h] [rbp-118h] BYREF
  unsigned int v126; // [rsp+84h] [rbp-10Ch] BYREF
  __int64 v127; // [rsp+88h] [rbp-108h] BYREF
  __int64 v128; // [rsp+90h] [rbp-100h] BYREF
  unsigned int *v129; // [rsp+98h] [rbp-F8h]
  __int64 v130; // [rsp+A0h] [rbp-F0h]
  __int64 v131; // [rsp+A8h] [rbp-E8h]
  unsigned int v132[10]; // [rsp+B0h] [rbp-E0h] BYREF
  char v133; // [rsp+D8h] [rbp-B8h]
  _BYTE *v134; // [rsp+E0h] [rbp-B0h] BYREF
  __int64 v135; // [rsp+E8h] [rbp-A8h]
  _BYTE v136[48]; // [rsp+F0h] [rbp-A0h] BYREF
  __m128i *v137; // [rsp+120h] [rbp-70h] BYREF
  __int64 v138; // [rsp+128h] [rbp-68h]
  _BYTE v139[96]; // [rsp+130h] [rbp-60h] BYREF

  v5 = *(unsigned int *)(a1 + 3432);
  v6 = *(_QWORD *)(a1 + 3416);
  v125 = a3;
  v119 = a5;
  if ( !(_DWORD)v5 )
    return;
  v7 = a1;
  v9 = (v5 - 1) & a2;
  v10 = 5LL * v9;
  v109 = (int *)(v6 + 88LL * v9);
  v11 = *v109;
  if ( *v109 != a2 )
  {
    for ( i = 1; ; i = v95 )
    {
      if ( v11 == -1 )
        return;
      v95 = i + 1;
      v9 = (v5 - 1) & (i + v9);
      v10 = v6 + 88LL * v9;
      v11 = *(_DWORD *)v10;
      if ( *(_DWORD *)v10 == a2 )
        break;
    }
    v109 = (int *)(v6 + 88LL * v9);
  }
  if ( v109 == (int *)(v6 + 88 * v5) )
    return;
  v12 = unk_5051170;
  *(_QWORD *)(*(_QWORD *)(a1 + 3136) + 8LL * a2) = unk_5051170;
  v13 = *(_QWORD *)(a1 + 16);
  v127 = 0;
  v14 = *(unsigned int *)(v13 + 40);
  if ( (_DWORD)v14 )
  {
    v15 = *(_QWORD *)(v13 + 32);
    v12 = v125;
    v16 = 0;
    v17 = 0;
    do
    {
      while ( 1 )
      {
        if ( v125 == *(_QWORD *)(v15 + 8 * v16) )
        {
          LODWORD(v127) = v16;
          if ( !v17 )
            break;
        }
        if ( v14 == ++v16 )
          goto LABEL_10;
      }
      ++v16;
      BYTE4(v127) = 1;
      v17 = 1;
    }
    while ( v14 != v16 );
LABEL_10:
    v119 = a5 | v17;
  }
  v18 = *((_QWORD *)v109 + 10);
  if ( !v119 )
  {
    if ( v18 )
    {
      v83 = 0;
      v84 = *((_QWORD *)v109 + 8);
      v85 = (__int64)(v109 + 12);
    }
    else
    {
      v83 = 1;
      v84 = *((_QWORD *)v109 + 1);
      v85 = v84 + 4LL * (unsigned int)v109[4];
    }
    if ( v83 )
      goto LABEL_148;
LABEL_142:
    if ( v85 == v84 )
    {
LABEL_154:
      sub_37C43E0(v7, a4, 0, v12, a5, v10);
      return;
    }
    for ( j = (unsigned int *)(v84 + 32); ; j = (unsigned int *)v84 )
    {
      v87 = *(unsigned int *)(v7 + 3464);
      v88 = *j;
      v89 = *(_QWORD *)(v7 + 3448);
      if ( (_DWORD)v87 )
      {
        v90 = (v87 - 1) & (37 * v88);
        v91 = (int *)(v89 + 88LL * v90);
        v92 = *v91;
        if ( v88 == *v91 )
          goto LABEL_146;
        v93 = 1;
        while ( v92 != -1 )
        {
          v106 = v93 + 1;
          v90 = (v87 - 1) & (v93 + v90);
          v91 = (int *)(v89 + 88LL * v90);
          v92 = *v91;
          if ( v88 == *v91 )
            goto LABEL_146;
          v93 = v106;
        }
      }
      v91 = (int *)(v89 + 88 * v87);
LABEL_146:
      sub_37BA020(v7, v88, (__int64)(v91 + 18), (__int64)&v125);
      if ( !v83 )
      {
        v84 = sub_220EF30(v84);
        goto LABEL_142;
      }
      v84 += 4;
LABEL_148:
      if ( v85 == v84 )
        goto LABEL_154;
    }
  }
  v128 = 0;
  v134 = v136;
  v135 = 0x600000000LL;
  v129 = 0;
  v19 = (__int64)(v109 + 12);
  v130 = 0;
  v131 = 0;
  v107 = v109 + 12;
  if ( v18 )
  {
    v123 = 0;
    v20 = *((_QWORD *)v109 + 8);
    v110 = (__int64)(v109 + 12);
  }
  else
  {
    v20 = *((_QWORD *)v109 + 1);
    v110 = v20 + 4LL * (unsigned int)v109[4];
    v123 = v119;
  }
  if ( v123 )
    goto LABEL_29;
LABEL_15:
  if ( v20 != v110 )
  {
    for ( k = (unsigned int *)(v20 + 32); ; k = (unsigned int *)v20 )
    {
      v22 = *k;
      v23 = *(unsigned int *)(v7 + 3464);
      v24 = *(_QWORD *)(v7 + 3448);
      v126 = v22;
      if ( (_DWORD)v23 )
      {
        v25 = (v23 - 1) & (37 * v22);
        v10 = 5LL * v25;
        v26 = (int *)(v24 + 88LL * v25);
        v27 = *v26;
        if ( (_DWORD)v22 == *v26 )
          goto LABEL_19;
        v10 = 1;
        while ( v27 != -1 )
        {
          v82 = v10 + 1;
          v25 = (v23 - 1) & (v10 + v25);
          v10 = 5LL * v25;
          v26 = (int *)(v24 + 88LL * v25);
          v27 = *v26;
          if ( (_DWORD)v22 == *v26 )
            goto LABEL_19;
          v10 = v82;
        }
      }
      v26 = (int *)(v24 + 88 * v23);
LABEL_19:
      v28 = (__int64)(v26 + 18);
      v137 = (__m128i *)v139;
      v138 = 0x100000000LL;
      if ( !BYTE4(v127) )
        goto LABEL_20;
      v133 = 0;
      v49 = v127;
      v132[0] = a2;
      v50 = (unsigned int)v26[4];
      v51 = v50;
      if ( (unsigned int)v50 > 1 )
      {
        v114 = v127;
        v118 = (unsigned int)v26[4];
        sub_C8D5F0((__int64)&v137, v139, v118, 0x30u, v28, v10);
        v49 = v114;
        v28 = (__int64)(v26 + 18);
        v50 = v118;
        v52 = &v137[3 * (unsigned int)v138];
      }
      else
      {
        if ( !v26[4] )
        {
          LODWORD(v138) = 0;
          v22 = v126;
          goto LABEL_20;
        }
        v52 = (__m128i *)v139;
      }
      do
      {
        if ( v52 )
        {
          v52->m128i_i32[0] = -1;
          v52[2].m128i_i8[8] = 0;
        }
        v52 += 3;
        --v50;
      }
      while ( v50 );
      v53 = (unsigned int)v26[4];
      LODWORD(v138) = v51 + v138;
      v54 = (const __m128i *)*((_QWORD *)v26 + 1);
      v55 = &v54[3 * v53];
      if ( v54 != v55 )
      {
        v115 = v20;
        v56 = (const __m128i *)*((_QWORD *)v26 + 1);
        v57 = v49;
        v111 = v7;
        v58 = v55;
        v59 = v137;
        v112 = v28;
        while ( 1 )
        {
          v60 = v56[2].m128i_i8[8];
          if ( v60 != v133 )
            goto LABEL_67;
          if ( v60 )
          {
            if ( !(unsigned __int8)sub_2EAB6C0((__int64)v56, (char *)v132) )
              goto LABEL_67;
LABEL_65:
            v56 += 3;
            v59->m128i_i32[0] = v57;
            v59 += 3;
            v59[-1].m128i_i8[8] = 0;
            if ( v58 == v56 )
              goto LABEL_68;
          }
          else
          {
            if ( v56->m128i_i32[0] == v132[0] )
              goto LABEL_65;
LABEL_67:
            v61 = _mm_loadu_si128(v56);
            v56 += 3;
            v59 += 3;
            v59[-3] = v61;
            v59[-2] = _mm_loadu_si128(v56 - 2);
            v59[-1].m128i_i64[0] = v56[-1].m128i_i64[0];
            v59[-1].m128i_i8[8] = v56[-1].m128i_i8[8];
            if ( v58 == v56 )
            {
LABEL_68:
              v20 = v115;
              v28 = v112;
              v7 = v111;
              break;
            }
          }
        }
      }
      v22 = v126;
LABEL_20:
      v29 = *(_QWORD *)(*(_QWORD *)(v7 + 32) + 32LL) + 48 * v22;
      sub_37BA660(*(_QWORD **)(v7 + 16), (__int64)&v137, v29, *(_QWORD *)(v29 + 40), v28);
      v31 = *(unsigned int *)(v7 + 3480);
      v33 = v32;
      v34 = v126 | v121 & 0xFFFFFFFF00000000LL;
      v35 = v31 + 1;
      v121 = v34;
      if ( v31 + 1 > (unsigned __int64)*(unsigned int *)(v7 + 3484) )
      {
        v116 = v33;
        sub_C8D5F0(v7 + 3472, (const void *)(v7 + 3488), v35, 0x10u, v30, v10);
        v31 = *(unsigned int *)(v7 + 3480);
        v33 = v116;
      }
      v19 = *(_QWORD *)(v7 + 3472) + 16 * v31;
      *(_QWORD *)v19 = v34;
      *(_QWORD *)(v19 + 8) = v33;
      ++*(_DWORD *)(v7 + 3480);
      if ( BYTE4(v127) )
      {
        sub_37B6E70((__int64)(v26 + 2), (__int64)&v137, v35, v19, v30, v10);
        v36 = v131;
        if ( (_DWORD)v131 )
        {
          v37 = v126;
          v10 = (unsigned int)(v131 - 1);
          a5 = (__int64)v129;
          v38 = v10 & (37 * v126);
          v19 = (__int64)&v129[v38];
          v39 = *(_DWORD *)v19;
          if ( *(_DWORD *)v19 == v126 )
            goto LABEL_25;
          v80 = 1;
          v81 = 0;
          while ( v39 != -1 )
          {
            if ( v39 != -2 || v81 )
              v19 = (__int64)v81;
            v38 = v10 & (v80 + v38);
            v39 = v129[v38];
            if ( v126 == v39 )
              goto LABEL_25;
            ++v80;
            v81 = (_DWORD *)v19;
            v19 = (__int64)&v129[v38];
          }
          if ( !v81 )
            v81 = (_DWORD *)v19;
          ++v128;
          v19 = (unsigned int)(v130 + 1);
          *(_QWORD *)v132 = v81;
          if ( 4 * (int)v19 < (unsigned int)(3 * v131) )
          {
            if ( (int)v131 - HIDWORD(v130) - (int)v19 > (unsigned int)v131 >> 3 )
            {
LABEL_131:
              LODWORD(v130) = v19;
              if ( *v81 != -1 )
                --HIDWORD(v130);
              *v81 = v37;
              goto LABEL_25;
            }
LABEL_136:
            sub_A08C50((__int64)&v128, v36);
            sub_22B31A0((__int64)&v128, (int *)&v126, v132);
            v37 = v126;
            v81 = *(_DWORD **)v132;
            v19 = (unsigned int)(v130 + 1);
            goto LABEL_131;
          }
        }
        else
        {
          ++v128;
          *(_QWORD *)v132 = 0;
        }
        v36 = 2 * v131;
        goto LABEL_136;
      }
      v62 = *((_QWORD *)v26 + 1);
      a5 = 48LL * (unsigned int)v26[4];
      v63 = v62 + a5;
      if ( v62 != v62 + a5 )
      {
        v64 = (int *)*((_QWORD *)v26 + 1);
        while ( *((_BYTE *)v64 + 40) )
        {
          v64 += 12;
          if ( (int *)v63 == v64 )
            goto LABEL_92;
        }
        if ( (int *)v63 != v64 )
        {
          a5 = a2;
          do
          {
            v65 = *v64;
            if ( *v64 != (_DWORD)a5 )
            {
              v19 = (unsigned int)v135;
              v66 = v135;
              if ( (unsigned int)v135 >= (unsigned __int64)HIDWORD(v135) )
              {
                v68 = ((unsigned __int64)v126 << 32) | v65;
                if ( HIDWORD(v135) < (unsigned __int64)(unsigned int)v135 + 1 )
                {
                  v113 = a5;
                  v117 = ((unsigned __int64)v126 << 32) | v65;
                  sub_C8D5F0((__int64)&v134, v136, (unsigned int)v135 + 1LL, 8u, a5, v10);
                  v19 = (unsigned int)v135;
                  a5 = v113;
                  v68 = v117;
                }
                *(_QWORD *)&v134[8 * v19] = v68;
                LODWORD(v135) = v135 + 1;
              }
              else
              {
                v19 = (__int64)&v134[8 * (unsigned int)v135];
                if ( v19 )
                {
                  *(_DWORD *)v19 = v65;
                  *(_DWORD *)(v19 + 4) = v126;
                  v66 = v135;
                }
                LODWORD(v135) = v66 + 1;
              }
            }
            v67 = v64 + 12;
            if ( (int *)v63 == v64 + 12 )
              break;
            while ( 1 )
            {
              v64 = v67;
              if ( !*((_BYTE *)v67 + 40) )
                break;
              v67 += 12;
              if ( (int *)v63 == v67 )
                goto LABEL_91;
            }
          }
          while ( (int *)v63 != v67 );
LABEL_91:
          v62 = *((_QWORD *)v26 + 1);
        }
      }
LABEL_92:
      if ( (int *)v62 != v26 + 6 )
        _libc_free(v62);
      *v26 = -2;
      --*(_DWORD *)(v7 + 3456);
      ++*(_DWORD *)(v7 + 3460);
LABEL_25:
      if ( v137 != (__m128i *)v139 )
        _libc_free((unsigned __int64)v137);
      if ( !v123 )
      {
        v20 = sub_220EF30(v20);
        goto LABEL_15;
      }
      v20 += 4;
LABEL_29:
      if ( v20 == v110 )
        break;
    }
  }
  v122 = &v134[8 * (unsigned int)v135];
  if ( v122 == v134 )
    goto LABEL_46;
  v124 = v7;
  v40 = v134;
  do
  {
    v41 = *(_QWORD *)(v124 + 3416);
    v42 = *(unsigned int *)(v124 + 3432);
    if ( (_DWORD)v42 )
    {
      v10 = (unsigned int)(v42 - 1);
      a5 = (unsigned int)v10 & *v40;
      v19 = 5 * a5;
      v43 = (int *)(v41 + 88 * a5);
      v44 = *v43;
      if ( *v43 == *v40 )
        goto LABEL_35;
      v19 = 1;
      while ( v44 != -1 )
      {
        v105 = v19 + 1;
        a5 = (unsigned int)v10 & ((_DWORD)v19 + (_DWORD)a5);
        v19 = 5LL * (unsigned int)a5;
        v43 = (int *)(v41 + 88LL * (unsigned int)a5);
        v44 = *v43;
        if ( *v40 == *v43 )
          goto LABEL_35;
        v19 = v105;
      }
    }
    v43 = (int *)(v41 + 88 * v42);
LABEL_35:
    if ( *((_QWORD *)v43 + 10) )
    {
      v10 = *((_QWORD *)v43 + 7);
      v69 = v43 + 12;
      if ( v10 )
      {
        v70 = v40[1];
        v71 = v43 + 12;
        v72 = *((_QWORD *)v43 + 7);
        while ( 1 )
        {
          while ( *(_DWORD *)(v72 + 32) < v70 )
          {
            v72 = *(_QWORD *)(v72 + 24);
            if ( !v72 )
              goto LABEL_104;
          }
          v73 = *(_QWORD *)(v72 + 16);
          if ( *(_DWORD *)(v72 + 32) <= v70 )
            break;
          v71 = (_QWORD *)v72;
          v72 = *(_QWORD *)(v72 + 16);
          if ( !v73 )
          {
LABEL_104:
            v74 = v71 == v69;
            goto LABEL_105;
          }
        }
        v75 = *(_QWORD *)(v72 + 24);
        if ( v75 )
        {
          do
          {
            while ( 1 )
            {
              v19 = *(_QWORD *)(v75 + 16);
              v76 = *(_QWORD *)(v75 + 24);
              if ( v70 < *(_DWORD *)(v75 + 32) )
                break;
              v75 = *(_QWORD *)(v75 + 24);
              if ( !v76 )
                goto LABEL_116;
            }
            v71 = (_QWORD *)v75;
            v75 = *(_QWORD *)(v75 + 16);
          }
          while ( v19 );
        }
LABEL_116:
        while ( v73 )
        {
          while ( 1 )
          {
            v19 = *(_QWORD *)(v73 + 16);
            v77 = *(_QWORD *)(v73 + 24);
            if ( v70 <= *(_DWORD *)(v73 + 32) )
              break;
            v73 = *(_QWORD *)(v73 + 24);
            if ( !v77 )
              goto LABEL_119;
          }
          v72 = v73;
          v73 = *(_QWORD *)(v73 + 16);
        }
LABEL_119:
        if ( *((_QWORD *)v43 + 8) == v72 && v69 == v71 )
        {
LABEL_107:
          sub_37B80B0(*((_QWORD *)v43 + 7));
          *((_QWORD *)v43 + 7) = 0;
          *((_QWORD *)v43 + 8) = v69;
          *((_QWORD *)v43 + 9) = v69;
          *((_QWORD *)v43 + 10) = 0;
          goto LABEL_44;
        }
        for ( ; (_QWORD *)v72 != v71; --*((_QWORD *)v43 + 10) )
        {
          v78 = (int *)v72;
          v72 = sub_220EF30(v72);
          v79 = sub_220F330(v78, v69);
          j_j___libc_free_0((unsigned __int64)v79);
        }
      }
      else
      {
        v74 = v119;
        v71 = v43 + 12;
LABEL_105:
        if ( *((_QWORD **)v43 + 8) == v71 && v74 )
          goto LABEL_107;
      }
    }
    else
    {
      v45 = (_DWORD *)*((_QWORD *)v43 + 1);
      v46 = &v45[v43[4]];
      v47 = v43[4];
      if ( v45 != v46 )
      {
        while ( *v45 != v40[1] )
        {
          if ( v46 == ++v45 )
            goto LABEL_44;
        }
        if ( v46 != v45 )
        {
          a5 = (__int64)(v45 + 1);
          if ( v46 != v45 + 1 )
          {
            memmove(v45, v45 + 1, (size_t)v46 - a5);
            v47 = v43[4];
          }
          v43[4] = v47 - 1;
        }
      }
    }
LABEL_44:
    v40 += 2;
  }
  while ( v122 != (_BYTE *)v40 );
  v7 = v124;
LABEL_46:
  if ( BYTE4(v127) )
  {
    v19 = v125;
    *(_QWORD *)(*(_QWORD *)(v7 + 3136) + 8LL * (unsigned int)v127) = v125;
  }
  sub_37C43E0(v7, a4, 0, v19, a5, v10);
  v109[4] = 0;
  sub_37B80B0(*((_QWORD *)v109 + 7));
  v48 = v130;
  *((_QWORD *)v109 + 7) = 0;
  *((_QWORD *)v109 + 10) = 0;
  *((_QWORD *)v109 + 8) = v107;
  *((_QWORD *)v109 + 9) = v107;
  if ( v48 )
  {
    v96 = v129;
    v97 = &v129[(unsigned int)v131];
    if ( v129 != v97 )
    {
      while ( 1 )
      {
        v98 = *v96;
        v99 = v96;
        if ( *v96 <= 0xFFFFFFFD )
          break;
        if ( v97 == ++v96 )
          goto LABEL_49;
      }
      if ( v97 != v96 )
      {
        v100 = v7 + 3408;
        do
        {
          v132[0] = v98;
          v101 = sub_37BEF10(v100, (int *)&v127);
          sub_2B5C0F0((__int64)&v137, (__int64)v101, v132, v102, v103);
          v104 = v99 + 1;
          if ( v99 + 1 == v97 )
            break;
          while ( 1 )
          {
            v98 = *v104;
            v99 = v104;
            if ( *v104 <= 0xFFFFFFFD )
              break;
            if ( v97 == ++v104 )
              goto LABEL_49;
          }
        }
        while ( v104 != v97 );
      }
    }
  }
LABEL_49:
  if ( v134 != v136 )
    _libc_free((unsigned __int64)v134);
  sub_C7D6A0((__int64)v129, 4LL * (unsigned int)v131, 4);
}
