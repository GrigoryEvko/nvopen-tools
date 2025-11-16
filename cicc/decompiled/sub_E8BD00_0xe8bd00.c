// Function: sub_E8BD00
// Address: 0xe8bd00
//
__int64 __fastcall sub_E8BD00(
        __int64 a1,
        __int64 a2,
        int *a3,
        __int64 a4,
        __int64 a5,
        unsigned __int64 a6,
        __int64 a7,
        __int64 a8)
{
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r14
  __int64 v16; // r9
  __int64 v17; // r13
  __int64 v18; // rax
  __m128i v19; // xmm0
  __m128i *v20; // rax
  __int64 v21; // rdx
  unsigned __int64 v22; // rcx
  _QWORD *v24; // r14
  __int64 v25; // rax
  __int64 v26; // rax
  __m128i si128; // xmm0
  __m128i *v28; // rax
  __int64 v29; // r8
  int *v30; // rdi
  _BYTE *v31; // r13
  __int64 v32; // rax
  __m128i v33; // xmm0
  __m128i *v34; // rax
  char v35; // si
  __m128i v36; // xmm5
  const __m128i *v37; // rbx
  __int64 v38; // rax
  unsigned __int64 v39; // rdx
  unsigned __int64 v40; // r8
  __m128i *v41; // rax
  __m128i v42; // xmm1
  __m128i v43; // xmm3
  unsigned __int64 v44; // rcx
  unsigned __int64 v45; // r13
  unsigned __int64 v46; // rsi
  int v47; // edi
  int v48; // edx
  unsigned __int64 v49; // rax
  __int64 v50; // r13
  __int64 v51; // rax
  __m128i v52; // xmm0
  unsigned __int64 v53; // rsi
  void *v54; // rax
  __int64 v55; // rax
  __m128i v56; // xmm0
  int v57; // r14d
  const __m128i *v58; // rdx
  __int64 v59; // rax
  unsigned __int64 v60; // rcx
  unsigned __int64 v61; // r8
  __m128i *v62; // rax
  bool v63; // zf
  __int64 v64; // rax
  __m128i v65; // xmm0
  __m128i v66; // xmm6
  __int64 v67; // rax
  __m128i v68; // xmm0
  __m128i *v69; // rcx
  __int64 v70; // r14
  __int64 v71; // rax
  __m128i v72; // xmm0
  __m128i *v73; // rax
  __int64 v74; // rsi
  unsigned __int64 v75; // rcx
  __m128i v76; // xmm6
  __int64 v77; // rdi
  const void *v78; // rsi
  char *v79; // rbx
  unsigned __int64 v80; // rdx
  const __m128i *v81; // r14
  __m128i *v82; // rax
  __int64 v83; // rdi
  const void *v84; // rsi
  char *v85; // rbx
  void *v86; // rax
  __int64 v87; // rax
  __m128i v88; // xmm0
  __int64 v89; // rax
  __m128i v90; // xmm0
  __m128i v91; // xmm3
  __m128i v92; // xmm6
  __int64 v93; // rdi
  const void *v94; // rsi
  char *v95; // r14
  __int64 v96; // [rsp+0h] [rbp-110h]
  _QWORD *v97; // [rsp+0h] [rbp-110h]
  unsigned __int64 v98; // [rsp+18h] [rbp-F8h]
  __int64 v99; // [rsp+20h] [rbp-F0h] BYREF
  __int64 v100; // [rsp+28h] [rbp-E8h]
  __int64 v101; // [rsp+30h] [rbp-E0h]
  int v102; // [rsp+38h] [rbp-D8h]
  __int64 v103; // [rsp+40h] [rbp-D0h] BYREF
  __int64 v104; // [rsp+48h] [rbp-C8h]
  __int64 v105; // [rsp+50h] [rbp-C0h]
  int v106; // [rsp+58h] [rbp-B8h]
  __m128i *v107; // [rsp+60h] [rbp-B0h] BYREF
  __int64 v108; // [rsp+68h] [rbp-A8h]
  __m128i v109; // [rsp+70h] [rbp-A0h] BYREF
  __int64 *v110; // [rsp+80h] [rbp-90h] BYREF
  __m128i *v111; // [rsp+88h] [rbp-88h]
  __int64 v112; // [rsp+90h] [rbp-80h] BYREF
  __m128i v113; // [rsp+98h] [rbp-78h] BYREF
  __int64 v114; // [rsp+B0h] [rbp-60h] BYREF
  __m128i *v115; // [rsp+B8h] [rbp-58h]
  unsigned __int64 v116; // [rsp+C0h] [rbp-50h]
  __m128i v117; // [rsp+C8h] [rbp-48h] BYREF
  char v118; // [rsp+D8h] [rbp-38h]

  v98 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64))(**(_QWORD **)(*(_QWORD *)(a2 + 296) + 8LL) + 56LL))(
          *(_QWORD *)(*(_QWORD *)(a2 + 296) + 8LL),
          a4,
          a5);
  if ( !BYTE4(v98) )
  {
    v114 = 23;
    v110 = &v112;
    v26 = sub_22409D0(&v110, &v114, 0);
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F813E0);
    v110 = (__int64 *)v26;
    v112 = v114;
    *(_DWORD *)(v26 + 16) = 1847619183;
    *(_WORD *)(v26 + 20) = 28001;
    *(_BYTE *)(v26 + 22) = 101;
    *(__m128i *)v26 = si128;
    v111 = (__m128i *)v114;
    *((_BYTE *)v110 + v114) = 0;
    v20 = (__m128i *)v110;
    if ( v110 == &v112 )
    {
      v42 = _mm_load_si128((const __m128i *)&v112);
      *(_BYTE *)a1 = 1;
      *(_QWORD *)(a1 + 8) = a1 + 24;
      v22 = (unsigned __int64)v111;
      v117 = v42;
    }
    else
    {
      v21 = v112;
      *(_BYTE *)a1 = 1;
      *(_QWORD *)(a1 + 8) = a1 + 24;
      v22 = (unsigned __int64)v111;
      v117.m128i_i64[0] = v21;
      if ( v20 != &v117 )
        goto LABEL_9;
    }
    goto LABEL_55;
  }
  if ( a6 )
  {
    sub_E9A370(a2, a6);
  }
  else
  {
    v24 = *(_QWORD **)(a2 + 8);
    v25 = sub_E6C430((__int64)v24, a4, HIDWORD(v98), v13, v14);
    a6 = sub_E808D0(v25, 0, v24, 0);
  }
  v15 = sub_E8BB10((_QWORD *)a2, a8);
  v99 = 0;
  v100 = 0;
  v101 = 0;
  v102 = 0;
  if ( !(unsigned __int8)sub_E81950(a3, (__int64)&v99, 0, 0) )
  {
    v114 = 32;
    v110 = &v112;
    v28 = (__m128i *)sub_22409D0(&v110, &v114, 0);
    v110 = (__int64 *)v28;
    v112 = v114;
    *v28 = _mm_load_si128((const __m128i *)&xmmword_3F813F0);
    v28[1] = _mm_load_si128((const __m128i *)&xmmword_3F81400);
    v111 = (__m128i *)v114;
    *((_BYTE *)v110 + v114) = 0;
    v20 = (__m128i *)v110;
    if ( v110 == &v112 )
    {
      v36 = _mm_load_si128((const __m128i *)&v112);
      *(_BYTE *)a1 = 0;
      *(_QWORD *)(a1 + 8) = a1 + 24;
      v22 = (unsigned __int64)v111;
      v117 = v36;
    }
    else
    {
      v21 = v112;
      *(_BYTE *)a1 = 0;
      *(_QWORD *)(a1 + 8) = a1 + 24;
      v22 = (unsigned __int64)v111;
      v117.m128i_i64[0] = v21;
      if ( v20 != &v117 )
        goto LABEL_9;
    }
    goto LABEL_55;
  }
  v17 = v99;
  if ( v99 )
  {
    if ( v100 )
      goto LABEL_7;
    v29 = *(_QWORD *)(v99 + 16);
    if ( *(_QWORD *)v29 )
    {
LABEL_20:
      v30 = *(int **)(v29 + 24);
      if ( (*(_BYTE *)(v29 + 9) & 0x70) != 0x20 )
      {
        v31 = *(_BYTE **)v29;
        if ( *(_BYTE *)(*(_QWORD *)v29 + 28LL) != 1 )
        {
          v114 = 37;
          v107 = &v109;
          v32 = sub_22409D0(&v107, &v114, 0);
          v107 = (__m128i *)v32;
          v109.m128i_i64[0] = v114;
          *(__m128i *)v32 = _mm_load_si128((const __m128i *)&xmmword_3F81440);
          v33 = _mm_load_si128((const __m128i *)&xmmword_3F81450);
          *(_DWORD *)(v32 + 32) = 1852140903;
          *(_BYTE *)(v32 + 36) = 116;
          *(__m128i *)(v32 + 16) = v33;
          v108 = v114;
          v107->m128i_i8[v114] = 0;
          v34 = v107;
          if ( v107 == &v109 )
          {
            v22 = v108;
            v113 = _mm_load_si128(&v109);
          }
          else
          {
            v22 = v108;
            v113.m128i_i64[0] = v109.m128i_i64[0];
            if ( v107 != &v113 )
            {
              v117.m128i_i64[0] = v109.m128i_i64[0];
              v35 = 0;
              goto LABEL_48;
            }
          }
          v34 = &v117;
          v35 = 0;
          v117 = _mm_loadu_si128(&v113);
LABEL_48:
          *(_BYTE *)a1 = v35;
          *(_BYTE *)(a1 + 40) = 0;
          *(_QWORD *)(a1 + 8) = a1 + 24;
          if ( v34 != &v117 )
          {
            *(_QWORD *)(a1 + 8) = v34;
            *(_QWORD *)(a1 + 24) = v117.m128i_i64[0];
            goto LABEL_10;
          }
          goto LABEL_55;
        }
        v57 = *(_QWORD *)(v29 + 24);
        goto LABEL_57;
      }
      *(_BYTE *)(v29 + 8) |= 8u;
      v96 = v29;
      v103 = 0;
      v104 = 0;
      v105 = 0;
      v106 = 0;
      if ( !(unsigned __int8)sub_E81950(v30, (__int64)&v103, 0, 0) )
      {
        v114 = 42;
        v107 = &v109;
        v64 = sub_22409D0(&v107, &v114, 0);
        v107 = (__m128i *)v64;
        v109.m128i_i64[0] = v114;
        *(__m128i *)v64 = _mm_load_si128((const __m128i *)&xmmword_3F81420);
        v65 = _mm_load_si128((const __m128i *)&xmmword_3F81430);
        qmemcpy((void *)(v64 + 32), "elocatable", 10);
        *(__m128i *)(v64 + 16) = v65;
LABEL_43:
        v108 = v114;
        v107->m128i_i8[v114] = 0;
        if ( v107 == &v109 )
        {
          v66 = _mm_load_si128(&v109);
          LOBYTE(v114) = 0;
          v115 = &v117;
          v53 = v108;
          v113 = v66;
        }
        else
        {
          LOBYTE(v114) = 0;
          v115 = &v117;
          v53 = v108;
          v113.m128i_i64[0] = v109.m128i_i64[0];
          if ( v107 != &v113 )
          {
            v115 = v107;
            v117.m128i_i64[0] = v109.m128i_i64[0];
LABEL_46:
            v116 = v53;
LABEL_47:
            v35 = v114;
            v34 = v115;
            v22 = v116;
            goto LABEL_48;
          }
        }
LABEL_100:
        v117 = _mm_loadu_si128(&v113);
        goto LABEL_46;
      }
      v50 = v103;
      if ( v103 )
      {
        if ( v104 )
        {
LABEL_42:
          v114 = 41;
          v107 = &v109;
          v51 = sub_22409D0(&v107, &v114, 0);
          v107 = (__m128i *)v51;
          v109.m128i_i64[0] = v114;
          *(__m128i *)v51 = _mm_load_si128((const __m128i *)&xmmword_3F81460);
          v52 = _mm_load_si128((const __m128i *)&xmmword_3F81470);
          *(_QWORD *)(v51 + 32) = 0x6C6261746E657365LL;
          *(_BYTE *)(v51 + 40) = 101;
          *(__m128i *)(v51 + 16) = v52;
          goto LABEL_43;
        }
        v70 = *(_QWORD *)(v103 + 16);
        if ( *(_QWORD *)v70 )
          goto LABEL_75;
        if ( (*(_BYTE *)(v70 + 9) & 0x70) == 0x20 && *(char *)(v70 + 8) >= 0 )
        {
          *(_BYTE *)(v70 + 8) |= 8u;
          v86 = sub_E807D0(*(_QWORD *)(v70 + 24));
          *(_QWORD *)v70 = v86;
          if ( v86 )
          {
            v70 = *(_QWORD *)(v50 + 16);
LABEL_75:
            if ( (*(_BYTE *)(v70 + 9) & 0x70) == 0x20 )
            {
              v114 = 44;
              v107 = &v109;
              v89 = sub_22409D0(&v107, &v114, 0);
              v107 = (__m128i *)v89;
              v109.m128i_i64[0] = v114;
              *(__m128i *)v89 = _mm_load_si128((const __m128i *)&xmmword_3F81480);
              v90 = _mm_load_si128((const __m128i *)&xmmword_3F81490);
              qmemcpy((void *)(v89 + 32), " is variable", 12);
              *(__m128i *)(v89 + 16) = v90;
              v108 = v114;
              v107->m128i_i8[v114] = 0;
              v73 = v107;
              if ( v107 == &v109 )
              {
                v92 = _mm_load_si128(&v109);
                LOBYTE(v114) = 0;
                v115 = &v117;
                v75 = v108;
                v113 = v92;
              }
              else
              {
                v74 = v109.m128i_i64[0];
                LOBYTE(v114) = 0;
                v115 = &v117;
                v75 = v108;
                v113.m128i_i64[0] = v109.m128i_i64[0];
                if ( v107 != &v113 )
                  goto LABEL_80;
              }
              goto LABEL_103;
            }
            v31 = *(_BYTE **)v70;
            if ( !*(_QWORD *)v70 || v31[28] != 1 )
            {
              v114 = 37;
              v107 = &v109;
              v71 = sub_22409D0(&v107, &v114, 0);
              v107 = (__m128i *)v71;
              v109.m128i_i64[0] = v114;
              *(__m128i *)v71 = _mm_load_si128((const __m128i *)&xmmword_3F81440);
              v72 = _mm_load_si128((const __m128i *)&xmmword_3F81450);
              *(_DWORD *)(v71 + 32) = 1852140903;
              *(_BYTE *)(v71 + 36) = 116;
              *(__m128i *)(v71 + 16) = v72;
              v108 = v114;
              v107->m128i_i8[v114] = 0;
              v73 = v107;
              if ( v107 == &v109 )
              {
                v91 = _mm_load_si128(&v109);
                LOBYTE(v114) = 0;
                v115 = &v117;
                v75 = v108;
                v113 = v91;
              }
              else
              {
                v74 = v109.m128i_i64[0];
                LOBYTE(v114) = 0;
                v115 = &v117;
                v75 = v108;
                v113.m128i_i64[0] = v109.m128i_i64[0];
                if ( v107 != &v113 )
                {
LABEL_80:
                  v115 = v73;
                  v117.m128i_i64[0] = v74;
LABEL_81:
                  v116 = v75;
                  goto LABEL_47;
                }
              }
LABEL_103:
              v117 = _mm_loadu_si128(&v113);
              goto LABEL_81;
            }
            v57 = v105 + *(_DWORD *)(v70 + 24);
LABEL_57:
            v118 = 0;
            v58 = (const __m128i *)&v110;
            v110 = (__int64 *)a6;
            HIDWORD(v111) = v98;
            LODWORD(v111) = v57 + v101;
            v112 = a7;
            v59 = *((unsigned int *)v31 + 26);
            v60 = *((_QWORD *)v31 + 12);
            v61 = v59 + 1;
            if ( v59 + 1 > (unsigned __int64)*((unsigned int *)v31 + 27) )
            {
              v77 = (__int64)(v31 + 96);
              v78 = v31 + 112;
              if ( v60 > (unsigned __int64)&v110 || (unsigned __int64)&v110 >= v60 + 24 * v59 )
              {
                sub_C8D5F0(v77, v78, v61, 0x18u, v61, v16);
                v60 = *((_QWORD *)v31 + 12);
                v59 = *((unsigned int *)v31 + 26);
                v58 = (const __m128i *)&v110;
              }
              else
              {
                v79 = (char *)&v110 - v60;
                sub_C8D5F0(v77, v78, v61, 0x18u, v61, v16);
                v60 = *((_QWORD *)v31 + 12);
                v59 = *((unsigned int *)v31 + 26);
                v58 = (const __m128i *)&v79[v60];
              }
            }
            v62 = (__m128i *)(v60 + 24 * v59);
            *v62 = _mm_loadu_si128(v58);
            v62[1].m128i_i64[0] = v58[1].m128i_i64[0];
            ++*((_DWORD *)v31 + 26);
            v63 = v118 == 0;
            *(_BYTE *)(a1 + 40) = 0;
            if ( !v63 )
            {
              v118 = 0;
              if ( v115 != &v117 )
                j_j___libc_free_0(v115, v117.m128i_i64[0] + 1);
            }
            return a1;
          }
        }
        v114 = 47;
        v107 = &v109;
        v87 = sub_22409D0(&v107, &v114, 0);
        v107 = (__m128i *)v87;
        v109.m128i_i64[0] = v114;
        *(__m128i *)v87 = _mm_load_si128((const __m128i *)&xmmword_3F81480);
        v88 = _mm_load_si128((const __m128i *)&xmmword_3F81490);
        qmemcpy((void *)(v87 + 32), " is not defined", 15);
        *(__m128i *)(v87 + 16) = v88;
        v108 = v114;
        v107->m128i_i8[v114] = 0;
        v111 = &v113;
        if ( v107 == &v109 )
        {
          v113 = _mm_load_si128(&v109);
        }
        else
        {
          v111 = v107;
          v113.m128i_i64[0] = v109.m128i_i64[0];
        }
        v69 = v111;
        LOBYTE(v114) = 0;
        v115 = &v117;
        v53 = v108;
        if ( v111 == &v113 )
          goto LABEL_100;
      }
      else
      {
        if ( v104 )
          goto LABEL_42;
        v31 = *(_BYTE **)v96;
        v57 = v105;
        if ( *(_QWORD *)v96
          || (*(_BYTE *)(v96 + 9) & 0x70) == 0x20
          && *(char *)(v96 + 8) >= 0
          && (*(_BYTE *)(v96 + 8) |= 8u, v31 = sub_E807D0(*(_QWORD *)(v96 + 24)), (*(_QWORD *)v96 = v31) != 0) )
        {
          if ( v31[28] == 1 )
            goto LABEL_57;
        }
        v114 = 37;
        v107 = &v109;
        v67 = sub_22409D0(&v107, &v114, 0);
        v107 = (__m128i *)v67;
        v109.m128i_i64[0] = v114;
        *(__m128i *)v67 = _mm_load_si128((const __m128i *)&xmmword_3F81440);
        v68 = _mm_load_si128((const __m128i *)&xmmword_3F81450);
        *(_DWORD *)(v67 + 32) = 1852140903;
        *(_BYTE *)(v67 + 36) = 116;
        *(__m128i *)(v67 + 16) = v68;
        v108 = v114;
        v107->m128i_i8[v114] = 0;
        v111 = &v113;
        if ( v107 == &v109 )
        {
          v113 = _mm_load_si128(&v109);
        }
        else
        {
          v111 = v107;
          v113.m128i_i64[0] = v109.m128i_i64[0];
        }
        v69 = v111;
        LOBYTE(v114) = 0;
        v115 = &v117;
        v53 = v108;
        if ( v111 == &v113 )
          goto LABEL_100;
      }
      v115 = v69;
      v117.m128i_i64[0] = v113.m128i_i64[0];
      goto LABEL_46;
    }
    if ( (*(_BYTE *)(v29 + 9) & 0x70) == 0x20 && *(char *)(v29 + 8) >= 0 )
    {
      *(_BYTE *)(v29 + 8) |= 8u;
      v97 = (_QWORD *)v29;
      v54 = sub_E807D0(*(_QWORD *)(v29 + 24));
      v29 = (__int64)v97;
      *v97 = v54;
      if ( v54 )
        goto LABEL_20;
      v29 = *(_QWORD *)(v17 + 16);
    }
    v44 = *(unsigned int *)(a2 + 320);
    v45 = *(_QWORD *)(a2 + 312);
    v46 = *(unsigned int *)(a2 + 324);
    v47 = v101;
    v48 = *(_DWORD *)(a2 + 320);
    v49 = v45 + 40 * v44;
    if ( v44 >= v46 )
    {
      v115 = (__m128i *)a6;
      v80 = v44 + 1;
      v117.m128i_i64[1] = v15;
      v81 = (const __m128i *)&v114;
      v114 = v29;
      v116 = __PAIR64__(v98, v101);
      v117.m128i_i64[0] = a7;
      if ( v46 < v44 + 1 )
      {
        v93 = a2 + 312;
        v94 = (const void *)(a2 + 328);
        if ( v45 > (unsigned __int64)&v114 || v49 <= (unsigned __int64)&v114 )
        {
          sub_C8D5F0(v93, v94, v80, 0x28u, v29, v16);
          v44 = *(unsigned int *)(a2 + 320);
          v45 = *(_QWORD *)(a2 + 312);
        }
        else
        {
          sub_C8D5F0(v93, v94, v80, 0x28u, v29, v16);
          v95 = (char *)&v114 - v45;
          v45 = *(_QWORD *)(a2 + 312);
          v44 = *(unsigned int *)(a2 + 320);
          v81 = (const __m128i *)&v95[v45];
        }
      }
      v82 = (__m128i *)(v45 + 40 * v44);
      *v82 = _mm_loadu_si128(v81);
      v82[1] = _mm_loadu_si128(v81 + 1);
      v82[2].m128i_i64[0] = v81[2].m128i_i64[0];
      ++*(_DWORD *)(a2 + 320);
    }
    else
    {
      if ( v49 )
      {
        *(_QWORD *)(v49 + 8) = a6;
        *(_QWORD *)v49 = v29;
        *(_DWORD *)(v49 + 20) = v98;
        *(_DWORD *)(v49 + 16) = v47;
        *(_QWORD *)(v49 + 24) = a7;
        *(_QWORD *)(v49 + 32) = v15;
        v48 = *(_DWORD *)(a2 + 320);
      }
      *(_DWORD *)(a2 + 320) = v48 + 1;
    }
    *(_BYTE *)(a1 + 40) = 0;
  }
  else
  {
    if ( v100 )
    {
LABEL_7:
      v114 = 34;
      v110 = &v112;
      v18 = sub_22409D0(&v110, &v114, 0);
      v110 = (__int64 *)v18;
      v112 = v114;
      *(__m128i *)v18 = _mm_load_si128((const __m128i *)&xmmword_3F813F0);
      v19 = _mm_load_si128((const __m128i *)&xmmword_3F81410);
      *(_WORD *)(v18 + 32) = 25964;
      *(__m128i *)(v18 + 16) = v19;
      v111 = (__m128i *)v114;
      *((_BYTE *)v110 + v114) = 0;
      v20 = (__m128i *)v110;
      if ( v110 == &v112 )
      {
        v43 = _mm_load_si128((const __m128i *)&v112);
        *(_BYTE *)a1 = 0;
        *(_QWORD *)(a1 + 8) = a1 + 24;
        v22 = (unsigned __int64)v111;
        v117 = v43;
      }
      else
      {
        v21 = v112;
        *(_BYTE *)a1 = 0;
        *(_QWORD *)(a1 + 8) = a1 + 24;
        v22 = (unsigned __int64)v111;
        v117.m128i_i64[0] = v21;
        if ( v20 != &v117 )
        {
LABEL_9:
          *(_QWORD *)(a1 + 8) = v20;
          *(_QWORD *)(a1 + 24) = v21;
LABEL_10:
          *(_QWORD *)(a1 + 16) = v22;
          *(_BYTE *)(a1 + 40) = 1;
          return a1;
        }
      }
LABEL_55:
      *(__m128i *)(a1 + 24) = _mm_loadu_si128(&v117);
      goto LABEL_10;
    }
    if ( v101 < 0 )
    {
      v114 = 25;
      v110 = &v112;
      v55 = sub_22409D0(&v110, &v114, 0);
      v56 = _mm_load_si128((const __m128i *)&xmmword_3F813F0);
      v110 = (__int64 *)v55;
      v112 = v114;
      *(_QWORD *)(v55 + 16) = 0x7669746167656E20LL;
      *(_BYTE *)(v55 + 24) = 101;
      *(__m128i *)v55 = v56;
      v111 = (__m128i *)v114;
      *((_BYTE *)v110 + v114) = 0;
      v20 = (__m128i *)v110;
      if ( v110 == &v112 )
      {
        v76 = _mm_load_si128((const __m128i *)&v112);
        *(_BYTE *)a1 = 0;
        *(_QWORD *)(a1 + 8) = a1 + 24;
        v22 = (unsigned __int64)v111;
        v117 = v76;
      }
      else
      {
        v21 = v112;
        *(_BYTE *)a1 = 0;
        *(_QWORD *)(a1 + 8) = a1 + 24;
        v22 = (unsigned __int64)v111;
        v117.m128i_i64[0] = v21;
        if ( v20 != &v117 )
          goto LABEL_9;
      }
      goto LABEL_55;
    }
    v115 = (__m128i *)__PAIR64__(v98, v101);
    v114 = a6;
    v37 = (const __m128i *)&v114;
    v116 = a7;
    v38 = *(unsigned int *)(v15 + 104);
    v39 = *(_QWORD *)(v15 + 96);
    v40 = v38 + 1;
    if ( v38 + 1 > (unsigned __int64)*(unsigned int *)(v15 + 108) )
    {
      v83 = v15 + 96;
      v84 = (const void *)(v15 + 112);
      if ( v39 > (unsigned __int64)&v114 || (unsigned __int64)&v114 >= v39 + 24 * v38 )
      {
        sub_C8D5F0(v83, v84, v40, 0x18u, v40, v16);
        v39 = *(_QWORD *)(v15 + 96);
        v38 = *(unsigned int *)(v15 + 104);
      }
      else
      {
        v85 = (char *)&v114 - v39;
        sub_C8D5F0(v83, v84, v40, 0x18u, v40, v16);
        v39 = *(_QWORD *)(v15 + 96);
        v38 = *(unsigned int *)(v15 + 104);
        v37 = (const __m128i *)&v85[v39];
      }
    }
    v41 = (__m128i *)(v39 + 24 * v38);
    *v41 = _mm_loadu_si128(v37);
    v41[1].m128i_i64[0] = v37[1].m128i_i64[0];
    ++*(_DWORD *)(v15 + 104);
    *(_BYTE *)(a1 + 40) = 0;
  }
  return a1;
}
