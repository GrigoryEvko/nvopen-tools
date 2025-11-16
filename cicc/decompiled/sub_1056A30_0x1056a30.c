// Function: sub_1056A30
// Address: 0x1056a30
//
__int64 __fastcall sub_1056A30(__int64 a1, __int64 a2)
{
  __int64 v2; // r15
  int v4; // esi
  __int64 v5; // rax
  __int64 v6; // rcx
  __int64 result; // rax
  __int64 v8; // rdx
  __int64 v9; // r12
  __int64 v10; // r13
  __int64 v11; // rsi
  __m128i *v12; // rdi
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  _BYTE *v17; // rax
  void *v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 *v22; // rbx
  __int64 *v23; // r12
  int v24; // eax
  __int64 v25; // r8
  __int64 v26; // rsi
  int v27; // edx
  unsigned int v28; // eax
  __int64 v29; // rdi
  void *v30; // rdx
  int v31; // r9d
  void *v32; // rdx
  void *v33; // rdx
  const char *v34; // rsi
  char v35; // r13
  _QWORD *v36; // rax
  _QWORD *v37; // rdx
  __int64 *v38; // rbx
  void *v39; // rdx
  __int64 v40; // r8
  unsigned __int64 v41; // rax
  void *v42; // rdx
  __m128i *v43; // rdx
  __m128i v44; // xmm0
  __int64 *v45; // rax
  __int64 v46; // rdx
  __int64 *v47; // r13
  __int64 v48; // rcx
  __int64 *v49; // r12
  __int64 v50; // r15
  __int64 v51; // r14
  __int64 *v52; // rax
  __int64 *v53; // r12
  __int64 *v54; // r13
  _WORD *v55; // rdx
  __int64 v56; // r10
  _BYTE *v57; // rax
  __int64 *v58; // rax
  __m128i *v59; // rdx
  __m128i v60; // xmm0
  __int64 *v61; // rbx
  __int64 v62; // r12
  __int64 v63; // rax
  void *v64; // rdx
  __int64 v65; // r8
  __int64 v66; // rax
  void *v67; // rdx
  __int64 v68; // r8
  __int64 v69; // rcx
  __int64 v70; // rcx
  __int64 v71; // r8
  __int64 *v72; // r9
  _WORD *v73; // rdx
  void *v74; // rdx
  __int64 v75; // r8
  __int64 v76; // rax
  __m128i v77; // xmm0
  __m128i *v78; // rdx
  __m128i v79; // xmm0
  __int64 *v80; // r12
  __int64 v81; // r13
  __int64 v82; // rax
  _BYTE *v83; // rax
  _WORD *v84; // rdx
  __int64 v85; // rcx
  __int64 v86; // rax
  __int64 *v87; // r13
  __int64 *v88; // r12
  char v89; // bl
  _DWORD *v90; // rdx
  __int64 v91; // rbx
  _BYTE *v92; // rax
  __m128i si128; // xmm0
  __int64 v94; // [rsp+8h] [rbp-228h]
  __int64 v95; // [rsp+20h] [rbp-210h]
  __int64 v96; // [rsp+20h] [rbp-210h]
  __int64 v97; // [rsp+20h] [rbp-210h]
  __int64 v98; // [rsp+28h] [rbp-208h]
  __int64 v99; // [rsp+28h] [rbp-208h]
  __int64 v100; // [rsp+28h] [rbp-208h]
  __int64 v101; // [rsp+28h] [rbp-208h]
  __int64 v102; // [rsp+28h] [rbp-208h]
  __int64 v103; // [rsp+30h] [rbp-200h]
  __int64 v104; // [rsp+30h] [rbp-200h]
  __int64 v105; // [rsp+30h] [rbp-200h]
  __int64 v106; // [rsp+38h] [rbp-1F8h]
  __int64 v107; // [rsp+38h] [rbp-1F8h]
  __int64 *v108; // [rsp+38h] [rbp-1F8h]
  __int64 v109; // [rsp+38h] [rbp-1F8h]
  __int64 v110; // [rsp+38h] [rbp-1F8h]
  __int64 *i; // [rsp+38h] [rbp-1F8h]
  unsigned __int8 v112[16]; // [rsp+50h] [rbp-1E0h] BYREF
  _QWORD v113[2]; // [rsp+60h] [rbp-1D0h] BYREF
  void (__fastcall *v114)(_QWORD *, _QWORD *, __int64); // [rsp+70h] [rbp-1C0h]
  void (__fastcall *v115)(_QWORD *, __int64); // [rsp+78h] [rbp-1B8h]
  _QWORD v116[2]; // [rsp+80h] [rbp-1B0h] BYREF
  void (__fastcall *v117)(_QWORD *, _QWORD *, __int64, __int64, __int64, __int64 *); // [rsp+90h] [rbp-1A0h]
  void (__fastcall *v118)(_QWORD *, __int64); // [rsp+98h] [rbp-198h]
  _QWORD v119[2]; // [rsp+A0h] [rbp-190h] BYREF
  void (__fastcall *v120)(_QWORD *, _QWORD *, __int64, __int64, __int64, __int64 *); // [rsp+B0h] [rbp-180h]
  void (__fastcall *v121)(_QWORD *, __int64); // [rsp+B8h] [rbp-178h]
  __m128i v122; // [rsp+C0h] [rbp-170h] BYREF
  void (__fastcall *v123)(__m128i *, __m128i *, __int64); // [rsp+D0h] [rbp-160h]
  void (__fastcall *v124)(__m128i *, __int64); // [rsp+D8h] [rbp-158h]
  _QWORD v125[2]; // [rsp+E0h] [rbp-150h] BYREF
  void (__fastcall *v126)(_QWORD *, _QWORD *, __int64); // [rsp+F0h] [rbp-140h]
  void (__fastcall *v127)(_QWORD *, __int64); // [rsp+F8h] [rbp-138h]
  _QWORD v128[2]; // [rsp+100h] [rbp-130h] BYREF
  void (__fastcall *v129)(_QWORD *, _QWORD *, __int64); // [rsp+110h] [rbp-120h]
  void (__fastcall *v130)(_QWORD *, __int64); // [rsp+118h] [rbp-118h]
  __int64 *v131; // [rsp+120h] [rbp-110h] BYREF
  __int64 v132; // [rsp+128h] [rbp-108h]
  _BYTE v133[64]; // [rsp+130h] [rbp-100h] BYREF
  __int64 (__fastcall **v134)(const __m128i **, const __m128i *, int); // [rsp+170h] [rbp-C0h] BYREF
  __int64 v135; // [rsp+178h] [rbp-B8h]
  __int64 (__fastcall *v136)(const __m128i **, const __m128i *, int); // [rsp+180h] [rbp-B0h] BYREF
  _QWORD *(__fastcall *v137)(__int64 *, __int64); // [rsp+188h] [rbp-A8h]

  v2 = a1;
  v4 = *(_DWORD *)(a1 + 256);
  *(_WORD *)v112 = 10;
  if ( v4 )
  {
    v87 = *(__int64 **)(a1 + 248);
    v88 = &v87[*(unsigned int *)(a1 + 264)];
    if ( v87 != v88 )
    {
      while ( *v87 == -8192 || *v87 == -4096 )
      {
        if ( v88 == ++v87 )
          goto LABEL_3;
      }
      v89 = 0;
      while ( v88 != v87 )
      {
        if ( !sub_E45340(*(_QWORD *)(v2 + 208), *v87) )
        {
          v90 = *(_DWORD **)(a2 + 32);
          if ( !v89 )
          {
            if ( *(_QWORD *)(a2 + 24) - (_QWORD)v90 <= 0x14u )
            {
              sub_CB6200(a2, "DIVERGENT ARGUMENTS:\n", 0x15u);
              v90 = *(_DWORD **)(a2 + 32);
            }
            else
            {
              si128 = _mm_load_si128((const __m128i *)&xmmword_3F8E890);
              v90[4] = 978539598;
              *((_BYTE *)v90 + 20) = 10;
              *(__m128i *)v90 = si128;
              v90 = (_DWORD *)(*(_QWORD *)(a2 + 32) + 21LL);
              *(_QWORD *)(a2 + 32) = v90;
            }
          }
          if ( *(_QWORD *)(a2 + 24) - (_QWORD)v90 <= 0xCu )
          {
            v91 = sub_CB6200(a2, "  DIVERGENT: ", 0xDu);
          }
          else
          {
            v91 = a2;
            qmemcpy(v90, "  DIVERGENT: ", 13);
            *(_QWORD *)(a2 + 32) += 13LL;
          }
          v11 = *(_QWORD *)(v2 + 208);
          v12 = (__m128i *)v113;
          sub_E45370(v113, v11, *v87);
          if ( !v114 )
            goto LABEL_168;
          v115(v113, v91);
          v92 = *(_BYTE **)(v91 + 32);
          if ( (unsigned __int64)v92 >= *(_QWORD *)(v91 + 24) )
          {
            sub_CB5D20(v91, 10);
          }
          else
          {
            *(_QWORD *)(v91 + 32) = v92 + 1;
            *v92 = 10;
          }
          if ( v114 )
            v114(v113, v113, 3);
          v89 = 1;
        }
        if ( ++v87 == v88 )
          break;
        while ( *v87 == -8192 || *v87 == -4096 )
        {
          if ( v88 == ++v87 )
            goto LABEL_3;
        }
      }
    }
LABEL_3:
    if ( *(_DWORD *)(v2 + 760) )
    {
      v78 = *(__m128i **)(a2 + 32);
      if ( *(_QWORD *)(a2 + 24) - (_QWORD)v78 <= 0x1Au )
      {
        sub_CB6200(a2, "CYCLES ASSSUMED DIVERGENT:\n", 0x1Bu);
      }
      else
      {
        v79 = _mm_load_si128((const __m128i *)&xmmword_3F8E8A0);
        qmemcpy(&v78[1], "DIVERGENT:\n", 11);
        *v78 = v79;
        *(_QWORD *)(a2 + 32) += 27LL;
      }
      v80 = *(__int64 **)(v2 + 752);
      for ( i = &v80[*(unsigned int *)(v2 + 760)]; i != v80; ++v80 )
      {
        v84 = *(_WORD **)(a2 + 32);
        v85 = *v80;
        if ( *(_QWORD *)(a2 + 24) - (_QWORD)v84 > 1u )
        {
          v81 = a2;
          *v84 = 8224;
          *(_QWORD *)(a2 + 32) += 2LL;
        }
        else
        {
          v97 = *v80;
          v86 = sub_CB6200(a2, (unsigned __int8 *)"  ", 2u);
          v85 = v97;
          v81 = v86;
        }
        v82 = *(_QWORD *)(v2 + 208);
        v134 = (__int64 (__fastcall **)(const __m128i **, const __m128i *, int))v85;
        v135 = v82;
        v136 = sub_E341B0;
        v137 = sub_E34BC0;
        sub_E34BC0((__int64 *)&v134, v81);
        v83 = *(_BYTE **)(v81 + 32);
        if ( (unsigned __int64)v83 >= *(_QWORD *)(v81 + 24) )
        {
          sub_CB5D20(v81, 10);
        }
        else
        {
          *(_QWORD *)(v81 + 32) = v83 + 1;
          *v83 = 10;
        }
        if ( v136 )
          v136((const __m128i **)&v134, (const __m128i *)&v134, 3);
      }
    }
    if ( *(_DWORD *)(v2 + 612) != *(_DWORD *)(v2 + 616) )
    {
      v43 = *(__m128i **)(a2 + 32);
      if ( *(_QWORD *)(a2 + 24) - (_QWORD)v43 <= 0x1Bu )
      {
        sub_CB6200(a2, "CYCLES WITH DIVERGENT EXIT:\n", 0x1Cu);
      }
      else
      {
        v44 = _mm_load_si128((const __m128i *)&xmmword_3F8E8B0);
        qmemcpy(&v43[1], "RGENT EXIT:\n", 12);
        *v43 = v44;
        *(_QWORD *)(a2 + 32) += 28LL;
      }
      v45 = *(__int64 **)(v2 + 600);
      v46 = *(_BYTE *)(v2 + 620) ? *(unsigned int *)(v2 + 612) : *(unsigned int *)(v2 + 608);
      v47 = &v45[v46];
      if ( v45 != v47 )
      {
        v48 = *v45;
        v49 = *(__int64 **)(v2 + 600);
        if ( (unsigned __int64)*v45 < 0xFFFFFFFFFFFFFFFELL )
        {
LABEL_80:
          if ( v47 != v45 )
          {
            v95 = v2;
            v50 = a2;
            v51 = v48;
            v52 = v49;
            v53 = v47;
            v54 = v52;
            do
            {
              v55 = *(_WORD **)(v50 + 32);
              if ( *(_QWORD *)(v50 + 24) - (_QWORD)v55 <= 1u )
              {
                v56 = sub_CB6200(v50, (unsigned __int8 *)"  ", 2u);
              }
              else
              {
                v56 = v50;
                *v55 = 8224;
                *(_QWORD *)(v50 + 32) += 2LL;
              }
              v109 = v56;
              v134 = (__int64 (__fastcall **)(const __m128i **, const __m128i *, int))v51;
              v135 = *(_QWORD *)(v95 + 208);
              v136 = sub_E341B0;
              v137 = sub_E34BC0;
              sub_E34BC0((__int64 *)&v134, v56);
              v57 = *(_BYTE **)(v109 + 32);
              if ( (unsigned __int64)v57 >= *(_QWORD *)(v109 + 24) )
              {
                sub_CB5D20(v109, 10);
              }
              else
              {
                *(_QWORD *)(v109 + 32) = v57 + 1;
                *v57 = 10;
              }
              if ( v136 )
                v136((const __m128i **)&v134, (const __m128i *)&v134, 3);
              v58 = v54 + 1;
              if ( v54 + 1 == v53 )
                break;
              while ( 1 )
              {
                v51 = *v58;
                v54 = v58;
                if ( (unsigned __int64)*v58 < 0xFFFFFFFFFFFFFFFELL )
                  break;
                if ( v53 == ++v58 )
                  goto LABEL_91;
              }
            }
            while ( v53 != v58 );
LABEL_91:
            a2 = v50;
            v2 = v95;
            if ( !*(_DWORD *)(v95 + 8) )
              goto LABEL_6;
            goto LABEL_92;
          }
        }
        else
        {
          while ( v47 != ++v45 )
          {
            v48 = *v45;
            v49 = v45;
            if ( (unsigned __int64)*v45 < 0xFFFFFFFFFFFFFFFELL )
              goto LABEL_80;
          }
        }
      }
    }
    if ( !*(_DWORD *)(v2 + 8) )
    {
LABEL_6:
      v5 = *(_QWORD *)(v2 + 216);
      v6 = *(_QWORD *)(v5 + 80);
      result = v5 + 72;
      v94 = result;
      v103 = v6;
      if ( result == v6 )
        return result;
      while ( 1 )
      {
        v8 = *(_QWORD *)(a2 + 32);
        v9 = v103 - 24;
        if ( !v103 )
          v9 = 0;
        if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v8) <= 6 )
        {
          v10 = sub_CB6200(a2, "\nBLOCK ", 7u);
        }
        else
        {
          *(_DWORD *)v8 = 1330397706;
          v10 = a2;
          *(_WORD *)(v8 + 4) = 19267;
          *(_BYTE *)(v8 + 6) = 32;
          *(_QWORD *)(a2 + 32) += 7LL;
        }
        v11 = *(_QWORD *)(v2 + 208);
        v12 = &v122;
        sub_E453B0(&v122, v11, v9);
        if ( !v123 )
          goto LABEL_168;
        v124(&v122, v10);
        v17 = *(_BYTE **)(v10 + 32);
        if ( (unsigned __int64)v17 >= *(_QWORD *)(v10 + 24) )
        {
          sub_CB5D20(v10, 10);
        }
        else
        {
          *(_QWORD *)(v10 + 32) = v17 + 1;
          *v17 = 10;
        }
        if ( v123 )
          v123(&v122, &v122, 3);
        v18 = *(void **)(a2 + 32);
        if ( *(_QWORD *)(a2 + 24) - (_QWORD)v18 <= 0xBu )
        {
          sub_CB6200(a2, "DEFINITIONS\n", 0xCu);
        }
        else
        {
          qmemcpy(v18, "DEFINITIONS\n", 12);
          *(_QWORD *)(a2 + 32) += 12LL;
        }
        v134 = &v136;
        v135 = 0x1000000000LL;
        sub_E45210((__int64)&v134, v9, (__int64)v18, v14, v15, v16);
        v22 = (__int64 *)v134;
        if ( &v134[(unsigned int)v135] != v134 )
          break;
LABEL_35:
        v33 = *(void **)(a2 + 32);
        if ( *(_QWORD *)(a2 + 24) - (_QWORD)v33 <= 0xBu )
        {
          sub_CB6200(a2, "TERMINATORS\n", 0xCu);
        }
        else
        {
          qmemcpy(v33, "TERMINATORS\n", 12);
          *(_QWORD *)(a2 + 32) += 12LL;
        }
        v34 = (const char *)v9;
        v131 = (__int64 *)v133;
        v132 = 0x800000000LL;
        sub_E452B0((__int64)&v131, v9, (__int64)v33, v19, v20, v21);
        v35 = *(_BYTE *)(v2 + 300);
        if ( v35 )
        {
          v36 = *(_QWORD **)(v2 + 280);
          v37 = &v36[*(unsigned int *)(v2 + 292)];
          if ( v36 == v37 )
          {
LABEL_63:
            v35 = 0;
          }
          else
          {
            while ( v9 != *v36 )
            {
              if ( v37 == ++v36 )
                goto LABEL_63;
            }
          }
        }
        else
        {
          v34 = (const char *)v9;
          v35 = sub_C8CA60(v2 + 272, v9) != 0;
        }
        v38 = v131;
        v108 = &v131[(unsigned int)v132];
        if ( v108 != v131 )
        {
          while ( 1 )
          {
            v39 = *(void **)(a2 + 32);
            v40 = *v38;
            v41 = *(_QWORD *)(a2 + 24) - (_QWORD)v39;
            if ( v35 )
            {
              if ( v41 <= 0xC )
              {
                v99 = *v38;
                sub_CB6200(a2, "  DIVERGENT: ", 0xDu);
                v40 = v99;
              }
              else
              {
                qmemcpy(v39, "  DIVERGENT: ", 13);
                *(_QWORD *)(a2 + 32) += 13LL;
              }
            }
            else if ( v41 <= 0xC )
            {
              v100 = *v38;
              sub_CB6200(a2, (unsigned __int8 *)"             ", 0xDu);
              v40 = v100;
            }
            else
            {
              memset(v39, 32, 13);
              *(_QWORD *)(a2 + 32) += 13LL;
            }
            v11 = *(_QWORD *)(v2 + 208);
            v12 = (__m128i *)v128;
            sub_E45390(v128, v11, v40);
            if ( !v129 )
              break;
            v130(v128, a2);
            v34 = (const char *)v112;
            sub_CB6200(a2, v112, 1u);
            if ( v129 )
            {
              v34 = (const char *)v128;
              v129(v128, v128, 3);
            }
            if ( v108 == ++v38 )
              goto LABEL_55;
          }
LABEL_168:
          sub_4263D6(v12, v11, v13);
        }
LABEL_55:
        v42 = *(void **)(a2 + 32);
        if ( *(_QWORD *)(a2 + 24) - (_QWORD)v42 <= 9u )
        {
          v34 = "END BLOCK\n";
          sub_CB6200(a2, "END BLOCK\n", 0xAu);
        }
        else
        {
          qmemcpy(v42, "END BLOCK\n", 10);
          *(_QWORD *)(a2 + 32) += 10LL;
        }
        if ( v131 != (__int64 *)v133 )
          _libc_free(v131, v34);
        if ( v134 != &v136 )
          _libc_free(v134, v34);
        result = *(_QWORD *)(v103 + 8);
        v103 = result;
        if ( v94 == result )
          return result;
      }
      v98 = v9;
      v23 = (__int64 *)&v134[(unsigned int)v135];
      while ( 1 )
      {
        v24 = *(_DWORD *)(v2 + 264);
        v25 = *v22;
        v26 = *(_QWORD *)(v2 + 248);
        if ( !v24 )
          goto LABEL_31;
        v27 = v24 - 1;
        v28 = (v24 - 1) & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
        v29 = *(_QWORD *)(v26 + 8LL * v28);
        if ( v25 != v29 )
          break;
LABEL_27:
        v30 = *(void **)(a2 + 32);
        if ( *(_QWORD *)(a2 + 24) - (_QWORD)v30 > 0xCu )
        {
          qmemcpy(v30, "  DIVERGENT: ", 13);
          *(_QWORD *)(a2 + 32) += 13LL;
        }
        else
        {
          v106 = *v22;
          sub_CB6200(a2, "  DIVERGENT: ", 0xDu);
          v25 = v106;
        }
LABEL_21:
        v11 = *(_QWORD *)(v2 + 208);
        v12 = (__m128i *)v125;
        sub_E45370(v125, v11, v25);
        if ( !v126 )
          goto LABEL_168;
        v127(v125, a2);
        sub_CB6200(a2, v112, 1u);
        if ( v126 )
          v126(v125, v125, 3);
        if ( v23 == ++v22 )
        {
          v9 = v98;
          goto LABEL_35;
        }
      }
      v31 = 1;
      while ( v29 != -4096 )
      {
        v28 = v27 & (v31 + v28);
        v29 = *(_QWORD *)(v26 + 8LL * v28);
        if ( v25 == v29 )
          goto LABEL_27;
        ++v31;
      }
LABEL_31:
      v32 = *(void **)(a2 + 32);
      if ( *(_QWORD *)(a2 + 24) - (_QWORD)v32 <= 0xCu )
      {
        v107 = *v22;
        sub_CB6200(a2, (unsigned __int8 *)"             ", 0xDu);
        v25 = v107;
      }
      else
      {
        memset(v32, 32, 13);
        *(_QWORD *)(a2 + 32) += 13LL;
      }
      goto LABEL_21;
    }
LABEL_92:
    v59 = *(__m128i **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v59 <= 0x1Au )
    {
      sub_CB6200(a2, "\nTEMPORAL DIVERGENCE LIST:\n", 0x1Bu);
    }
    else
    {
      v60 = _mm_load_si128((const __m128i *)&xmmword_3F8E8C0);
      qmemcpy(&v59[1], "ENCE LIST:\n", 11);
      *v59 = v60;
      *(_QWORD *)(a2 + 32) += 27LL;
    }
    v61 = *(__int64 **)v2;
    v96 = *(_QWORD *)v2 + 24LL * *(unsigned int *)(v2 + 8);
    if ( v96 != *(_QWORD *)v2 )
    {
      do
      {
        v74 = *(void **)(a2 + 32);
        v75 = v61[2];
        v105 = *v61;
        v110 = v61[1];
        if ( *(_QWORD *)(a2 + 24) - (_QWORD)v74 > 0xEu )
        {
          v62 = a2;
          qmemcpy(v74, "Value         :", 15);
          *(_QWORD *)(a2 + 32) += 15LL;
        }
        else
        {
          v102 = v61[2];
          v76 = sub_CB6200(a2, "Value         :", 0xFu);
          v75 = v102;
          v62 = v76;
        }
        v11 = *(_QWORD *)(v2 + 208);
        v12 = (__m128i *)v116;
        sub_E45370(v116, v11, v75);
        if ( !v117 )
          goto LABEL_168;
        v118(v116, v62);
        v63 = sub_CB6200(v62, v112, 1u);
        v64 = *(void **)(v63 + 32);
        v65 = v63;
        if ( *(_QWORD *)(v63 + 24) - (_QWORD)v64 <= 0xEu )
        {
          v65 = sub_CB6200(v63, "Used by       :", 0xFu);
        }
        else
        {
          qmemcpy(v64, "Used by       :", 15);
          *(_QWORD *)(v63 + 32) += 15LL;
        }
        v11 = *(_QWORD *)(v2 + 208);
        v101 = v65;
        v12 = (__m128i *)v119;
        sub_E45390(v119, v11, v110);
        if ( !v120 )
          goto LABEL_168;
        v121(v119, v101);
        v66 = sub_CB6200(v101, v112, 1u);
        v67 = *(void **)(v66 + 32);
        v68 = v66;
        if ( *(_QWORD *)(v66 + 24) - (_QWORD)v67 <= 0xEu )
        {
          v68 = sub_CB6200(v66, "Outside cycle :", 0xFu);
        }
        else
        {
          qmemcpy(v67, "Outside cycle :", 15);
          *(_QWORD *)(v66 + 32) += 15LL;
        }
        v69 = v105;
        v104 = v68;
        v135 = *(_QWORD *)(v2 + 208);
        v136 = sub_E341B0;
        v134 = (__int64 (__fastcall **)(const __m128i **, const __m128i *, int))v69;
        v137 = sub_E34BC0;
        sub_E34BC0((__int64 *)&v134, v68);
        v71 = v104;
        v72 = (__int64 *)&v134;
        v73 = *(_WORD **)(v104 + 32);
        if ( *(_QWORD *)(v104 + 24) - (_QWORD)v73 <= 1u )
        {
          sub_CB6200(v104, (unsigned __int8 *)"\n\n", 2u);
          v72 = (__int64 *)&v134;
        }
        else
        {
          *v73 = 2570;
          *(_QWORD *)(v104 + 32) += 2LL;
        }
        if ( v136 )
          v136((const __m128i **)&v134, (const __m128i *)&v134, 3);
        if ( v120 )
          v120(v119, v119, 3, v70, v71, v72);
        if ( v117 )
          v117(v116, v116, 3, v70, v71, v72);
        v61 += 3;
      }
      while ( (__int64 *)v96 != v61 );
    }
    goto LABEL_6;
  }
  if ( *(_DWORD *)(a1 + 296) != *(_DWORD *)(a1 + 292) || *(_DWORD *)(a1 + 616) != *(_DWORD *)(a1 + 612) )
    goto LABEL_3;
  result = *(_QWORD *)(a2 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - result) <= 0x12 )
    return sub_CB6200(a2, "ALL VALUES UNIFORM\n", 0x13u);
  v77 = _mm_load_si128((const __m128i *)&xmmword_3F8E880);
  *(_BYTE *)(result + 18) = 10;
  *(_WORD *)(result + 16) = 19794;
  *(__m128i *)result = v77;
  *(_QWORD *)(a2 + 32) += 19LL;
  return result;
}
