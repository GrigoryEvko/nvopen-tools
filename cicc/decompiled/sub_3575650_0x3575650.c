// Function: sub_3575650
// Address: 0x3575650
//
void __fastcall sub_3575650(__int64 a1, __int64 a2)
{
  __int64 v2; // r15
  __int64 v3; // r14
  int v4; // ecx
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // r12
  __int64 v8; // rsi
  __m128i *v9; // rdi
  __int64 v10; // rdx
  _BYTE *v11; // rax
  void *v12; // rdx
  int *v13; // rbx
  int v14; // eax
  int v15; // r12d
  __int64 v16; // rsi
  int v17; // edx
  unsigned int v18; // eax
  int v19; // edi
  void *v20; // rdx
  int v21; // r8d
  void *v22; // rdx
  void *v23; // rdx
  char v24; // r13
  _QWORD *v25; // rax
  _QWORD *v26; // rdx
  __int64 *v27; // rbx
  void *v28; // rdx
  __int64 v29; // r8
  unsigned __int64 v30; // rax
  void *v31; // rdx
  __m128i *v32; // rdx
  __m128i v33; // xmm0
  __int64 *v34; // rax
  __int64 v35; // rdx
  __int64 *v36; // r13
  __int64 v37; // rcx
  __int64 *v38; // r12
  __int64 v39; // r14
  __int64 v40; // r15
  __int64 *v41; // rax
  __int64 *v42; // r12
  __int64 *v43; // r13
  _WORD *v44; // rdx
  __int64 v45; // r10
  _BYTE *v46; // rax
  __int64 *v47; // rax
  __m128i *v48; // rdx
  __m128i v49; // xmm0
  __int64 *v50; // rbx
  __int64 v51; // r12
  __int64 v52; // rax
  void *v53; // rdx
  __int64 v54; // r8
  __int64 v55; // rax
  void *v56; // rdx
  __int64 v57; // r8
  __int64 v58; // rcx
  __int64 v59; // rcx
  __int64 v60; // r8
  __int64 (__fastcall ***v61)(const __m128i **, const __m128i *, int); // r9
  _WORD *v62; // rdx
  void *v63; // rdx
  __m128i *v64; // rax
  __m128i v65; // xmm0
  __m128i *v66; // rdx
  __m128i v67; // xmm0
  __int64 *v68; // r12
  __int64 v69; // r13
  __int64 v70; // rax
  _BYTE *v71; // rax
  _WORD *v72; // rdx
  __int64 v73; // rcx
  __int64 v74; // rax
  int *v75; // r13
  int *v76; // r12
  char v77; // bl
  _DWORD *v78; // rdx
  __int64 v79; // rbx
  _BYTE *v80; // rax
  __m128i si128; // xmm0
  __int64 v82; // [rsp+8h] [rbp-208h]
  __int64 v83; // [rsp+20h] [rbp-1F0h]
  __int64 v84; // [rsp+20h] [rbp-1F0h]
  __int64 v85; // [rsp+20h] [rbp-1F0h]
  __int64 v86; // [rsp+28h] [rbp-1E8h]
  __int64 v87; // [rsp+28h] [rbp-1E8h]
  __int64 v88; // [rsp+28h] [rbp-1E8h]
  __int64 v89; // [rsp+30h] [rbp-1E0h]
  __int64 v90; // [rsp+30h] [rbp-1E0h]
  __int64 v91; // [rsp+30h] [rbp-1E0h]
  int *v92; // [rsp+38h] [rbp-1D8h]
  __int64 *v93; // [rsp+38h] [rbp-1D8h]
  __int64 v94; // [rsp+38h] [rbp-1D8h]
  __int64 v95; // [rsp+38h] [rbp-1D8h]
  __int64 *i; // [rsp+38h] [rbp-1D8h]
  __int64 v97; // [rsp+50h] [rbp-1C0h]
  unsigned __int8 v98[16]; // [rsp+70h] [rbp-1A0h] BYREF
  _BYTE v99[16]; // [rsp+80h] [rbp-190h] BYREF
  void (__fastcall *v100)(_BYTE *, _BYTE *, __int64); // [rsp+90h] [rbp-180h]
  void (__fastcall *v101)(_BYTE *, __int64); // [rsp+98h] [rbp-178h]
  _BYTE v102[16]; // [rsp+A0h] [rbp-170h] BYREF
  void (__fastcall *v103)(_BYTE *, _BYTE *, __int64, __int64, __int64, __int64 (__fastcall ***)(const __m128i **, const __m128i *, int)); // [rsp+B0h] [rbp-160h]
  void (__fastcall *v104)(_BYTE *, __int64); // [rsp+B8h] [rbp-158h]
  _QWORD v105[2]; // [rsp+C0h] [rbp-150h] BYREF
  void (__fastcall *v106)(_QWORD *, _QWORD *, __int64, __int64, __int64, __int64 (__fastcall ***)(const __m128i **, const __m128i *, int)); // [rsp+D0h] [rbp-140h]
  void (__fastcall *v107)(_QWORD *, __int64); // [rsp+D8h] [rbp-138h]
  __m128i v108; // [rsp+E0h] [rbp-130h] BYREF
  void (__fastcall *v109)(__m128i *, __m128i *, __int64); // [rsp+F0h] [rbp-120h]
  void (__fastcall *v110)(__m128i *, __int64); // [rsp+F8h] [rbp-118h]
  _BYTE v111[16]; // [rsp+100h] [rbp-110h] BYREF
  void (__fastcall *v112)(_BYTE *, _BYTE *, __int64); // [rsp+110h] [rbp-100h]
  void (__fastcall *v113)(_BYTE *, __int64); // [rsp+118h] [rbp-F8h]
  _QWORD v114[2]; // [rsp+120h] [rbp-F0h] BYREF
  void (__fastcall *v115)(_QWORD *, _QWORD *, __int64); // [rsp+130h] [rbp-E0h]
  void (__fastcall *v116)(_QWORD *, __int64); // [rsp+138h] [rbp-D8h]
  int *v117; // [rsp+140h] [rbp-D0h] BYREF
  __int64 v118; // [rsp+148h] [rbp-C8h]
  _BYTE v119[64]; // [rsp+150h] [rbp-C0h] BYREF
  __int64 (__fastcall **v120)(const __m128i **, const __m128i *, int); // [rsp+190h] [rbp-80h] BYREF
  __int64 v121; // [rsp+198h] [rbp-78h]
  __int64 (__fastcall *v122)(const __m128i **, const __m128i *, int); // [rsp+1A0h] [rbp-70h] BYREF
  _QWORD *(__fastcall *v123)(__int64 *, __int64); // [rsp+1A8h] [rbp-68h]

  v2 = a2;
  v3 = a1;
  v4 = *(_DWORD *)(a1 + 256);
  v98[0] = 0;
  if ( v4 )
  {
    v75 = *(int **)(a1 + 248);
    v76 = &v75[*(unsigned int *)(a1 + 264)];
    if ( v75 != v76 )
    {
      if ( (unsigned int)*v75 <= 0xFFFFFFFD )
      {
LABEL_135:
        if ( v76 != v75 )
        {
          v77 = 0;
          do
          {
            if ( !sub_2EE71A0(*(_QWORD *)(v3 + 208), *v75) )
            {
              v78 = *(_DWORD **)(v2 + 32);
              if ( !v77 )
              {
                if ( *(_QWORD *)(v2 + 24) - (_QWORD)v78 <= 0x14u )
                {
                  sub_CB6200(v2, "DIVERGENT ARGUMENTS:\n", 0x15u);
                  v78 = *(_DWORD **)(v2 + 32);
                }
                else
                {
                  si128 = _mm_load_si128((const __m128i *)&xmmword_3F8E890);
                  v78[4] = 978539598;
                  *((_BYTE *)v78 + 20) = 10;
                  *(__m128i *)v78 = si128;
                  v78 = (_DWORD *)(*(_QWORD *)(v2 + 32) + 21LL);
                  *(_QWORD *)(v2 + 32) = v78;
                }
              }
              if ( *(_QWORD *)(v2 + 24) - (_QWORD)v78 <= 0xCu )
              {
                v79 = sub_CB6200(v2, "  DIVERGENT: ", 0xDu);
              }
              else
              {
                v79 = v2;
                qmemcpy(v78, "  DIVERGENT: ", 13);
                *(_QWORD *)(v2 + 32) += 13LL;
              }
              v8 = *(_QWORD *)(v3 + 208);
              v9 = (__m128i *)v99;
              sub_2EE7340((__int64)v99, v8, *v75);
              if ( !v100 )
                goto LABEL_165;
              v101(v99, v79);
              v80 = *(_BYTE **)(v79 + 32);
              if ( (unsigned __int64)v80 >= *(_QWORD *)(v79 + 24) )
              {
                sub_CB5D20(v79, 10);
              }
              else
              {
                *(_QWORD *)(v79 + 32) = v80 + 1;
                *v80 = 10;
              }
              if ( v100 )
                v100(v99, v99, 3);
              v77 = 1;
            }
            if ( ++v75 == v76 )
              break;
            while ( (unsigned int)*v75 > 0xFFFFFFFD )
            {
              if ( v76 == ++v75 )
                goto LABEL_3;
            }
          }
          while ( v76 != v75 );
        }
      }
      else
      {
        while ( v76 != ++v75 )
        {
          if ( (unsigned int)*v75 <= 0xFFFFFFFD )
            goto LABEL_135;
        }
      }
    }
LABEL_3:
    if ( *(_DWORD *)(v3 + 760) )
    {
      v66 = *(__m128i **)(v2 + 32);
      if ( *(_QWORD *)(v2 + 24) - (_QWORD)v66 <= 0x1Au )
      {
        sub_CB6200(v2, "CYCLES ASSSUMED DIVERGENT:\n", 0x1Bu);
      }
      else
      {
        v67 = _mm_load_si128((const __m128i *)&xmmword_3F8E8A0);
        qmemcpy(&v66[1], "DIVERGENT:\n", 11);
        *v66 = v67;
        *(_QWORD *)(v2 + 32) += 27LL;
      }
      v68 = *(__int64 **)(v3 + 752);
      for ( i = &v68[*(unsigned int *)(v3 + 760)]; i != v68; ++v68 )
      {
        v72 = *(_WORD **)(v2 + 32);
        v73 = *v68;
        if ( *(_QWORD *)(v2 + 24) - (_QWORD)v72 > 1u )
        {
          v69 = v2;
          *v72 = 8224;
          *(_QWORD *)(v2 + 32) += 2LL;
        }
        else
        {
          v85 = *v68;
          v74 = sub_CB6200(v2, (unsigned __int8 *)"  ", 2u);
          v73 = v85;
          v69 = v74;
        }
        v70 = *(_QWORD *)(v3 + 208);
        v120 = (__int64 (__fastcall **)(const __m128i **, const __m128i *, int))v73;
        v121 = v70;
        v122 = sub_2E5D7C0;
        v123 = sub_2E5F810;
        sub_2E5F810((__int64 *)&v120, v69);
        v71 = *(_BYTE **)(v69 + 32);
        if ( (unsigned __int64)v71 >= *(_QWORD *)(v69 + 24) )
        {
          sub_CB5D20(v69, 10);
        }
        else
        {
          *(_QWORD *)(v69 + 32) = v71 + 1;
          *v71 = 10;
        }
        if ( v122 )
          v122((const __m128i **)&v120, (const __m128i *)&v120, 3);
      }
    }
    if ( *(_DWORD *)(v3 + 612) != *(_DWORD *)(v3 + 616) )
    {
      v32 = *(__m128i **)(v2 + 32);
      if ( *(_QWORD *)(v2 + 24) - (_QWORD)v32 <= 0x1Bu )
      {
        sub_CB6200(v2, "CYCLES WITH DIVERGENT EXIT:\n", 0x1Cu);
      }
      else
      {
        v33 = _mm_load_si128((const __m128i *)&xmmword_3F8E8B0);
        qmemcpy(&v32[1], "RGENT EXIT:\n", 12);
        *v32 = v33;
        *(_QWORD *)(v2 + 32) += 28LL;
      }
      v34 = *(__int64 **)(v3 + 600);
      v35 = *(_BYTE *)(v3 + 620) ? *(unsigned int *)(v3 + 612) : *(unsigned int *)(v3 + 608);
      v36 = &v34[v35];
      if ( v34 != v36 )
      {
        v37 = *v34;
        v38 = *(__int64 **)(v3 + 600);
        if ( (unsigned __int64)*v34 < 0xFFFFFFFFFFFFFFFELL )
        {
LABEL_77:
          if ( v36 != v34 )
          {
            v83 = v3;
            v39 = v2;
            v40 = v37;
            v41 = v38;
            v42 = v36;
            v43 = v41;
            do
            {
              v44 = *(_WORD **)(v39 + 32);
              if ( *(_QWORD *)(v39 + 24) - (_QWORD)v44 <= 1u )
              {
                v45 = sub_CB6200(v39, (unsigned __int8 *)"  ", 2u);
              }
              else
              {
                v45 = v39;
                *v44 = 8224;
                *(_QWORD *)(v39 + 32) += 2LL;
              }
              v94 = v45;
              v120 = (__int64 (__fastcall **)(const __m128i **, const __m128i *, int))v40;
              v121 = *(_QWORD *)(v83 + 208);
              v122 = sub_2E5D7C0;
              v123 = sub_2E5F810;
              sub_2E5F810((__int64 *)&v120, v45);
              v46 = *(_BYTE **)(v94 + 32);
              if ( (unsigned __int64)v46 >= *(_QWORD *)(v94 + 24) )
              {
                sub_CB5D20(v94, 10);
              }
              else
              {
                *(_QWORD *)(v94 + 32) = v46 + 1;
                *v46 = 10;
              }
              if ( v122 )
                v122((const __m128i **)&v120, (const __m128i *)&v120, 3);
              v47 = v43 + 1;
              if ( v43 + 1 == v42 )
                break;
              while ( 1 )
              {
                v40 = *v47;
                v43 = v47;
                if ( (unsigned __int64)*v47 < 0xFFFFFFFFFFFFFFFELL )
                  break;
                if ( v42 == ++v47 )
                  goto LABEL_88;
              }
            }
            while ( v42 != v47 );
LABEL_88:
            v2 = v39;
            v3 = v83;
            if ( !*(_DWORD *)(v83 + 8) )
              goto LABEL_6;
            goto LABEL_89;
          }
        }
        else
        {
          while ( v36 != ++v34 )
          {
            v37 = *v34;
            v38 = v34;
            if ( (unsigned __int64)*v34 < 0xFFFFFFFFFFFFFFFELL )
              goto LABEL_77;
          }
        }
      }
    }
    if ( !*(_DWORD *)(v3 + 8) )
    {
LABEL_6:
      v5 = *(_QWORD *)(v3 + 216);
      v82 = v5 + 320;
      v89 = *(_QWORD *)(v5 + 328);
      if ( v5 + 320 == v89 )
        return;
      while ( 1 )
      {
        v6 = *(_QWORD *)(v2 + 32);
        if ( (unsigned __int64)(*(_QWORD *)(v2 + 24) - v6) <= 6 )
        {
          v7 = sub_CB6200(v2, "\nBLOCK ", 7u);
        }
        else
        {
          *(_DWORD *)v6 = 1330397706;
          v7 = v2;
          *(_WORD *)(v6 + 4) = 19267;
          *(_BYTE *)(v6 + 6) = 32;
          *(_QWORD *)(v2 + 32) += 7LL;
        }
        v8 = *(_QWORD *)(v3 + 208);
        v9 = &v108;
        sub_2EE72D0(&v108, v8, v89);
        if ( !v109 )
          goto LABEL_165;
        v110(&v108, v7);
        v11 = *(_BYTE **)(v7 + 32);
        if ( (unsigned __int64)v11 >= *(_QWORD *)(v7 + 24) )
        {
          sub_CB5D20(v7, 10);
        }
        else
        {
          *(_QWORD *)(v7 + 32) = v11 + 1;
          *v11 = 10;
        }
        if ( v109 )
          v109(&v108, &v108, 3);
        v12 = *(void **)(v2 + 32);
        if ( *(_QWORD *)(v2 + 24) - (_QWORD)v12 <= 0xBu )
        {
          sub_CB6200(v2, "DEFINITIONS\n", 0xCu);
        }
        else
        {
          qmemcpy(v12, "DEFINITIONS\n", 12);
          *(_QWORD *)(v2 + 32) += 12LL;
        }
        v117 = (int *)v119;
        v118 = 0x1000000000LL;
        sub_2EE6FC0((__int64)&v117, v89);
        v13 = v117;
        if ( &v117[(unsigned int)v118] != v117 )
          break;
LABEL_32:
        v23 = *(void **)(v2 + 32);
        if ( *(_QWORD *)(v2 + 24) - (_QWORD)v23 <= 0xBu )
        {
          sub_CB6200(v2, "TERMINATORS\n", 0xCu);
        }
        else
        {
          qmemcpy(v23, "TERMINATORS\n", 12);
          *(_QWORD *)(v2 + 32) += 12LL;
        }
        v120 = &v122;
        v121 = 0x800000000LL;
        sub_2EE70F0((__int64)&v120, v89);
        v24 = *(_BYTE *)(v3 + 300);
        if ( v24 )
        {
          v25 = *(_QWORD **)(v3 + 280);
          v26 = &v25[*(unsigned int *)(v3 + 292)];
          if ( v25 == v26 )
          {
LABEL_60:
            v24 = 0;
          }
          else
          {
            while ( v89 != *v25 )
            {
              if ( v26 == ++v25 )
                goto LABEL_60;
            }
          }
        }
        else
        {
          v24 = sub_C8CA60(v3 + 272, v89) != 0;
        }
        v27 = (__int64 *)v120;
        v93 = (__int64 *)&v120[(unsigned int)v121];
        if ( v93 != (__int64 *)v120 )
        {
          while ( 1 )
          {
            v28 = *(void **)(v2 + 32);
            v29 = *v27;
            v30 = *(_QWORD *)(v2 + 24) - (_QWORD)v28;
            if ( v24 )
            {
              if ( v30 <= 0xC )
              {
                v86 = *v27;
                sub_CB6200(v2, "  DIVERGENT: ", 0xDu);
                v29 = v86;
              }
              else
              {
                qmemcpy(v28, "  DIVERGENT: ", 13);
                *(_QWORD *)(v2 + 32) += 13LL;
              }
            }
            else if ( v30 <= 0xC )
            {
              v87 = *v27;
              sub_CB6200(v2, (unsigned __int8 *)"             ", 0xDu);
              v29 = v87;
            }
            else
            {
              memset(v28, 32, 13);
              *(_QWORD *)(v2 + 32) += 13LL;
            }
            v8 = *(_QWORD *)(v3 + 208);
            v9 = (__m128i *)v114;
            sub_2EE7320(v114, v8, v29);
            if ( !v115 )
              break;
            v116(v114, v2);
            sub_CB6200(v2, v98, 0);
            if ( v115 )
              v115(v114, v114, 3);
            if ( v93 == ++v27 )
              goto LABEL_52;
          }
LABEL_165:
          sub_4263D6(v9, v8, v10);
        }
LABEL_52:
        v31 = *(void **)(v2 + 32);
        if ( *(_QWORD *)(v2 + 24) - (_QWORD)v31 <= 9u )
        {
          sub_CB6200(v2, "END BLOCK\n", 0xAu);
        }
        else
        {
          qmemcpy(v31, "END BLOCK\n", 10);
          *(_QWORD *)(v2 + 32) += 10LL;
        }
        if ( v120 != &v122 )
          _libc_free((unsigned __int64)v120);
        if ( v117 != (int *)v119 )
          _libc_free((unsigned __int64)v117);
        v89 = *(_QWORD *)(v89 + 8);
        if ( v82 == v89 )
          return;
      }
      v92 = &v117[(unsigned int)v118];
      while ( 1 )
      {
        v14 = *(_DWORD *)(v3 + 264);
        v15 = *v13;
        v16 = *(_QWORD *)(v3 + 248);
        if ( !v14 )
          goto LABEL_29;
        v17 = v14 - 1;
        v18 = (v14 - 1) & (37 * v15);
        v19 = *(_DWORD *)(v16 + 4LL * v18);
        if ( v15 != v19 )
          break;
LABEL_25:
        v20 = *(void **)(v2 + 32);
        if ( *(_QWORD *)(v2 + 24) - (_QWORD)v20 > 0xCu )
        {
          qmemcpy(v20, "  DIVERGENT: ", 13);
          *(_QWORD *)(v2 + 32) += 13LL;
        }
        else
        {
          sub_CB6200(v2, "  DIVERGENT: ", 0xDu);
        }
LABEL_19:
        v8 = *(_QWORD *)(v3 + 208);
        v9 = (__m128i *)v111;
        sub_2EE7340((__int64)v111, v8, v15);
        if ( !v112 )
          goto LABEL_165;
        v113(v111, v2);
        sub_CB6200(v2, v98, 0);
        if ( v112 )
          v112(v111, v111, 3);
        if ( v92 == ++v13 )
          goto LABEL_32;
      }
      v21 = 1;
      while ( v19 != -1 )
      {
        v18 = v17 & (v21 + v18);
        v19 = *(_DWORD *)(v16 + 4LL * v18);
        if ( v15 == v19 )
          goto LABEL_25;
        ++v21;
      }
LABEL_29:
      v22 = *(void **)(v2 + 32);
      if ( *(_QWORD *)(v2 + 24) - (_QWORD)v22 <= 0xCu )
      {
        sub_CB6200(v2, (unsigned __int8 *)"             ", 0xDu);
      }
      else
      {
        memset(v22, 32, 13);
        *(_QWORD *)(v2 + 32) += 13LL;
      }
      goto LABEL_19;
    }
LABEL_89:
    v48 = *(__m128i **)(v2 + 32);
    if ( *(_QWORD *)(v2 + 24) - (_QWORD)v48 <= 0x1Au )
    {
      sub_CB6200(v2, "\nTEMPORAL DIVERGENCE LIST:\n", 0x1Bu);
    }
    else
    {
      v49 = _mm_load_si128((const __m128i *)&xmmword_3F8E8C0);
      qmemcpy(&v48[1], "ENCE LIST:\n", 11);
      *v48 = v49;
      *(_QWORD *)(v2 + 32) += 27LL;
    }
    v50 = *(__int64 **)v3;
    v84 = *(_QWORD *)v3 + 24LL * *(unsigned int *)(v3 + 8);
    if ( v84 != *(_QWORD *)v3 )
    {
      do
      {
        v63 = *(void **)(v2 + 32);
        v97 = v50[2];
        v91 = *v50;
        v95 = v50[1];
        if ( *(_QWORD *)(v2 + 24) - (_QWORD)v63 > 0xEu )
        {
          v51 = v2;
          qmemcpy(v63, "Value         :", 15);
          *(_QWORD *)(v2 + 32) += 15LL;
        }
        else
        {
          v51 = sub_CB6200(v2, "Value         :", 0xFu);
        }
        v8 = *(_QWORD *)(v3 + 208);
        v9 = (__m128i *)v102;
        sub_2EE7340((__int64)v102, v8, v97);
        if ( !v103 )
          goto LABEL_165;
        v104(v102, v51);
        v52 = sub_CB6200(v51, v98, 0);
        v53 = *(void **)(v52 + 32);
        v54 = v52;
        if ( *(_QWORD *)(v52 + 24) - (_QWORD)v53 <= 0xEu )
        {
          v54 = sub_CB6200(v52, "Used by       :", 0xFu);
        }
        else
        {
          qmemcpy(v53, "Used by       :", 15);
          *(_QWORD *)(v52 + 32) += 15LL;
        }
        v8 = *(_QWORD *)(v3 + 208);
        v88 = v54;
        v9 = (__m128i *)v105;
        sub_2EE7320(v105, v8, v95);
        if ( !v106 )
          goto LABEL_165;
        v107(v105, v88);
        v55 = sub_CB6200(v88, v98, 0);
        v56 = *(void **)(v55 + 32);
        v57 = v55;
        if ( *(_QWORD *)(v55 + 24) - (_QWORD)v56 <= 0xEu )
        {
          v57 = sub_CB6200(v55, "Outside cycle :", 0xFu);
        }
        else
        {
          qmemcpy(v56, "Outside cycle :", 15);
          *(_QWORD *)(v55 + 32) += 15LL;
        }
        v58 = v91;
        v90 = v57;
        v121 = *(_QWORD *)(v3 + 208);
        v122 = sub_2E5D7C0;
        v120 = (__int64 (__fastcall **)(const __m128i **, const __m128i *, int))v58;
        v123 = sub_2E5F810;
        sub_2E5F810((__int64 *)&v120, v57);
        v60 = v90;
        v61 = &v120;
        v62 = *(_WORD **)(v90 + 32);
        if ( *(_QWORD *)(v90 + 24) - (_QWORD)v62 <= 1u )
        {
          sub_CB6200(v90, (unsigned __int8 *)"\n\n", 2u);
          v61 = &v120;
        }
        else
        {
          *v62 = 2570;
          *(_QWORD *)(v90 + 32) += 2LL;
        }
        if ( v122 )
          v122((const __m128i **)&v120, (const __m128i *)&v120, 3);
        if ( v106 )
          v106(v105, v105, 3, v59, v60, v61);
        if ( v103 )
          v103(v102, v102, 3, v59, v60, v61);
        v50 += 3;
      }
      while ( (__int64 *)v84 != v50 );
    }
    goto LABEL_6;
  }
  if ( *(_DWORD *)(a1 + 296) != *(_DWORD *)(a1 + 292) || *(_DWORD *)(a1 + 616) != *(_DWORD *)(a1 + 612) )
    goto LABEL_3;
  v64 = *(__m128i **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v64 <= 0x12u )
  {
    sub_CB6200(a2, "ALL VALUES UNIFORM\n", 0x13u);
  }
  else
  {
    v65 = _mm_load_si128((const __m128i *)&xmmword_3F8E880);
    v64[1].m128i_i8[2] = 10;
    v64[1].m128i_i16[0] = 19794;
    *v64 = v65;
    *(_QWORD *)(a2 + 32) += 19LL;
  }
}
