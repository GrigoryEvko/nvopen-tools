// Function: sub_23C9090
// Address: 0x23c9090
//
__int64 __fastcall sub_23C9090(__int64 a1, __int64 a2, _QWORD *a3)
{
  __int64 v4; // rdx
  char v5; // al
  size_t v6; // rbx
  __int32 v7; // r15d
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rcx
  size_t v10; // rax
  size_t v11; // rax
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rdx
  unsigned __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rcx
  size_t v17; // r12
  void *v18; // rcx
  const void *v19; // r14
  size_t v20; // rbx
  int v21; // eax
  __int64 v22; // rcx
  __int64 v23; // rax
  __int64 v24; // r8
  __int64 *v25; // r14
  int v26; // eax
  __int64 v27; // rdx
  __int64 v28; // rax
  __int64 i; // rsi
  unsigned __int64 *v30; // r12
  const char *v31; // rax
  __int64 *v32; // rsi
  __int64 *v33; // rax
  __int64 *v34; // rdi
  __int64 v35; // rcx
  __int64 v36; // rdi
  unsigned __int64 v37; // rax
  unsigned __int64 v38; // rax
  __int64 *v39; // rsi
  _QWORD *v40; // rdi
  __int64 v41; // rcx
  __int64 v42; // rdx
  __int64 result; // rax
  size_t v44; // rcx
  __int64 *v45; // rdi
  size_t v46; // rcx
  __int64 v47; // rdx
  __int64 v48; // rsi
  __int64 v49; // rax
  unsigned __int64 v50; // rax
  _BYTE *v51; // rdi
  size_t v52; // rcx
  __int64 v53; // rdx
  __m128i v54; // xmm1
  _QWORD *v55; // r15
  _BYTE *v56; // rdi
  _QWORD *v57; // rax
  size_t v58; // rsi
  __int64 v59; // rcx
  __int64 v60; // rdi
  __int64 v61; // rax
  __int64 v62; // r8
  unsigned int v63; // r9d
  __int64 *v64; // rcx
  __int64 v65; // rbx
  __int64 *v66; // rax
  __int64 v67; // rax
  unsigned int v68; // r9d
  _QWORD *v69; // r10
  _QWORD *v70; // rcx
  __int64 *v71; // rax
  size_t v72; // rdx
  size_t v73; // rdx
  size_t v74; // rdx
  size_t v75; // rdx
  size_t v76; // rdx
  _QWORD *v77; // [rsp+0h] [rbp-2C0h]
  __int64 *v78; // [rsp+8h] [rbp-2B8h]
  _QWORD *v79; // [rsp+8h] [rbp-2B8h]
  void *v80; // [rsp+10h] [rbp-2B0h]
  size_t v81; // [rsp+18h] [rbp-2A8h]
  __int64 v82; // [rsp+20h] [rbp-2A0h]
  char *v83; // [rsp+28h] [rbp-298h]
  _QWORD *v84; // [rsp+38h] [rbp-288h]
  unsigned __int64 v85; // [rsp+48h] [rbp-278h]
  unsigned int v86; // [rsp+50h] [rbp-270h]
  unsigned int v87; // [rsp+50h] [rbp-270h]
  char v88; // [rsp+54h] [rbp-26Ch]
  __int32 v89; // [rsp+58h] [rbp-268h]
  __int64 v90; // [rsp+60h] [rbp-260h] BYREF
  unsigned __int64 v91; // [rsp+68h] [rbp-258h] BYREF
  __m128i v92; // [rsp+70h] [rbp-250h] BYREF
  size_t v93[2]; // [rsp+80h] [rbp-240h]
  __m128i v94; // [rsp+90h] [rbp-230h] BYREF
  __m128i v95; // [rsp+A0h] [rbp-220h]
  unsigned __int64 v96[2]; // [rsp+C0h] [rbp-200h] BYREF
  __int64 v97; // [rsp+D0h] [rbp-1F0h] BYREF
  __int64 *v98; // [rsp+E0h] [rbp-1E0h] BYREF
  size_t v99; // [rsp+E8h] [rbp-1D8h]
  _QWORD v100[2]; // [rsp+F0h] [rbp-1D0h] BYREF
  _QWORD v101[4]; // [rsp+100h] [rbp-1C0h] BYREF
  __int16 v102; // [rsp+120h] [rbp-1A0h]
  _QWORD v103[4]; // [rsp+130h] [rbp-190h] BYREF
  __int16 v104; // [rsp+150h] [rbp-170h]
  _QWORD *v105; // [rsp+160h] [rbp-160h] BYREF
  size_t v106; // [rsp+168h] [rbp-158h]
  _QWORD v107[2]; // [rsp+170h] [rbp-150h] BYREF
  __int16 v108; // [rsp+180h] [rbp-140h]
  __int64 *v109; // [rsp+190h] [rbp-130h] BYREF
  size_t v110; // [rsp+198h] [rbp-128h]
  _QWORD v111[2]; // [rsp+1A0h] [rbp-120h] BYREF
  __int16 v112; // [rsp+1B0h] [rbp-110h]
  _QWORD v113[2]; // [rsp+1C0h] [rbp-100h] BYREF
  char *v114; // [rsp+1D0h] [rbp-F0h]
  size_t v115; // [rsp+1D8h] [rbp-E8h]
  __int16 v116; // [rsp+1E0h] [rbp-E0h]
  __int64 v117[2]; // [rsp+1F0h] [rbp-D0h] BYREF
  __m128i v118; // [rsp+200h] [rbp-C0h]
  __int16 v119; // [rsp+210h] [rbp-B0h]
  __int64 v120; // [rsp+220h] [rbp-A0h] BYREF
  size_t v121; // [rsp+228h] [rbp-98h]
  _QWORD v122[2]; // [rsp+230h] [rbp-90h] BYREF
  __int16 v123; // [rsp+240h] [rbp-80h]
  _QWORD *v124; // [rsp+250h] [rbp-70h] BYREF
  size_t n; // [rsp+258h] [rbp-68h]
  _QWORD src[2]; // [rsp+260h] [rbp-60h] BYREF
  char v127; // [rsp+270h] [rbp-50h]
  __int32 v128; // [rsp+27Ch] [rbp-44h]
  __int64 v129; // [rsp+280h] [rbp-40h] BYREF
  unsigned __int64 v130; // [rsp+288h] [rbp-38h]

  sub_23C8E10((__int64)&v124, a1, "*", 1u, 1, 1);
  if ( (n & 1) == 0 )
  {
    v117[0] = 1;
    v85 = (unsigned __int64)v124;
LABEL_3:
    v4 = *(_QWORD *)(a2 + 8);
    v88 = 1;
    if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v4) > 0x16 )
    {
      if ( *(_QWORD *)v4 ^ 0x6169636570732123LL | *(_QWORD *)(v4 + 8) ^ 0x6C2D657361632D6CLL
        || *(_DWORD *)(v4 + 16) != 762606441
        || *(_WORD *)(v4 + 20) != 12662
        || (v5 = 0, *(_BYTE *)(v4 + 22) != 10) )
      {
        v5 = 1;
      }
      v88 = v5;
    }
    sub_C7C840((__int64)&v124, a2, 1, 35);
    if ( !v127 )
      return 1;
    v84 = a3;
    while ( 1 )
    {
      v6 = 0;
      v7 = v128;
      v8 = sub_C935B0(&v129, byte_3F15413, 6, 0);
      v9 = v130;
      if ( v8 < v130 )
      {
        v9 = v8;
        v6 = v130 - v8;
      }
      v120 = v129 + v9;
      v121 = v6;
      v10 = sub_C93740(&v120, byte_3F15413, 6, 0xFFFFFFFFFFFFFFFFLL) + 1;
      v92.m128i_i64[0] = v120;
      if ( v10 > v121 )
        v10 = v121;
      v11 = v121 - v6 + v10;
      if ( v11 > v121 )
        v11 = v121;
      v92.m128i_i64[1] = v11;
      if ( !v11 )
        goto LABEL_64;
      if ( *(_BYTE *)v120 == 91 )
      {
        if ( *(_BYTE *)(v120 + v11 - 1) != 93 )
        {
          v113[0] = "malformed section header on line ";
          v117[0] = (__int64)v113;
          LODWORD(v114) = v7;
          v119 = 770;
          v122[0] = v120;
          v118.m128i_i64[0] = (__int64)": ";
          v122[1] = v11;
          v116 = 2307;
          v120 = (__int64)v117;
          v123 = 1282;
          sub_CA0F50((__int64 *)&v109, (void **)&v120);
          v45 = (__int64 *)*v84;
          if ( v109 == v111 )
          {
            v73 = v110;
            if ( v110 )
            {
              if ( v110 == 1 )
                *(_BYTE *)v45 = v111[0];
              else
                memcpy(v45, v111, v110);
              v73 = v110;
              v45 = (__int64 *)*v84;
            }
            v84[1] = v73;
            *((_BYTE *)v45 + v73) = 0;
            v45 = v109;
            goto LABEL_71;
          }
          v46 = v110;
          v47 = v111[0];
          if ( v45 == v84 + 2 )
          {
            *v84 = v109;
            v84[1] = v46;
            v84[2] = v47;
          }
          else
          {
            v48 = v84[2];
            *v84 = v109;
            v84[1] = v46;
            v84[2] = v47;
            if ( v45 )
            {
              v109 = v45;
              v111[0] = v48;
              goto LABEL_71;
            }
          }
          v109 = v111;
          v45 = v111;
LABEL_71:
          v110 = 0;
          *(_BYTE *)v45 = 0;
          if ( v109 != v111 )
            j_j___libc_free_0((unsigned __int64)v109);
          return 0;
        }
        v44 = v11 - 2;
        if ( v11 - 2 > v11 - 1 )
          v44 = 0;
        sub_23C8E10((__int64)&v120, a1, (_BYTE *)(v120 + 1), v44, v7, v88);
        if ( (v121 & 1) != 0 )
        {
          LOBYTE(v121) = v121 & 0xFD;
          v49 = v120;
          v120 = 0;
          v113[0] = v49 | 1;
          v50 = v49 & 0xFFFFFFFFFFFFFFFELL;
          if ( v50 )
          {
            v30 = (unsigned __int64 *)v117;
            v32 = v117;
            v117[0] = v50 | 1;
            v113[0] = 0;
            sub_C64870((__int64)&v120, v117);
            v51 = (_BYTE *)*v84;
            if ( (_QWORD *)v120 == v122 )
            {
              v75 = v121;
              if ( v121 )
              {
                if ( v121 == 1 )
                {
                  *v51 = v122[0];
                }
                else
                {
                  v32 = v122;
                  memcpy(v51, v122, v121);
                }
                v75 = v121;
                v51 = (_BYTE *)*v84;
              }
              v84[1] = v75;
              v51[v75] = 0;
              v51 = (_BYTE *)v120;
              goto LABEL_79;
            }
            v32 = v84 + 2;
            v52 = v121;
            v53 = v122[0];
            if ( v51 == (_BYTE *)(v84 + 2) )
            {
              *v84 = v120;
              v84[1] = v52;
              v84[2] = v53;
            }
            else
            {
              v32 = (__int64 *)v84[2];
              *v84 = v120;
              v84[1] = v52;
              v84[2] = v53;
              if ( v51 )
              {
                v120 = (__int64)v51;
                v122[0] = v32;
LABEL_79:
                v121 = 0;
                *v51 = 0;
                if ( (_QWORD *)v120 != v122 )
                {
                  v32 = (__int64 *)(v122[0] + 1LL);
                  j_j___libc_free_0(v120);
                }
                if ( (v117[0] & 1) == 0 && (v117[0] & 0xFFFFFFFFFFFFFFFELL) == 0 )
                {
                  if ( (v113[0] & 1) != 0 || (v113[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
                    sub_C63C30(v113, (__int64)v32);
                  return 0;
                }
LABEL_145:
                sub_C63C30(v30, (__int64)v32);
              }
            }
            v120 = (__int64)v122;
            v51 = v122;
            goto LABEL_79;
          }
        }
        else
        {
          v113[0] = 1;
          v85 = v120;
        }
        goto LABEL_64;
      }
      v12 = sub_C931B0(v92.m128i_i64, ":", 1u, 0);
      if ( v12 == -1 )
      {
        v54 = _mm_loadu_si128(&v92);
        v89 = v7;
        v94 = 0u;
        v55 = v84;
        *(__m128i *)v93 = v54;
        goto LABEL_87;
      }
      v13 = v12 + 1;
      if ( v12 + 1 > v92.m128i_i64[1] )
      {
        v93[0] = v92.m128i_i64[0];
        if ( v12 > v92.m128i_i64[1] )
          v12 = v92.m128i_u64[1];
        v89 = v7;
        v55 = v84;
        v94 = (__m128i)(unsigned __int64)(v92.m128i_i64[1] + v92.m128i_i64[0]);
        v93[1] = v12;
LABEL_87:
        v109 = (__int64 *)"malformed line ";
        v119 = 1282;
        LODWORD(v111[0]) = v89;
        v113[0] = &v109;
        v114 = ": '";
        v117[0] = (__int64)v113;
        v112 = 2307;
        v118 = v92;
        v116 = 770;
        v120 = (__int64)v117;
        v122[0] = "'";
        v123 = 770;
        sub_CA0F50((__int64 *)&v105, (void **)&v120);
        v56 = (_BYTE *)*v55;
        v57 = (_QWORD *)*v55;
        if ( v105 == v107 )
        {
          v74 = v106;
          if ( v106 )
          {
            if ( v106 == 1 )
              *v56 = v107[0];
            else
              memcpy(v56, v107, v106);
            v56 = (_BYTE *)*v55;
            v74 = v106;
          }
          v55[1] = v74;
          v56[v74] = 0;
          v57 = v105;
        }
        else
        {
          v58 = v106;
          v59 = v107[0];
          if ( v57 == v55 + 2 )
          {
            *v55 = v105;
            v55[1] = v58;
            v55[2] = v59;
          }
          else
          {
            v60 = v55[2];
            *v55 = v105;
            v55[1] = v58;
            v55[2] = v59;
            if ( v57 )
            {
              v105 = v57;
              v107[0] = v60;
              goto LABEL_91;
            }
          }
          v105 = v107;
          v57 = v107;
        }
LABEL_91:
        v106 = 0;
        *(_BYTE *)v57 = 0;
        if ( v105 != v107 )
        {
          j_j___libc_free_0((unsigned __int64)v105);
          return 0;
        }
        return 0;
      }
      v93[0] = v92.m128i_i64[0];
      if ( v12 > v92.m128i_i64[1] )
        v12 = v92.m128i_u64[1];
      v94.m128i_i64[1] = v92.m128i_i64[1] - v13;
      v94.m128i_i64[0] = v13 + v92.m128i_i64[0];
      v93[1] = v12;
      if ( v92.m128i_i64[1] == v13 )
      {
        v89 = v7;
        v55 = v84;
        goto LABEL_87;
      }
      v14 = sub_C931B0(v94.m128i_i64, "=", 1u, 0);
      if ( v14 == -1 )
      {
        v17 = 0;
        v80 = 0;
        v83 = (char *)v94.m128i_i64[0];
        v95 = _mm_loadu_si128(&v94);
        v81 = v94.m128i_u64[1];
      }
      else
      {
        v15 = v94.m128i_i64[1];
        v16 = v14 + 1;
        v83 = (char *)v94.m128i_i64[0];
        if ( v14 + 1 > v94.m128i_i64[1] )
        {
          v16 = v94.m128i_i64[1];
          v17 = 0;
        }
        else
        {
          v17 = v94.m128i_i64[1] - v16;
        }
        v18 = (void *)(v94.m128i_i64[0] + v16);
        if ( v14 <= v94.m128i_i64[1] )
          v15 = v14;
        v80 = v18;
        v81 = v15;
      }
      v19 = (const void *)v93[0];
      v20 = v93[1];
      v21 = sub_C92610();
      v22 = (unsigned int)sub_C92740(v85 + 8, v19, v20, v21);
      v23 = *(_QWORD *)(v85 + 8);
      v24 = *(_QWORD *)(v23 + 8 * v22);
      if ( v24 )
      {
        if ( v24 != -8 )
          goto LABEL_30;
        --*(_DWORD *)(v85 + 24);
      }
      v79 = (_QWORD *)(v23 + 8 * v22);
      v87 = v22;
      v67 = sub_C7D670(v20 + 33, 8);
      v68 = v87;
      v69 = v79;
      v70 = (_QWORD *)v67;
      if ( v20 )
      {
        v77 = (_QWORD *)v67;
        memcpy((void *)(v67 + 32), v19, v20);
        v68 = v87;
        v69 = v79;
        v70 = v77;
      }
      *((_BYTE *)v70 + v20 + 32) = 0;
      *v70 = v20;
      v70[1] = 0;
      v70[2] = 0;
      v70[3] = 0x3800000000LL;
      *v69 = v70;
      ++*(_DWORD *)(v85 + 20);
      v71 = (__int64 *)(*(_QWORD *)(v85 + 8) + 8LL * (unsigned int)sub_C929D0((__int64 *)(v85 + 8), v68));
      v24 = *v71;
      if ( *v71 == -8 || !v24 )
      {
        do
        {
          do
          {
            v24 = v71[1];
            ++v71;
          }
          while ( !v24 );
        }
        while ( v24 == -8 );
      }
LABEL_30:
      v25 = (__int64 *)(v24 + 8);
      v82 = v24;
      v26 = sub_C92610();
      v27 = (unsigned int)sub_C92740((__int64)v25, v80, v17, v26);
      v28 = *(_QWORD *)(v82 + 8);
      i = *(_QWORD *)(v28 + 8 * v27);
      if ( i )
      {
        if ( i != -8 )
          goto LABEL_32;
        --*(_DWORD *)(v82 + 24);
      }
      v78 = (__int64 *)(v28 + 8 * v27);
      v86 = v27;
      v61 = sub_C7D670(v17 + 57, 8);
      v62 = v82;
      v63 = v86;
      v64 = v78;
      v65 = v61;
      if ( v17 )
      {
        memcpy((void *)(v61 + 56), v80, v17);
        v62 = v82;
        v63 = v86;
        v64 = v78;
      }
      *(_BYTE *)(v65 + v17 + 56) = 0;
      *(_OWORD *)(v65 + 24) = 0;
      *(_QWORD *)v65 = v17;
      *(_BYTE *)(v65 + 28) = 88;
      *(_OWORD *)(v65 + 8) = 0;
      *(_OWORD *)(v65 + 40) = 0;
      *v64 = v65;
      ++*(_DWORD *)(v62 + 20);
      v66 = (__int64 *)(*(_QWORD *)(v62 + 8) + 8LL * (unsigned int)sub_C929D0(v25, v63));
      for ( i = *v66; !i; ++v66 )
LABEL_101:
        i = v66[1];
      if ( i == -8 )
        goto LABEL_101;
LABEL_32:
      v95.m128i_i64[1] = v81;
      v95.m128i_i64[0] = (__int64)v83;
      sub_23C7DD0(&v90, i + 8, v83, v81, v7, v88 & 1);
      if ( (v90 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      {
        v30 = &v91;
        v91 = v90 & 0xFFFFFFFFFFFFFFFELL | 1;
        v90 = 0;
        sub_C64870((__int64)v96, (__int64 *)&v91);
        v31 = "glob";
        if ( !v88 )
          v31 = "regex";
        v112 = 770;
        v32 = &v120;
        v101[2] = v31;
        v102 = 771;
        v103[0] = v101;
        v105 = v103;
        v101[0] = "malformed ";
        v109 = (__int64 *)&v105;
        v103[2] = " in line ";
        v113[0] = &v109;
        LODWORD(v107[0]) = v7;
        v115 = v81;
        v111[0] = ": '";
        v117[0] = (__int64)v113;
        v116 = 1282;
        v118.m128i_i64[0] = (__int64)"': ";
        v104 = 770;
        v120 = (__int64)v117;
        v122[0] = v96;
        v103[1] = 0;
        v106 = 0;
        v108 = 2306;
        v110 = 0;
        v113[1] = 0;
        v114 = v83;
        v117[1] = 0;
        v119 = 770;
        v121 = 0;
        v123 = 1026;
        sub_CA0F50((__int64 *)&v98, (void **)&v120);
        v33 = (__int64 *)*v84;
        v34 = (__int64 *)*v84;
        if ( v98 == v100 )
        {
          v76 = v99;
          if ( v99 )
          {
            if ( v99 == 1 )
            {
              *(_BYTE *)v34 = v100[0];
            }
            else
            {
              v32 = v100;
              memcpy(v34, v100, v99);
            }
            v34 = (__int64 *)*v84;
            v76 = v99;
          }
          v84[1] = v76;
          *((_BYTE *)v34 + v76) = 0;
          v33 = v98;
        }
        else
        {
          v35 = v100[0];
          v32 = (__int64 *)v99;
          if ( v33 == v84 + 2 )
          {
            *v84 = v98;
            v84[1] = v32;
            v84[2] = v35;
          }
          else
          {
            v36 = v84[2];
            *v84 = v98;
            v84[1] = v32;
            v84[2] = v35;
            if ( v33 )
            {
              v98 = v33;
              v100[0] = v36;
              goto LABEL_39;
            }
          }
          v98 = v100;
          v33 = v100;
        }
LABEL_39:
        v99 = 0;
        *(_BYTE *)v33 = 0;
        if ( v98 != v100 )
        {
          v32 = (__int64 *)(v100[0] + 1LL);
          j_j___libc_free_0((unsigned __int64)v98);
        }
        if ( (__int64 *)v96[0] != &v97 )
        {
          v32 = (__int64 *)(v97 + 1);
          j_j___libc_free_0(v96[0]);
        }
        if ( (v91 & 1) == 0 && (v91 & 0xFFFFFFFFFFFFFFFELL) == 0 )
        {
          if ( (v90 & 1) != 0 || (v90 & 0xFFFFFFFFFFFFFFFELL) != 0 )
            sub_C63C30(&v90, (__int64)v32);
          return 0;
        }
        goto LABEL_145;
      }
LABEL_64:
      sub_C7C5C0((__int64)&v124);
      if ( !v127 )
        return 1;
    }
  }
  LOBYTE(n) = n & 0xFD;
  v37 = (unsigned __int64)v124;
  v124 = 0;
  v117[0] = v37 | 1;
  v38 = v37 & 0xFFFFFFFFFFFFFFFELL;
  if ( !v38 )
    goto LABEL_3;
  v117[0] = 0;
  v39 = &v120;
  v120 = v38 | 1;
  sub_C64870((__int64)&v124, &v120);
  v40 = (_QWORD *)*a3;
  if ( v124 == src )
  {
    v72 = n;
    if ( n )
    {
      if ( n == 1 )
      {
        *(_BYTE *)v40 = src[0];
      }
      else
      {
        v39 = src;
        memcpy(v40, src, n);
      }
      v72 = n;
      v40 = (_QWORD *)*a3;
    }
    a3[1] = v72;
    *((_BYTE *)v40 + v72) = 0;
    v40 = v124;
    goto LABEL_53;
  }
  v41 = src[0];
  v39 = (__int64 *)n;
  if ( v40 == a3 + 2 )
  {
    *a3 = v124;
    a3[1] = v39;
    a3[2] = v41;
    goto LABEL_128;
  }
  v42 = a3[2];
  *a3 = v124;
  a3[1] = v39;
  a3[2] = v41;
  if ( !v40 )
  {
LABEL_128:
    v124 = src;
    v40 = src;
    goto LABEL_53;
  }
  v124 = v40;
  src[0] = v42;
LABEL_53:
  n = 0;
  *(_BYTE *)v40 = 0;
  if ( v124 != src )
  {
    v39 = (__int64 *)(src[0] + 1LL);
    j_j___libc_free_0((unsigned __int64)v124);
  }
  if ( (v120 & 1) != 0 || (v120 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    sub_C63C30(&v120, (__int64)v39);
  if ( (v117[0] & 1) != 0 || (result = 0, (v117[0] & 0xFFFFFFFFFFFFFFFELL) != 0) )
    sub_C63C30(v117, (__int64)v39);
  return result;
}
