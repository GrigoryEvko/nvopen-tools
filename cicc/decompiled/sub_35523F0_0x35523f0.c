// Function: sub_35523F0
// Address: 0x35523f0
//
__int64 __fastcall sub_35523F0(_QWORD *a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // r8
  __int64 v4; // rcx
  __int64 v5; // rdx
  __int64 v6; // rsi
  __int64 v7; // r12
  __int64 v8; // rax
  unsigned int *v9; // r9
  unsigned __int64 v10; // rcx
  _QWORD *v11; // rax
  unsigned __int64 v12; // r8
  __int64 **v13; // rax
  __int64 v14; // rdi
  __int64 v15; // rbx
  _BYTE *v16; // r12
  _BYTE *v17; // r13
  _BYTE *v18; // rbx
  unsigned int v19; // r13d
  int v20; // r13d
  __int16 *v21; // rbx
  int *v22; // r12
  int *v23; // rax
  int v24; // edx
  _BYTE *v25; // r13
  unsigned int *v26; // r14
  unsigned int *v27; // rax
  __int64 **v28; // rdx
  __int64 **v29; // r13
  __int64 v30; // rax
  _BYTE *v31; // r15
  _BYTE *v32; // r12
  _BYTE *v33; // rbx
  unsigned int v34; // r15d
  unsigned int *v35; // rax
  unsigned int *v36; // rdx
  _BYTE *v37; // r15
  _BYTE *v38; // rsi
  __int64 v39; // rdx
  __int64 v40; // rdx
  __int64 v41; // rcx
  __int64 v42; // r8
  __int64 v43; // r9
  const void *v44; // r12
  signed __int64 v45; // r15
  char *v46; // r14
  char *v47; // r13
  unsigned __int64 v48; // rax
  char *v49; // rbx
  __int64 v50; // rdx
  __int64 v51; // rcx
  char *v52; // rax
  char *v53; // rsi
  char *v54; // rbx
  unsigned __int64 *v55; // rsi
  __int64 v56; // rax
  __int64 v57; // rax
  unsigned int v58; // r15d
  unsigned int *v59; // rax
  unsigned int *v60; // rdx
  int v61; // edx
  unsigned __int64 v62; // rax
  __int64 *v63; // rsi
  __int64 v64; // rdx
  __int64 v65; // rdx
  unsigned __int64 v66; // rsi
  int v67; // eax
  __int64 v68; // rdx
  unsigned __int64 v69; // rsi
  int v70; // eax
  unsigned __int64 v71; // rax
  __int64 *v72; // rsi
  unsigned __int64 v73; // rdx
  unsigned int *v74; // rbx
  char v75; // r13
  __int64 v76; // rax
  __int64 v77; // rax
  __int64 v78; // rdx
  const __m128i *v79; // r15
  __m128i *v80; // rax
  unsigned int *v81; // r12
  char v82; // r13
  __int64 v83; // rax
  __int64 v84; // rax
  __int64 v85; // rdx
  const __m128i *v86; // rax
  __m128i *v87; // rdx
  char *v88; // rsi
  char *v89; // r15
  unsigned __int64 v90; // [rsp-8h] [rbp-5B8h]
  __int16 *v92; // [rsp+40h] [rbp-570h]
  int v93; // [rsp+4Ch] [rbp-564h]
  __int64 v94; // [rsp+60h] [rbp-550h]
  unsigned __int64 v95; // [rsp+70h] [rbp-540h]
  __int64 **v96; // [rsp+78h] [rbp-538h]
  unsigned int *v97; // [rsp+88h] [rbp-528h]
  _BYTE *v98; // [rsp+90h] [rbp-520h]
  char *v99; // [rsp+90h] [rbp-520h]
  unsigned int *v100; // [rsp+90h] [rbp-520h]
  _BYTE *v101; // [rsp+98h] [rbp-518h]
  _BYTE *v102; // [rsp+98h] [rbp-518h]
  __int64 **v103; // [rsp+A0h] [rbp-510h]
  _QWORD *v104; // [rsp+A0h] [rbp-510h]
  _QWORD *v105; // [rsp+A0h] [rbp-510h]
  _QWORD *v106; // [rsp+A8h] [rbp-508h]
  unsigned __int64 v107; // [rsp+B0h] [rbp-500h]
  __int64 **v108; // [rsp+B8h] [rbp-4F8h]
  unsigned int v109; // [rsp+CCh] [rbp-4E4h] BYREF
  unsigned int v110; // [rsp+D0h] [rbp-4E0h] BYREF
  __int64 v111; // [rsp+D8h] [rbp-4D8h]
  __int64 v112; // [rsp+E0h] [rbp-4D0h]
  unsigned int *v113; // [rsp+F0h] [rbp-4C0h] BYREF
  __int64 v114; // [rsp+F8h] [rbp-4B8h]
  _BYTE v115[16]; // [rsp+100h] [rbp-4B0h] BYREF
  __int64 v116; // [rsp+110h] [rbp-4A0h] BYREF
  __int64 v117; // [rsp+118h] [rbp-498h] BYREF
  unsigned __int64 v118; // [rsp+120h] [rbp-490h]
  __int64 *v119; // [rsp+128h] [rbp-488h]
  __int64 *v120; // [rsp+130h] [rbp-480h]
  __int64 v121; // [rsp+138h] [rbp-478h]
  unsigned __int64 v122; // [rsp+140h] [rbp-470h] BYREF
  __int64 v123; // [rsp+148h] [rbp-468h]
  _BYTE v124[192]; // [rsp+150h] [rbp-460h] BYREF
  _QWORD v125[7]; // [rsp+210h] [rbp-3A0h] BYREF
  __int16 v126; // [rsp+248h] [rbp-368h]
  char v127; // [rsp+24Ah] [rbp-366h]
  __int64 v128; // [rsp+250h] [rbp-360h]
  unsigned __int64 v129; // [rsp+258h] [rbp-358h]
  __int64 v130; // [rsp+260h] [rbp-350h]
  __int64 v131; // [rsp+268h] [rbp-348h]
  _BYTE *v132; // [rsp+270h] [rbp-340h]
  __int64 v133; // [rsp+278h] [rbp-338h]
  _BYTE v134[192]; // [rsp+280h] [rbp-330h] BYREF
  unsigned __int64 v135; // [rsp+340h] [rbp-270h]
  int v136; // [rsp+348h] [rbp-268h]
  int v137; // [rsp+350h] [rbp-260h]
  _BYTE *v138; // [rsp+358h] [rbp-258h]
  __int64 v139; // [rsp+360h] [rbp-250h]
  _BYTE v140[32]; // [rsp+368h] [rbp-248h] BYREF
  unsigned __int64 v141; // [rsp+388h] [rbp-228h]
  int v142; // [rsp+390h] [rbp-220h]
  unsigned __int64 v143; // [rsp+398h] [rbp-218h]
  __int64 v144; // [rsp+3A0h] [rbp-210h]
  __int64 v145; // [rsp+3A8h] [rbp-208h]
  unsigned __int64 v146[3]; // [rsp+3B0h] [rbp-200h] BYREF
  _BYTE *v147; // [rsp+3C8h] [rbp-1E8h]
  __int64 v148; // [rsp+3D0h] [rbp-1E0h]
  _BYTE v149[192]; // [rsp+3D8h] [rbp-1D8h] BYREF
  _BYTE *v150; // [rsp+498h] [rbp-118h]
  __int64 v151; // [rsp+4A0h] [rbp-110h]
  _BYTE v152[192]; // [rsp+4A8h] [rbp-108h] BYREF
  __int64 v153; // [rsp+568h] [rbp-48h]
  __int64 v154; // [rsp+570h] [rbp-40h]

  v107 = *(_QWORD *)a2;
  result = *(_QWORD *)a2 + 88LL * *(unsigned int *)(a2 + 8);
  v94 = result;
  if ( *(_QWORD *)a2 == result )
    return result;
  while ( 2 )
  {
    if ( *(_DWORD *)(v107 + 40) <= 2u )
      goto LABEL_3;
    v125[6] = v146;
    v147 = v149;
    v132 = v134;
    v150 = v152;
    v126 = 1;
    memset(v146, 0, sizeof(v146));
    v148 = 0x800000000LL;
    v151 = 0x800000000LL;
    v153 = 0;
    v154 = 0;
    memset(v125, 0, 48);
    v127 = 0;
    v128 = 0;
    v129 = 0;
    v130 = 0;
    v131 = 0;
    v133 = 0x800000000LL;
    v135 = 0;
    v136 = 0;
    v137 = 0;
    v138 = v140;
    v139 = 0x800000000LL;
    v3 = a1[113];
    v4 = a1[437];
    v141 = 0;
    v5 = a1[438];
    v6 = a1[4];
    v142 = 0;
    v143 = 0;
    v144 = 0;
    v145 = 0;
    sub_2F796A0((__int64)v125, v6, v5, v4, v3, v3 + 48, 0, 1);
    v7 = a1[4];
    v8 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(v7 + 16) + 200LL))(*(_QWORD *)(v7 + 16));
    v10 = v107;
    v95 = v8;
    v11 = *(_QWORD **)(v7 + 32);
    v123 = 0x800000000LL;
    v12 = v90;
    v106 = v11;
    v122 = (unsigned __int64)v124;
    v113 = (unsigned int *)v115;
    v114 = 0x400000000LL;
    LODWORD(v117) = 0;
    v118 = 0;
    v119 = &v117;
    v120 = &v117;
    v121 = 0;
    v13 = *(__int64 ***)(v107 + 32);
    v96 = &v13[*(unsigned int *)(v107 + 40)];
    if ( v13 == v96 )
    {
      v38 = v124;
      v39 = 0;
      goto LABEL_65;
    }
    v108 = *(__int64 ***)(v107 + 32);
    do
    {
      v14 = **v108;
      if ( *(_WORD *)(v14 + 68) == 68 )
        goto LABEL_42;
      if ( !*(_WORD *)(v14 + 68) )
        goto LABEL_42;
      v15 = *(_QWORD *)(v14 + 32);
      v16 = (_BYTE *)(v15 + 40LL * (*(_DWORD *)(v14 + 40) & 0xFFFFFF));
      v17 = (_BYTE *)(v15 + 40LL * (unsigned int)sub_2E88FE0(v14));
      if ( v16 == v17 )
        goto LABEL_42;
      while ( 1 )
      {
        v18 = v17;
        if ( (unsigned __int8)sub_2E2FA70(v17) )
          break;
        v17 += 40;
        if ( v16 == v17 )
          goto LABEL_42;
      }
      while ( 1 )
      {
        if ( v16 == v18 )
          goto LABEL_42;
        v19 = *((_DWORD *)v18 + 2);
        v109 = v19;
        if ( (v19 & 0x80000000) == 0 )
          break;
        if ( v121 )
        {
          sub_2DCBF00((__int64)&v116, &v109);
          goto LABEL_27;
        }
        v10 = (unsigned int)v114;
        v26 = &v113[(unsigned int)v114];
        if ( v113 == v26 )
        {
          v12 = (unsigned __int64)&v116;
          if ( (unsigned int)v114 > 3uLL )
            goto LABEL_166;
        }
        else
        {
          v27 = v113;
          while ( v19 != *v27 )
          {
            if ( v26 == ++v27 )
              goto LABEL_38;
          }
          if ( v26 != v27 )
            goto LABEL_27;
LABEL_38:
          if ( (unsigned int)v114 > 3uLL )
          {
            v102 = v16;
            v81 = v113;
            do
            {
              v84 = sub_2DCC990(&v116, (__int64)&v117, v81);
              if ( v85 )
              {
                v82 = v84 || (__int64 *)v85 == &v117 || *v81 < *(_DWORD *)(v85 + 32);
                v105 = (_QWORD *)v85;
                v83 = sub_22077B0(0x28u);
                *(_DWORD *)(v83 + 32) = *v81;
                sub_220F040(v82, v83, v105, &v117);
                ++v121;
              }
              ++v81;
            }
            while ( v26 != v81 );
            v16 = v102;
LABEL_166:
            LODWORD(v114) = 0;
            sub_2DCBF00((__int64)&v116, &v109);
            goto LABEL_27;
          }
        }
        if ( (unsigned __int64)(unsigned int)v114 + 1 > HIDWORD(v114) )
        {
          sub_C8D5F0((__int64)&v113, v115, (unsigned int)v114 + 1LL, 4u, v12, (__int64)v9);
          v26 = &v113[(unsigned int)v114];
        }
        *v26 = v19;
        v25 = v18 + 40;
        LODWORD(v114) = v114 + 1;
        if ( v18 + 40 == v16 )
          goto LABEL_42;
LABEL_30:
        while ( 1 )
        {
          v18 = v25;
          if ( (unsigned __int8)sub_2E2FA70(v25) )
            break;
          v25 += 40;
          if ( v16 == v25 )
            goto LABEL_42;
        }
      }
      if ( !*(_BYTE *)(*(_QWORD *)(*(_QWORD *)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*v106 + 16LL) + 200LL))(*(_QWORD *)(*v106 + 16LL))
                                             + 248)
                                 + 16LL)
                     + v19) )
        goto LABEL_27;
      v10 = v19;
      if ( (*(_QWORD *)(v106[48] + 8LL * (v19 >> 6)) & (1LL << v19)) != 0 )
        goto LABEL_27;
      v10 = v95;
      v101 = v16;
      v98 = v18;
      v20 = *(_DWORD *)(*(_QWORD *)(v95 + 8) + 24LL * v109 + 16) & 0xFFF;
      v21 = (__int16 *)(*(_QWORD *)(v95 + 56) + 2LL * (*(_DWORD *)(*(_QWORD *)(v95 + 8) + 24LL * v109 + 16) >> 12));
      while ( v21 )
      {
        v110 = v20;
        if ( !v121 )
        {
          v9 = v113;
          v22 = (int *)&v113[(unsigned int)v114];
          if ( v113 == (unsigned int *)v22 )
          {
            if ( (unsigned int)v114 <= 3uLL )
              goto LABEL_141;
          }
          else
          {
            v23 = (int *)v113;
            while ( v20 != *v23 )
            {
              if ( v22 == ++v23 )
                goto LABEL_140;
            }
            if ( v22 != v23 )
              goto LABEL_25;
LABEL_140:
            if ( (unsigned int)v114 <= 3uLL )
            {
LABEL_141:
              v73 = (unsigned int)v114 + 1LL;
              if ( v73 > HIDWORD(v114) )
              {
                sub_C8D5F0((__int64)&v113, v115, v73, 4u, v12, (__int64)v113);
                v22 = (int *)&v113[(unsigned int)v114];
              }
              *v22 = v20;
              LODWORD(v114) = v114 + 1;
              goto LABEL_25;
            }
            v92 = v21;
            v74 = v113;
            v93 = v20;
            do
            {
              v77 = sub_2DCC990(&v116, (__int64)&v117, v74);
              if ( v78 )
              {
                v75 = v77 || (__int64 *)v78 == &v117 || *v74 < *(_DWORD *)(v78 + 32);
                v104 = (_QWORD *)v78;
                v76 = sub_22077B0(0x28u);
                *(_DWORD *)(v76 + 32) = *v74;
                sub_220F040(v75, v76, v104, &v117);
                ++v121;
              }
              ++v74;
            }
            while ( v22 != (int *)v74 );
            v20 = v93;
            v21 = v92;
          }
          LODWORD(v114) = 0;
        }
        sub_2DCBE50((__int64)&v116, &v110);
LABEL_25:
        v24 = *v21++;
        v20 += v24;
        if ( !(_WORD)v24 )
          break;
      }
      v16 = v101;
      v18 = v98;
LABEL_27:
      v25 = v18 + 40;
      if ( v18 + 40 != v16 )
        goto LABEL_30;
LABEL_42:
      ++v108;
    }
    while ( v96 != v108 );
    v28 = *(__int64 ***)(v107 + 32);
    v103 = &v28[*(unsigned int *)(v107 + 40)];
    if ( v28 != v103 )
    {
      v29 = *(__int64 ***)(v107 + 32);
      while ( 1 )
      {
        v30 = **v29;
        v31 = *(_BYTE **)(v30 + 32);
        v32 = &v31[40 * (*(_DWORD *)(v30 + 40) & 0xFFFFFF)];
        if ( v31 != v32 )
        {
          while ( 1 )
          {
            v33 = v31;
            if ( sub_2DADC00(v31) )
              break;
            v31 += 40;
            if ( v32 == v31 )
              goto LABEL_63;
          }
          if ( v32 != v31 )
            break;
        }
LABEL_63:
        if ( v103 == ++v29 )
          goto LABEL_64;
      }
      while ( 2 )
      {
        if ( (((v33[3] & 0x10) != 0) & (v33[3] >> 6)) != 0 )
          goto LABEL_58;
        v34 = *((_DWORD *)v33 + 2);
        if ( (v34 & 0x80000000) != 0 )
        {
          if ( !v121 )
          {
            v35 = v113;
            v36 = &v113[(unsigned int)v114];
            if ( v113 != v36 )
            {
              while ( v34 != *v35 )
              {
                if ( v36 == ++v35 )
                  goto LABEL_121;
              }
              if ( v36 != v35 )
                goto LABEL_58;
            }
LABEL_121:
            v65 = (unsigned int)v123;
            v66 = v122;
            v67 = v123;
            v10 = v122 + 24LL * (unsigned int)v123;
            if ( (unsigned int)v123 >= (unsigned __int64)HIDWORD(v123) )
            {
              v12 = (unsigned int)v123 + 1LL;
              v110 = *((_DWORD *)v33 + 2);
              v79 = (const __m128i *)&v110;
              v111 = 0;
              v112 = 0;
              if ( HIDWORD(v123) < v12 )
              {
                if ( v122 > (unsigned __int64)&v110 || v10 <= (unsigned __int64)&v110 )
                {
                  sub_C8D5F0((__int64)&v122, v124, (unsigned int)v123 + 1LL, 0x18u, v12, (__int64)v9);
                  v66 = v122;
                  v65 = (unsigned int)v123;
                  v79 = (const __m128i *)&v110;
                }
                else
                {
                  v89 = (char *)&v110 - v122;
                  sub_C8D5F0((__int64)&v122, v124, (unsigned int)v123 + 1LL, 0x18u, v12, (__int64)v9);
                  v66 = v122;
                  v65 = (unsigned int)v123;
                  v79 = (const __m128i *)&v89[v122];
                }
              }
              v80 = (__m128i *)(v66 + 24 * v65);
              *v80 = _mm_loadu_si128(v79);
              v80[1].m128i_i64[0] = v79[1].m128i_i64[0];
              LODWORD(v123) = v123 + 1;
            }
            else
            {
              if ( v10 )
              {
                *(_DWORD *)v10 = v34;
                *(_QWORD *)(v10 + 8) = 0;
                *(_QWORD *)(v10 + 16) = 0;
                v67 = v123;
              }
              LODWORD(v123) = v67 + 1;
            }
            goto LABEL_58;
          }
          v62 = v118;
          if ( !v118 )
            goto LABEL_121;
          v63 = &v117;
          do
          {
            while ( 1 )
            {
              v10 = *(_QWORD *)(v62 + 16);
              v64 = *(_QWORD *)(v62 + 24);
              if ( v34 <= *(_DWORD *)(v62 + 32) )
                break;
              v62 = *(_QWORD *)(v62 + 24);
              if ( !v64 )
                goto LABEL_119;
            }
            v63 = (__int64 *)v62;
            v62 = *(_QWORD *)(v62 + 16);
          }
          while ( v10 );
LABEL_119:
          if ( v63 == &v117 || v34 < *((_DWORD *)v63 + 8) )
            goto LABEL_121;
LABEL_58:
          v37 = v33 + 40;
          if ( v33 + 40 == v32 )
            goto LABEL_63;
          while ( 1 )
          {
            v33 = v37;
            if ( sub_2DADC00(v37) )
              break;
            v37 += 40;
            if ( v32 == v37 )
              goto LABEL_63;
          }
          if ( v37 == v32 )
            goto LABEL_63;
          continue;
        }
        break;
      }
      v12 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*v106 + 16LL) + 200LL))(*(_QWORD *)(*v106 + 16LL));
      if ( !*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v12 + 248) + 16LL) + v34) )
        goto LABEL_58;
      v10 = v34;
      if ( (*(_QWORD *)(v106[48] + 8LL * (v34 >> 6)) & (1LL << v34)) != 0 )
        goto LABEL_58;
      v12 = *(_DWORD *)(*(_QWORD *)(v95 + 8) + 24LL * v34 + 16) & 0xFFF;
      v9 = (unsigned int *)(*(_QWORD *)(v95 + 56) + 2LL * (*(_DWORD *)(*(_QWORD *)(v95 + 8) + 24LL * v34 + 16) >> 12));
      v58 = *(_DWORD *)(*(_QWORD *)(v95 + 8) + 24LL * v34 + 16) & 0xFFF;
      while ( 2 )
      {
        if ( !v9 )
          goto LABEL_58;
        if ( v121 )
        {
          v71 = v118;
          if ( v118 )
          {
            v72 = &v117;
            do
            {
              v10 = *(_QWORD *)(v71 + 16);
              if ( v58 > *(_DWORD *)(v71 + 32) )
              {
                v71 = *(_QWORD *)(v71 + 24);
              }
              else
              {
                v72 = (__int64 *)v71;
                v71 = *(_QWORD *)(v71 + 16);
              }
            }
            while ( v71 );
            if ( v72 != &v117 && v58 >= *((_DWORD *)v72 + 8) )
              goto LABEL_112;
          }
        }
        else
        {
          v59 = v113;
          v60 = &v113[(unsigned int)v114];
          if ( v113 != v60 )
          {
            while ( v58 != *v59 )
            {
              if ( v60 == ++v59 )
                goto LABEL_127;
            }
            if ( v59 != v60 )
            {
LABEL_112:
              v61 = *(__int16 *)v9;
              v9 = (unsigned int *)((char *)v9 + 2);
              v58 += v61;
              if ( !(_WORD)v61 )
                goto LABEL_58;
              continue;
            }
          }
        }
        break;
      }
LABEL_127:
      v68 = (unsigned int)v123;
      v69 = v122;
      v70 = v123;
      v10 = v122 + 24LL * (unsigned int)v123;
      if ( (unsigned int)v123 >= (unsigned __int64)HIDWORD(v123) )
      {
        v12 = (unsigned int)v123 + 1LL;
        v110 = v58;
        v86 = (const __m128i *)&v110;
        v111 = 0;
        v112 = 0;
        if ( HIDWORD(v123) < v12 )
        {
          if ( v122 > (unsigned __int64)&v110 || v10 <= (unsigned __int64)&v110 )
          {
            v100 = v9;
            sub_C8D5F0((__int64)&v122, v124, (unsigned int)v123 + 1LL, 0x18u, v12, (__int64)v9);
            v69 = v122;
            v68 = (unsigned int)v123;
            v86 = (const __m128i *)&v110;
            v9 = v100;
          }
          else
          {
            v97 = v9;
            v99 = (char *)&v110 - v122;
            sub_C8D5F0((__int64)&v122, v124, (unsigned int)v123 + 1LL, 0x18u, v12, (__int64)v9);
            v69 = v122;
            v68 = (unsigned int)v123;
            v9 = v97;
            v86 = (const __m128i *)&v99[v122];
          }
        }
        v87 = (__m128i *)(v69 + 24 * v68);
        *v87 = _mm_loadu_si128(v86);
        v87[1].m128i_i64[0] = v86[1].m128i_i64[0];
        LODWORD(v123) = v123 + 1;
      }
      else
      {
        if ( v10 )
        {
          *(_DWORD *)v10 = v58;
          *(_QWORD *)(v10 + 8) = 0;
          *(_QWORD *)(v10 + 16) = 0;
          v70 = v123;
        }
        LODWORD(v123) = v70 + 1;
      }
      goto LABEL_112;
    }
LABEL_64:
    v38 = (_BYTE *)v122;
    v39 = (unsigned int)v123;
LABEL_65:
    sub_2F76D20((__int64)v125, (__int64)v38, v39, v10, v12, (__int64)v9);
    sub_353ED70(v118);
    if ( v113 != (unsigned int *)v115 )
      _libc_free((unsigned __int64)v113);
    if ( (_BYTE *)v122 != v124 )
      _libc_free(v122);
    sub_2F75730((__int64)v125, (__int64)v38, v40, v41, v42, v43);
    v44 = *(const void **)(v107 + 32);
    v45 = 8LL * *(unsigned int *)(v107 + 40);
    if ( v45 )
    {
      v46 = (char *)sub_22077B0(v45);
      v47 = &v46[v45];
      memcpy(v46, v44, v45);
      if ( &v46[v45] != v46 )
      {
        _BitScanReverse64(&v48, v45 >> 3);
        sub_353E140(v46, v47, 2LL * (int)(63 - (v48 ^ 0x3F)));
        if ( (unsigned __int64)v45 <= 0x80 )
        {
          sub_353D670(v46, v47);
        }
        else
        {
          v49 = v46 + 128;
          sub_353D670(v46, v46 + 128);
          if ( v47 != v46 + 128 )
          {
            do
            {
              while ( 1 )
              {
                v50 = *((_QWORD *)v49 - 1);
                v51 = *(_QWORD *)v49;
                v52 = v49 - 8;
                if ( *(_DWORD *)(*(_QWORD *)v49 + 200LL) > *(_DWORD *)(v50 + 200) )
                  break;
                v88 = v49;
                v49 += 8;
                *(_QWORD *)v88 = v51;
                if ( v47 == v49 )
                  goto LABEL_76;
              }
              do
              {
                *((_QWORD *)v52 + 1) = v50;
                v53 = v52;
                v50 = *((_QWORD *)v52 - 1);
                v52 -= 8;
              }
              while ( *(_DWORD *)(v51 + 200) > *(_DWORD *)(v50 + 200) );
              v49 += 8;
              *(_QWORD *)v53 = v51;
            }
            while ( v47 != v49 );
          }
        }
LABEL_76:
        v54 = v46;
        while ( 1 )
        {
          v55 = *(unsigned __int64 **)v54;
          v56 = **(_QWORD **)v54;
          if ( !v56 )
            BUG();
          if ( (*(_BYTE *)v56 & 4) == 0 && (*(_BYTE *)(v56 + 44) & 8) != 0 )
          {
            do
              v56 = *(_QWORD *)(v56 + 8);
            while ( (*(_BYTE *)(v56 + 44) & 8) != 0 );
          }
          v57 = *(_QWORD *)(v56 + 8);
          v122 = 0;
          v128 = v57;
          LODWORD(v123) = 0;
          sub_2F778F0(v125, *v55, 0, (__int64)&v122, 0, 0, v146[0]);
          if ( (_WORD)v122 )
            break;
          v54 += 8;
          sub_2F78B80((__int64)v125, 0);
          if ( v47 == v54 )
            goto LABEL_83;
        }
        *(_QWORD *)(v107 + 72) = *(_QWORD *)v54;
      }
LABEL_83:
      j_j___libc_free_0((unsigned __int64)v46);
    }
    if ( v143 )
      j_j___libc_free_0(v143);
    if ( v141 )
      _libc_free(v141);
    if ( v138 != v140 )
      _libc_free((unsigned __int64)v138);
    if ( v135 )
      _libc_free(v135);
    if ( v132 != v134 )
      _libc_free((unsigned __int64)v132);
    if ( v129 )
      j_j___libc_free_0(v129);
    if ( v150 != v152 )
      _libc_free((unsigned __int64)v150);
    if ( v147 != v149 )
      _libc_free((unsigned __int64)v147);
    if ( v146[0] )
      j_j___libc_free_0(v146[0]);
LABEL_3:
    v107 += 88LL;
    result = v107;
    if ( v94 != v107 )
      continue;
    return result;
  }
}
