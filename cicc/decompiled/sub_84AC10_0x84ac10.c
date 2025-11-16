// Function: sub_84AC10
// Address: 0x84ac10
//
__int64 __fastcall sub_84AC10(
        __int64 a1,
        __int64 m128i_i64,
        __int64 a3,
        _BOOL4 a4,
        __m128i *a5,
        _QWORD *a6,
        _QWORD *a7,
        int a8,
        unsigned int a9,
        int a10,
        unsigned int a11,
        int a12,
        FILE *a13,
        unsigned int a14,
        _DWORD *a15,
        _DWORD *a16,
        _DWORD *a17,
        _DWORD *a18,
        __int64 *a19,
        _QWORD *a20)
{
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // r13
  __int64 v26; // r12
  __int64 v27; // rdx
  __int64 v28; // rax
  __m128i *v29; // rax
  __int64 v30; // rdi
  char v31; // al
  bool v32; // zf
  unsigned int v33; // eax
  int v34; // r12d
  char v35; // bl
  __int64 v36; // r14
  __int64 v37; // rdx
  __int64 v38; // rcx
  __int64 v39; // rbx
  __int64 *v40; // r9
  __int64 v41; // r8
  __int64 v42; // r13
  __int64 v43; // rdx
  __int64 v44; // rcx
  __int64 v45; // r8
  __int64 *v46; // r9
  __int64 v48; // rax
  char v49; // al
  __int64 m; // rax
  _QWORD *v51; // r12
  __int64 n; // rbx
  char v53; // dl
  __int64 v54; // rax
  __int64 v55; // rax
  __int64 ii; // r13
  _BYTE *v57; // rax
  __int64 *v58; // rax
  __int64 v59; // rdx
  __int64 v60; // r12
  __int64 v61; // r13
  __int64 v62; // rbx
  char v63; // al
  char v64; // al
  char v65; // al
  __int64 v66; // r15
  __int64 v67; // rax
  __int64 *v68; // rax
  int v69; // r14d
  char v70; // bl
  __int64 *v71; // r15
  __int64 jj; // r12
  __int64 *v73; // r13
  __int64 v74; // rbx
  __int64 v75; // rax
  const __m128i *v76; // rax
  __int64 v77; // r15
  __int64 v78; // r14
  __m128i *v79; // rcx
  __int64 v80; // rax
  bool v81; // bl
  __int64 v82; // rax
  int v83; // eax
  int v84; // eax
  __int64 v85; // rax
  __int64 v86; // r14
  __int64 v87; // rax
  __int64 v88; // rdi
  __int64 v89; // rax
  __int64 v90; // rax
  __int64 v91; // rdi
  char v92; // al
  __int64 v93; // rdx
  _QWORD *k; // r12
  __int64 v95; // rax
  __int64 v96; // rax
  __int64 v97; // r12
  unsigned int v98; // r13d
  __int64 v99; // r12
  __int64 j; // rax
  __int64 v101; // rdx
  __int64 v102; // rcx
  __int64 v103; // r8
  _QWORD *v104; // rbx
  __int64 v105; // rdi
  __int64 i; // rax
  __int64 v108; // r12
  int v109; // eax
  __int64 v110; // rax
  bool v111; // bl
  int v112; // eax
  int v113; // eax
  int v114; // eax
  __int64 v115; // [rsp+0h] [rbp-170h]
  _QWORD *v117; // [rsp+10h] [rbp-160h]
  bool v118; // [rsp+18h] [rbp-158h]
  _BOOL4 v119; // [rsp+1Ch] [rbp-154h]
  unsigned int v121; // [rsp+30h] [rbp-140h]
  bool v122; // [rsp+37h] [rbp-139h]
  int v123; // [rsp+58h] [rbp-118h]
  __int64 v124; // [rsp+60h] [rbp-110h]
  _QWORD *v125; // [rsp+70h] [rbp-100h]
  __int64 v126; // [rsp+78h] [rbp-F8h]
  _BOOL4 v128; // [rsp+84h] [rbp-ECh]
  int v130; // [rsp+98h] [rbp-D8h] BYREF
  int v131; // [rsp+9Ch] [rbp-D4h] BYREF
  __int64 v132; // [rsp+A0h] [rbp-D0h] BYREF
  __int64 *v133; // [rsp+A8h] [rbp-C8h] BYREF
  __int64 v134; // [rsp+B0h] [rbp-C0h] BYREF
  __int64 v135; // [rsp+B8h] [rbp-B8h] BYREF
  _QWORD *v136; // [rsp+C0h] [rbp-B0h] BYREF
  _QWORD *v137; // [rsp+C8h] [rbp-A8h] BYREF
  __m128i v138[10]; // [rsp+D0h] [rbp-A0h] BYREF

  v126 = a1;
  v121 = m128i_i64;
  v20 = sub_82BD70(a1, m128i_i64, a3);
  v25 = *(_QWORD *)(v20 + 1024);
  v26 = v20;
  if ( v25 == *(_QWORD *)(v20 + 1016) )
    sub_8332F0(v20, m128i_i64, v21, v22, v23, (__int64 *)v24);
  v27 = *(_QWORD *)(v26 + 1008);
  v28 = v27 + 40 * v25;
  if ( v28 )
  {
    *(_BYTE *)v28 &= 0xFCu;
    *(_QWORD *)(v28 + 8) = 0;
    *(_QWORD *)(v28 + 16) = 0;
    *(_QWORD *)(v28 + 24) = 0;
    *(_QWORD *)(v28 + 32) = 0;
  }
  *(_QWORD *)(v26 + 1024) = v25 + 1;
  if ( a17 )
    *a17 = 0;
  if ( a18 )
    *a18 = 0;
  if ( a15 )
    *a15 = 0;
  if ( a16 )
    *a16 = 0;
  v118 = 1;
  if ( a12 != 7 )
    v118 = dword_4F07518 == 0;
  v117 = 0;
  v115 = 0;
  while ( 2 )
  {
    v29 = 0;
    v134 = 0;
    v130 = 0;
    v131 = 0;
    if ( a4 )
      v29 = a5;
    v133 = 0;
    a5 = v29;
    if ( a19 )
    {
      a15 = 0;
      *a19 = 0;
    }
    v30 = v126;
    v122 = v126 != 0;
    if ( !v126 || a10 )
    {
      v119 = 0;
      if ( dword_4F04C44 != -1 )
      {
LABEL_28:
        if ( a5 )
        {
          v30 = (__int64)a5;
          if ( (unsigned int)sub_82F020(a5->m128i_i64) )
            goto LABEL_30;
        }
        v60 = (__int64)a6;
        while ( 1 )
        {
          if ( !v60 )
          {
LABEL_126:
            v30 = v121;
            if ( v121 )
            {
              v30 = a3;
              if ( (unsigned int)sub_89A370(a3) )
                goto LABEL_30;
              if ( a12 != 3 )
                goto LABEL_128;
            }
            else if ( a12 != 3 )
            {
LABEL_128:
              if ( v126 )
              {
                v30 = v126;
                if ( sub_82EEC0(v126) )
                {
LABEL_130:
                  v122 = 0;
                  v34 = 0;
                  goto LABEL_31;
                }
              }
              if ( a12 != 1 )
              {
LABEL_136:
                v23 = (unsigned int)qword_4F077B4;
                if ( (_DWORD)qword_4F077B4 )
                {
                  if ( !v126 )
                    goto LABEL_140;
                  if ( *(_BYTE *)(v126 + 80) == 13 )
                    goto LABEL_140;
                  v30 = a9;
                  if ( !a9 )
                    goto LABEL_140;
                  if ( (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 4) == 0 )
                    goto LABEL_44;
LABEL_143:
                  v22 = dword_4D047C8;
                  if ( dword_4D047C8 )
                    goto LABEL_144;
LABEL_43:
                  if ( v126 )
                  {
LABEL_44:
                    v128 = 0;
                    if ( !a9 )
                    {
LABEL_45:
                      if ( a15 && !a4 )
                      {
                        v27 = v126;
                        v49 = *(_BYTE *)(v126 + 80);
                        if ( v49 == 16 )
                        {
                          v27 = **(_QWORD **)(v126 + 88);
                          v49 = *(_BYTE *)(v27 + 80);
                        }
                        if ( v49 == 24 )
                        {
                          v27 = *(_QWORD *)(v27 + 88);
                          v49 = *(_BYTE *)(v27 + 80);
                        }
                        if ( (unsigned __int8)(v49 - 10) <= 1u )
                        {
                          for ( i = *(_QWORD *)(*(_QWORD *)(v27 + 88) + 152LL);
                                *(_BYTE *)(i + 140) == 12;
                                i = *(_QWORD *)(i + 160) )
                          {
                            ;
                          }
                          if ( (*(_BYTE *)(*(_QWORD *)(i + 168) + 20LL) & 2) == 0 )
                          {
                            if ( a9
                              || (m128i_i64 = v121,
                                  v30 = v126,
                                  (unsigned int)sub_828ED0(v126, v121, 0, 0, v128, 0, 0, 0, 0)) )
                            {
                              v34 = v128;
                              a9 = 0;
                              *a15 = 1;
                              v35 = v128;
                              v36 = v126;
                              goto LABEL_32;
                            }
                          }
                        }
                      }
                      v30 = v126;
                      sub_8360D0(
                        v126,
                        v121,
                        a3,
                        (__int64)a6,
                        a7,
                        a4,
                        a5,
                        0,
                        (a8 & 0x401) == 1 && a7 == 0,
                        1u,
                        0,
                        v128,
                        0,
                        a9,
                        0,
                        a8,
                        a12,
                        (__int64 *)&v133,
                        (__int64)&v134,
                        &v130,
                        &v131);
                      v122 = 1;
                      a9 = 0;
                      v123 = 1;
                      goto LABEL_53;
                    }
                    goto LABEL_199;
                  }
LABEL_194:
                  v122 = 0;
                  v128 = 0;
                  v126 = 0;
                  v123 = 0;
                  goto LABEL_53;
                }
                goto LABEL_142;
              }
              goto LABEL_30;
            }
            v95 = qword_4F04C68[0] + 776LL * dword_4F04C64;
            if ( (*(_BYTE *)(v95 + 12) & 0x10) != 0 )
            {
              if ( v126 )
              {
                v30 = v126;
                if ( sub_82EEC0(v126) )
                  goto LABEL_130;
                goto LABEL_136;
              }
              v27 = (unsigned int)qword_4F077B4;
              if ( !(_DWORD)qword_4F077B4 )
              {
                if ( (*(_BYTE *)(v95 + 6) & 4) != 0 )
                  goto LABEL_143;
                goto LABEL_194;
              }
LABEL_140:
              m128i_i64 = dword_4D047A8;
              if ( !dword_4D047A8 )
              {
                m128i_i64 = 0x2000000000LL;
                if ( (*(_QWORD *)(qword_4D03C50 + 16LL) & 0x2002400000LL) == 0x2000000000LL )
                  goto LABEL_130;
              }
LABEL_142:
              if ( (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 4) != 0 )
                goto LABEL_143;
              goto LABEL_43;
            }
LABEL_30:
            v34 = 1;
LABEL_31:
            v35 = v122;
            v36 = 0;
            *a17 = 1;
            *a20 = 0;
            goto LABEL_32;
          }
          v65 = *(_BYTE *)(v60 + 8);
          if ( v65 )
          {
            v62 = *(_QWORD *)(v60 + 16);
            if ( v62 )
              goto LABEL_30;
            if ( v65 == 1 )
            {
              v30 = *(_QWORD *)(v60 + 24);
              if ( (unsigned int)sub_82EC50(v30, m128i_i64, v27) )
                goto LABEL_30;
            }
            else if ( v65 != 2 )
            {
              sub_721090();
            }
            goto LABEL_93;
          }
          v61 = *(_QWORD *)(v60 + 24);
          v62 = v61 + 8;
          v30 = v61 + 8;
          if ( sub_82ED00(v61 + 8, m128i_i64) )
            goto LABEL_30;
          v22 = dword_4F077BC;
          if ( !dword_4F077BC )
            goto LABEL_111;
          v63 = *(_BYTE *)(v61 + 24);
          if ( v63 != 2 )
          {
            if ( v63 != 1 )
              goto LABEL_93;
            goto LABEL_113;
          }
          if ( *(_BYTE *)(v61 + 325) != 12 )
            goto LABEL_93;
          v30 = v61 + 152;
          if ( sub_712690(v61 + 152) )
            goto LABEL_30;
          m128i_i64 = dword_4F077BC;
          if ( !dword_4F077BC )
          {
LABEL_111:
            if ( !(_DWORD)qword_4F077B4 )
              goto LABEL_93;
          }
          if ( *(_BYTE *)(v61 + 24) != 1 )
            goto LABEL_93;
LABEL_113:
          v66 = *(_QWORD *)(v61 + 152);
          if ( *(_BYTE *)(v66 + 24) == 3 )
          {
            v30 = *(_QWORD *)(v61 + 8);
            if ( (unsigned int)sub_8D23E0(v30) )
              goto LABEL_30;
          }
          if ( dword_4F077BC )
          {
            v24 = (unsigned int)qword_4F077B4;
            if ( !(_DWORD)qword_4F077B4 && qword_4F077A8 <= 0x9EFBu && *(_BYTE *)(v66 + 24) == 1 )
            {
              if ( (unsigned __int8)(*(_BYTE *)(v66 + 56) - 3) > 1u
                || (v66 = *(_QWORD *)(v66 + 72), *(_BYTE *)(v66 + 24) == 1) )
              {
                if ( (*(_BYTE *)(v66 + 27) & 2) == 0 )
                {
                  v27 = *(unsigned __int8 *)(v66 + 56);
                  v67 = *(_QWORD *)(v66 + 72);
                  if ( (_BYTE)v27 != 95 )
                  {
                    if ( (_BYTE)v27 != 107 )
                    {
                      if ( (_BYTE)v27 == 105
                        && *(_BYTE *)(v67 + 24) == 2
                        && *(_BYTE *)(*(_QWORD *)(v67 + 56) + 173LL) == 12 )
                      {
                        goto LABEL_30;
                      }
                      goto LABEL_93;
                    }
                    v67 = *(_QWORD *)(v67 + 16);
                    if ( !v67 )
                      goto LABEL_93;
                  }
                  if ( *(_BYTE *)(v67 + 24) == 3 && (*(_BYTE *)(*(_QWORD *)(v67 + 56) + 172LL) & 1) != 0 )
                    goto LABEL_30;
                }
              }
            }
          }
LABEL_93:
          if ( dword_4F04C64 == -1 )
            goto LABEL_99;
          v27 = (__int64)qword_4F04C68;
          if ( (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 7) & 1) == 0 )
            goto LABEL_99;
          if ( *(_QWORD *)(v60 + 16) )
            goto LABEL_30;
          if ( !v62 )
            goto LABEL_99;
          v64 = *(_BYTE *)(v62 + 16);
          if ( v64 == 1 )
          {
            v68 = *(__int64 **)(v62 + 144);
LABEL_132:
            if ( !v68 )
              goto LABEL_99;
            goto LABEL_133;
          }
          if ( v64 != 2 )
            goto LABEL_99;
          v68 = *(__int64 **)(v62 + 288);
          if ( !v68 )
          {
            if ( *(_BYTE *)(v62 + 317) != 12 || *(_BYTE *)(v62 + 320) != 1 )
              goto LABEL_99;
            v30 = v62 + 144;
            v68 = sub_72E9A0(v62 + 144);
            goto LABEL_132;
          }
LABEL_133:
          if ( (*((_BYTE *)v68 + 26) & 4) != 0 )
            goto LABEL_30;
LABEL_99:
          if ( !*(_QWORD *)v60 )
            goto LABEL_126;
          if ( *(_BYTE *)(*(_QWORD *)v60 + 8LL) == 3 )
          {
            v30 = v60;
            v60 = sub_6BBB10((_QWORD *)v60);
          }
          else
          {
            v60 = *(_QWORD *)v60;
          }
        }
      }
    }
    else
    {
      v31 = *(_BYTE *)(v126 + 80);
      m128i_i64 = v31 == 13;
      v119 = v31 == 13;
      if ( a9 && v31 != 13 )
      {
        v32 = (unsigned int)sub_8287B0((_BYTE *)v126) == 0;
        v33 = 0;
        if ( v32 )
          v33 = a9;
        a9 = v33;
      }
      if ( dword_4F04C44 != -1 )
        goto LABEL_28;
    }
    v27 = qword_4F04C68[0];
    v48 = qword_4F04C68[0] + 776LL * dword_4F04C64;
    if ( (*(_BYTE *)(v48 + 6) & 6) != 0 || *(_BYTE *)(v48 + 4) == 12 )
      goto LABEL_28;
    if ( !dword_4D047C8 || unk_4F04C48 == -1 || (*(_BYTE *)(qword_4F04C68[0] + 776LL * unk_4F04C48 + 10) & 1) == 0 )
      goto LABEL_43;
LABEL_144:
    if ( !a9 )
    {
      v128 = a12 == 3;
      if ( !v126 )
      {
        v123 = 0;
        v122 = 0;
        goto LABEL_53;
      }
      goto LABEL_45;
    }
    if ( !v126 )
    {
      v122 = 0;
      v128 = 0;
      v123 = 0;
      goto LABEL_53;
    }
    v30 = a14;
    v85 = sub_878D80(a14, 0, a9);
    m128i_i64 = v85 == 0;
    v128 = v85 == 0;
    if ( v85 )
    {
      v110 = *(_QWORD *)(v85 + 32);
      if ( v110 )
      {
        v126 = v110;
        a9 = 1;
        goto LABEL_45;
      }
    }
LABEL_199:
    v86 = (__int64)a6;
    v135 = 0;
    v136 = 0;
    v137 = 0;
    while ( v86 )
    {
      if ( *(_BYTE *)(v86 + 8) )
        goto LABEL_200;
      v88 = *(_QWORD *)(v86 + 24);
      if ( *(_BYTE *)(v88 + 24) != 3 )
      {
        sub_7D38C0(*(_QWORD *)(v88 + 8), &v135);
LABEL_200:
        v87 = *(_QWORD *)v86;
        if ( !*(_QWORD *)v86 )
          break;
        goto LABEL_201;
      }
      sub_82C920(v88 + 8, &v135, &v137, &v136);
      v87 = *(_QWORD *)v86;
      if ( !*(_QWORD *)v86 )
        break;
LABEL_201:
      if ( *(_BYTE *)(v87 + 8) == 3 )
        v86 = sub_6BBB10((_QWORD *)v86);
      else
        v86 = v87;
    }
    sub_878710(v126, v138);
    v89 = 0;
    if ( !(a10 | v119) )
      v89 = v126;
    v124 = v89;
    v90 = sub_7D4C80(v89, v138, &v135, &v137, &v136, a11);
    v125 = (_QWORD *)v90;
    if ( a15 && v90 )
    {
      v91 = *(_QWORD *)(v90 + 8);
      v92 = *(_BYTE *)(v91 + 80);
      v93 = v91;
      if ( v92 == 16 )
      {
        v93 = **(_QWORD **)(v91 + 88);
        v92 = *(_BYTE *)(v93 + 80);
      }
      if ( v92 == 24 )
      {
        v93 = *(_QWORD *)(v93 + 88);
        v92 = *(_BYTE *)(v93 + 80);
      }
      if ( (unsigned __int8)(v92 - 10) <= 1u )
      {
        for ( j = *(_QWORD *)(*(_QWORD *)(v93 + 88) + 152LL); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
          ;
        if ( (*(_BYTE *)(*(_QWORD *)(j + 168) + 20LL) & 2) == 0 )
        {
          m128i_i64 = v121;
          if ( (unsigned int)sub_828ED0(v91, v121, 0, v91 != v124, v128, 0, 0, 0, 0) )
          {
            v104 = (_QWORD *)*v125;
            if ( !*v125 )
            {
LABEL_319:
              v34 = v128;
              v30 = (__int64)v125;
              sub_878490(v125);
              *a15 = 1;
              v115 = v125[1];
              v36 = v115;
              v35 = v128 && v122;
              goto LABEL_32;
            }
            while ( 1 )
            {
              v105 = v104[1];
              m128i_i64 = v125[1];
              if ( !v105 || !m128i_i64 )
                break;
              if ( m128i_i64 != v105 && !sub_828A00(v105, m128i_i64, v101, v102, v103) )
                goto LABEL_302;
              v104 = (_QWORD *)*v104;
              if ( !v104 )
                goto LABEL_319;
            }
            v91 = v125[1];
          }
          else
          {
LABEL_302:
            v91 = v125[1];
          }
        }
      }
LABEL_216:
      for ( k = v125; ; v91 = k[1] )
      {
        sub_8360D0(
          v91,
          v121,
          a3,
          (__int64)a6,
          0,
          a4,
          a5,
          0,
          0,
          1u,
          v124 != v91 || v125 != k,
          v128,
          0,
          0,
          0,
          a8,
          a12,
          (__int64 *)&v133,
          (__int64)&v134,
          &v130,
          &v131);
        if ( unk_4F04C48 != -1 && *(char *)(qword_4F04C68[0] + 776LL * unk_4F04C48 + 12) < 0 )
          break;
        k = (_QWORD *)*k;
        if ( !k )
          break;
      }
      v123 = 1;
    }
    else
    {
      v123 = 0;
      if ( v90 )
      {
        v91 = *(_QWORD *)(v90 + 8);
        goto LABEL_216;
      }
    }
    v30 = (__int64)v125;
    sub_878490(v125);
LABEL_53:
    if ( a19 )
    {
      LODWORD(v135) = 0;
      LODWORD(v136) = 0;
      for ( m = a5->m128i_i64[0]; *(_BYTE *)(m + 140) == 12; m = *(_QWORD *)(m + 160) )
        ;
      v51 = *(_QWORD **)(*(_QWORD *)(*(_QWORD *)m + 96LL) + 40LL);
      if ( v51 )
      {
        for ( n = v51[1]; n; n = v51[1] )
        {
          v53 = *(_BYTE *)(n + 80);
          v54 = n;
          if ( v53 == 16 )
          {
            v54 = **(_QWORD **)(n + 88);
            v53 = *(_BYTE *)(v54 + 80);
          }
          if ( v53 == 24 )
            v54 = *(_QWORD *)(v54 + 88);
          v55 = *(_QWORD *)(v54 + 88);
          for ( ii = *(_QWORD *)(v55 + 152); *(_BYTE *)(ii + 140) == 12; ii = *(_QWORD *)(ii + 160) )
            ;
          if ( (*(_BYTE *)(v55 + 194) & 1) == 0 )
          {
            v77 = *(_QWORD *)(ii + 160);
            v30 = v77;
            if ( (unsigned int)sub_8D32B0(v77) )
            {
              v78 = sub_8D46C0(v77);
              if ( (unsigned int)sub_8D2FB0(v77) )
              {
                if ( (unsigned int)sub_8D2E30(v78) )
                  v78 = sub_8D46C0(v78);
              }
              while ( *(_BYTE *)(v78 + 140) == 12 )
                v78 = *(_QWORD *)(v78 + 160);
              v30 = v78;
              if ( (unsigned int)sub_8D2310(v78) )
              {
                v79 = sub_82EAF0(ii, n, 1);
                v30 = (__int64)a5;
                if ( v79 )
                  sub_8399C0((__int64)a5, 0, 0, v79, ii, (__int64)v138);
                else
                  sub_838020(
                    (__int64)a5,
                    0,
                    *(__m128i **)(**(_QWORD **)(ii + 168) + 8LL),
                    **(_QWORD **)(ii + 168),
                    0,
                    0,
                    v138);
                if ( v138[0].m128i_i32[2] != 7 )
                {
                  v30 = 0;
                  sub_833B90(
                    0,
                    0,
                    0,
                    0,
                    n,
                    v78,
                    a6,
                    1,
                    a5,
                    0,
                    0,
                    0,
                    1u,
                    0,
                    0,
                    0,
                    0,
                    0,
                    (int)&dword_400000,
                    0,
                    (__int64 *)&v133,
                    &v135,
                    &v136,
                    &v137);
                }
              }
            }
          }
          v51 = (_QWORD *)*v51;
          if ( !v51 )
            break;
        }
      }
    }
    m128i_i64 = (__int64)&dword_4F04C44;
    v57 = (_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64);
    if ( dword_4F04C44 == -1 && (v57[6] & 6) == 0 && v57[4] != 12 || (v57[12] & 0x10) != 0 || (v58 = v133) == 0 )
    {
LABEL_149:
      m128i_i64 = (__int64)a13;
      v30 = (__int64)&v133;
      sub_82D8D0((__int64 *)&v133, (__int64)a13, &v132, (_DWORD *)&v132 + 1, v23, v24);
      v69 = v132;
      *a20 = 0;
      if ( v69 )
      {
        v70 = v117 != 0 && !v118;
        goto LABEL_151;
      }
      v80 = sub_82BD70(&v133, a13, v27);
      if ( *(_QWORD *)(v80 + 1024) && (**(_BYTE **)(v80 + 1008) & 1) != 0 )
      {
        v70 = v117 != 0 && !v118;
        goto LABEL_151;
      }
      v71 = v133;
      if ( v133 )
      {
        v81 = !v118;
        if ( !HIDWORD(v132) )
        {
          m128i_i64 = (__int64)a20;
          jj = v133[1];
          v115 = v133[2];
          *a20 = v133[15];
          v71[15] = 0;
          if ( (*((_BYTE *)v71 + 145) & 8) != 0 )
            *a16 = 1;
          v96 = v71[7];
          if ( v96 )
          {
            m128i_i64 = (__int64)a19;
            *a19 = v96;
          }
          v70 = v117 != 0 && v81;
          goto LABEL_152;
        }
        if ( !*v133 && (v82 = v133[1]) != 0 && (*(_BYTE *)(v82 + 82) & 4) != 0 )
        {
          v83 = sub_6E5430();
          LOBYTE(v27) = v117 != 0;
          v70 = v117 != 0 && v81;
          if ( v83 )
          {
            m128i_i64 = (__int64)a13;
            v30 = 266;
            sub_6854C0(0x10Au, a13, v126);
          }
        }
        else
        {
          do
          {
            if ( v71[7] )
            {
              v108 = a5->m128i_i64[0];
              if ( (a5[1].m128i_i8[2] & 2) != 0 )
              {
                v30 = a5->m128i_i64[0];
                v108 = sub_8D46C0(a5->m128i_i64[0]);
                if ( !(unsigned int)sub_6E5430() )
                  goto LABEL_190;
              }
              else if ( !(unsigned int)sub_6E5430() )
              {
                goto LABEL_190;
              }
              m128i_i64 = (__int64)a13;
              v30 = 982;
              v117 = sub_67DA80(0x3D6u, a13, v108);
              goto LABEL_190;
            }
            v71 = (__int64 *)*v71;
          }
          while ( v71 );
          if ( (unsigned int)sub_6E5430() )
          {
            v30 = dword_3C1E360[a12];
            m128i_i64 = (__int64)a13;
            v117 = sub_67E020(v30, a13, v126);
          }
LABEL_190:
          v84 = sub_6E5430();
          LOBYTE(v27) = v117 != 0;
          v70 = v117 != 0 && v81;
          if ( v84 )
          {
            m128i_i64 = (__int64)a5;
            v30 = (__int64)v133;
            sub_82E650(v133, a5, (__int64)a6, 0, v117);
          }
        }
LABEL_151:
        v71 = v133;
        for ( jj = 0; v71; qword_4D03C68 = v73 )
        {
LABEL_152:
          v73 = v71;
          v71 = (__int64 *)*v71;
          sub_725130((__int64 *)v73[5]);
          v30 = v73[15];
          sub_82D8A0((_QWORD *)v30);
          *v73 = (__int64)qword_4D03C68;
        }
        if ( v70 )
        {
          v74 = sub_82BD70(v30, m128i_i64, v27);
          if ( (unsigned int)sub_6E5430() )
          {
            if ( *(__int64 *)(v74 + 1024) <= 1 )
            {
              v27 = *(_QWORD *)(v74 + 1008);
              if ( (*(_BYTE *)v27 & 1) == 0 )
              {
                v118 = 0;
                *(_BYTE *)v27 |= 1u;
                continue;
              }
            }
          }
          v36 = jj;
          v34 = v128;
        }
        else
        {
          v36 = jj;
          v34 = v128;
          v35 = v128 && v122;
          if ( !v117 )
            goto LABEL_32;
        }
        v75 = sub_82BD70(v30, m128i_i64, v27);
        v76 = (const __m128i *)(*(_QWORD *)(v75 + 1008) + 8 * (5LL * *(_QWORD *)(v75 + 1024) - 5));
        if ( v76[1].m128i_i64[0] )
        {
          m128i_i64 = (__int64)v76[1].m128i_i64;
          sub_67E370((__int64)v117, v76 + 1);
        }
        v30 = (__int64)v117;
        sub_685910((__int64)v117, (FILE *)m128i_i64);
        v35 = v34 & v122;
LABEL_32:
        if ( dword_4D047C8 )
          goto LABEL_82;
        goto LABEL_33;
      }
      if ( v134 )
      {
LABEL_242:
        if ( a5 )
        {
          m128i_i64 = (__int64)a5;
          v97 = a5->m128i_i64[0];
          if ( v131 )
          {
            v98 = 1087;
            if ( v97 )
              goto LABEL_245;
LABEL_308:
            v30 = 0;
            m128i_i64 = (__int64)v138;
            v97 = 0;
            v98 = 1087;
            if ( (unsigned int)sub_830940(0, v138[0].m128i_i64) )
              v97 = v138[0].m128i_i64[0];
LABEL_245:
            v70 = !v118;
            if ( !v117 )
            {
              if ( (unsigned int)sub_6E5430() )
              {
                v117 = sub_67E020(v98, a13, v126);
                sub_82E4F0(v97, (__int64)a6, v117);
                v30 = v134;
                m128i_i64 = (__int64)v117;
                sub_87CA90(v134, v117);
                v70 &= v117 != 0;
              }
              else
              {
                v70 = 0;
              }
            }
LABEL_246:
            if ( a3 )
            {
              v99 = *(_QWORD *)(a3 + 48);
              if ( v99 )
              {
                if ( *(_QWORD *)(v99 + 96) && (unsigned int)sub_6E5430() )
                {
                  v30 = v99 + 8;
                  sub_6E1690(v99 + 8);
                }
              }
            }
            goto LABEL_151;
          }
        }
        else
        {
          v97 = 0;
          if ( v131 )
            goto LABEL_308;
        }
        m128i_i64 = (__int64)dword_3C1E3A0;
        v98 = dword_3C1E3A0[a12];
        if ( (unsigned __int8)(*(_BYTE *)(v126 + 80) - 10) <= 1u && (v98 == 304 || v98 == 384) )
        {
          if ( (*(_BYTE *)(v126 + 104) & 1) != 0 )
          {
            v30 = v126;
            v109 = sub_8796F0(v126);
          }
          else
          {
            v109 = (*(_BYTE *)(*(_QWORD *)(v126 + 88) + 208LL) & 4) != 0;
          }
          if ( !v109 )
            v98 = 1767;
        }
        goto LABEL_245;
      }
      if ( v130 )
      {
        v113 = sub_6E5430();
        LOBYTE(v27) = v117 != 0;
        v70 = v117 != 0 && !v118;
        if ( v113 )
        {
          m128i_i64 = (__int64)a13;
          v30 = 245;
          sub_6851C0(0xF5u, a13);
        }
        goto LABEL_246;
      }
      if ( ((v123 ^ 1) & v119) != 0 )
      {
        if ( (*(_BYTE *)(qword_4D03C50 + 18LL) & 0x40) != 0 )
        {
          v70 = v117 != 0 && !v118;
        }
        else
        {
          v30 = v126;
          v70 = 0;
          sub_886080(v126);
        }
        if ( !(unsigned int)sub_6E5430() )
          goto LABEL_246;
      }
      else
      {
        v111 = !v118;
        if ( !a10 )
        {
          if ( v126 )
            goto LABEL_242;
          v112 = sub_6E5430();
          LOBYTE(v27) = v117 != 0;
          v70 = v117 != 0 && v111;
          if ( v112 )
          {
            m128i_i64 = (__int64)a13;
            v30 = 980;
            sub_6851C0(0x3D4u, a13);
          }
          goto LABEL_246;
        }
        v114 = sub_6E5430();
        LOBYTE(v27) = v117 != 0;
        v70 = v117 != 0 && v111;
        if ( !v114 )
          goto LABEL_246;
      }
      m128i_i64 = (__int64)a13;
      v30 = dword_3C1E320[a12];
      sub_6851A0(v30, a13, *(_QWORD *)(*(_QWORD *)v126 + 8LL));
      goto LABEL_246;
    }
    break;
  }
  while ( 2 )
  {
    v59 = v58[1];
    if ( !v59 || (*(_BYTE *)(v59 + 81) & 0x10) == 0 || (*(_BYTE *)(*(_QWORD *)(v59 + 64) + 177LL) & 0x20) == 0 )
      goto LABEL_72;
    v22 = *(unsigned __int8 *)(v59 + 80);
    if ( (_BYTE)v22 == 20 )
    {
      v27 = *(_QWORD *)(v59 + 88);
      if ( (*(_BYTE *)(v27 + 160) & 0x20) != 0 )
        break;
      v22 = *(_QWORD *)(v27 + 176);
      if ( *(_QWORD *)(v22 + 216) )
        break;
      v27 = *(_QWORD *)(*(_QWORD *)(v27 + 104) + 176LL);
      if ( v27 )
      {
        if ( *(_QWORD *)(v27 + 16) )
          break;
      }
      goto LABEL_72;
    }
    if ( (_BYTE)v22 != 10 || (v27 = *(_QWORD *)(v59 + 88), !*(_QWORD *)(v27 + 216)) )
    {
LABEL_72:
      v58 = (__int64 *)*v58;
      if ( !v58 )
        goto LABEL_149;
      continue;
    }
    break;
  }
  v35 = v122;
  v34 = 1;
  v36 = 0;
  *a17 = 1;
  *a20 = 0;
  if ( dword_4D047C8 )
  {
LABEL_82:
    if ( sub_827E90(v30, m128i_i64, v27, v22) && !v34 )
    {
      if ( a9 )
      {
        if ( !v36 || (v30 = v36, !(unsigned int)sub_880A60(v36)) )
        {
          m128i_i64 = a14;
          v30 = v115;
          sub_878E70(v115, a14, 0, 0, 0);
        }
      }
    }
  }
LABEL_33:
  if ( v35 && (*(_BYTE *)(v126 + 81) & 0x10) == 0 )
    *(_BYTE *)(*(_QWORD *)v126 + 73LL) |= 8u;
  v39 = sub_82BD70(v30, m128i_i64, v27);
  v41 = *(_QWORD *)(v39 + 1008);
  v42 = *(_QWORD *)(v41 + 8 * (5LL * *(_QWORD *)(v39 + 1024) - 5) + 32);
  if ( v42 )
  {
    sub_823A00(*(_QWORD *)v42, 16LL * (unsigned int)(*(_DWORD *)(v42 + 8) + 1), v37, v38, v41, v40);
    sub_823A00(v42, 16, v43, v44, v45, v46);
  }
  --*(_QWORD *)(v39 + 1024);
  return v36;
}
