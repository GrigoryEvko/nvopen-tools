// Function: sub_EEDD80
// Address: 0xeedd80
//
__int64 __fastcall sub_EEDD80(char **a1, __int64 a2)
{
  signed __int64 v2; // rax
  unsigned __int8 *v3; // rdx
  __int64 v4; // r8
  char *v5; // r9
  __int64 v6; // r12
  unsigned __int8 *v7; // r13
  char *v8; // rax
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  char *v13; // r12
  char *v14; // r13
  char *v15; // rbx
  __int64 v16; // rax
  char *v17; // rax
  char *v18; // rcx
  __int64 v19; // r13
  unsigned __int64 v20; // rcx
  char *v21; // rax
  __int64 v22; // rsi
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  char *v28; // r12
  __int64 v29; // rdx
  __int64 v30; // rbx
  char *v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rcx
  __int64 v34; // r8
  __int64 v35; // r9
  char v36; // al
  unsigned __int64 *v37; // r13
  __int64 v38; // rdx
  char *v39; // rax
  char v40; // dl
  char *v41; // rax
  char v42; // al
  _BYTE **v43; // rsi
  _QWORD *v44; // rax
  __int64 v45; // r9
  __int64 *v46; // rax
  __int64 v47; // rax
  char v48; // r14
  __int64 v49; // rax
  __int64 v50; // rdx
  __int64 v51; // rcx
  __int64 v52; // r8
  __int64 v53; // r9
  __int64 v54; // r10
  char *v55; // rax
  __int64 v56; // rdx
  __int64 v57; // rcx
  __int64 v58; // r8
  __int64 v59; // r9
  __int64 v60; // rcx
  __int64 v61; // r8
  __int64 v62; // r9
  __int64 v63; // rdx
  __int64 v64; // r10
  char *v65; // r12
  char *v66; // rbx
  __int64 v67; // rax
  unsigned __int64 v68; // r10
  unsigned __int64 v69; // rdx
  unsigned __int64 v70; // r10
  __int64 v71; // rax
  __int64 v72; // rdx
  __int64 v73; // rcx
  __int64 v74; // r8
  __int64 v75; // r9
  __int64 v76; // rdx
  __int64 v77; // rcx
  __int64 v78; // r9
  __int64 v79; // r10
  __int64 v80; // r8
  unsigned __int64 *v81; // rbx
  __int64 v82; // rax
  unsigned __int64 v83; // r10
  unsigned __int64 v84; // rdx
  unsigned __int64 v85; // r10
  __int64 v86; // rax
  _QWORD *v87; // rax
  __int64 v88; // r9
  __int64 *v89; // rax
  __int64 v90; // rax
  __int64 v91; // rbx
  __int64 v92; // rax
  char *v93; // rcx
  char v94; // r14
  __int64 v95; // rax
  char *v96; // rax
  __int64 v97; // rax
  __int64 v98; // r9
  __int64 v99; // rax
  __int64 v100; // r9
  __int64 *v101; // rdx
  int v102; // [rsp+0h] [rbp-1A0h]
  unsigned __int64 v103; // [rsp+0h] [rbp-1A0h]
  int v104; // [rsp+0h] [rbp-1A0h]
  unsigned __int64 v105; // [rsp+0h] [rbp-1A0h]
  __int64 v106; // [rsp+10h] [rbp-190h]
  __int64 v107; // [rsp+10h] [rbp-190h]
  __int64 v108; // [rsp+18h] [rbp-188h]
  char *v109; // [rsp+18h] [rbp-188h]
  __int64 v110; // [rsp+20h] [rbp-180h]
  __int64 v111; // [rsp+20h] [rbp-180h]
  __int64 v112; // [rsp+20h] [rbp-180h]
  char v113; // [rsp+28h] [rbp-178h]
  char *v114; // [rsp+28h] [rbp-178h]
  unsigned __int8 *v115; // [rsp+30h] [rbp-170h]
  signed __int64 v116; // [rsp+38h] [rbp-168h]
  __int64 v117; // [rsp+40h] [rbp-160h]
  __int64 v118; // [rsp+48h] [rbp-158h]
  _QWORD *v119; // [rsp+48h] [rbp-158h]
  __int64 v120; // [rsp+48h] [rbp-158h]
  char *v121; // [rsp+48h] [rbp-158h]
  char *v122; // [rsp+50h] [rbp-150h]
  char v123; // [rsp+50h] [rbp-150h]
  char *v124; // [rsp+58h] [rbp-148h]
  _QWORD *v125; // [rsp+58h] [rbp-148h]
  __int64 v126; // [rsp+58h] [rbp-148h]
  char *v127; // [rsp+58h] [rbp-148h]
  char *v128; // [rsp+58h] [rbp-148h]
  __int64 *v129; // [rsp+68h] [rbp-138h] BYREF
  __int64 *v130; // [rsp+70h] [rbp-130h] BYREF
  __int64 v131; // [rsp+78h] [rbp-128h]
  _QWORD v132[3]; // [rsp+80h] [rbp-120h] BYREF
  _OWORD v133[4]; // [rsp+98h] [rbp-108h] BYREF
  char v134[8]; // [rsp+D8h] [rbp-C8h] BYREF
  _BYTE *v135; // [rsp+E0h] [rbp-C0h] BYREF
  __int64 v136; // [rsp+E8h] [rbp-B8h]
  _BYTE v137[176]; // [rsp+F0h] [rbp-B0h] BYREF

  if ( a2 )
    a1[84] = a1[83];
  if ( (unsigned __int8)sub_EE3B50((const void **)a1, 2u, "Ut") )
  {
    v2 = sub_EE32C0(a1, 0);
    v5 = 0;
    v6 = v2;
    v7 = v3;
    v8 = *a1;
    if ( a1[1] != *a1 && *v8 == 95 )
    {
      *a1 = v8 + 1;
      v42 = *((_BYTE *)a1 + 937);
      v135 = v137;
      v123 = v42;
      v136 = 0x2000000000LL;
      sub_EE3C10((__int64)&v135, 0x33u, v6, v3, v4, 0);
      v43 = &v135;
      v44 = sub_C65B40((__int64)(a1 + 113), (__int64)&v135, (__int64 *)&v130, (__int64)off_497B2F0);
      v5 = (char *)v44;
      if ( v44 )
      {
        v45 = (__int64)(v44 + 1);
        if ( v135 != v137 )
        {
          v125 = v44 + 1;
          _libc_free(v135, &v135);
          v45 = (__int64)v125;
        }
        v135 = (_BYTE *)v45;
        v126 = v45;
        v46 = sub_EE6840((__int64)(a1 + 118), (__int64 *)&v135);
        v5 = (char *)v126;
        if ( v46 )
        {
          v47 = v46[1];
          if ( v47 )
            v5 = (char *)v47;
        }
        if ( a1[116] == v5 )
          *((_BYTE *)a1 + 936) = 1;
      }
      else
      {
        if ( v123 )
        {
          v97 = sub_CD1D40((__int64 *)a1 + 101, 40, 3);
          *(_QWORD *)v97 = 0;
          v43 = (_BYTE **)v97;
          v98 = v97 + 8;
          *(_WORD *)(v97 + 16) = 16435;
          LOBYTE(v97) = *(_BYTE *)(v97 + 18);
          v43[3] = (_BYTE *)v6;
          v43[4] = v7;
          v128 = (char *)v98;
          *((_BYTE *)v43 + 18) = v97 & 0xF0 | 5;
          v43[1] = &unk_49E00E8;
          sub_C657C0((__int64 *)a1 + 113, (__int64 *)v43, v130, (__int64)off_497B2F0);
          v5 = v128;
        }
        if ( v135 != v137 )
        {
          v127 = v5;
          _libc_free(v135, v43);
          v5 = v127;
        }
        a1[115] = v5;
      }
    }
    return (__int64)v5;
  }
  if ( (unsigned __int8)sub_EE3B50((const void **)a1, 2u, "Ul") )
  {
    v13 = a1[84];
    v14 = a1[83];
    v15 = a1[98];
    v130 = (__int64 *)a1;
    v132[2] = v134;
    v124 = v15;
    v16 = (v13 - v14) >> 3;
    v132[0] = v133;
    a1[98] = (char *)v16;
    v131 = v16;
    v132[1] = v133;
    memset(v133, 0, sizeof(v133));
    if ( v13 != a1[85] )
    {
LABEL_9:
      a1[84] = v13 + 8;
      *(_QWORD *)v13 = v132;
      v17 = *a1;
      v18 = a1[1];
      v19 = (a1[3] - a1[2]) >> 3;
      if ( *a1 != v18 )
      {
        v20 = v18 - v17;
        do
        {
          if ( *v17 != 84 )
            break;
          if ( v20 <= 1 )
            break;
          v21 = (char *)memchr("yptnk", v17[1], 5u);
          if ( !v21 || v21 == "" )
            break;
          v22 = (__int64)v132;
          v23 = sub_EF6290(a1, v132);
          v135 = (_BYTE *)v23;
          if ( !v23 )
            goto LABEL_45;
          sub_E18380((__int64)(a1 + 2), (__int64 *)&v135, v24, v25, v26, v27);
          v17 = *a1;
          v20 = a1[1] - *a1;
        }
        while ( *a1 != a1[1] );
      }
      v22 = v19;
      v28 = (char *)sub_EE6060(a1, v19);
      v30 = v29;
      if ( !v29 )
        a1[84] -= 8;
      v118 = 0;
      v31 = *a1;
      if ( *a1 != a1[1] && *v31 == 81 )
      {
        v94 = *((_BYTE *)a1 + 778);
        *((_BYTE *)a1 + 778) = 1;
        *a1 = v31 + 1;
        v95 = sub_EEA9F0((__int64)a1);
        *((_BYTE *)a1 + 778) = v94;
        v118 = v95;
        if ( !v95 )
          goto LABEL_29;
      }
      v22 = 1;
      if ( !(unsigned __int8)sub_EE3B50((const void **)a1, 1u, "v") )
      {
        while ( 1 )
        {
          v23 = sub_EF1F20(a1);
          v135 = (_BYTE *)v23;
          if ( !v23 )
            break;
          v22 = (__int64)&v135;
          sub_E18380((__int64)(a1 + 2), (__int64 *)&v135, v32, v33, v34, v35);
          if ( a1[1] != *a1 )
          {
            v36 = **a1;
            if ( v36 == 69 || v36 == 81 )
              goto LABEL_26;
          }
        }
LABEL_45:
        v5 = (char *)v23;
        goto LABEL_30;
      }
LABEL_26:
      v22 = v19;
      v37 = (unsigned __int64 *)sub_EE6060(a1, v19);
      v117 = v38;
      v39 = *a1;
      if ( *a1 == a1[1] )
        goto LABEL_29;
      v40 = *v39;
      if ( *v39 == 81 )
      {
        v48 = *((_BYTE *)a1 + 778);
        *((_BYTE *)a1 + 778) = 1;
        *a1 = v39 + 1;
        v49 = sub_EEA9F0((__int64)a1);
        *((_BYTE *)a1 + 778) = v48;
        if ( !v49 )
          goto LABEL_29;
        v39 = *a1;
        if ( *a1 == a1[1] )
          goto LABEL_29;
        v40 = *v39;
      }
      if ( v40 == 69 )
      {
        v22 = 0;
        *a1 = v39 + 1;
        v116 = sub_EE32C0(a1, 0);
        v55 = *a1;
        v115 = (unsigned __int8 *)v50;
        if ( *a1 != a1[1] && *v55 == 95 )
        {
          *a1 = v55 + 1;
          v108 = v54;
          v113 = *((_BYTE *)a1 + 937);
          v135 = v137;
          v136 = 0x2000000000LL;
          sub_D953B0((__int64)&v135, 52, v50, v51, v52, v53);
          sub_D953B0((__int64)&v135, v30, v56, v57, v58, v59);
          v61 = (__int64)&v28[8 * v30];
          v62 = (__int64)v28;
          v63 = (unsigned int)v136;
          v64 = v108;
          if ( (char *)v61 != v28 )
          {
            v110 = v108;
            v109 = v28;
            v65 = &v28[8 * v30];
            v106 = v30;
            v66 = (char *)v62;
            do
            {
              v67 = (unsigned int)v63;
              v68 = *(_QWORD *)v66;
              v69 = (unsigned int)v63 + 1LL;
              if ( v69 > HIDWORD(v136) )
              {
                v103 = *(_QWORD *)v66;
                sub_C8D5F0((__int64)&v135, v137, v69, 4u, v61, v62);
                v67 = (unsigned int)v136;
                v68 = v103;
              }
              *(_DWORD *)&v135[4 * v67] = v68;
              v70 = HIDWORD(v68);
              v60 = HIDWORD(v136);
              LODWORD(v136) = v136 + 1;
              v71 = (unsigned int)v136;
              if ( (unsigned __int64)(unsigned int)v136 + 1 > HIDWORD(v136) )
              {
                v102 = v70;
                sub_C8D5F0((__int64)&v135, v137, (unsigned int)v136 + 1LL, 4u, v61, v62);
                v71 = (unsigned int)v136;
                LODWORD(v70) = v102;
              }
              v66 += 8;
              *(_DWORD *)&v135[4 * v71] = v70;
              v63 = (unsigned int)(v136 + 1);
              LODWORD(v136) = v136 + 1;
            }
            while ( v65 != v66 );
            v64 = v110;
            v28 = v109;
            v30 = v106;
          }
          v111 = v64;
          sub_D953B0((__int64)&v135, v118, v63, v60, v61, v62);
          sub_D953B0((__int64)&v135, v117, v72, v73, v74, v75);
          v79 = v111;
          v80 = (__int64)&v37[v117];
          if ( v37 != (unsigned __int64 *)v80 )
          {
            LODWORD(v76) = v136;
            v107 = v30;
            v81 = v37;
            do
            {
              v82 = (unsigned int)v76;
              v83 = *v81;
              v84 = (unsigned int)v76 + 1LL;
              if ( v84 > HIDWORD(v136) )
              {
                v105 = *v81;
                sub_C8D5F0((__int64)&v135, v137, v84, 4u, v80, v78);
                v82 = (unsigned int)v136;
                v83 = v105;
              }
              *(_DWORD *)&v135[4 * v82] = v83;
              v85 = HIDWORD(v83);
              v77 = HIDWORD(v136);
              LODWORD(v136) = v136 + 1;
              v86 = (unsigned int)v136;
              if ( (unsigned __int64)(unsigned int)v136 + 1 > HIDWORD(v136) )
              {
                v104 = v85;
                sub_C8D5F0((__int64)&v135, v137, (unsigned int)v136 + 1LL, 4u, v80, v78);
                v86 = (unsigned int)v136;
                LODWORD(v85) = v104;
              }
              ++v81;
              *(_DWORD *)&v135[4 * v86] = v85;
              v76 = (unsigned int)(v136 + 1);
              LODWORD(v136) = v136 + 1;
            }
            while ( &v37[v117] != v81 );
            v79 = v111;
            v30 = v107;
          }
          v112 = v79;
          sub_D953B0((__int64)&v135, v79, v76, v77, v80, v78);
          if ( v116 )
            sub_C653C0((__int64)&v135, v115, v116);
          else
            sub_C653C0((__int64)&v135, 0, 0);
          v22 = (__int64)&v135;
          v87 = sub_C65B40((__int64)(a1 + 113), (__int64)&v135, (__int64 *)&v129, (__int64)off_497B2F0);
          v5 = (char *)v87;
          if ( v87 )
          {
            v88 = (__int64)(v87 + 1);
            if ( v135 != v137 )
            {
              v119 = v87 + 1;
              _libc_free(v135, &v135);
              v88 = (__int64)v119;
            }
            v22 = (__int64)&v135;
            v135 = (_BYTE *)v88;
            v120 = v88;
            v89 = sub_EE6840((__int64)(a1 + 118), (__int64 *)&v135);
            v5 = (char *)v120;
            if ( v89 )
            {
              v90 = v89[1];
              if ( v90 )
                v5 = (char *)v90;
            }
            if ( a1[116] == v5 )
              *((_BYTE *)a1 + 936) = 1;
          }
          else
          {
            if ( v113 )
            {
              v99 = sub_CD1D40((__int64 *)a1 + 101, 88, 3);
              *(_QWORD *)v99 = 0;
              v22 = v99;
              v100 = v99 + 8;
              v101 = v129;
              *(_WORD *)(v99 + 16) = 16436;
              LOBYTE(v99) = *(_BYTE *)(v99 + 18);
              *(_QWORD *)(v22 + 24) = v28;
              *(_QWORD *)(v22 + 32) = v30;
              *(_QWORD *)(v22 + 48) = v37;
              *(_BYTE *)(v22 + 18) = v99 & 0xF0 | 5;
              *(_QWORD *)(v22 + 64) = v112;
              v114 = (char *)v100;
              *(_QWORD *)(v22 + 8) = &unk_49E0148;
              *(_QWORD *)(v22 + 40) = v118;
              *(_QWORD *)(v22 + 56) = v117;
              *(_QWORD *)(v22 + 72) = v116;
              *(_QWORD *)(v22 + 80) = v115;
              sub_C657C0((__int64 *)a1 + 113, (__int64 *)v22, v101, (__int64)off_497B2F0);
              v5 = v114;
            }
            if ( v135 != v137 )
            {
              v121 = v5;
              _libc_free(v135, v22);
              v5 = v121;
            }
            a1[115] = v5;
          }
          goto LABEL_30;
        }
      }
LABEL_29:
      v5 = 0;
LABEL_30:
      v130[84] = v130[83] + 8 * v131;
      if ( (_OWORD *)v132[0] != v133 )
      {
        v122 = v5;
        _libc_free(v132[0], v22);
        v5 = v122;
      }
      a1[98] = v124;
      return (__int64)v5;
    }
    v91 = 16 * v16;
    if ( v14 == (char *)(a1 + 86) )
    {
      v96 = (char *)malloc(v91, 2, v134, v10, v11, v12);
      v93 = v96;
      if ( v96 )
      {
        if ( v13 != v14 )
          v93 = (char *)memmove(v96, v14, v13 - v14);
        a1[83] = v93;
        goto LABEL_80;
      }
    }
    else
    {
      v92 = realloc(v14);
      a1[83] = (char *)v92;
      v93 = (char *)v92;
      if ( v92 )
      {
LABEL_80:
        v13 = &v93[v13 - v14];
        a1[85] = &v93[v91];
        goto LABEL_9;
      }
    }
    abort();
  }
  if ( !(unsigned __int8)sub_EE3B50((const void **)a1, 2u, "Ub") )
    return 0;
  sub_EE32C0(a1, 0);
  v41 = *a1;
  if ( *a1 == a1[1] || *v41 != 95 )
    return 0;
  *a1 = v41 + 1;
  return sub_EE68C0((__int64)(a1 + 101), "'block-literal'");
}
