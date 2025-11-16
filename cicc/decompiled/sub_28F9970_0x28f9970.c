// Function: sub_28F9970
// Address: 0x28f9970
//
void __fastcall sub_28F9970(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  _QWORD *v3; // rbx
  _BYTE *v4; // r8
  size_t v5; // r14
  bool v6; // r12
  unsigned __int64 *v7; // rax
  char v8; // al
  unsigned __int8 *v9; // r12
  __int64 v10; // rdi
  __int64 v11; // rax
  __int64 v12; // rax
  unsigned __int64 *v13; // rdi
  unsigned __int8 **v14; // rax
  unsigned __int8 *v15; // rsi
  __int64 v16; // rdx
  __int64 v17; // r12
  __int64 v18; // rax
  __int64 *v19; // rcx
  __int64 v20; // rbx
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rcx
  __int64 v24; // rcx
  __m128i *v25; // r14
  __int64 v26; // rsi
  bool v27; // r15
  bool v28; // al
  char v29; // r14
  unsigned int v30; // r15d
  unsigned __int64 v31; // rdx
  __int64 v32; // rdx
  _QWORD *v33; // rax
  unsigned __int8 *v34; // rdi
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // r14
  unsigned __int8 *v38; // rbx
  __int64 j; // r13
  _BYTE *v40; // rdx
  __int64 v41; // rdx
  __int64 v42; // r8
  unsigned __int8 v43; // dl
  char v44; // al
  __int64 v45; // rax
  _BYTE *v46; // rax
  __int64 v47; // rax
  __int64 v48; // rdx
  __int64 v49; // r14
  __int64 v50; // rax
  __int64 v51; // rcx
  __int64 v52; // rdx
  __int64 v53; // rax
  const char *v54; // rbx
  int v55; // esi
  const char *v56; // r15
  __int64 v57; // r8
  __int64 v58; // r9
  const char *v59; // r15
  int v60; // esi
  const char *v61; // r15
  int v62; // esi
  unsigned __int8 *v63; // r15
  __int64 v64; // rax
  unsigned __int8 *v65; // rbx
  const char *v66; // r13
  int v67; // edx
  int v68; // eax
  __int64 v69; // rcx
  unsigned __int8 *v70; // rbx
  const char *v71; // r15
  __int64 v72; // rax
  __int64 v73; // rax
  __int64 v74; // rax
  __int64 v75; // r14
  _BYTE *v76; // rdx
  __int64 v77; // rdx
  __int64 v78; // rcx
  unsigned __int8 **v79; // rdi
  unsigned __int8 **i; // rsi
  __int64 *v81; // r13
  __int64 v82; // rax
  __int64 *v83; // r15
  unsigned __int8 *v84; // r14
  unsigned __int8 **v85; // rax
  __int64 v86; // rax
  unsigned __int8 *v87; // rax
  unsigned __int8 *v88; // r13
  unsigned __int8 **v89; // rax
  unsigned __int8 **v90; // rsi
  __int64 *v91; // rax
  __m128i *v92; // rbx
  __int64 v93; // rsi
  __int64 m128i_i64; // r13
  __int64 v95; // rdx
  __int64 v96; // rax
  __int64 v97; // rax
  __int64 v98; // rax
  bool v99; // bl
  __int64 v100; // rsi
  unsigned __int8 *v101; // rsi
  __int64 v102; // rsi
  unsigned __int8 *v103; // rsi
  int v104; // edx
  unsigned __int8 **v105; // rax
  unsigned __int8 *v106; // rbx
  const char *v107; // r15
  unsigned __int8 *v108; // rbx
  const char *v109; // r15
  unsigned int v110; // esi
  unsigned __int8 **v111; // rax
  __int64 v112; // [rsp+0h] [rbp-140h]
  bool v113; // [rsp+Fh] [rbp-131h]
  __int64 v114; // [rsp+10h] [rbp-130h]
  _BYTE *srcc; // [rsp+28h] [rbp-118h]
  unsigned __int8 *src; // [rsp+28h] [rbp-118h]
  unsigned __int8 *srca; // [rsp+28h] [rbp-118h]
  unsigned __int8 *srcb; // [rsp+28h] [rbp-118h]
  unsigned __int64 v120; // [rsp+30h] [rbp-110h] BYREF
  unsigned __int64 v121; // [rsp+38h] [rbp-108h]
  unsigned int v122; // [rsp+40h] [rbp-100h]
  unsigned __int64 v123; // [rsp+48h] [rbp-F8h]
  unsigned int v124; // [rsp+50h] [rbp-F0h]
  unsigned __int8 **v125; // [rsp+60h] [rbp-E0h] BYREF
  unsigned __int64 v126; // [rsp+68h] [rbp-D8h]
  unsigned __int8 *v127; // [rsp+70h] [rbp-D0h] BYREF
  unsigned __int64 v128; // [rsp+78h] [rbp-C8h]
  unsigned int v129; // [rsp+80h] [rbp-C0h]
  __m128i v130; // [rsp+B0h] [rbp-90h] BYREF
  unsigned __int64 v131; // [rsp+C0h] [rbp-80h] BYREF
  __int64 v132; // [rsp+C8h] [rbp-78h]
  unsigned __int8 *v133; // [rsp+D0h] [rbp-70h] BYREF
  unsigned __int8 *v134; // [rsp+D8h] [rbp-68h]
  __int64 v135; // [rsp+E0h] [rbp-60h]
  __int64 v136; // [rsp+E8h] [rbp-58h]
  __int16 v137; // [rsp+F0h] [rbp-50h]

  if ( (unsigned __int8)(*(_BYTE *)a2 - 41) > 0x12u )
    return;
  v2 = a2;
  v130.m128i_i64[0] = (__int64)&v131;
  v3 = (_QWORD *)sub_B43CA0(a2);
  v4 = (_BYTE *)v3[29];
  v5 = v3[30];
  v6 = v4 == 0 && &v4[v5] != 0;
  if ( v6 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v125 = (unsigned __int8 **)v3[30];
  if ( v5 > 0xF )
  {
    srcc = v4;
    v12 = sub_22409D0((__int64)&v130, (unsigned __int64 *)&v125, 0);
    v4 = srcc;
    v130.m128i_i64[0] = v12;
    v13 = (unsigned __int64 *)v12;
    v131 = (unsigned __int64)v125;
LABEL_29:
    memcpy(v13, v4, v5);
    v5 = (size_t)v125;
    v7 = (unsigned __int64 *)v130.m128i_i64[0];
    goto LABEL_7;
  }
  if ( v5 == 1 )
  {
    LOBYTE(v131) = *v4;
    v7 = &v131;
    goto LABEL_7;
  }
  if ( v5 )
  {
    v13 = &v131;
    goto LABEL_29;
  }
  v7 = &v131;
LABEL_7:
  v130.m128i_i64[1] = v5;
  *((_BYTE *)v7 + v5) = 0;
  v133 = (unsigned __int8 *)v3[33];
  v134 = (unsigned __int8 *)v3[34];
  v135 = v3[35];
  if ( (unsigned int)((_DWORD)v133 - 42) > 1 )
  {
    if ( (unsigned __int64 *)v130.m128i_i64[0] != &v131 )
    {
      j_j___libc_free_0(v130.m128i_u64[0]);
      if ( *(_BYTE *)a2 != 54 )
        goto LABEL_12;
LABEL_32:
      v14 = (unsigned __int8 **)sub_986520(a2);
      if ( *v14[4] != 17 )
        goto LABEL_12;
      if ( !sub_28ED300(*v14, 17) )
      {
        v86 = *(_QWORD *)(a2 + 16);
        if ( !v86
          || *(_QWORD *)(v86 + 8)
          || !sub_28ED300(*(unsigned __int8 **)(v86 + 24), 17)
          && (v6 || !sub_28ED300(*(unsigned __int8 **)(*(_QWORD *)(a2 + 16) + 24LL), 13)) )
        {
          goto LABEL_12;
        }
      }
      v15 = (unsigned __int8 *)sub_AD64C0(*(_QWORD *)(a2 + 8), 1, 0);
      if ( (*(_BYTE *)(v2 + 7) & 0x40) != 0 )
        v16 = *(_QWORD *)(v2 - 8);
      else
        v16 = v2 - 32LL * (*(_DWORD *)(v2 + 4) & 0x7FFFFFF);
      v17 = *(_QWORD *)(v16 + 32);
      v18 = sub_AABE40(0x19u, v15, (unsigned __int8 *)v17);
      LOWORD(v133) = 257;
      if ( (*(_BYTE *)(v2 + 7) & 0x40) != 0 )
        v19 = *(__int64 **)(v2 - 8);
      else
        v19 = (__int64 *)(v2 - 32LL * (*(_DWORD *)(v2 + 4) & 0x7FFFFFF));
      v20 = sub_B504D0(17, *v19, v18, (__int64)&v130, v2 + 24, 0);
      v21 = sub_ACADE0(*(__int64 ***)(v2 + 8));
      if ( (*(_BYTE *)(v2 + 7) & 0x40) != 0 )
        v22 = *(_QWORD *)(v2 - 8);
      else
        v22 = v2 - 32LL * (*(_DWORD *)(v2 + 4) & 0x7FFFFFF);
      if ( *(_QWORD *)v22 )
      {
        v23 = *(_QWORD *)(v22 + 8);
        **(_QWORD **)(v22 + 16) = v23;
        if ( v23 )
          *(_QWORD *)(v23 + 16) = *(_QWORD *)(v22 + 16);
      }
      *(_QWORD *)v22 = v21;
      if ( v21 )
      {
        v24 = *(_QWORD *)(v21 + 16);
        *(_QWORD *)(v22 + 8) = v24;
        if ( v24 )
          *(_QWORD *)(v24 + 16) = v22 + 8;
        *(_QWORD *)(v22 + 16) = v21 + 16;
        *(_QWORD *)(v21 + 16) = v22;
      }
      v25 = (__m128i *)(v20 + 48);
      sub_BD6B90((unsigned __int8 *)v20, (unsigned __int8 *)v2);
      sub_BD84D0(v2, v20);
      v26 = *(_QWORD *)(v2 + 48);
      v130.m128i_i64[0] = v26;
      if ( v26 )
      {
        sub_B96E90((__int64)&v130, v26, 1);
        if ( v25 == &v130 )
        {
          if ( v130.m128i_i64[0] )
            sub_B91220((__int64)&v130, v130.m128i_i64[0]);
          goto LABEL_51;
        }
        v102 = *(_QWORD *)(v20 + 48);
        if ( !v102 )
        {
LABEL_214:
          v103 = (unsigned __int8 *)v130.m128i_i64[0];
          *(_QWORD *)(v20 + 48) = v130.m128i_i64[0];
          if ( v103 )
            sub_B976B0((__int64)&v130, v103, v20 + 48);
          goto LABEL_51;
        }
      }
      else if ( v25 == &v130 || (v102 = *(_QWORD *)(v20 + 48)) == 0 )
      {
LABEL_51:
        v27 = sub_B44900(v2);
        v28 = sub_B448F0(v2);
        v29 = v28;
        if ( v27 )
        {
          if ( v28 )
            goto LABEL_56;
          v30 = *(_DWORD *)(v17 + 32);
          if ( v30 > 0x40 )
          {
            if ( v30 - (unsigned int)sub_C444A0(v17 + 24) > 0x40 )
              goto LABEL_57;
            v31 = **(_QWORD **)(v17 + 24);
          }
          else
          {
            v31 = *(_QWORD *)(v17 + 24);
          }
          if ( (unsigned int)((*(_DWORD *)(*(_QWORD *)(v2 + 8) + 8LL) >> 8) - 1) > v31 )
LABEL_56:
            sub_B44850((unsigned __int8 *)v20, 1);
        }
LABEL_57:
        sub_B447F0((unsigned __int8 *)v20, v29);
        v32 = v2;
        v2 = v20;
        sub_D68D20((__int64)&v130, 0, v32);
        sub_28F19A0(a1 + 64, &v130);
        sub_D68D70(&v130);
        *(_BYTE *)(a1 + 752) = 1;
        goto LABEL_12;
      }
      sub_B91220(v20 + 48, v102);
      goto LABEL_214;
    }
    v8 = *(_BYTE *)a2;
LABEL_11:
    if ( v8 != 54 )
      goto LABEL_12;
    goto LABEL_32;
  }
  if ( (unsigned __int64 *)v130.m128i_i64[0] != &v131 )
    j_j___libc_free_0(v130.m128i_u64[0]);
  v8 = *(_BYTE *)a2;
  v6 = 1;
  if ( *(_BYTE *)a2 != 42 )
    goto LABEL_11;
  v45 = *(_QWORD *)(a2 + 16);
  if ( v45 )
  {
    if ( !*(_QWORD *)(v45 + 8) )
    {
      v46 = *(_BYTE **)sub_986520(a2);
      if ( *v46 == 54 && **((_BYTE **)v46 - 4) == 17 )
        return;
    }
  }
LABEL_12:
  if ( sub_B46D50((unsigned __int8 *)v2) )
    sub_28EFCB0(a1, v2);
  v9 = sub_28F3780(a1, (unsigned __int8 *)v2);
  if ( !v9 )
    v9 = (unsigned __int8 *)v2;
  if ( (unsigned __int8)sub_920620((__int64)v9) && (!sub_B451B0((__int64)v9) || !sub_B451E0((__int64)v9)) )
    return;
  v10 = *((_QWORD *)v9 + 1);
  if ( (unsigned int)*(unsigned __int8 *)(v10 + 8) - 17 <= 1 )
    v10 = **(_QWORD **)(v10 + 16);
  v113 = sub_BCAC40(v10, 1);
  if ( v113 )
    return;
  if ( *v9 != 58 )
    goto LABEL_23;
  v49 = sub_986520((__int64)v9);
  v50 = 32LL * (*((_DWORD *)v9 + 1) & 0x7FFFFFF);
  v51 = v49 + v50;
  v52 = v50 >> 5;
  v53 = v50 >> 7;
  v112 = v51;
  if ( v53 )
  {
    v114 = v49 + (v53 << 7);
LABEL_86:
    v54 = "\r";
    v55 = 13;
    v56 = "\r";
    src = *(unsigned __int8 **)v49;
    while ( !sub_28ED300(src, v55) )
    {
      v56 += 4;
      if ( jpt_2901BE4 == (_UNKNOWN *__ptr32 *)v56 )
      {
        v59 = "\r";
        v60 = 13;
        srca = *(unsigned __int8 **)(v49 + 32);
        while ( 1 )
        {
          if ( sub_28ED300(srca, v60) )
          {
            v49 += 32;
            goto LABEL_102;
          }
          v59 += 4;
          if ( jpt_2901BE4 == (_UNKNOWN *__ptr32 *)v59 )
            break;
          v60 = *(_DWORD *)v59;
        }
        v61 = "\r";
        v62 = 13;
        srcb = *(unsigned __int8 **)(v49 + 64);
        while ( 1 )
        {
          if ( sub_28ED300(srcb, v62) )
          {
            v49 += 64;
            goto LABEL_102;
          }
          v61 += 4;
          if ( jpt_2901BE4 == (_UNKNOWN *__ptr32 *)v61 )
            break;
          v62 = *(_DWORD *)v61;
        }
        v63 = *(unsigned __int8 **)(v49 + 96);
        if ( sub_28ED300(v63, 13) )
        {
LABEL_101:
          v49 += 96;
          goto LABEL_102;
        }
        while ( 1 )
        {
          v54 += 4;
          if ( jpt_2901BE4 == (_UNKNOWN *__ptr32 *)v54 )
            break;
          if ( sub_28ED300(v63, *(_DWORD *)v54) )
            goto LABEL_101;
        }
        v49 += 128;
        if ( v114 == v49 )
        {
          v52 = (v112 - v49) >> 5;
          goto LABEL_112;
        }
        goto LABEL_86;
      }
      v55 = *(_DWORD *)v56;
    }
    goto LABEL_102;
  }
LABEL_112:
  if ( v52 == 2 )
    goto LABEL_228;
  if ( v52 == 3 )
  {
    v108 = *(unsigned __int8 **)v49;
    v109 = "\r";
    if ( sub_28ED300(*(unsigned __int8 **)v49, 13) )
      goto LABEL_102;
    while ( 1 )
    {
      v109 += 4;
      if ( jpt_2901BE4 == (_UNKNOWN *__ptr32 *)v109 )
        break;
      if ( sub_28ED300(v108, *(_DWORD *)v109) )
        goto LABEL_102;
    }
    v49 += 32;
LABEL_228:
    v106 = *(unsigned __int8 **)v49;
    v107 = "\r";
    if ( sub_28ED300(*(unsigned __int8 **)v49, 13) )
      goto LABEL_102;
    while ( 1 )
    {
      v107 += 4;
      if ( jpt_2901BE4 == (_UNKNOWN *__ptr32 *)v107 )
        break;
      if ( sub_28ED300(v106, *(_DWORD *)v107) )
        goto LABEL_102;
    }
    v49 += 32;
    goto LABEL_115;
  }
  if ( v52 != 1 )
    goto LABEL_103;
LABEL_115:
  v70 = *(unsigned __int8 **)v49;
  v71 = "\r";
  if ( !sub_28ED300(*(unsigned __int8 **)v49, 13) )
  {
    do
    {
      v71 += 4;
      if ( jpt_2901BE4 == (_UNKNOWN *__ptr32 *)v71 )
        goto LABEL_103;
    }
    while ( !sub_28ED300(v70, *(_DWORD *)v71) );
  }
LABEL_102:
  if ( v112 == v49 )
  {
LABEL_103:
    v64 = *((_QWORD *)v9 + 2);
    if ( *(_QWORD *)(v64 + 8) )
      goto LABEL_23;
    v65 = *(unsigned __int8 **)(v64 + 24);
    v66 = "\r";
    v67 = 13;
    v68 = *v65;
    if ( (unsigned __int8)v68 <= 0x1Cu )
      goto LABEL_107;
    while ( 1 )
    {
      if ( (unsigned int)(v68 - 42) <= 0x11 )
      {
        v69 = *((_QWORD *)v65 + 2);
        if ( v69 )
        {
          if ( !*(_QWORD *)(v69 + 8)
            && v67 == v68 - 29
            && (!(unsigned __int8)sub_920620((__int64)v65) || sub_B451B0((__int64)v65) && sub_B451E0((__int64)v65)) )
          {
            break;
          }
        }
      }
      do
      {
LABEL_107:
        v66 += 4;
        if ( v66 == (const char *)jpt_2901BE4 )
          goto LABEL_23;
        v68 = *v65;
        v67 = *(_DWORD *)v66;
      }
      while ( (unsigned __int8)v68 <= 0x1Cu );
    }
  }
  BYTE4(v132) = 1;
  v126 = 0x800000000LL;
  v125 = &v127;
  v130.m128i_i64[0] = 0;
  v130.m128i_i64[1] = (__int64)&v133;
  v131 = 8;
  LODWORD(v132) = 0;
  if ( *v9 <= 0x1Cu )
    goto LABEL_177;
  v133 = v9;
  LODWORD(v77) = 1;
  v78 = 1;
  HIDWORD(v131) = 1;
  v79 = &v127;
  v130.m128i_i64[0] = 1;
  v127 = v9;
  LODWORD(v126) = 1;
  while ( 1 )
  {
    while ( 1 )
    {
      for ( i = &v79[(unsigned int)v77]; ; --i )
      {
        if ( !(_DWORD)v77 )
        {
          v113 = 1;
          goto LABEL_172;
        }
        v81 = (__int64 *)*(i - 1);
        v77 = (unsigned int)(v77 - 1);
        LODWORD(v126) = v77;
        if ( *(_BYTE *)v81 != 61 )
          break;
      }
      if ( (unsigned int)*(unsigned __int8 *)v81 - 29 > 0x20 )
        break;
      if ( *(_BYTE *)v81 == 54 )
        goto LABEL_162;
      if ( *(_BYTE *)v81 != 58 )
        goto LABEL_172;
      v82 = 32LL * (*((_DWORD *)v81 + 1) & 0x7FFFFFF);
      if ( (*((_BYTE *)v81 + 7) & 0x40) != 0 )
      {
        v83 = (__int64 *)*(v81 - 1);
        v81 = &v83[(unsigned __int64)v82 / 8];
      }
      else
      {
        v83 = &v81[v82 / 0xFFFFFFFFFFFFFFF8LL];
      }
      if ( v83 != v81 )
      {
LABEL_147:
        v84 = (unsigned __int8 *)*v83;
        if ( *(_BYTE *)*v83 <= 0x1Cu )
          goto LABEL_172;
        if ( !(_BYTE)v78 )
          goto LABEL_222;
        v85 = (unsigned __int8 **)v130.m128i_i64[1];
        v77 = v130.m128i_i64[1] + 8LL * HIDWORD(v131);
        if ( v130.m128i_i64[1] == v77 )
        {
LABEL_241:
          if ( HIDWORD(v131) < (unsigned int)v131 )
          {
            ++HIDWORD(v131);
            *(_QWORD *)v77 = v84;
            v78 = BYTE4(v132);
            ++v130.m128i_i64[0];
            goto LABEL_223;
          }
LABEL_222:
          sub_C8CC70((__int64)&v130, *v83, v77, v78, v57, v58);
          v78 = BYTE4(v132);
          if ( (_BYTE)v77 )
          {
LABEL_223:
            v104 = v126;
            if ( (unsigned int)v126 >= (unsigned __int64)HIDWORD(v126) )
            {
              if ( HIDWORD(v126) < (unsigned __int64)(unsigned int)v126 + 1 )
                sub_C8D5F0((__int64)&v125, &v127, (unsigned int)v126 + 1LL, 8u, v57, v58);
              v77 = (unsigned int)v126;
              v125[(unsigned int)v126] = v84;
              v78 = BYTE4(v132);
              LODWORD(v126) = v126 + 1;
            }
            else
            {
              v105 = &v125[(unsigned int)v126];
              if ( v105 )
              {
                *v105 = v84;
                v104 = v126;
                v78 = BYTE4(v132);
              }
              v77 = (unsigned int)(v104 + 1);
              LODWORD(v126) = v77;
            }
          }
        }
        else
        {
          while ( v84 != *v85 )
          {
            if ( (unsigned __int8 **)v77 == ++v85 )
              goto LABEL_241;
          }
        }
        v83 += 4;
        if ( v81 == v83 )
        {
          v79 = v125;
          LODWORD(v77) = v126;
          continue;
        }
        goto LABEL_147;
      }
    }
    if ( *(_BYTE *)v81 != 68 )
      break;
LABEL_162:
    v87 = (*((_BYTE *)v81 + 7) & 0x40) != 0
        ? (unsigned __int8 *)*(v81 - 1)
        : (unsigned __int8 *)&v81[-4 * (*((_DWORD *)v81 + 1) & 0x7FFFFFF)];
    v88 = *(unsigned __int8 **)v87;
    if ( **(_BYTE **)v87 <= 0x1Cu )
      break;
    if ( !(_BYTE)v78 )
      goto LABEL_248;
    v89 = (unsigned __int8 **)v130.m128i_i64[1];
    v90 = (unsigned __int8 **)(v130.m128i_i64[1] + 8LL * HIDWORD(v131));
    if ( (unsigned __int8 **)v130.m128i_i64[1] != v90 )
    {
      while ( v88 != *v89 )
      {
        if ( v90 == ++v89 )
          goto LABEL_253;
      }
LABEL_170:
      v79 = v125;
      continue;
    }
LABEL_253:
    if ( HIDWORD(v131) < (unsigned int)v131 )
    {
      ++HIDWORD(v131);
      *v90 = v88;
      v110 = v126;
      ++v130.m128i_i64[0];
    }
    else
    {
LABEL_248:
      sub_C8CC70((__int64)&v130, (__int64)v88, v77, v78, v57, v58);
      v110 = v126;
      v78 = BYTE4(v132);
      v58 = v77;
      LODWORD(v77) = v126;
      if ( !(_BYTE)v58 )
        goto LABEL_170;
    }
    if ( v110 >= (unsigned __int64)HIDWORD(v126) )
    {
      if ( HIDWORD(v126) < (unsigned __int64)v110 + 1 )
        sub_C8D5F0((__int64)&v125, &v127, v110 + 1LL, 8u, v57, v58);
      v125[(unsigned int)v126] = v88;
      v79 = v125;
      v78 = BYTE4(v132);
      LODWORD(v77) = v126 + 1;
      LODWORD(v126) = v126 + 1;
    }
    else
    {
      v79 = v125;
      v111 = &v125[v110];
      if ( v111 )
      {
        *v111 = v88;
        v79 = v125;
        v110 = v126;
      }
      LODWORD(v77) = v110 + 1;
      v78 = BYTE4(v132);
      LODWORD(v126) = v110 + 1;
    }
  }
LABEL_172:
  if ( !(_BYTE)v78 )
    _libc_free(v130.m128i_u64[1]);
  if ( v125 != &v127 )
    _libc_free((unsigned __int64)v125);
  if ( !v113 )
  {
LABEL_177:
    if ( (v9[1] & 2) == 0 )
    {
      v96 = sub_B43CC0((__int64)v9);
      v137 = 257;
      v130 = (__m128i)(unsigned __int64)v96;
      v131 = 0;
      v132 = 0;
      v133 = 0;
      v134 = v9;
      v135 = 0;
      v136 = 0;
      v97 = *(_QWORD *)(sub_986520((__int64)v9) + 32);
      LODWORD(v127) = 1;
      v126 = 0;
      v129 = 1;
      v125 = (unsigned __int8 **)(v97 & 0xFFFFFFFFFFFFFFFBLL);
      v128 = 0;
      v98 = *(_QWORD *)sub_986520((__int64)v9);
      v122 = 1;
      v121 = 0;
      v124 = 1;
      v120 = v98 & 0xFFFFFFFFFFFFFFFBLL;
      v123 = 0;
      v99 = sub_9ACC00((__int64)&v120, (__int64)&v125, &v130);
      if ( v124 > 0x40 && v123 )
        j_j___libc_free_0_0(v123);
      if ( v122 > 0x40 && v121 )
        j_j___libc_free_0_0(v121);
      if ( v129 > 0x40 && v128 )
        j_j___libc_free_0_0(v128);
      if ( (unsigned int)v127 > 0x40 && v126 )
        j_j___libc_free_0_0(v126);
      if ( !v99 )
        goto LABEL_23;
    }
    LOWORD(v133) = 257;
    if ( (v9[7] & 0x40) != 0 )
      v91 = (__int64 *)*((_QWORD *)v9 - 1);
    else
      v91 = (__int64 *)&v9[-32 * (*((_DWORD *)v9 + 1) & 0x7FFFFFF)];
    v92 = (__m128i *)sub_28E9200(*v91, v91[4], (__int64)&v130, (__int64)(v9 + 24), 0, 0, (__int64)v9);
    sub_B44850((unsigned __int8 *)v92, 1);
    sub_B447F0((unsigned __int8 *)v92, 1);
    sub_BD6B90((unsigned __int8 *)v92, v9);
    sub_BD84D0((__int64)v9, (__int64)v92);
    v93 = *((_QWORD *)v9 + 6);
    v130.m128i_i64[0] = v93;
    if ( v93 )
    {
      m128i_i64 = (__int64)v92[3].m128i_i64;
      sub_B96E90((__int64)&v130, v93, 1);
      if ( &v92[3] == &v130 )
      {
        if ( v130.m128i_i64[0] )
          sub_B91220((__int64)&v130, v130.m128i_i64[0]);
        goto LABEL_184;
      }
      v100 = v92[3].m128i_i64[0];
      if ( !v100 )
      {
LABEL_203:
        v101 = (unsigned __int8 *)v130.m128i_i64[0];
        v92[3].m128i_i64[0] = v130.m128i_i64[0];
        if ( v101 )
          sub_B976B0((__int64)&v130, v101, m128i_i64);
        goto LABEL_184;
      }
    }
    else
    {
      m128i_i64 = (__int64)v92[3].m128i_i64;
      if ( &v92[3] == &v130 || (v100 = v92[3].m128i_i64[0]) == 0 )
      {
LABEL_184:
        v95 = (__int64)v9;
        v9 = (unsigned __int8 *)v92;
        sub_D68D20((__int64)&v130, 0, v95);
        sub_28F19A0(a1 + 64, &v130);
        sub_D68D70(&v130);
        *(_BYTE *)(a1 + 752) = 1;
        goto LABEL_23;
      }
    }
    sub_B91220(m128i_i64, v100);
    goto LABEL_203;
  }
LABEL_23:
  if ( *v9 != 44 )
  {
    if ( (*v9 & 0xFB) != 0x29 )
      goto LABEL_25;
    if ( !(unsigned __int8)sub_28EECD0((__int64)v9) )
    {
      if ( (unsigned __int8)sub_28EE9F0(v9) )
      {
        v33 = (_QWORD *)sub_986520((__int64)v9);
        v34 = (unsigned __int8 *)((unsigned int)*v9 - 42 > 0x11 ? *v33 : v33[4]);
        if ( sub_28ED300(v34, 18) )
        {
          v35 = *((_QWORD *)v9 + 2);
          if ( !v35 || *(_QWORD *)(v35 + 8) || !sub_28ED300(*(unsigned __int8 **)(v35 + 24), 18) )
          {
            v36 = sub_28EB420(v9);
            v37 = *(_QWORD *)(v36 + 16);
            v38 = (unsigned __int8 *)v36;
            for ( j = a1 + 64; v37; v37 = *(_QWORD *)(v37 + 8) )
            {
              v40 = *(_BYTE **)(v37 + 24);
              if ( (unsigned __int8)(*v40 - 42) <= 0x11u )
              {
                sub_D68D20((__int64)&v130, 0, (__int64)v40);
                sub_28F19A0(j, &v130);
                sub_D68D70(&v130);
              }
            }
            goto LABEL_71;
          }
        }
      }
      goto LABEL_25;
    }
LABEL_83:
    v47 = sub_28F3F90((unsigned __int64)v9, a1 + 64);
    v48 = (__int64)v9;
    v9 = (unsigned __int8 *)v47;
    sub_D68D20((__int64)&v130, 0, v48);
    sub_28F19A0(a1 + 64, &v130);
    sub_D68D70(&v130);
    *(_BYTE *)(a1 + 752) = 1;
    goto LABEL_25;
  }
  if ( (unsigned __int8)sub_28EECD0((__int64)v9) )
    goto LABEL_83;
  v130.m128i_i64[0] = 0;
  if ( *v9 == 44 )
  {
    if ( (unsigned __int8)sub_28EB290((__int64 **)&v130, (__int64)v9) )
    {
      v72 = sub_986520((__int64)v9);
      if ( sub_28ED300(*(unsigned __int8 **)(v72 + 32), 17) )
      {
        v73 = *((_QWORD *)v9 + 2);
        if ( !v73 || *(_QWORD *)(v73 + 8) || !sub_28ED300(*(unsigned __int8 **)(v73 + 24), 17) )
        {
          v74 = sub_28EB420(v9);
          v75 = *(_QWORD *)(v74 + 16);
          v38 = (unsigned __int8 *)v74;
          for ( j = a1 + 64; v75; v75 = *(_QWORD *)(v75 + 8) )
          {
            v76 = *(_BYTE **)(v75 + 24);
            if ( (unsigned __int8)(*v76 - 42) <= 0x11u )
            {
              sub_D68D20((__int64)&v130, 0, (__int64)v76);
              sub_28F19A0(j, &v130);
              sub_D68D70(&v130);
            }
          }
LABEL_71:
          v41 = (__int64)v9;
          v9 = v38;
          sub_D68D20((__int64)&v130, 0, v41);
          sub_28F19A0(j, &v130);
          sub_D68D70(&v130);
          *(_BYTE *)(a1 + 752) = 1;
        }
      }
    }
  }
LABEL_25:
  if ( sub_B46CC0(v9) )
  {
    v11 = *((_QWORD *)v9 + 2);
    if ( !v11 || *(_QWORD *)(v11 + 8) )
      goto LABEL_27;
    v42 = *(_QWORD *)(v11 + 24);
    v43 = *v9;
    v44 = *(_BYTE *)v42;
    if ( *v9 != *(_BYTE *)v42 )
    {
      if ( v43 == 42 )
      {
        if ( v44 == 44 )
          return;
      }
      else if ( v43 == 43 && v44 == 45 )
      {
        return;
      }
LABEL_27:
      sub_28F9120(a1, v9);
      return;
    }
    if ( (unsigned __int8 *)v42 != v9 && *(_QWORD *)(v42 + 40) == *((_QWORD *)v9 + 5) )
    {
      sub_D68D20((__int64)&v130, 0, v42);
      sub_28F19A0(a1 + 64, &v130);
      sub_D68D70(&v130);
    }
  }
}
