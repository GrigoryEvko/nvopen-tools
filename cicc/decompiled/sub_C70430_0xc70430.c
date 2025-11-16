// Function: sub_C70430
// Address: 0xc70430
//
__int64 __fastcall sub_C70430(__int64 a1, char a2, char a3, char a4, __int64 a5, __int64 a6)
{
  unsigned int v10; // eax
  bool v11; // al
  unsigned int v12; // edx
  int v13; // eax
  bool v14; // al
  unsigned int v15; // r14d
  int v16; // eax
  bool v17; // al
  unsigned __int64 v18; // rdi
  bool v19; // cc
  unsigned int v20; // eax
  __int64 v21; // rdi
  unsigned int v22; // r14d
  unsigned int v24; // r14d
  bool v25; // al
  unsigned int v26; // r14d
  unsigned int v27; // edx
  int v28; // eax
  unsigned __int64 v29; // rax
  unsigned int v30; // r8d
  unsigned int v31; // esi
  unsigned int v32; // esi
  unsigned __int64 v33; // rdx
  unsigned int v34; // esi
  unsigned __int64 v35; // rdx
  unsigned int v36; // eax
  unsigned __int64 v37; // rdx
  unsigned __int64 v38; // rdx
  unsigned int v39; // r8d
  __int64 v40; // rdi
  unsigned int v41; // eax
  unsigned __int64 v42; // rdx
  unsigned __int64 v43; // rdx
  unsigned int v44; // r8d
  __int64 v45; // rdi
  unsigned __int64 v46; // rdx
  unsigned __int64 v47; // rdi
  __int64 v48; // rax
  __int64 v49; // rbx
  unsigned int v51; // esi
  unsigned __int64 v52; // rdx
  unsigned int v53; // eax
  unsigned __int64 v54; // rdx
  unsigned __int64 v55; // rdx
  unsigned int v56; // r8d
  __int64 v57; // rdi
  unsigned int v58; // eax
  unsigned __int64 v59; // rdx
  unsigned __int64 v60; // rdx
  unsigned int v61; // r8d
  __int64 v62; // rdi
  unsigned int v63; // esi
  unsigned __int64 v64; // rdx
  __int64 v65; // rdx
  unsigned __int64 v66; // rax
  __int64 v67; // rcx
  unsigned int v68; // eax
  unsigned __int64 v69; // rax
  unsigned __int64 v70; // rax
  unsigned int v71; // eax
  unsigned int v72; // r8d
  int v73; // eax
  unsigned __int64 v74; // rax
  unsigned int v75; // edx
  unsigned int v76; // esi
  unsigned int v77; // r13d
  unsigned __int64 v78; // rbx
  int v79; // ebx
  unsigned int v80; // esi
  unsigned __int64 v81; // rdx
  unsigned __int64 v82; // rax
  unsigned int v83; // esi
  unsigned __int64 v84; // rdx
  __int64 v85; // rax
  unsigned int v86; // r13d
  int v87; // ebx
  unsigned __int64 v88; // rbx
  unsigned int v89; // esi
  unsigned __int64 v90; // rdx
  unsigned __int64 v91; // rax
  unsigned int v92; // esi
  __int64 v93; // rdx
  __int64 v94; // rax
  unsigned int v95; // eax
  unsigned __int64 v96; // rsi
  unsigned int v97; // edx
  unsigned __int64 v98; // rcx
  unsigned __int64 v99; // rdi
  unsigned int v100; // eax
  __int64 v101; // rdi
  __int64 v102; // rax
  __int64 v103; // rsi
  __int64 v104; // rax
  unsigned int v105; // edx
  int v106; // eax
  unsigned int v107; // esi
  unsigned __int64 v108; // rdx
  unsigned __int64 v109; // rax
  unsigned int v110; // r8d
  unsigned int v111; // edx
  unsigned int v112; // eax
  unsigned int v113; // esi
  unsigned __int64 v114; // rdx
  unsigned __int64 v115; // rax
  __int64 v116; // rsi
  __int64 v117; // rsi
  __int64 v118; // rax
  __int64 v119; // rsi
  __int64 v120; // rax
  unsigned __int64 v121; // rax
  unsigned __int64 v122; // rax
  int v123; // [rsp+0h] [rbp-D0h]
  unsigned int v124; // [rsp+4h] [rbp-CCh]
  unsigned int v125; // [rsp+4h] [rbp-CCh]
  unsigned int v126; // [rsp+8h] [rbp-C8h]
  unsigned int v127; // [rsp+8h] [rbp-C8h]
  const void **v128; // [rsp+8h] [rbp-C8h]
  __int64 *v129; // [rsp+18h] [rbp-B8h]
  unsigned int v130; // [rsp+20h] [rbp-B0h]
  unsigned int v132; // [rsp+28h] [rbp-A8h]
  const void **v133; // [rsp+28h] [rbp-A8h]
  unsigned int v134; // [rsp+28h] [rbp-A8h]
  char *v135; // [rsp+30h] [rbp-A0h] BYREF
  unsigned int v136; // [rsp+38h] [rbp-98h]
  unsigned __int64 v137; // [rsp+40h] [rbp-90h] BYREF
  unsigned int v138; // [rsp+48h] [rbp-88h]
  unsigned __int64 v139; // [rsp+50h] [rbp-80h] BYREF
  unsigned int v140; // [rsp+58h] [rbp-78h]
  unsigned __int64 v141; // [rsp+60h] [rbp-70h] BYREF
  unsigned int v142; // [rsp+68h] [rbp-68h]
  unsigned __int64 v143; // [rsp+70h] [rbp-60h] BYREF
  unsigned int v144; // [rsp+78h] [rbp-58h]
  unsigned __int64 v145; // [rsp+80h] [rbp-50h] BYREF
  unsigned int v146; // [rsp+88h] [rbp-48h]
  __int64 v147; // [rsp+90h] [rbp-40h]
  int v148; // [rsp+98h] [rbp-38h]

  v10 = *(_DWORD *)(a5 + 8);
  v130 = v10;
  *(_DWORD *)(a1 + 8) = v10;
  v129 = (__int64 *)(a1 + 16);
  if ( v10 > 0x40 )
  {
    sub_C43690(a1, 0, 0);
    *(_DWORD *)(a1 + 24) = v130;
    sub_C43690((__int64)v129, 0, 0);
  }
  else
  {
    *(_QWORD *)a1 = 0;
    *(_DWORD *)(a1 + 24) = v10;
    *(_QWORD *)(a1 + 16) = 0;
  }
  v132 = *(_DWORD *)(a5 + 8);
  if ( v132 <= 0x40 )
    v11 = *(_QWORD *)a5 == 0;
  else
    v11 = v132 == (unsigned int)sub_C444A0(a5);
  if ( v11 )
  {
    v22 = *(_DWORD *)(a5 + 24);
    if ( v22 <= 0x40 ? *(_QWORD *)(a5 + 16) == 0 : v22 == (unsigned int)sub_C444A0(a5 + 16) )
    {
      v24 = *(_DWORD *)(a6 + 8);
      if ( v24 <= 0x40 )
        v25 = *(_QWORD *)a6 == 0;
      else
        v25 = v24 == (unsigned int)sub_C444A0(a6);
      if ( v25 )
      {
        v26 = *(_DWORD *)(a6 + 24);
        if ( v26 <= 0x40 )
        {
          if ( !*(_QWORD *)(a6 + 16) )
            return a1;
        }
        else if ( v26 == (unsigned int)sub_C444A0(a6 + 16) )
        {
          return a1;
        }
      }
      goto LABEL_29;
    }
  }
  v12 = *(_DWORD *)(a6 + 8);
  if ( v12 <= 0x40 )
  {
    v14 = *(_QWORD *)a6 == 0;
  }
  else
  {
    v126 = *(_DWORD *)(a6 + 8);
    v13 = sub_C444A0(a6);
    v12 = v126;
    v14 = v126 == v13;
  }
  if ( v14 )
  {
    v15 = *(_DWORD *)(a6 + 24);
    if ( v15 <= 0x40 )
    {
      v17 = *(_QWORD *)(a6 + 16) == 0;
    }
    else
    {
      v127 = v12;
      v16 = sub_C444A0(a6 + 16);
      v12 = v127;
      v17 = v15 == v16;
    }
    if ( v17 )
    {
LABEL_29:
      if ( a4 )
      {
        if ( a2 )
          goto LABEL_31;
        v128 = (const void **)(a6 + 16);
LABEL_156:
        v146 = v132;
        if ( v132 > 0x40 )
        {
          sub_C43780((__int64)&v145, (const void **)a5);
          v132 = v146;
          if ( v146 > 0x40 )
          {
            sub_C43D10((__int64)&v145);
            v132 = v146;
            v70 = v145;
            goto LABEL_160;
          }
          v69 = v145;
        }
        else
        {
          v69 = *(_QWORD *)a5;
        }
        v70 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v132) & ~v69;
        if ( !v132 )
          v70 = 0;
LABEL_160:
        v141 = v70;
        v71 = *(_DWORD *)(a6 + 24);
        v142 = v132;
        v146 = v71;
        if ( v71 > 0x40 )
          sub_C43780((__int64)&v145, v128);
        else
          v145 = *(_QWORD *)(a6 + 16);
        sub_C49A20((__int64)&v139, (__int64)&v141, (__int64 *)&v145);
        if ( v146 > 0x40 && v145 )
          j_j___libc_free_0_0(v145);
        if ( v142 > 0x40 && v141 )
          j_j___libc_free_0_0(v141);
        if ( a3 )
        {
          sub_C44740((__int64)&v145, (char **)&v139, v130 - 1);
          v110 = v146;
          v111 = v130 - 1;
          if ( v146 <= 0x40 )
          {
            if ( v145 )
            {
              _BitScanReverse64(&v121, v145);
              v110 = v146 - 64 + (v121 ^ 0x3F);
            }
          }
          else
          {
            v112 = sub_C444A0((__int64)&v145);
            v111 = v130 - 1;
            v110 = v112;
            if ( v145 )
            {
              v125 = v112;
              j_j___libc_free_0_0(v145);
              v110 = v125;
              v111 = v130 - 1;
            }
          }
          v113 = v111 - v110;
          if ( v110 )
          {
            if ( v113 > 0x3F || v111 > 0x40 )
            {
              sub_C43C90((_QWORD *)a1, v113, v111);
            }
            else
            {
              v114 = *(_QWORD *)a1;
              v115 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v110) << v113;
              if ( *(_DWORD *)(a1 + 8) > 0x40u )
                *(_QWORD *)v114 |= v115;
              else
                *(_QWORD *)a1 = v114 | v115;
            }
          }
        }
        v72 = v140;
        if ( v140 > 0x40 )
        {
          v134 = v140;
          v73 = sub_C444A0((__int64)&v139);
          v75 = *(_DWORD *)(a1 + 8);
          v72 = v134;
          v76 = v75 - v73;
          if ( v75 - v73 == v75 )
            goto LABEL_177;
        }
        else
        {
          v73 = v140;
          if ( v139 )
          {
            _BitScanReverse64(&v74, v139);
            v73 = v140 - 64 + (v74 ^ 0x3F);
          }
          v75 = *(_DWORD *)(a1 + 8);
          v76 = v75 - v73;
          if ( v75 - v73 == v75 )
            goto LABEL_179;
        }
        if ( v76 > 0x3F || v75 > 0x40 )
        {
          sub_C43C90((_QWORD *)a1, v76, v75);
          v72 = v140;
        }
        else
        {
          *(_QWORD *)a1 |= 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v73) << v76;
        }
        if ( v72 <= 0x40 )
        {
LABEL_179:
          if ( !a3 )
            goto LABEL_105;
          v136 = 1;
          v135 = 0;
          v138 = 1;
          v137 = 0;
          goto LABEL_112;
        }
LABEL_177:
        if ( v139 )
          j_j___libc_free_0_0(v139);
        goto LABEL_179;
      }
      if ( !a3 )
        goto LABEL_105;
      v136 = 1;
      v135 = 0;
      v138 = 1;
      v137 = 0;
      v128 = (const void **)(a6 + 16);
      if ( a2 )
      {
        v133 = (const void **)(a5 + 16);
        goto LABEL_55;
      }
      goto LABEL_112;
    }
  }
  if ( !a2 )
  {
    v142 = v12;
    if ( v12 > 0x40 )
      sub_C43780((__int64)&v141, (const void **)a6);
    else
      v141 = *(_QWORD *)a6;
    v128 = (const void **)(a6 + 16);
    v95 = *(_DWORD *)(a6 + 24);
    v144 = v95;
    if ( v95 > 0x40 )
    {
      sub_C43780((__int64)&v143, v128);
      v96 = v143;
      v95 = v144;
    }
    else
    {
      v96 = *(_QWORD *)(a6 + 16);
    }
    v97 = v142;
    v98 = v141;
    v142 = v95;
    v141 = v96;
    v144 = v97;
    v143 = v98;
    sub_C6EF30((__int64)&v145, a5, (__int64)&v141, 0, 1u);
    if ( *(_DWORD *)(a1 + 8) > 0x40u )
    {
      v99 = *(_QWORD *)a1;
      if ( *(_QWORD *)a1 )
        j_j___libc_free_0_0(v99);
    }
    v19 = *(_DWORD *)(a1 + 24) <= 0x40u;
    *(_QWORD *)a1 = v145;
    v100 = v146;
    v146 = 0;
    *(_DWORD *)(a1 + 8) = v100;
    if ( v19 || (v101 = *(_QWORD *)(a1 + 16)) == 0 )
    {
      *(_QWORD *)(a1 + 16) = v147;
      *(_DWORD *)(a1 + 24) = v148;
    }
    else
    {
      j_j___libc_free_0_0(v101);
      v19 = v146 <= 0x40;
      *(_QWORD *)(a1 + 16) = v147;
      *(_DWORD *)(a1 + 24) = v148;
      if ( !v19 && v145 )
        j_j___libc_free_0_0(v145);
    }
    if ( v144 > 0x40 && v143 )
      j_j___libc_free_0_0(v143);
    if ( v142 > 0x40 && v141 )
      j_j___libc_free_0_0(v141);
    if ( a4 )
    {
      v132 = *(_DWORD *)(a5 + 8);
      goto LABEL_156;
    }
    if ( !a3 )
      goto LABEL_105;
    v136 = 1;
    v135 = 0;
    v138 = 1;
    v137 = 0;
LABEL_112:
    v140 = *(_DWORD *)(a5 + 24);
    if ( v140 > 0x40 )
      sub_C43780((__int64)&v139, (const void **)(a5 + 16));
    else
      v139 = *(_QWORD *)(a5 + 16);
    v51 = *(_DWORD *)(a5 + 8);
    v52 = *(_QWORD *)a5;
    if ( v51 > 0x40 )
      v52 = *(_QWORD *)(v52 + 8LL * ((v51 - 1) >> 6));
    if ( (v52 & (1LL << ((unsigned __int8)v51 - 1))) == 0 )
    {
      v104 = 1LL << ((unsigned __int8)v140 - 1);
      if ( v140 > 0x40 )
        *(_QWORD *)(v139 + 8LL * ((v140 - 1) >> 6)) |= v104;
      else
        v139 |= v104;
    }
    v53 = *(_DWORD *)(a6 + 8);
    v146 = v53;
    if ( v53 > 0x40 )
    {
      sub_C43780((__int64)&v145, (const void **)a6);
      v53 = v146;
      if ( v146 > 0x40 )
      {
        sub_C43D10((__int64)&v145);
        v53 = v146;
        v55 = v145;
        goto LABEL_121;
      }
      v54 = v145;
    }
    else
    {
      v54 = *(_QWORD *)a6;
    }
    v55 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v53) & ~v54;
    if ( !v53 )
      v55 = 0;
LABEL_121:
    v56 = *(_DWORD *)(a6 + 24);
    v57 = *(_QWORD *)(a6 + 16);
    v142 = v53;
    v141 = v55;
    if ( v56 > 0x40 )
      v57 = *(_QWORD *)(v57 + 8LL * ((v56 - 1) >> 6));
    if ( (v57 & (1LL << ((unsigned __int8)v56 - 1))) == 0 )
    {
      v103 = ~(1LL << ((unsigned __int8)v53 - 1));
      if ( v53 > 0x40 )
        *(_QWORD *)(v55 + 8LL * ((v53 - 1) >> 6)) &= v103;
      else
        v141 = v55 & v103;
    }
    sub_C46CE0((__int64)&v145, (__int64)&v139, (__int64)&v141);
    if ( v136 > 0x40 && v135 )
      j_j___libc_free_0_0(v135);
    v135 = (char *)v145;
    v136 = v146;
    if ( v142 > 0x40 && v141 )
      j_j___libc_free_0_0(v141);
    if ( v140 > 0x40 && v139 )
      j_j___libc_free_0_0(v139);
    v58 = *(_DWORD *)(a5 + 8);
    v146 = v58;
    if ( v58 > 0x40 )
    {
      sub_C43780((__int64)&v145, (const void **)a5);
      v58 = v146;
      if ( v146 > 0x40 )
      {
        sub_C43D10((__int64)&v145);
        v58 = v146;
        v60 = v145;
        goto LABEL_137;
      }
      v59 = v145;
    }
    else
    {
      v59 = *(_QWORD *)a5;
    }
    v60 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v58) & ~v59;
    if ( !v58 )
      v60 = 0;
LABEL_137:
    v61 = *(_DWORD *)(a5 + 24);
    v62 = *(_QWORD *)(a5 + 16);
    v140 = v58;
    v139 = v60;
    if ( v61 > 0x40 )
      v62 = *(_QWORD *)(v62 + 8LL * ((v61 - 1) >> 6));
    if ( (v62 & (1LL << ((unsigned __int8)v61 - 1))) == 0 )
    {
      v116 = ~(1LL << ((unsigned __int8)v58 - 1));
      if ( v58 > 0x40 )
        *(_QWORD *)(v60 + 8LL * ((v58 - 1) >> 6)) &= v116;
      else
        v139 = v60 & v116;
    }
    v142 = *(_DWORD *)(a6 + 24);
    if ( v142 > 0x40 )
      sub_C43780((__int64)&v141, v128);
    else
      v141 = *(_QWORD *)(a6 + 16);
    v63 = *(_DWORD *)(a6 + 8);
    v64 = *(_QWORD *)a6;
    if ( v63 > 0x40 )
      v64 = *(_QWORD *)(v64 + 8LL * ((v63 - 1) >> 6));
    if ( (v64 & (1LL << ((unsigned __int8)v63 - 1))) == 0 )
    {
      v120 = 1LL << ((unsigned __int8)v142 - 1);
      if ( v142 > 0x40 )
        *(_QWORD *)(v141 + 8LL * ((v142 - 1) >> 6)) |= v120;
      else
        v141 |= v120;
    }
    sub_C46CE0((__int64)&v145, (__int64)&v139, (__int64)&v141);
    if ( v138 > 0x40 )
      goto LABEL_89;
    goto LABEL_91;
  }
  sub_C6EF30((__int64)&v145, a5, a6, 1u, 0);
  if ( *(_DWORD *)(a1 + 8) > 0x40u )
  {
    v18 = *(_QWORD *)a1;
    if ( *(_QWORD *)a1 )
      j_j___libc_free_0_0(v18);
  }
  v19 = *(_DWORD *)(a1 + 24) <= 0x40u;
  *(_QWORD *)a1 = v145;
  v20 = v146;
  v146 = 0;
  *(_DWORD *)(a1 + 8) = v20;
  if ( v19 || (v21 = *(_QWORD *)(a1 + 16)) == 0 )
  {
    *(_QWORD *)(a1 + 16) = v147;
    *(_DWORD *)(a1 + 24) = v148;
  }
  else
  {
    j_j___libc_free_0_0(v21);
    v19 = v146 <= 0x40;
    *(_QWORD *)(a1 + 16) = v147;
    *(_DWORD *)(a1 + 24) = v148;
    if ( !v19 && v145 )
      j_j___libc_free_0_0(v145);
  }
  if ( a4 )
  {
LABEL_31:
    v133 = (const void **)(a5 + 16);
    v142 = *(_DWORD *)(a5 + 24);
    if ( v142 > 0x40 )
      sub_C43780((__int64)&v141, v133);
    else
      v141 = *(_QWORD *)(a5 + 16);
    v128 = (const void **)(a6 + 16);
    v146 = *(_DWORD *)(a6 + 24);
    if ( v146 > 0x40 )
      sub_C43780((__int64)&v145, v128);
    else
      v145 = *(_QWORD *)(a6 + 16);
    sub_C49B30((__int64)&v139, (__int64)&v141, (__int64 *)&v145);
    if ( v146 > 0x40 && v145 )
      j_j___libc_free_0_0(v145);
    if ( v142 > 0x40 && v141 )
      j_j___libc_free_0_0(v141);
    if ( !a3 )
    {
LABEL_42:
      v27 = v140;
      if ( v140 > 0x40 )
      {
        v124 = v140;
        v28 = sub_C44500((__int64)&v139);
        v30 = *(_DWORD *)(a1 + 24);
        v27 = v124;
        v31 = v30 - v28;
        if ( v30 == v30 - v28 )
          goto LABEL_51;
      }
      else
      {
        if ( !v140 )
          goto LABEL_53;
        v28 = 64;
        if ( v139 << (64 - (unsigned __int8)v140) != -1 )
        {
          _BitScanReverse64(&v29, ~(v139 << (64 - (unsigned __int8)v140)));
          v28 = v29 ^ 0x3F;
        }
        v30 = *(_DWORD *)(a1 + 24);
        v31 = v30 - v28;
        if ( v30 == v30 - v28 )
          goto LABEL_53;
      }
      if ( v31 > 0x3F || v30 > 0x40 )
      {
        sub_C43C90(v129, v31, v30);
        v27 = v140;
      }
      else
      {
        *(_QWORD *)(a1 + 16) |= 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v28) << v31;
      }
      if ( v27 <= 0x40 )
      {
LABEL_53:
        if ( !a3 )
          goto LABEL_105;
        v136 = 1;
        v135 = 0;
        v138 = 1;
        v137 = 0;
        goto LABEL_55;
      }
LABEL_51:
      if ( v139 )
        j_j___libc_free_0_0(v139);
      goto LABEL_53;
    }
    sub_C44740((__int64)&v145, (char **)&v139, v130 - 1);
    v105 = v130 - 1;
    if ( v146 <= 0x40 )
    {
      if ( !v146 )
        goto LABEL_42;
      if ( v145 << (64 - (unsigned __int8)v146) == -1 )
      {
        v107 = v130 - 65;
        LOBYTE(v106) = 64;
LABEL_273:
        if ( v107 > 0x3F || v105 > 0x40 )
        {
          sub_C43C90(v129, v107, v105);
        }
        else
        {
          v108 = *(_QWORD *)(a1 + 16);
          v109 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v106) << v107;
          if ( *(_DWORD *)(a1 + 24) > 0x40u )
            *(_QWORD *)v108 |= v109;
          else
            *(_QWORD *)(a1 + 16) = v108 | v109;
        }
        goto LABEL_42;
      }
      _BitScanReverse64(&v122, ~(v145 << (64 - (unsigned __int8)v146)));
      v106 = v122 ^ 0x3F;
    }
    else
    {
      v106 = sub_C44500((__int64)&v145);
      v105 = v130 - 1;
      if ( v145 )
      {
        v123 = v106;
        j_j___libc_free_0_0(v145);
        v106 = v123;
        v105 = v130 - 1;
      }
    }
    v107 = v105 - v106;
    if ( !v106 )
      goto LABEL_42;
    goto LABEL_273;
  }
  if ( !a3 )
    goto LABEL_105;
  v136 = 1;
  v133 = (const void **)(a5 + 16);
  v128 = (const void **)(a6 + 16);
  v135 = 0;
  v138 = 1;
  v137 = 0;
LABEL_55:
  v140 = *(_DWORD *)(a5 + 24);
  if ( v140 > 0x40 )
    sub_C43780((__int64)&v139, v133);
  else
    v139 = *(_QWORD *)(a5 + 16);
  v32 = *(_DWORD *)(a5 + 8);
  v33 = *(_QWORD *)a5;
  if ( v32 > 0x40 )
    v33 = *(_QWORD *)(v33 + 8LL * ((v32 - 1) >> 6));
  if ( (v33 & (1LL << ((unsigned __int8)v32 - 1))) == 0 )
  {
    v102 = 1LL << ((unsigned __int8)v140 - 1);
    if ( v140 > 0x40 )
      *(_QWORD *)(v139 + 8LL * ((v140 - 1) >> 6)) |= v102;
    else
      v139 |= v102;
  }
  v142 = *(_DWORD *)(a6 + 24);
  if ( v142 > 0x40 )
    sub_C43780((__int64)&v141, v128);
  else
    v141 = *(_QWORD *)(a6 + 16);
  v34 = *(_DWORD *)(a6 + 8);
  v35 = *(_QWORD *)a6;
  if ( v34 > 0x40 )
    v35 = *(_QWORD *)(v35 + 8LL * ((v34 - 1) >> 6));
  if ( (v35 & (1LL << ((unsigned __int8)v34 - 1))) == 0 )
  {
    v118 = 1LL << ((unsigned __int8)v142 - 1);
    if ( v142 > 0x40 )
      *(_QWORD *)(v141 + 8LL * ((v142 - 1) >> 6)) |= v118;
    else
      v141 |= v118;
  }
  sub_C46090((__int64)&v145, (__int64)&v139, (__int64)&v141);
  if ( v136 > 0x40 && v135 )
    j_j___libc_free_0_0(v135);
  v135 = (char *)v145;
  v136 = v146;
  if ( v142 > 0x40 && v141 )
    j_j___libc_free_0_0(v141);
  if ( v140 > 0x40 && v139 )
    j_j___libc_free_0_0(v139);
  v36 = *(_DWORD *)(a5 + 8);
  v146 = v36;
  if ( v36 > 0x40 )
  {
    sub_C43780((__int64)&v145, (const void **)a5);
    v36 = v146;
    if ( v146 > 0x40 )
    {
      sub_C43D10((__int64)&v145);
      v36 = v146;
      v38 = v145;
      goto LABEL_78;
    }
    v37 = v145;
  }
  else
  {
    v37 = *(_QWORD *)a5;
  }
  v38 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v36) & ~v37;
  if ( !v36 )
    v38 = 0;
LABEL_78:
  v39 = *(_DWORD *)(a5 + 24);
  v40 = *(_QWORD *)(a5 + 16);
  v140 = v36;
  v139 = v38;
  if ( v39 > 0x40 )
    v40 = *(_QWORD *)(v40 + 8LL * ((v39 - 1) >> 6));
  if ( (v40 & (1LL << ((unsigned __int8)v39 - 1))) == 0 )
  {
    v117 = ~(1LL << ((unsigned __int8)v36 - 1));
    if ( v36 > 0x40 )
      *(_QWORD *)(v38 + 8LL * ((v36 - 1) >> 6)) &= v117;
    else
      v139 = v38 & v117;
  }
  v41 = *(_DWORD *)(a6 + 8);
  v146 = v41;
  if ( v41 > 0x40 )
  {
    sub_C43780((__int64)&v145, (const void **)a6);
    v41 = v146;
    if ( v146 > 0x40 )
    {
      sub_C43D10((__int64)&v145);
      v41 = v146;
      v43 = v145;
      goto LABEL_85;
    }
    v42 = v145;
  }
  else
  {
    v42 = *(_QWORD *)a6;
  }
  v43 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v41) & ~v42;
  if ( !v41 )
    v43 = 0;
LABEL_85:
  v44 = *(_DWORD *)(a6 + 24);
  v45 = *(_QWORD *)(a6 + 16);
  v142 = v41;
  v141 = v43;
  if ( v44 > 0x40 )
    v45 = *(_QWORD *)(v45 + 8LL * ((v44 - 1) >> 6));
  if ( (v45 & (1LL << ((unsigned __int8)v44 - 1))) == 0 )
  {
    v119 = ~(1LL << ((unsigned __int8)v41 - 1));
    if ( v41 > 0x40 )
      *(_QWORD *)(v43 + 8LL * ((v41 - 1) >> 6)) &= v119;
    else
      v141 = v43 & v119;
  }
  sub_C46090((__int64)&v145, (__int64)&v139, (__int64)&v141);
  if ( v138 > 0x40 )
  {
LABEL_89:
    if ( v137 )
      j_j___libc_free_0_0(v137);
  }
LABEL_91:
  v137 = v145;
  v138 = v146;
  if ( v142 > 0x40 && v141 )
    j_j___libc_free_0_0(v141);
  if ( v140 > 0x40 && v139 )
    j_j___libc_free_0_0(v139);
  v46 = (unsigned __int64)v135;
  if ( v136 > 0x40 )
    v46 = *(_QWORD *)&v135[8 * ((v136 - 1) >> 6)];
  if ( (v46 & (1LL << ((unsigned __int8)v136 - 1))) == 0 )
  {
    v77 = v130 - 1;
    sub_C44740((__int64)&v145, &v135, v130 - 1);
    if ( v146 > 0x40 )
    {
      v79 = sub_C44500((__int64)&v145);
      if ( v145 )
        j_j___libc_free_0_0(v145);
    }
    else
    {
      if ( !v146 )
        goto LABEL_202;
      if ( v145 << (64 - (unsigned __int8)v146) == -1 )
      {
        LOBYTE(v79) = 64;
        v80 = v130 - 65;
        goto LABEL_198;
      }
      _BitScanReverse64(&v78, ~(v145 << (64 - (unsigned __int8)v146)));
      v79 = v78 ^ 0x3F;
    }
    v80 = v77 - v79;
    if ( v79 )
    {
LABEL_198:
      if ( v80 > 0x3F || v77 > 0x40 )
      {
        sub_C43C90(v129, v80, v77);
      }
      else
      {
        v81 = *(_QWORD *)(a1 + 16);
        v82 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v79) << v80;
        if ( *(_DWORD *)(a1 + 24) > 0x40u )
          *(_QWORD *)v81 |= v82;
        else
          *(_QWORD *)(a1 + 16) = v81 | v82;
      }
    }
LABEL_202:
    v83 = *(_DWORD *)(a1 + 8);
    v84 = *(_QWORD *)a1;
    v85 = 1LL << ((unsigned __int8)v83 - 1);
    if ( v83 > 0x40 )
      *(_QWORD *)(v84 + 8LL * ((v83 - 1) >> 6)) |= v85;
    else
      *(_QWORD *)a1 = v84 | v85;
  }
  v47 = v137;
  v48 = 1LL << ((unsigned __int8)v138 - 1);
  if ( v138 > 0x40 )
  {
    if ( (*(_QWORD *)(v137 + 8LL * ((v138 - 1) >> 6)) & v48) == 0 )
    {
LABEL_190:
      if ( v47 )
        j_j___libc_free_0_0(v47);
      goto LABEL_102;
    }
LABEL_204:
    v86 = v130 - 1;
    sub_C44740((__int64)&v145, (char **)&v137, v130 - 1);
    v87 = v146;
    if ( v146 > 0x40 )
    {
      v87 = sub_C444A0((__int64)&v145);
      if ( v145 )
        j_j___libc_free_0_0(v145);
    }
    else if ( v145 )
    {
      _BitScanReverse64(&v88, v145);
      v87 = v146 - 64 + (v88 ^ 0x3F);
    }
    v89 = v86 - v87;
    if ( v87 )
    {
      if ( v89 > 0x3F || v86 > 0x40 )
      {
        sub_C43C90((_QWORD *)a1, v89, v86);
      }
      else
      {
        v90 = *(_QWORD *)a1;
        v91 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v87) << v89;
        if ( *(_DWORD *)(a1 + 8) > 0x40u )
          *(_QWORD *)v90 |= v91;
        else
          *(_QWORD *)a1 = v90 | v91;
      }
    }
    v92 = *(_DWORD *)(a1 + 24);
    v93 = *(_QWORD *)(a1 + 16);
    v94 = 1LL << ((unsigned __int8)v92 - 1);
    if ( v92 > 0x40 )
      *(_QWORD *)(v93 + 8LL * ((v92 - 1) >> 6)) |= v94;
    else
      *(_QWORD *)(a1 + 16) = v93 | v94;
    if ( v138 <= 0x40 )
      goto LABEL_102;
    v47 = v137;
    goto LABEL_190;
  }
  if ( (v48 & v137) != 0 )
    goto LABEL_204;
LABEL_102:
  if ( v136 > 0x40 && v135 )
    j_j___libc_free_0_0(v135);
LABEL_105:
  v49 = *(unsigned int *)(a1 + 8);
  if ( (unsigned int)v49 <= 0x40 )
  {
    if ( (*(_QWORD *)(a1 + 16) & *(_QWORD *)a1) == 0 )
      return a1;
    *(_QWORD *)a1 = -1;
    v65 = -1;
  }
  else
  {
    if ( !(unsigned __int8)sub_C446A0((__int64 *)a1, v129) )
      return a1;
    memset(*(void **)a1, -1, 8 * (((unsigned __int64)(unsigned int)v49 + 63) >> 6));
    v49 = *(unsigned int *)(a1 + 8);
    v65 = *(_QWORD *)a1;
  }
  v66 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v49;
  if ( (_DWORD)v49 )
  {
    if ( (unsigned int)v49 > 0x40 )
    {
      v67 = (unsigned int)((unsigned __int64)(v49 + 63) >> 6) - 1;
      *(_QWORD *)(v65 + 8 * v67) &= v66;
      goto LABEL_152;
    }
  }
  else
  {
    v66 = 0;
  }
  *(_QWORD *)a1 = v65 & v66;
LABEL_152:
  v68 = *(_DWORD *)(a1 + 24);
  if ( v68 > 0x40 )
    memset(*(void **)(a1 + 16), 0, 8 * (((unsigned __int64)v68 + 63) >> 6));
  else
    *(_QWORD *)(a1 + 16) = 0;
  return a1;
}
