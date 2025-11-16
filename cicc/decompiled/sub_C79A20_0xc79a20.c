// Function: sub_C79A20
// Address: 0xc79a20
//
__int64 __fastcall sub_C79A20(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  unsigned int v7; // ebx
  unsigned int v8; // r11d
  unsigned __int64 v9; // rax
  unsigned int v10; // esi
  unsigned __int64 v11; // rax
  __int64 v12; // rdx
  unsigned int v13; // r14d
  int v14; // eax
  bool v15; // al
  unsigned int v16; // edx
  int v17; // eax
  bool v18; // al
  unsigned int v19; // esi
  unsigned __int64 v20; // r10
  unsigned int v21; // eax
  unsigned __int64 v22; // rcx
  __int64 v23; // r8
  unsigned int v24; // ebx
  __int64 v25; // r9
  unsigned __int64 v26; // rax
  const void **v27; // r11
  bool v28; // al
  int v29; // eax
  unsigned int v30; // esi
  char v31; // bl
  unsigned int v32; // eax
  unsigned int v33; // eax
  bool v34; // zf
  unsigned int v35; // eax
  __int64 v37; // rax
  __int64 v38; // rsi
  unsigned __int64 v39; // rdx
  __int64 v40; // rax
  unsigned int v41; // eax
  void *v42; // rdx
  unsigned int v43; // ecx
  unsigned int v44; // r9d
  __int64 v45; // rdi
  __int64 v46; // rcx
  unsigned __int64 v47; // r11
  unsigned int v48; // edi
  unsigned __int64 v49; // rdx
  unsigned int v50; // edi
  unsigned __int64 v51; // rdx
  unsigned int v52; // ebx
  bool v53; // al
  unsigned int v54; // edi
  __int64 v55; // r8
  unsigned __int64 v56; // rax
  unsigned __int64 v57; // rdi
  unsigned __int64 v58; // rax
  unsigned __int64 v59; // rax
  unsigned int v60; // r8d
  __int64 v61; // rdi
  unsigned int v62; // eax
  unsigned __int64 v63; // rdx
  unsigned __int64 v64; // rdi
  unsigned __int64 v65; // rdx
  unsigned __int64 v66; // rdi
  unsigned __int64 v67; // rdx
  unsigned int v68; // r8d
  __int64 v69; // rdi
  int v70; // eax
  __int64 v71; // rdx
  unsigned __int64 v72; // rax
  unsigned __int64 v73; // rdi
  unsigned __int64 v74; // rdx
  unsigned __int64 v75; // rdx
  unsigned __int64 v76; // rax
  unsigned int v77; // eax
  unsigned int v78; // r9d
  unsigned __int64 v79; // rax
  unsigned __int64 v80; // rsi
  unsigned __int64 v81; // rax
  unsigned __int64 v82; // rax
  unsigned int v83; // r8d
  __int64 v84; // rsi
  bool v85; // bl
  unsigned int v86; // ecx
  __int64 v87; // rdx
  unsigned __int64 v88; // rax
  unsigned __int64 v89; // rsi
  unsigned __int64 v90; // rax
  unsigned int v91; // eax
  unsigned int v92; // esi
  unsigned __int64 v93; // rdi
  int v94; // eax
  int v95; // eax
  bool v96; // al
  __int64 v97; // r8
  unsigned __int64 v98; // rax
  int v99; // eax
  unsigned __int64 v100; // rax
  unsigned int v101; // esi
  __int64 v102; // rsi
  unsigned __int64 v103; // rax
  unsigned int v104; // eax
  unsigned int v105; // r8d
  unsigned __int64 v106; // rdi
  unsigned int v107; // edx
  int v108; // ebx
  unsigned __int64 v109; // rdx
  unsigned __int64 v110; // rdx
  unsigned __int64 v111; // rax
  __int64 v112; // rax
  __int64 v113; // rdx
  __int64 v114; // rdx
  __int64 v115; // rdx
  unsigned __int64 v116; // rdx
  unsigned __int64 v117; // rdx
  unsigned __int64 v118; // rsi
  unsigned __int64 v119; // rax
  __int64 v120; // rdx
  __int64 v121; // rsi
  __int64 v122; // rsi
  __int64 v123; // rsi
  unsigned __int64 v124; // [rsp+0h] [rbp-100h]
  unsigned int v125; // [rsp+0h] [rbp-100h]
  unsigned int v126; // [rsp+0h] [rbp-100h]
  unsigned int v127; // [rsp+0h] [rbp-100h]
  unsigned int v128; // [rsp+8h] [rbp-F8h]
  __int64 v129; // [rsp+8h] [rbp-F8h]
  unsigned int v130; // [rsp+8h] [rbp-F8h]
  unsigned int v131; // [rsp+8h] [rbp-F8h]
  unsigned int v132; // [rsp+8h] [rbp-F8h]
  unsigned int v133; // [rsp+8h] [rbp-F8h]
  unsigned __int64 v134; // [rsp+8h] [rbp-F8h]
  unsigned int v135; // [rsp+10h] [rbp-F0h]
  unsigned int v136; // [rsp+10h] [rbp-F0h]
  __int64 v137; // [rsp+10h] [rbp-F0h]
  __int64 v138; // [rsp+18h] [rbp-E8h]
  unsigned int v139; // [rsp+18h] [rbp-E8h]
  unsigned int v140; // [rsp+18h] [rbp-E8h]
  unsigned int v142; // [rsp+28h] [rbp-D8h]
  unsigned int v143; // [rsp+28h] [rbp-D8h]
  unsigned __int64 v144; // [rsp+30h] [rbp-D0h] BYREF
  unsigned int v145; // [rsp+38h] [rbp-C8h]
  unsigned __int64 v146; // [rsp+40h] [rbp-C0h] BYREF
  unsigned int v147; // [rsp+48h] [rbp-B8h]
  __int128 v148; // [rsp+50h] [rbp-B0h] BYREF
  __int64 v149; // [rsp+60h] [rbp-A0h]
  void *v150; // [rsp+70h] [rbp-90h] BYREF
  unsigned int v151; // [rsp+78h] [rbp-88h]
  void *s; // [rsp+80h] [rbp-80h] BYREF
  unsigned int v153; // [rsp+88h] [rbp-78h]
  unsigned __int64 v154; // [rsp+90h] [rbp-70h] BYREF
  unsigned int v155; // [rsp+98h] [rbp-68h]
  void *v156; // [rsp+A0h] [rbp-60h] BYREF
  unsigned int v157; // [rsp+A8h] [rbp-58h]
  unsigned __int64 v158; // [rsp+B0h] [rbp-50h] BYREF
  unsigned int v159; // [rsp+B8h] [rbp-48h]
  void *v160; // [rsp+C0h] [rbp-40h]
  unsigned int v161; // [rsp+C8h] [rbp-38h]

  v7 = *(_DWORD *)(a2 + 8);
  v8 = v7 - 1;
  v138 = 1LL << ((unsigned __int8)v7 - 1);
  v9 = *(_QWORD *)a2;
  if ( v7 > 0x40 )
  {
    if ( (*(_QWORD *)(v9 + 8LL * (v8 >> 6)) & v138) == 0 )
    {
      v151 = *(_DWORD *)(a2 + 8);
      goto LABEL_58;
    }
  }
  else if ( (v9 & (1LL << ((unsigned __int8)v7 - 1))) == 0 )
  {
    v151 = *(_DWORD *)(a2 + 8);
    goto LABEL_6;
  }
  v10 = *(_DWORD *)(a3 + 8);
  v11 = *(_QWORD *)a3;
  v12 = 1LL << ((unsigned __int8)v10 - 1);
  if ( v10 <= 0x40 )
  {
    if ( (v11 & v12) == 0 )
      goto LABEL_5;
LABEL_48:
    sub_C79480((unsigned __int64 *)a1, a2, a3, a4);
    return a1;
  }
  if ( (*(_QWORD *)(v11 + 8LL * ((v10 - 1) >> 6)) & v12) != 0 )
    goto LABEL_48;
LABEL_5:
  v151 = v7;
  if ( v7 > 0x40 )
  {
LABEL_58:
    sub_C43690((__int64)&v150, 0, 0);
    v153 = v7;
    sub_C43690((__int64)&s, 0, 0);
    v13 = *(_DWORD *)(a2 + 8);
    v8 = v7 - 1;
    goto LABEL_7;
  }
LABEL_6:
  v150 = 0;
  v13 = v7;
  v153 = v7;
  s = 0;
LABEL_7:
  if ( !v13 )
    goto LABEL_50;
  if ( v13 <= 0x40 )
  {
    v15 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v13) == *(_QWORD *)a2;
  }
  else
  {
    v142 = v8;
    v14 = sub_C445E0(a2);
    v8 = v142;
    v15 = v14 == v13;
  }
  if ( v15
    || (v16 = *(_DWORD *)(a3 + 8)) == 0
    || (v16 <= 0x40
      ? (v18 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v16) == *(_QWORD *)a3)
      : (v128 = *(_DWORD *)(a3 + 8), v143 = v8, v17 = sub_C445E0(a3), v16 = v128, v8 = v143, v18 = v128 == v17),
        v18) )
  {
LABEL_50:
    v37 = v151;
    if ( v151 > 0x40 )
    {
      memset(v150, -1, 8 * (((unsigned __int64)v151 + 63) >> 6));
      v37 = v151;
      v38 = (__int64)v150;
    }
    else
    {
      v150 = (void *)-1LL;
      v38 = -1;
    }
    v39 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v37;
    if ( (_DWORD)v37 )
    {
      if ( (unsigned int)v37 > 0x40 )
      {
        v40 = (unsigned int)((unsigned __int64)(v37 + 63) >> 6) - 1;
        *(_QWORD *)(v38 + 8 * v40) &= v39;
        v41 = v153;
        v42 = 0;
        if ( v153 <= 0x40 )
        {
LABEL_55:
          v43 = v151;
          *(_DWORD *)(a1 + 24) = v41;
          *(_QWORD *)(a1 + 16) = v42;
          *(_DWORD *)(a1 + 8) = v43;
          *(_QWORD *)a1 = v150;
          return a1;
        }
LABEL_61:
        memset(s, 0, 8 * (((unsigned __int64)v41 + 63) >> 6));
        v42 = s;
        v41 = v153;
        goto LABEL_55;
      }
    }
    else
    {
      v39 = 0;
    }
    v41 = v153;
    v150 = (void *)(v38 & v39);
    v42 = 0;
    if ( v153 <= 0x40 )
      goto LABEL_55;
    goto LABEL_61;
  }
  v19 = *(_DWORD *)(a2 + 24);
  v20 = *(_QWORD *)(a2 + 16);
  v149 = 0;
  v21 = v19 - 1;
  v148 = 0;
  v22 = v20;
  v23 = 1LL << ((unsigned __int8)v19 - 1);
  if ( v19 > 0x40 )
    v22 = *(_QWORD *)(v20 + 8LL * (v21 >> 6));
  if ( (v22 & v23) != 0 )
  {
    v44 = *(_DWORD *)(a3 + 24);
    v125 = v44 - 1;
    v129 = 1LL << ((unsigned __int8)v44 - 1);
    v45 = *(_QWORD *)(a3 + 16);
    v46 = v45;
    if ( v44 > 0x40 )
      v46 = *(_QWORD *)(v45 + 8LL * (v125 >> 6));
    if ( (v46 & v129) == 0 )
    {
      v47 = *(_QWORD *)a3;
      if ( v16 > 0x40 )
        v47 = *(_QWORD *)(v47 + 8LL * ((v16 - 1) >> 6));
      if ( (v47 & (1LL << ((unsigned __int8)v16 - 1))) == 0 )
        goto LABEL_18;
      if ( a4 )
      {
LABEL_70:
        v147 = v44;
        if ( v44 > 0x40 )
          sub_C43780((__int64)&v146, (const void **)(a3 + 16));
        else
          v146 = *(_QWORD *)(a3 + 16);
        v48 = *(_DWORD *)(a3 + 8);
        v49 = *(_QWORD *)a3;
        if ( v48 > 0x40 )
          v49 = *(_QWORD *)(v49 + 8LL * ((v48 - 1) >> 6));
        if ( (v49 & (1LL << ((unsigned __int8)v48 - 1))) == 0 )
        {
          v115 = 1LL << ((unsigned __int8)v147 - 1);
          if ( v147 > 0x40 )
            *(_QWORD *)(v146 + 8LL * ((v147 - 1) >> 6)) |= v115;
          else
            v146 |= v115;
        }
        v155 = *(_DWORD *)(a2 + 24);
        if ( v155 > 0x40 )
          sub_C43780((__int64)&v154, (const void **)(a2 + 16));
        else
          v154 = *(_QWORD *)(a2 + 16);
        v50 = *(_DWORD *)(a2 + 8);
        v51 = *(_QWORD *)a2;
        if ( v50 > 0x40 )
          v51 = *(_QWORD *)(v51 + 8LL * ((v50 - 1) >> 6));
        if ( (v51 & (1LL << ((unsigned __int8)v50 - 1))) == 0 )
        {
          v113 = 1LL << ((unsigned __int8)v155 - 1);
          if ( v155 > 0x40 )
            *(_QWORD *)(v154 + 8LL * ((v155 - 1) >> 6)) |= v113;
          else
            v154 |= v113;
        }
        v52 = v147;
        if ( v147 <= 0x40 )
          v53 = v146 == 0;
        else
          v53 = v52 == (unsigned int)sub_C444A0((__int64)&v146);
        if ( v53 )
        {
          v159 = v155;
          if ( v155 > 0x40 )
            sub_C43780((__int64)&v158, (const void **)&v154);
          else
            v158 = v154;
          goto LABEL_117;
        }
        goto LABEL_164;
      }
      v159 = v13;
      if ( v13 > 0x40 )
      {
        sub_C43780((__int64)&v158, (const void **)a2);
        v13 = v159;
        if ( v159 > 0x40 )
        {
          sub_C43D10((__int64)&v158);
          v19 = *(_DWORD *)(a2 + 24);
          v13 = v159;
          v75 = v158;
          v20 = *(_QWORD *)(a2 + 16);
          v21 = v19 - 1;
          v23 = 1LL << ((unsigned __int8)v19 - 1);
LABEL_137:
          v145 = v13;
          v144 = v75;
          if ( v19 > 0x40 )
            v76 = *(_QWORD *)(v20 + 8LL * (v21 >> 6));
          else
            v76 = v20;
          if ( (v76 & v23) == 0 )
          {
            v112 = ~(1LL << ((unsigned __int8)v13 - 1));
            if ( v13 <= 0x40 )
            {
              v109 = v112 & v75;
              goto LABEL_247;
            }
            *(_QWORD *)(v75 + 8LL * ((v13 - 1) >> 6)) &= v112;
            v13 = v145;
          }
          if ( v13 > 0x40 )
          {
            sub_C43D10((__int64)&v144);
            goto LABEL_142;
          }
          v109 = v144;
LABEL_247:
          v110 = ~v109;
          v111 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v13;
          if ( !v13 )
            v111 = 0;
          v144 = v111 & v110;
LABEL_142:
          sub_C46250((__int64)&v144);
          v77 = v145;
          v78 = *(_DWORD *)(a3 + 8);
          v145 = 0;
          v147 = v77;
          v159 = v78;
          v146 = v144;
          if ( v78 > 0x40 )
          {
            sub_C43780((__int64)&v158, (const void **)a3);
            v78 = v159;
            if ( v159 > 0x40 )
            {
              sub_C43D10((__int64)&v158);
              v78 = v159;
              v82 = v158;
              goto LABEL_147;
            }
            v79 = v158;
          }
          else
          {
            v79 = *(_QWORD *)a3;
          }
          v80 = ~v79;
          v81 = 0;
          if ( v78 )
            v81 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v78;
          v82 = v80 & v81;
LABEL_147:
          v83 = *(_DWORD *)(a3 + 24);
          v84 = *(_QWORD *)(a3 + 16);
          v155 = v78;
          v154 = v82;
          if ( v83 > 0x40 )
            v84 = *(_QWORD *)(v84 + 8LL * ((v83 - 1) >> 6));
          if ( (v84 & (1LL << ((unsigned __int8)v83 - 1))) == 0 )
          {
            v114 = ~(1LL << ((unsigned __int8)v78 - 1));
            if ( v78 <= 0x40 )
            {
              v154 = v114 & v82;
              v85 = (int)sub_C49970((__int64)&v146, &v154) >= 0;
LABEL_153:
              if ( v147 > 0x40 && v146 )
                j_j___libc_free_0_0(v146);
              if ( v145 > 0x40 && v144 )
                j_j___libc_free_0_0(v144);
              if ( !v85 )
                goto LABEL_125;
              v44 = *(_DWORD *)(a3 + 24);
              goto LABEL_70;
            }
            *(_QWORD *)(v82 + 8LL * ((v78 - 1) >> 6)) &= v114;
            v78 = v155;
          }
          v139 = v78;
          v85 = (int)sub_C49970((__int64)&v146, &v154) >= 0;
          if ( v139 > 0x40 && v154 )
            j_j___libc_free_0_0(v154);
          goto LABEL_153;
        }
        v19 = *(_DWORD *)(a2 + 24);
        v21 = v19 - 1;
        v73 = ~v158;
        v23 = 1LL << ((unsigned __int8)v19 - 1);
        v74 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v159;
        if ( !v159 )
          v74 = 0;
        v20 = *(_QWORD *)(a2 + 16);
      }
      else
      {
        v73 = ~*(_QWORD *)a2;
        v74 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v13;
      }
      v75 = v73 & v74;
      goto LABEL_137;
    }
    v159 = v16;
    if ( v16 <= 0x40 )
    {
      v88 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v16;
      v89 = ~*(_QWORD *)a3;
LABEL_169:
      v90 = v89 & v88;
      goto LABEL_170;
    }
    v132 = v8;
    sub_C43780((__int64)&v158, (const void **)a3);
    v16 = v159;
    v8 = v132;
    if ( v159 <= 0x40 )
    {
      v44 = *(_DWORD *)(a3 + 24);
      v89 = ~v158;
      v125 = v44 - 1;
      v129 = 1LL << ((unsigned __int8)v44 - 1);
      v88 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v159;
      if ( v159 )
      {
        v45 = *(_QWORD *)(a3 + 16);
        goto LABEL_169;
      }
      v45 = *(_QWORD *)(a3 + 16);
      v90 = 0;
    }
    else
    {
      sub_C43D10((__int64)&v158);
      v44 = *(_DWORD *)(a3 + 24);
      v16 = v159;
      v90 = v158;
      v8 = v132;
      v125 = v44 - 1;
      v129 = 1LL << ((unsigned __int8)v44 - 1);
      v45 = *(_QWORD *)(a3 + 16);
    }
LABEL_170:
    v147 = v16;
    v146 = v90;
    if ( v44 > 0x40 )
      v45 = *(_QWORD *)(v45 + 8LL * (v125 >> 6));
    if ( (v45 & v129) == 0 )
    {
      v102 = ~(1LL << ((unsigned __int8)v16 - 1));
      if ( v16 > 0x40 )
        *(_QWORD *)(v90 + 8LL * ((v16 - 1) >> 6)) &= v102;
      else
        v146 = v102 & v90;
    }
    v91 = *(_DWORD *)(a2 + 24);
    v155 = v91;
    if ( v91 > 0x40 )
    {
      v133 = v8;
      sub_C43780((__int64)&v154, (const void **)(a2 + 16));
      v91 = v155;
      v8 = v133;
    }
    else
    {
      v154 = *(_QWORD *)(a2 + 16);
    }
    v92 = *(_DWORD *)(a2 + 8);
    v93 = *(_QWORD *)a2;
    if ( v92 > 0x40 )
      v93 = *(_QWORD *)(v93 + 8LL * ((v92 - 1) >> 6));
    if ( (v93 & (1LL << ((unsigned __int8)v92 - 1))) == 0 )
    {
      v86 = v91 - 1;
      v87 = 1LL << ((unsigned __int8)v91 - 1);
      if ( v91 <= 0x40 )
      {
        v154 |= v87;
        goto LABEL_163;
      }
      *(_QWORD *)(v154 + 8LL * (v86 >> 6)) |= v87;
      v91 = v155;
    }
    v86 = v91 - 1;
    if ( v91 > 0x40 )
    {
      v126 = v8;
      v130 = v91 - 1;
      if ( (*(_QWORD *)(v154 + 8LL * (v86 >> 6)) & (1LL << v86)) == 0 )
        goto LABEL_164;
      v94 = sub_C44590((__int64)&v154);
      v8 = v126;
      if ( v94 != v130 )
        goto LABEL_164;
LABEL_181:
      if ( !v147
        || (v147 <= 0x40
          ? (v96 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v147) == v146)
          : (v127 = v147, v131 = v8, v95 = sub_C445E0((__int64)&v146), v8 = v131, v96 = v127 == v95),
            v96) )
      {
        v159 = v7;
        v97 = ~v138;
        if ( v7 > 0x40 )
        {
          v137 = ~v138;
          v140 = v8;
          sub_C43690((__int64)&v158, -1, 1);
          v97 = v137;
          if ( v159 > 0x40 )
          {
            *(_QWORD *)(v158 + 8LL * (v140 >> 6)) &= v137;
            goto LABEL_117;
          }
        }
        else
        {
          v98 = 0xFFFFFFFFFFFFFFFFLL >> (63 - ((v7 - 1) & 0x3F));
          if ( !v7 )
            v98 = 0;
          v158 = v98;
        }
        v158 &= v97;
LABEL_117:
        if ( (_BYTE)v149 )
        {
          if ( DWORD2(v148) > 0x40 && (_QWORD)v148 )
            j_j___libc_free_0_0(v148);
          *(_QWORD *)&v148 = v158;
          DWORD2(v148) = v159;
        }
        else
        {
          LOBYTE(v149) = 1;
          DWORD2(v148) = v159;
          *(_QWORD *)&v148 = v158;
        }
        if ( v155 > 0x40 && v154 )
          j_j___libc_free_0_0(v154);
        if ( v147 > 0x40 && v146 )
          j_j___libc_free_0_0(v146);
        goto LABEL_125;
      }
LABEL_164:
      sub_C4A3E0((__int64)&v158, (__int64)&v154, (__int64)&v146);
      goto LABEL_117;
    }
LABEL_163:
    if ( v154 != 1LL << v86 )
      goto LABEL_164;
    goto LABEL_181;
  }
LABEL_18:
  v24 = v13 - 1;
  v25 = 1LL << ((unsigned __int8)v13 - 1);
  v26 = *(_QWORD *)a2;
  if ( v13 > 0x40 )
    v26 = *(_QWORD *)(v26 + 8LL * (v24 >> 6));
  if ( (v26 & v25) == 0 )
    goto LABEL_24;
  v27 = (const void **)(a2 + 16);
  v28 = v20 == 0;
  if ( v19 > 0x40 )
  {
    v124 = v20;
    v135 = v16;
    v29 = sub_C444A0(a2 + 16);
    v27 = (const void **)(a2 + 16);
    v16 = v135;
    v25 = 1LL << ((unsigned __int8)v13 - 1);
    v20 = v124;
    v28 = v19 == v29;
  }
  if ( v28 )
    goto LABEL_24;
  v54 = *(_DWORD *)(a3 + 24);
  v55 = *(_QWORD *)(a3 + 16);
  if ( v54 > 0x40 )
    v55 = *(_QWORD *)(v55 + 8LL * ((v54 - 1) >> 6));
  if ( (v55 & (1LL << ((unsigned __int8)v54 - 1))) == 0 )
    goto LABEL_24;
  if ( a4 )
  {
LABEL_100:
    v159 = v16;
    if ( v16 > 0x40 )
    {
      sub_C43780((__int64)&v158, (const void **)a3);
      v16 = v159;
      if ( v159 > 0x40 )
      {
        sub_C43D10((__int64)&v158);
        v16 = v159;
        v59 = v158;
        goto LABEL_105;
      }
      v56 = v158;
    }
    else
    {
      v56 = *(_QWORD *)a3;
    }
    v57 = ~v56;
    v58 = 0;
    if ( v16 )
      v58 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v16;
    v59 = v57 & v58;
LABEL_105:
    v60 = *(_DWORD *)(a3 + 24);
    v61 = *(_QWORD *)(a3 + 16);
    v147 = v16;
    v146 = v59;
    if ( v60 > 0x40 )
      v61 = *(_QWORD *)(v61 + 8LL * ((v60 - 1) >> 6));
    if ( (v61 & (1LL << ((unsigned __int8)v60 - 1))) == 0 )
    {
      v121 = ~(1LL << ((unsigned __int8)v16 - 1));
      if ( v16 > 0x40 )
        *(_QWORD *)(v59 + 8LL * ((v16 - 1) >> 6)) &= v121;
      else
        v146 = v59 & v121;
    }
    v62 = *(_DWORD *)(a2 + 8);
    v159 = v62;
    if ( v62 > 0x40 )
    {
      sub_C43780((__int64)&v158, (const void **)a2);
      v62 = v159;
      if ( v159 > 0x40 )
      {
        sub_C43D10((__int64)&v158);
        v62 = v159;
        v67 = v158;
LABEL_113:
        v68 = *(_DWORD *)(a2 + 24);
        v69 = *(_QWORD *)(a2 + 16);
        v155 = v62;
        v154 = v67;
        if ( v68 > 0x40 )
          v69 = *(_QWORD *)(v69 + 8LL * ((v68 - 1) >> 6));
        if ( (v69 & (1LL << ((unsigned __int8)v68 - 1))) == 0 )
        {
          v122 = ~(1LL << ((unsigned __int8)v62 - 1));
          if ( v62 > 0x40 )
            *(_QWORD *)(v67 + 8LL * ((v62 - 1) >> 6)) &= v122;
          else
            v154 = v67 & v122;
        }
        sub_C4A3E0((__int64)&v158, (__int64)&v154, (__int64)&v146);
        goto LABEL_117;
      }
      v63 = v158;
    }
    else
    {
      v63 = *(_QWORD *)a2;
    }
    v64 = v63;
    v65 = 0;
    v66 = ~v64;
    if ( v62 )
      v65 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v62;
    v67 = v66 & v65;
    goto LABEL_113;
  }
  v147 = v19;
  if ( v19 > 0x40 )
  {
    sub_C43780((__int64)&v146, v27);
    v13 = *(_DWORD *)(a2 + 8);
    v24 = v13 - 1;
    v25 = 1LL << ((unsigned __int8)v13 - 1);
  }
  else
  {
    v146 = v20;
  }
  v103 = *(_QWORD *)a2;
  if ( v13 > 0x40 )
    v103 = *(_QWORD *)(v103 + 8LL * (v24 >> 6));
  if ( (v103 & v25) == 0 )
  {
    v120 = 1LL << ((unsigned __int8)v147 - 1);
    if ( v147 > 0x40 )
      *(_QWORD *)(v146 + 8LL * ((v147 - 1) >> 6)) |= v120;
    else
      v146 |= v120;
  }
  v104 = *(_DWORD *)(a3 + 24);
  v155 = v104;
  if ( v104 > 0x40 )
  {
    sub_C43780((__int64)&v154, (const void **)(a3 + 16));
    v104 = v155;
  }
  else
  {
    v154 = *(_QWORD *)(a3 + 16);
  }
  v105 = *(_DWORD *)(a3 + 8);
  v106 = *(_QWORD *)a3;
  if ( v105 > 0x40 )
    v106 = *(_QWORD *)(v106 + 8LL * ((v105 - 1) >> 6));
  if ( (v106 & (1LL << ((unsigned __int8)v105 - 1))) == 0 )
  {
    v123 = 1LL << ((unsigned __int8)v104 - 1);
    if ( v104 <= 0x40 )
    {
      v116 = v123 | v154;
      goto LABEL_261;
    }
    *(_QWORD *)(v154 + 8LL * ((v104 - 1) >> 6)) |= v123;
    v104 = v155;
  }
  if ( v104 > 0x40 )
  {
    sub_C43D10((__int64)&v154);
    goto LABEL_221;
  }
  v116 = v154;
LABEL_261:
  v117 = ~v116;
  v118 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v104;
  v34 = v104 == 0;
  v119 = 0;
  if ( !v34 )
    v119 = v118;
  v154 = v119 & v117;
LABEL_221:
  sub_C46250((__int64)&v154);
  v107 = v155;
  v155 = 0;
  v159 = v107;
  v136 = v107;
  v158 = v154;
  v134 = v154;
  v108 = sub_C49970((__int64)&v146, &v158);
  if ( v136 > 0x40 )
  {
    if ( v134 )
    {
      j_j___libc_free_0_0(v134);
      if ( v155 > 0x40 )
      {
        if ( v154 )
          j_j___libc_free_0_0(v154);
      }
    }
  }
  if ( v147 > 0x40 && v146 )
    j_j___libc_free_0_0(v146);
  if ( v108 >= 0 )
  {
    v16 = *(_DWORD *)(a3 + 8);
    goto LABEL_100;
  }
LABEL_125:
  if ( !(_BYTE)v149 )
    goto LABEL_24;
  v70 = DWORD2(v148);
  v71 = 1LL << (BYTE8(v148) - 1);
  if ( DWORD2(v148) > 0x40 )
  {
    if ( (*(_QWORD *)(v148 + 8LL * ((unsigned int)(DWORD2(v148) - 1) >> 6)) & v71) == 0 )
    {
      v70 = sub_C444A0((__int64)&v148);
      goto LABEL_130;
    }
    v99 = sub_C44500((__int64)&v148);
LABEL_193:
    v101 = v153 - v99;
    if ( v153 != v153 - v99 )
    {
      if ( v101 <= 0x3F && v153 <= 0x40 )
      {
        v30 = v151;
        s = (void *)((0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v99) << ((unsigned __int8)v153
                                                                            - (unsigned __int8)v99))
                   | (unsigned __int64)s);
        goto LABEL_25;
      }
      sub_C43C90(&s, v101, v153);
    }
LABEL_24:
    v30 = v151;
    goto LABEL_25;
  }
  if ( (v71 & (unsigned __int64)v148) != 0 )
  {
    if ( !DWORD2(v148) )
      goto LABEL_24;
    v99 = 64;
    if ( (_QWORD)v148 << (64 - BYTE8(v148)) != -1 )
    {
      _BitScanReverse64(&v100, ~((_QWORD)v148 << (64 - BYTE8(v148))));
      v99 = v100 ^ 0x3F;
    }
    goto LABEL_193;
  }
  if ( (_QWORD)v148 )
  {
    _BitScanReverse64(&v72, v148);
    v70 = DWORD2(v148) - 64 + (v72 ^ 0x3F);
  }
LABEL_130:
  v30 = v151 - v70;
  if ( v151 != v151 - v70 )
  {
    if ( v30 <= 0x3F && v151 <= 0x40 )
    {
      v155 = v151;
      v31 = a4;
      v150 = (void *)((0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v70) << v30) | (unsigned __int64)v150);
      goto LABEL_26;
    }
    sub_C43C90(&v150, v30, v151);
    v30 = v151;
  }
LABEL_25:
  v155 = v30;
  v31 = a4;
  if ( v30 > 0x40 )
  {
    sub_C43780((__int64)&v154, (const void **)&v150);
    goto LABEL_27;
  }
LABEL_26:
  v154 = (unsigned __int64)v150;
LABEL_27:
  v157 = v153;
  if ( v153 > 0x40 )
    sub_C43780((__int64)&v156, (const void **)&s);
  else
    v156 = s;
  sub_C6FD10((__int64)&v158, (__int64)&v154, a2, a3, v31);
  if ( v151 > 0x40 && v150 )
    j_j___libc_free_0_0(v150);
  v150 = (void *)v158;
  v32 = v159;
  v159 = 0;
  v151 = v32;
  if ( v153 > 0x40 && s )
  {
    j_j___libc_free_0_0(s);
    s = v160;
    v153 = v161;
    if ( v159 > 0x40 && v158 )
      j_j___libc_free_0_0(v158);
  }
  else
  {
    s = v160;
    v153 = v161;
  }
  if ( v157 > 0x40 && v156 )
    j_j___libc_free_0_0(v156);
  if ( v155 > 0x40 && v154 )
    j_j___libc_free_0_0(v154);
  v33 = v151;
  v34 = (_BYTE)v149 == 0;
  v151 = 0;
  *(_DWORD *)(a1 + 8) = v33;
  *(_QWORD *)a1 = v150;
  v35 = v153;
  v153 = 0;
  *(_DWORD *)(a1 + 24) = v35;
  *(_QWORD *)(a1 + 16) = s;
  if ( !v34 )
  {
    LOBYTE(v149) = 0;
    if ( DWORD2(v148) > 0x40 )
    {
      if ( (_QWORD)v148 )
      {
        j_j___libc_free_0_0(v148);
        if ( v153 > 0x40 )
        {
          if ( s )
            j_j___libc_free_0_0(s);
        }
      }
    }
  }
  if ( v151 > 0x40 && v150 )
    j_j___libc_free_0_0(v150);
  return a1;
}
