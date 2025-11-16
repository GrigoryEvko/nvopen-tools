// Function: sub_3029BF0
// Address: 0x3029bf0
//
void __fastcall sub_3029BF0(__int64 a1, _BYTE *a2, __int64 a3, char a4, __int64 a5)
{
  _BYTE *v7; // rdi
  __int64 v10; // rax
  __int64 v11; // rdx
  const char *v12; // rax
  unsigned __int64 v13; // rdx
  const char *v14; // rax
  unsigned __int64 v15; // rdx
  _BYTE *v16; // rdi
  __int64 v17; // r15
  char v18; // al
  _BYTE *v19; // rdi
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 v23; // r8
  __int16 v24; // ax
  char v25; // cl
  __int64 v26; // r8
  unsigned __int8 v27; // cl
  __int64 v28; // rax
  __int64 v29; // rdx
  unsigned __int64 v30; // r15
  int v31; // eax
  __int64 v32; // rax
  __int64 v33; // r12
  const char *v34; // rax
  size_t v35; // rdx
  __int64 v36; // rax
  __int64 v37; // r13
  const char *v38; // rax
  size_t v39; // rdx
  __int64 v40; // rax
  _QWORD *v41; // r13
  unsigned int v42; // r15d
  signed __int64 v43; // rbx
  int v44; // r13d
  __int64 v45; // rax
  __int64 v46; // rax
  int v47; // eax
  __int64 v48; // rax
  unsigned __int8 *v49; // r14
  int v50; // eax
  unsigned int v51; // eax
  const char *v52; // rax
  __int64 v53; // rdx
  __int64 v54; // rcx
  __int64 v55; // r8
  __int64 v56; // r9
  __int64 v57; // rcx
  __int64 v58; // r8
  __int64 v59; // r9
  __int64 v60; // rcx
  __int64 v61; // r8
  __int64 v62; // r9
  char v63; // al
  __int64 v64; // rax
  __int64 v65; // rax
  const char *v66; // rax
  unsigned __int64 v67; // rdx
  const char *v68; // rax
  unsigned __int64 v69; // rdx
  bool v70; // al
  __int64 v71; // r8
  size_t v72; // rcx
  char v73; // al
  unsigned int v74; // r8d
  const char *v75; // rsi
  __int64 v76; // rax
  __int64 v77; // rax
  __int64 v78; // rax
  __int64 v79; // rdx
  __int64 v80; // rcx
  __int64 v81; // r8
  __int64 v82; // r9
  __int64 v83; // rax
  __int64 v84; // rax
  __int64 v85; // rax
  __int64 v86; // rax
  unsigned __int64 v87; // rbx
  unsigned __int64 *v88; // r14
  __int64 v89; // r12
  const char *v90; // rax
  __int64 v91; // rdi
  unsigned __int64 *v92; // r12
  size_t v93; // rdx
  __int64 v94; // rax
  unsigned __int64 *v95; // rax
  unsigned __int64 *v96; // rax
  __int64 v97; // rsi
  unsigned __int64 *v98; // rax
  unsigned __int64 *v99; // rdx
  char v100; // di
  _BYTE *v101; // rsi
  unsigned __int64 v102; // rdi
  char *v103; // rcx
  __int64 v104; // rsi
  char *v105; // rdi
  __int64 i; // rsi
  signed __int64 v107; // rax
  __int64 v108; // rax
  __int64 v109; // rax
  __int64 v110; // rax
  __int64 v111; // rax
  __int64 v112; // rax
  __int64 v113; // rdx
  __int64 v114; // rcx
  __int64 v115; // r8
  __int64 v116; // r9
  const char *v117; // rax
  __int64 v118; // rdx
  __int64 v119; // rcx
  __int64 v120; // r8
  __int64 v121; // r9
  unsigned __int64 v122; // [rsp+0h] [rbp-240h]
  int v123; // [rsp+Ch] [rbp-234h]
  __int64 v124; // [rsp+10h] [rbp-230h]
  char v125; // [rsp+18h] [rbp-228h]
  __int64 v126; // [rsp+18h] [rbp-228h]
  size_t v127; // [rsp+18h] [rbp-228h]
  size_t n; // [rsp+20h] [rbp-220h]
  size_t na; // [rsp+20h] [rbp-220h]
  size_t nb; // [rsp+20h] [rbp-220h]
  _BYTE *v131; // [rsp+28h] [rbp-218h] BYREF
  __m128i v132; // [rsp+30h] [rbp-210h] BYREF
  const char *v133; // [rsp+40h] [rbp-200h]
  __int64 v134; // [rsp+48h] [rbp-1F8h]
  __int16 v135; // [rsp+50h] [rbp-1F0h]
  __m128i v136[2]; // [rsp+60h] [rbp-1E0h] BYREF
  char v137; // [rsp+80h] [rbp-1C0h]
  char v138; // [rsp+81h] [rbp-1BFh]
  __m128i v139[3]; // [rsp+90h] [rbp-1B0h] BYREF
  __m128i v140; // [rsp+C0h] [rbp-180h] BYREF
  const char *v141; // [rsp+D0h] [rbp-170h]
  __int64 v142; // [rsp+D8h] [rbp-168h]
  __int16 v143; // [rsp+E0h] [rbp-160h]
  __m128i v144[2]; // [rsp+F0h] [rbp-150h] BYREF
  char v145; // [rsp+110h] [rbp-130h]
  char v146; // [rsp+111h] [rbp-12Fh]
  __m128i v147[2]; // [rsp+120h] [rbp-120h] BYREF
  char v148; // [rsp+140h] [rbp-100h]
  char v149; // [rsp+141h] [rbp-FFh]
  __m128i v150; // [rsp+150h] [rbp-F0h] BYREF
  size_t v151; // [rsp+160h] [rbp-E0h] BYREF
  __int64 v152; // [rsp+168h] [rbp-D8h]
  char *v153; // [rsp+170h] [rbp-D0h]
  __int64 v154; // [rsp+178h] [rbp-C8h]
  _BYTE v155[16]; // [rsp+180h] [rbp-C0h] BYREF
  _BYTE *v156; // [rsp+190h] [rbp-B0h]
  __int64 v157; // [rsp+198h] [rbp-A8h]
  _BYTE v158[32]; // [rsp+1A0h] [rbp-A0h] BYREF
  _BYTE *v159; // [rsp+1C0h] [rbp-80h]
  __int64 v160; // [rsp+1C8h] [rbp-78h]
  _BYTE v161[32]; // [rsp+1D0h] [rbp-70h] BYREF
  int v162; // [rsp+1F0h] [rbp-50h]
  __int64 v163; // [rsp+1F8h] [rbp-48h]
  char v164; // [rsp+200h] [rbp-40h]

  v7 = a2;
  v131 = a2;
  if ( (a2[35] & 4) != 0 )
  {
    v10 = sub_B31D10((__int64)a2, (__int64)a2, a3);
    if ( v11 == 13
      && *(_QWORD *)v10 == 0x74656D2E6D766C6CLL
      && *(_DWORD *)(v10 + 8) == 1952539745
      && *(_BYTE *)(v10 + 12) == 97 )
    {
      return;
    }
    v7 = v131;
  }
  v12 = sub_BD5D20((__int64)v7);
  if ( v13 <= 4 || *(_DWORD *)v12 != 1836477548 || v12[4] != 46 )
  {
    v14 = sub_BD5D20((__int64)v131);
    if ( v15 <= 4 || *(_DWORD *)v14 != 1836480110 || v14[4] != 46 )
    {
      v16 = v131;
      n = sub_31DA930(a1);
      v17 = *((_QWORD *)v131 + 3);
      v18 = v131[32] & 0xF;
      if ( v18 )
      {
        if ( *(_DWORD *)(a5 + 336) > 0x31u && v18 == 10 )
        {
          if ( *(_DWORD *)(*((_QWORD *)v131 + 1) + 8LL) >> 8 != 1 )
          {
LABEL_15:
            sub_904010(a3, ".weak ");
            v16 = v131;
            goto LABEL_16;
          }
          sub_904010(a3, ".common ");
          v16 = v131;
        }
        else if ( ((v18 + 15) & 0xFu) <= 4 || v18 == 10 )
        {
          goto LABEL_15;
        }
      }
      else
      {
        if ( sub_B2FC80((__int64)v131) )
          sub_904010(a3, ".extern ");
        else
          sub_904010(a3, ".visible ");
        v16 = v131;
      }
LABEL_16:
      if ( (unsigned __int8)sub_CE8750(v16) )
      {
        v33 = sub_904010(a3, ".global .texref ");
        v34 = sub_CEF7A0((__int64)v131);
      }
      else
      {
        if ( !(unsigned __int8)sub_CE87C0(v131) )
        {
          if ( sub_B2FC80((__int64)v131) )
          {
            sub_3024A30(a1, (__int64)v131, a3, a5);
            sub_904010(a3, ";\n");
            return;
          }
          if ( !(unsigned __int8)sub_CE8830(v131) )
          {
            v19 = v131;
            if ( (v131[32] & 0xF) == 8 )
            {
              v66 = sub_BD5D20((__int64)v131);
              if ( v67 > 0xB && *(_QWORD *)v66 == 0x72706C6C6F726E75LL && *((_DWORD *)v66 + 2) == 1634559841 )
                return;
              v68 = sub_BD5D20((__int64)v131);
              if ( v69 > 7 && *(_QWORD *)v68 == 0x656D616E656C6966LL )
                return;
              v19 = v131;
              if ( !*((_QWORD *)v131 + 2) )
                return;
            }
            if ( !a4 && (v19[32] & 0xFu) - 7 <= 1 && *(_DWORD *)(*((_QWORD *)v19 + 1) + 8LL) >> 8 == 3 )
            {
              v150.m128i_i64[0] = 0;
              if ( (unsigned __int8)sub_30209A0((__int64)v19, v150.m128i_i64) )
              {
                v87 = v150.m128i_i64[0];
                if ( v150.m128i_i64[0] )
                {
                  v88 = (unsigned __int64 *)(a1 + 1152);
                  v89 = sub_904010(a3, "// ");
                  v90 = sub_BD5D20((__int64)v131);
                  v91 = v89;
                  v92 = (unsigned __int64 *)(a1 + 1152);
                  v94 = sub_A51340(v91, v90, v93);
                  sub_904010(v94, " has been demoted\n");
                  v95 = *(unsigned __int64 **)(a1 + 1160);
                  if ( !v95 )
                    goto LABEL_131;
                  do
                  {
                    if ( v95[4] < v87 )
                    {
                      v95 = (unsigned __int64 *)v95[3];
                    }
                    else
                    {
                      v92 = v95;
                      v95 = (unsigned __int64 *)v95[2];
                    }
                  }
                  while ( v95 );
                  if ( v88 == v92 || v92[4] > v87 )
                  {
LABEL_131:
                    v96 = (unsigned __int64 *)sub_22077B0(0x40u);
                    v97 = (__int64)v92;
                    v96[4] = v87;
                    v92 = v96;
                    v96[5] = 0;
                    v96[6] = 0;
                    v96[7] = 0;
                    v98 = sub_3029AF0((_QWORD *)(a1 + 1144), v97, v96 + 4);
                    if ( v99 )
                    {
                      v100 = v88 == v99 || v98 || v87 < v99[4];
                      sub_220F040(v100, (__int64)v92, v99, (_QWORD *)(a1 + 1152));
                      ++*(_QWORD *)(a1 + 1184);
                    }
                    else
                    {
                      v102 = (unsigned __int64)v92;
                      v92 = v98;
                      j_j___libc_free_0(v102);
                    }
                  }
                  v101 = (_BYTE *)v92[6];
                  if ( v101 == (_BYTE *)v92[7] )
                  {
                    sub_3028380((__int64)(v92 + 5), v101, &v131);
                  }
                  else
                  {
                    if ( v101 )
                      *(_QWORD *)v101 = v131;
                    v92[6] += 8LL;
                  }
                  return;
                }
              }
            }
            sub_904010(a3, ".");
            sub_3024560(a1, *(_DWORD *)(*((_QWORD *)v131 + 1) + 8LL) >> 8, a3, v20, v21, v22);
            if ( (unsigned __int8)sub_CE8AD0(v131) )
            {
              if ( *(_DWORD *)(a5 + 336) <= 0x27u || *(_DWORD *)(a5 + 340) <= 0x12Bu )
                sub_C64ED0(".attribute(.managed) requires PTX version >= 4.0 and sm_30", 1u);
              sub_904010(a3, " .attribute(.managed)");
            }
            if ( (unsigned __int8)sub_CE95F0(v131) )
            {
              v150.m128i_i64[1] = 0;
              v150.m128i_i64[0] = (__int64)&v151;
              LOBYTE(v151) = 0;
              if ( (unsigned __int8)sub_CE9F90((__int64)v131, v147) )
              {
                v64 = sub_904010(a3, " .attribute(.unified(");
                v65 = sub_CB59D0(v64, v147[0].m128i_u64[0]);
                sub_904010(v65, "))");
              }
              else if ( (unsigned __int8)sub_CE9650((__int64)v131, &v150) )
              {
                v83 = sub_904010(a3, " .attribute(.unified(");
                v84 = sub_CB6200(v83, (unsigned __int8 *)v150.m128i_i64[0], v150.m128i_u64[1]);
                sub_904010(v84, "))");
              }
              else
              {
                sub_904010(a3, " .attribute(.unified)");
              }
              sub_2240A30((unsigned __int64 *)&v150);
            }
            v23 = sub_904010(a3, " .align ");
            v24 = (*((_WORD *)v131 + 17) >> 1) & 0x3F;
            if ( v24 )
            {
              v124 = v23;
              v125 = v24 - 1;
              sub_AE5260(n, v17);
              v25 = v125;
              v26 = v124;
            }
            else
            {
              v126 = v23;
              v63 = sub_AE5260(n, v17);
              v26 = v126;
              v25 = v63;
            }
            sub_CB59D0(v26, 1LL << v25);
            v27 = *(_BYTE *)(v17 + 8);
            if ( v27 <= 3u )
              goto LABEL_77;
            if ( v27 == 5 )
              goto LABEL_33;
            if ( (v27 & 0xFD) == 4 || v27 == 14 )
            {
LABEL_77:
              sub_904010(a3, " .");
              if ( sub_BCAC40(v17, 1) )
              {
                sub_904010(a3, "u8");
              }
              else
              {
                sub_30246F0((__int64)&v150, a1, v17, 0);
                sub_CB6200(a3, (unsigned __int8 *)v150.m128i_i64[0], v150.m128i_u64[1]);
                sub_2240A30((unsigned __int64 *)&v150);
              }
              sub_904010(a3, " ");
              v48 = sub_31DB510(a1, v131);
              sub_EA12C0(v48, a3, *(_BYTE **)(a1 + 208));
              if ( !sub_B2FC80((__int64)v131) )
              {
                v49 = (unsigned __int8 *)*((_QWORD *)v131 - 4);
                v50 = *(_DWORD *)(*((_QWORD *)v131 + 1) + 8LL) >> 8;
                if ( v50 == 4 || v50 == 1 )
                {
                  if ( !sub_AC30F0((__int64)v49) && (unsigned int)*v49 - 12 > 1 )
                  {
                    sub_904010(a3, " = ");
                    sub_3027EB0(a1, (__int64)v49, a3);
                  }
                }
                else if ( !sub_AC30F0((__int64)v49) && (unsigned int)**((unsigned __int8 **)v131 - 4) - 12 > 1 )
                {
                  v149 = 1;
                  v147[0].m128i_i64[0] = (__int64)")";
                  v148 = 3;
                  v51 = *(_DWORD *)(*((_QWORD *)v131 + 1) + 8LL);
                  v138 = 1;
                  v143 = 265;
                  v137 = 3;
                  v140.m128i_i32[0] = v51 >> 8;
                  v136[0].m128i_i64[0] = (__int64)"' is not allowed in addrspace(";
                  v52 = sub_BD5D20((__int64)v131);
                  v135 = 1283;
                  v134 = v53;
                  v133 = v52;
                  v132.m128i_i64[0] = (__int64)"initial value of '";
                  sub_9C6370(v139, &v132, v136, v54, v55, v56);
                  sub_9C6370(v144, v139, &v140, v57, v58, v59);
                  sub_9C6370(&v150, v144, v147, v60, v61, v62);
                  sub_C64D30((__int64)&v150, 1u);
                }
              }
              goto LABEL_37;
            }
            if ( v27 == 12 )
            {
              if ( (unsigned int)sub_BCB060(v17) > 0x40 )
                goto LABEL_33;
              goto LABEL_77;
            }
            if ( v27 <= 0x11u && ((1LL << v27) & 0x39020) != 0 )
            {
LABEL_33:
              v28 = sub_9208B0(n, v17);
              v150.m128i_i64[1] = v29;
              v150.m128i_i64[0] = (unsigned __int64)(v28 + 7) >> 3;
              v30 = sub_CA1930(&v150);
              v31 = *(_DWORD *)(*((_QWORD *)v131 + 1) + 8LL) >> 8;
              if ( (v31 == 4 || v31 == 1)
                && !sub_B2FC80((__int64)v131)
                && (unsigned int)**((unsigned __int8 **)v131 - 4) - 12 > 1 )
              {
                na = *((_QWORD *)v131 - 4);
                v70 = sub_AC30F0(na);
                v71 = na;
                if ( !v70 )
                {
                  v150.m128i_i32[0] = v30;
                  v150.m128i_i64[1] = 0;
                  v151 = 0;
                  v152 = 0;
                  if ( (_DWORD)v30 )
                  {
                    v127 = na;
                    v150.m128i_i64[1] = sub_22077B0((unsigned int)v30);
                    v152 = v150.m128i_i64[1] + (unsigned int)v30;
                    nb = v152;
                    memset((void *)v150.m128i_i64[1], 0, (unsigned int)v30);
                    v72 = nb;
                    v71 = v127;
                  }
                  else
                  {
                    v72 = 0;
                  }
                  v153 = v155;
                  v151 = v72;
                  v154 = 0x400000000LL;
                  v157 = 0x400000000LL;
                  v160 = 0x400000000LL;
                  v73 = *(_BYTE *)(a1 + 1192);
                  v156 = v158;
                  v159 = v161;
                  v162 = 0;
                  v163 = a1;
                  v164 = v73;
                  sub_3026630(a1, v71, (__int64)&v150);
                  if ( !(_DWORD)v157 )
                  {
                    v75 = " .b8 ";
LABEL_107:
                    sub_904010(a3, v75);
                    v76 = sub_31DB510(a1, v131);
                    sub_EA12C0(v76, a3, *(_BYTE **)(a1 + 208));
                    v77 = sub_904010(a3, "[");
                    v78 = sub_CB59D0(v77, v30);
                    sub_904010(v78, "] = {");
                    sub_3027B30((__int64)&v150, a3, v79, v80, v81, v82);
                    sub_904010(a3, "}");
LABEL_108:
                    if ( v159 != v161 )
                      _libc_free((unsigned __int64)v159);
                    if ( v156 != v158 )
                      _libc_free((unsigned __int64)v156);
                    if ( v153 != v155 )
                      _libc_free((unsigned __int64)v153);
                    if ( v150.m128i_i64[1] )
                      j_j___libc_free_0(v150.m128i_u64[1]);
                    goto LABEL_37;
                  }
                  v74 = *(_DWORD *)(*(_QWORD *)(a1 + 208) + 8LL);
                  if ( v30 % v74 )
                  {
LABEL_106:
                    v75 = " .u8 ";
                    if ( *(_DWORD *)(a5 + 336) <= 0x46u )
                    {
                      v146 = 1;
                      v144[0].m128i_i64[0] = (__int64)"' requires at least PTX ISA version 7.1";
                      v145 = 3;
                      v117 = sub_BD5D20((__int64)v131);
                      v143 = 1283;
                      v142 = v118;
                      v140.m128i_i64[0] = (__int64)"initialized packed aggregate with pointers '";
                      v141 = v117;
                      sub_9C6370(v147, &v140, v144, v119, v120, v121);
                      sub_C64D30((__int64)v147, 1u);
                    }
                    goto LABEL_107;
                  }
                  v103 = v153;
                  v104 = 4LL * (unsigned int)v154;
                  v105 = &v153[v104];
                  for ( i = v104 >> 4; i; --i )
                  {
                    if ( *(_DWORD *)v103 % v74 )
                      goto LABEL_158;
                    if ( *((_DWORD *)v103 + 1) % v74 )
                    {
                      v103 += 4;
                      goto LABEL_158;
                    }
                    if ( *((_DWORD *)v103 + 2) % v74 )
                    {
                      v103 += 8;
                      goto LABEL_158;
                    }
                    if ( *((_DWORD *)v103 + 3) % v74 )
                    {
                      v103 += 12;
                      goto LABEL_158;
                    }
                    v103 += 16;
                  }
                  v107 = v105 - v103;
                  if ( v105 - v103 != 8 )
                  {
                    if ( v107 != 12 )
                    {
                      if ( v107 != 4 )
                        goto LABEL_159;
LABEL_157:
                      if ( !(*(_DWORD *)v103 % v74) )
                        goto LABEL_159;
LABEL_158:
                      if ( v105 != v103 )
                        goto LABEL_106;
LABEL_159:
                      v122 = v30 / *(unsigned int *)(*(_QWORD *)(a1 + 208) + 8LL);
                      v123 = *(_DWORD *)(*(_QWORD *)(a1 + 208) + 8LL);
                      v108 = sub_904010(a3, " .u");
                      v109 = sub_CB59D0(v108, (unsigned int)(8 * v123));
                      sub_904010(v109, " ");
                      v110 = sub_31DB510(a1, v131);
                      sub_EA12C0(v110, a3, *(_BYTE **)(a1 + 208));
                      v111 = sub_904010(a3, "[");
                      v112 = sub_CB59D0(v111, v122);
                      sub_904010(v112, "] = {");
                      sub_30279F0((__int64)&v150, a3, v113, v114, v115, v116);
                      sub_904010(a3, "}");
                      goto LABEL_108;
                    }
                    if ( *(_DWORD *)v103 % v74 )
                      goto LABEL_158;
                    v103 += 4;
                  }
                  if ( *(_DWORD *)v103 % v74 )
                    goto LABEL_158;
                  v103 += 4;
                  goto LABEL_157;
                }
              }
              sub_904010(a3, " .b8 ");
              v32 = sub_31DB510(a1, v131);
              sub_EA12C0(v32, a3, *(_BYTE **)(a1 + 208));
              if ( v30 )
              {
                v85 = sub_904010(a3, "[");
                v86 = sub_CB59D0(v85, v30);
                sub_904010(v86, "]");
              }
LABEL_37:
              sub_904010(a3, ";\n");
              return;
            }
LABEL_171:
            BUG();
          }
          v37 = sub_904010(a3, ".global .samplerref ");
          v38 = sub_CEF7C0((__int64)v131);
          sub_A51340(v37, v38, v39);
          if ( sub_B2FC80((__int64)v131) )
            goto LABEL_37;
          v40 = *((_QWORD *)v131 - 4);
          if ( !v40 || *(_BYTE *)v40 != 17 )
            goto LABEL_37;
          v41 = *(_QWORD **)(v40 + 24);
          if ( *(_DWORD *)(v40 + 32) > 0x40u )
            v41 = (_QWORD *)*v41;
          v42 = (unsigned int)v41;
          v43 = 0;
          sub_904010(a3, " = { ");
          v44 = (unsigned __int8)v41 & 7;
          do
          {
            v45 = sub_904010(a3, "addr_mode_");
            v46 = sub_CB59F0(v45, v43);
            sub_904010(v46, " = ");
            switch ( v44 )
            {
              case 0:
              case 3:
                sub_904010(a3, "wrap");
                break;
              case 1:
                sub_904010(a3, "clamp_to_border");
                break;
              case 2:
                sub_904010(a3, "clamp_to_edge");
                break;
              case 4:
                sub_904010(a3, "mirror");
                break;
              default:
                break;
            }
            ++v43;
            sub_904010(a3, ", ");
          }
          while ( v43 != 3 );
          sub_904010(a3, "filter_mode = ");
          v47 = (v42 >> 4) & 3;
          if ( v47 == 1 )
          {
            sub_904010(a3, "linear");
          }
          else
          {
            if ( v47 == 2 )
              goto LABEL_171;
            sub_904010(a3, "nearest");
          }
          if ( (v42 & 8) == 0 )
            sub_904010(a3, ", force_unnormalized_coords = 1");
          sub_904010(a3, " }");
          goto LABEL_37;
        }
        v33 = sub_904010(a3, ".global .surfref ");
        v34 = sub_CEF7B0((__int64)v131);
      }
      v36 = sub_A51340(v33, v34, v35);
      sub_904010(v36, ";\n");
    }
  }
}
