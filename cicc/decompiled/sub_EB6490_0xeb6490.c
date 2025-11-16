// Function: sub_EB6490
// Address: 0xeb6490
//
char __fastcall sub_EB6490(__int64 a1, __int64 *a2, __int64 *a3, __int64 a4)
{
  __int64 v8; // rbx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rdx
  char v12; // r15
  const char *v14; // rax
  char v15; // al
  int v16; // edx
  __int64 v17; // r9
  __int64 v18; // rax
  bool v19; // zf
  __m128i v20; // kr10_16
  __m128i v21; // xmm1
  unsigned __int16 v22; // r15
  __int64 v23; // rbx
  int v24; // eax
  int v25; // eax
  __int64 v26; // r9
  __int64 v27; // rdx
  __int64 v28; // rax
  __int64 v29; // r8
  _BYTE *v30; // rax
  _BYTE *v31; // rdi
  __int64 (*v32)(); // rax
  __int64 v33; // rax
  unsigned __int64 v34; // rbx
  void *v35; // rax
  void *v36; // rbx
  __int64 v37; // r8
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 v40; // rbx
  __int128 v41; // rax
  unsigned __int64 v42; // rax
  unsigned __int64 v43; // rcx
  unsigned __int64 v44; // rsi
  char *v45; // r9
  unsigned __int64 v46; // rdx
  const char *v47; // rsi
  __int64 v48; // rdi
  __int16 v49; // r10
  char v50; // al
  __int64 v51; // rbx
  unsigned __int64 v52; // rax
  __int64 v53; // rax
  _BYTE *v54; // rcx
  __int64 v55; // rdx
  unsigned __int64 v56; // rsi
  __m128i v57; // xmm3
  __m128i v58; // xmm4
  int v59; // eax
  __int64 v60; // rdx
  __m128i v61; // xmm5
  __m128i v62; // xmm6
  __int64 v63; // rdi
  __int64 v64; // rax
  __int64 v65; // rbx
  __int64 v66; // rax
  const char *v67; // r15
  char v68; // al
  unsigned __int64 v69; // r8
  __int64 v70; // r10
  unsigned __int64 v71; // rdx
  __int64 v72; // rax
  __m128i v73; // xmm2
  __int64 v74; // rdi
  __int64 v75; // rax
  _DWORD *v76; // rax
  _DWORD *v77; // rax
  _DWORD *v78; // rax
  __int64 v79; // r9
  _DWORD *v80; // rax
  void **v81; // r12
  __int64 v82; // rax
  __int64 v83; // rax
  unsigned __int64 v84; // r15
  __int64 v85; // rbx
  __int64 v86; // rdi
  __int64 v87; // rax
  __int64 v88; // rax
  __int64 v89; // rdx
  __int64 v90; // rcx
  __int64 v91; // r8
  __int64 v92; // r15
  __int64 v93; // rdi
  __int64 v94; // rax
  void *v95; // rax
  char v96; // al
  __int64 v97; // rsi
  unsigned __int64 v98; // rax
  char v99; // r13
  __int64 v100; // rbx
  __int64 v101; // rax
  __int64 v102; // rdi
  __int64 v103; // rsi
  __int64 v104; // r8
  __int64 v105; // rdx
  __int64 v106; // rcx
  int v107; // eax
  unsigned __int64 v108; // [rsp+8h] [rbp-118h]
  char *v109; // [rsp+10h] [rbp-110h]
  __m128i v110; // [rsp+10h] [rbp-110h]
  __int64 v111; // [rsp+18h] [rbp-108h]
  __int64 v112; // [rsp+20h] [rbp-100h]
  unsigned __int64 v113; // [rsp+20h] [rbp-100h]
  unsigned __int16 v114; // [rsp+20h] [rbp-100h]
  __int64 v115; // [rsp+20h] [rbp-100h]
  __int64 v116; // [rsp+20h] [rbp-100h]
  int v117; // [rsp+20h] [rbp-100h]
  __int64 v118; // [rsp+28h] [rbp-F8h]
  int v119; // [rsp+28h] [rbp-F8h]
  __int64 v120; // [rsp+28h] [rbp-F8h]
  __int64 v121; // [rsp+28h] [rbp-F8h]
  unsigned __int64 v122; // [rsp+28h] [rbp-F8h]
  __int64 v123; // [rsp+28h] [rbp-F8h]
  __int64 v124; // [rsp+28h] [rbp-F8h]
  __int64 v125; // [rsp+28h] [rbp-F8h]
  __int64 v126; // [rsp+28h] [rbp-F8h]
  unsigned __int64 v127; // [rsp+28h] [rbp-F8h]
  __int64 v128; // [rsp+28h] [rbp-F8h]
  __int64 v129; // [rsp+28h] [rbp-F8h]
  __int64 v130; // [rsp+28h] [rbp-F8h]
  __int64 v131; // [rsp+28h] [rbp-F8h]
  __int64 v132; // [rsp+28h] [rbp-F8h]
  __int64 v133; // [rsp+28h] [rbp-F8h]
  __int64 v134; // [rsp+28h] [rbp-F8h]
  __m128i v135; // [rsp+40h] [rbp-E0h] BYREF
  _OWORD v136[2]; // [rsp+50h] [rbp-D0h] BYREF
  __int128 v137; // [rsp+70h] [rbp-B0h]
  const char *v138; // [rsp+80h] [rbp-A0h] BYREF
  unsigned __int64 v139; // [rsp+88h] [rbp-98h]
  const char *v140; // [rsp+90h] [rbp-90h] BYREF
  unsigned __int64 v141; // [rsp+98h] [rbp-88h]
  __int16 v142; // [rsp+A0h] [rbp-80h]
  _BYTE v143[24]; // [rsp+B0h] [rbp-70h] BYREF
  __m128i v144; // [rsp+C8h] [rbp-58h]
  __int64 v145; // [rsp+D8h] [rbp-48h]
  __int64 v146; // [rsp+E0h] [rbp-40h]

  v8 = a1 + 40;
  v10 = sub_ECD690(a1 + 40);
  v11 = **(unsigned int **)(a1 + 48);
  switch ( (int)v11 )
  {
    case 1:
      return 1;
    case 2:
    case 3:
    case 24:
    case 27:
    case 46:
      v112 = v10;
      v119 = **(_DWORD **)(a1 + 48);
      v135 = 0u;
      v15 = sub_EB61F0(a1, v135.m128i_i64);
      v16 = v119;
      v17 = v112;
      if ( v15 )
      {
        v76 = (_DWORD *)sub_ECD7B0(a1);
        v17 = v112;
        if ( *v76 == 27 || (v77 = (_DWORD *)sub_ECD7B0(a1), v16 = v119, v17 = v112, *v77 == 24) )
        {
          v128 = v17;
          v78 = (_DWORD *)sub_ECD7B0(a1);
          v79 = v128;
          if ( *v78 == 27 && *(_BYTE *)(*(_QWORD *)(a1 + 240) + 32LL)
            || (v80 = (_DWORD *)sub_ECD7B0(a1), v79 = v128, *v80 == 24) && *(_BYTE *)(*(_QWORD *)(a1 + 240) + 22LL) )
          {
            v131 = v79;
            sub_EABFE0(a1);
            v92 = sub_E6C430(*(_QWORD *)(a1 + 224), (__int64)&v135, v89, v90, v91);
            (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a1 + 232) + 208LL))(
              *(_QWORD *)(a1 + 232),
              v92,
              0);
            v93 = v92;
            v12 = 0;
            *a2 = sub_E808D0(v93, 0, *(_QWORD **)(a1 + 224), 0);
            *a3 = v131;
          }
          else
          {
            *(_QWORD *)v143 = "invalid token in expression";
            v144.m128i_i16[4] = 259;
            return sub_ECDA70(a1, v128, v143, 0, 0);
          }
          return v12;
        }
      }
      v18 = *(_QWORD *)(a1 + 240);
      v136[1] = 0;
      v19 = *(_BYTE *)(v18 + 352) == 0;
      v137 = 0;
      if ( v19 )
      {
        if ( v16 == 3 )
        {
          if ( **(_DWORD **)(a1 + 48) != 46 )
          {
LABEL_10:
            v20 = v135;
            v21 = _mm_loadu_si128(&v135);
            *a3 = v135.m128i_i64[0] + v135.m128i_i64[1];
            v136[0] = v21;
            if ( v20.m128i_i64[1] )
            {
LABEL_11:
              v22 = 0;
LABEL_12:
              v111 = v17;
              v23 = *(_QWORD *)(a1 + 224);
              v24 = sub_C92610();
              v25 = sub_C92860((__int64 *)(v23 + 1408), (const void *)v20.m128i_i64[0], v20.m128i_u64[1], v24);
              v26 = v111;
              if ( v25 == -1
                || (v27 = *(_QWORD *)(v23 + 1408),
                    v28 = v27 + 8LL * v25,
                    v28 == v27 + 8LL * *(unsigned int *)(v23 + 1416))
                || (v29 = *(_QWORD *)(*(_QWORD *)v28 + 8LL)) == 0 )
              {
                v85 = *(_QWORD *)(a1 + 224);
                if ( *(_BYTE *)(*(_QWORD *)(a1 + 240) + 22LL) )
                {
                  sub_C93170((__int64 *)&v138, (__int64)v136);
                  v144.m128i_i16[4] = 261;
                  *(_QWORD *)v143 = v138;
                  *(_QWORD *)&v143[8] = v139;
                  v94 = sub_E6C460(v85, (const char **)v143);
                  v26 = v111;
                  v29 = v94;
                  if ( v138 != (const char *)&v140 )
                  {
                    v132 = v94;
                    j_j___libc_free_0(v138, v140 + 1);
                    v26 = v111;
                    v29 = v132;
                  }
                }
                else
                {
                  v86 = *(_QWORD *)(a1 + 224);
                  v144.m128i_i16[4] = 261;
                  *(_OWORD *)v143 = v136[0];
                  v87 = sub_E6C460(v86, (const char **)v143);
                  v26 = v111;
                  v29 = v87;
                }
              }
              if ( (*(_BYTE *)(v29 + 9) & 0x70) == 0x20 )
              {
                v30 = *(_BYTE **)(v29 + 24);
                if ( *v30 == 1 )
                {
                  if ( !v22 )
                    goto LABEL_20;
                }
                else if ( *v30 == 4 )
                {
                  v31 = v30 - 8;
                  v32 = *(__int64 (**)())(*((_QWORD *)v30 - 1) + 56LL);
                  if ( v32 != sub_E4C910 )
                  {
                    v116 = v26;
                    v133 = v29;
                    v96 = ((__int64 (__fastcall *)(_BYTE *))v32)(v31);
                    v29 = v133;
                    v26 = v116;
                    if ( v96 )
                    {
                      if ( v22 )
                      {
                        v97 = *a3;
                        *(_QWORD *)v143 = "unexpected modifier on variable reference";
                        v144.m128i_i16[4] = 259;
                        return sub_ECDA70(a1, v97, v143, 0, 0);
                      }
                      v30 = *(_BYTE **)(v133 + 24);
LABEL_20:
                      *a2 = (__int64)v30;
                      return 0;
                    }
                  }
                }
              }
              v30 = (_BYTE *)sub_E808D0(v29, v22, *(_QWORD **)(a1 + 224), v26);
              goto LABEL_20;
            }
            goto LABEL_89;
          }
          v126 = v17;
          sub_EABFE0(a1);
          v66 = sub_ECD690(v8);
          v138 = 0;
          v139 = 0;
          v67 = (const char *)v66;
          v68 = sub_EB61F0(a1, (__int64 *)&v138);
          v17 = v126;
          if ( v68 )
          {
            *(_QWORD *)v143 = "expected symbol variant after '@'";
            v144.m128i_i16[4] = 259;
            return sub_ECDA70(a1, v67, v143, 0, 0);
          }
          v69 = v135.m128i_u64[1];
          v70 = v135.m128i_i64[0];
          v67 = v138;
          v71 = v139;
          v72 = v135.m128i_i64[1];
        }
        else
        {
          v129 = v17;
          v143[0] = 64;
          v83 = sub_C931B0(v135.m128i_i64, v143, 1u, 0);
          v17 = v129;
          v69 = v83;
          if ( v83 == -1 )
          {
            v69 = v135.m128i_u64[1];
            v70 = v135.m128i_i64[0];
            v71 = 0;
            v67 = 0;
            v72 = v135.m128i_i64[1];
          }
          else
          {
            v84 = v83 + 1;
            v72 = v135.m128i_i64[1];
            v70 = v135.m128i_i64[0];
            if ( v84 > v135.m128i_i64[1] )
            {
              v84 = v135.m128i_u64[1];
              v71 = 0;
            }
            else
            {
              v71 = v135.m128i_i64[1] - v84;
            }
            v67 = (const char *)(v135.m128i_i64[0] + v84);
            if ( v69 > v135.m128i_i64[1] )
              v69 = v135.m128i_u64[1];
          }
          *(_QWORD *)&v137 = v67;
          *((_QWORD *)&v137 + 1) = v71;
        }
LABEL_69:
        v73 = _mm_loadu_si128(&v135);
        *a3 = v70 + v72;
        v136[0] = v73;
        if ( !v72 )
        {
LABEL_89:
          *(_QWORD *)v143 = "expected a symbol reference";
          v144.m128i_i16[4] = 259;
          v82 = sub_ECD690(v8);
          return sub_ECDA70(a1, v82, v143, 0, 0);
        }
        if ( !v71 )
          goto LABEL_101;
        v74 = *(_QWORD *)(a1 + 240);
        v115 = v17;
        *((_QWORD *)&v137 + 1) = v71;
        v127 = v71;
        v110.m128i_i64[0] = v70;
        v110.m128i_i64[1] = v69;
        *(_QWORD *)&v137 = v67;
        v75 = sub_106EF90(v74, v67, v71);
        v17 = v115;
        if ( BYTE4(v75) )
        {
          v22 = v75;
          v136[0] = v110;
          v20 = v110;
          goto LABEL_12;
        }
        v88 = *(_QWORD *)(a1 + 240);
        if ( *(_BYTE *)(v88 + 180) )
        {
          if ( !*(_BYTE *)(v88 + 352) )
          {
LABEL_101:
            v20 = (__m128i)v136[0];
            goto LABEL_11;
          }
        }
        v140 = v67;
        v138 = "invalid variant '";
        *(_QWORD *)v143 = &v138;
        v142 = 1283;
        v141 = v127;
        *(_QWORD *)&v143[16] = "'";
        v144.m128i_i16[4] = 770;
        return sub_ECDA70(a1, v67, v143, 0, 0);
      }
      if ( **(_DWORD **)(a1 + 48) != 17 )
        goto LABEL_10;
      v130 = v17;
      sub_EABFE0(a1);
      v138 = 0;
      v139 = 0;
      sub_EB61F0(a1, (__int64 *)&v138);
      v144.m128i_i16[4] = 259;
      *(_QWORD *)v143 = "expected ')'";
      v12 = sub_ECE210(a1, 18, v143);
      if ( !v12 )
      {
        v69 = v135.m128i_u64[1];
        v70 = v135.m128i_i64[0];
        v67 = v138;
        v71 = v139;
        v17 = v130;
        v72 = v135.m128i_i64[1];
        goto LABEL_69;
      }
      return v12;
    case 4:
      v38 = sub_ECD7B0(a1);
      v124 = sub_ECD6A0(v38);
      v39 = sub_ECD7B0(a1);
      if ( *(_DWORD *)(v39 + 32) <= 0x40u )
        v40 = *(_QWORD *)(v39 + 24);
      else
        v40 = **(_QWORD **)(v39 + 24);
      v12 = 0;
      *a2 = sub_E81A90(v40, *(_QWORD **)(a1 + 224), 0, 0);
      *a3 = sub_ECD6B0(*(_QWORD *)(a1 + 48));
      sub_EABFE0(a1);
      if ( **(_DWORD **)(a1 + 48) != 2 )
        return v12;
      v41 = *(_OWORD *)(sub_ECD7B0(a1) + 8);
      v143[0] = 64;
      v136[0] = v41;
      v42 = sub_C931B0((__int64 *)v136, v143, 1u, 0);
      if ( v42 == -1 )
      {
        v43 = *((_QWORD *)&v136[0] + 1);
        v45 = *(char **)&v136[0];
        v49 = 0;
        goto LABEL_44;
      }
      v43 = *((_QWORD *)&v136[0] + 1);
      v44 = v42 + 1;
      v45 = *(char **)&v136[0];
      if ( v42 + 1 > *((_QWORD *)&v136[0] + 1) )
      {
        v44 = *((_QWORD *)&v136[0] + 1);
        v46 = 0;
      }
      else
      {
        v46 = *((_QWORD *)&v136[0] + 1) - v44;
      }
      v47 = (const char *)(*(_QWORD *)&v136[0] + v44);
      if ( v42 <= *((_QWORD *)&v136[0] + 1) )
        v43 = v42;
      if ( v42 >= *((_QWORD *)&v136[0] + 1) )
      {
        v49 = 0;
        goto LABEL_44;
      }
      v48 = *(_QWORD *)(a1 + 240);
      v108 = v43;
      v109 = *(char **)&v136[0];
      *(_QWORD *)&v137 = v47;
      *((_QWORD *)&v137 + 1) = v46;
      v113 = v46;
      v135.m128i_i64[0] = sub_106EF90(v48, v47, v46);
      v45 = v109;
      v43 = v108;
      if ( v135.m128i_i8[4] )
      {
        *(_QWORD *)&v136[0] = v109;
        v49 = v135.m128i_i16[0];
        *((_QWORD *)&v136[0] + 1) = v108;
LABEL_44:
        if ( v43 == 1 )
        {
          v50 = *v45;
          if ( *v45 == 102 || v50 == 98 )
          {
            v114 = v49;
            v51 = sub_E70DF0(*(_QWORD *)(a1 + 224), v40, v50 == 98);
            v52 = sub_E808D0(v51, v114, *(_QWORD **)(a1 + 224), 0);
            v19 = *((_QWORD *)&v136[0] + 1) == 1;
            *a2 = v52;
            if ( v19 && **(_BYTE **)&v136[0] == 98 && !*(_QWORD *)v51 )
            {
              if ( (*(_BYTE *)(v51 + 9) & 0x70) != 0x20
                || *(char *)(v51 + 8) < 0
                || (*(_BYTE *)(v51 + 8) |= 8u, v95 = sub_E807D0(*(_QWORD *)(v51 + 24)), (*(_QWORD *)v51 = v95) == 0) )
              {
                *(_QWORD *)v143 = "directional label undefined";
                v144.m128i_i16[4] = 259;
                return sub_ECDA70(a1, v124, v143, 0, 0);
              }
            }
            v53 = *(_QWORD *)(a1 + 512);
            *(_QWORD *)v143 = v51;
            v54 = v143;
            v55 = *(unsigned int *)(a1 + 536);
            v56 = *(unsigned int *)(a1 + 540);
            v57 = _mm_loadu_si128((const __m128i *)(a1 + 480));
            v145 = v53;
            v58 = _mm_loadu_si128((const __m128i *)(a1 + 496));
            *(__m128i *)&v143[8] = v57;
            v146 = v124;
            v59 = v55;
            v144 = v58;
            if ( v55 + 1 > v56 )
            {
              v98 = *(_QWORD *)(a1 + 528);
              if ( v98 > (unsigned __int64)v143 || (unsigned __int64)v143 >= v98 + 56 * v55 )
              {
                v100 = -1;
                v99 = 0;
              }
              else
              {
                v99 = 1;
                v100 = 0x6DB6DB6DB6DB6DB7LL * ((__int64)&v143[-v98] >> 3);
              }
              v101 = sub_C8D7D0(a1 + 528, a1 + 544, v55 + 1, 0x38u, (unsigned __int64 *)&v138, v55 + 1);
              v102 = *(_QWORD *)(a1 + 528);
              v103 = a1 + 544;
              v104 = v101;
              v105 = v102;
              v106 = v102 + 56LL * *(unsigned int *)(a1 + 536);
              if ( v102 != v106 )
              {
                do
                {
                  if ( v101 )
                  {
                    *(_QWORD *)v101 = *(_QWORD *)v105;
                    *(__m128i *)(v101 + 8) = _mm_loadu_si128((const __m128i *)(v105 + 8));
                    *(__m128i *)(v101 + 24) = _mm_loadu_si128((const __m128i *)(v105 + 24));
                    *(_QWORD *)(v101 + 40) = *(_QWORD *)(v105 + 40);
                    *(_QWORD *)(v101 + 48) = *(_QWORD *)(v105 + 48);
                  }
                  v105 += 56;
                  v101 += 56;
                }
                while ( v106 != v105 );
                v102 = *(_QWORD *)(a1 + 528);
              }
              v107 = (int)v138;
              if ( v103 != v102 )
              {
                v117 = (int)v138;
                v134 = v104;
                _libc_free(v102, v103);
                v107 = v117;
                v104 = v134;
              }
              v55 = *(unsigned int *)(a1 + 536);
              *(_DWORD *)(a1 + 540) = v107;
              v54 = v143;
              *(_QWORD *)(a1 + 528) = v104;
              v59 = v55;
              if ( v99 )
                v54 = (_BYTE *)(v104 + 56 * v100);
            }
            v60 = *(_QWORD *)(a1 + 528) + 56 * v55;
            if ( v60 )
            {
              *(_QWORD *)v60 = *(_QWORD *)v54;
              v61 = _mm_loadu_si128((const __m128i *)(v54 + 8));
              v62 = _mm_loadu_si128((const __m128i *)(v54 + 24));
              *(_QWORD *)(v60 + 40) = *((_QWORD *)v54 + 5);
              *(__m128i *)(v60 + 8) = v61;
              *(__m128i *)(v60 + 24) = v62;
              *(_QWORD *)(v60 + 48) = *((_QWORD *)v54 + 6);
              v59 = *(_DWORD *)(a1 + 536);
            }
            v63 = *(_QWORD *)(a1 + 48);
            *(_DWORD *)(a1 + 536) = v59 + 1;
            *a3 = sub_ECD6B0(v63);
            sub_EABFE0(a1);
          }
        }
        return 0;
      }
      v141 = v113;
      v142 = 1283;
      v138 = "invalid variant '";
      v140 = v47;
      *(_QWORD *)v143 = &v138;
      v144.m128i_i16[4] = 770;
      *(_QWORD *)&v143[16] = "'";
      return sub_ECE0E0(a1, v143, 0, 0);
    case 5:
      v144.m128i_i8[9] = 1;
      v14 = "literal value out of range for directive";
      goto LABEL_6;
    case 6:
      v33 = sub_ECD7B0(a1);
      v34 = *(_QWORD *)(v33 + 16);
      v122 = *(_QWORD *)(v33 + 8);
      v35 = sub_C33320();
      sub_C43310((void **)v143, v35, v122, v34);
      v36 = sub_C33340();
      if ( *(void **)v143 == v36 )
        sub_C3E660((__int64)&v138, (__int64)v143);
      else
        sub_C3A850((__int64)&v138, (__int64 *)v143);
      v37 = (__int64)v138;
      if ( (unsigned int)v139 > 0x40 )
      {
        v123 = *(_QWORD *)v138;
        j_j___libc_free_0_0(v138);
        v37 = v123;
      }
      *a2 = sub_E81A90(v37, *(_QWORD **)(a1 + 224), 0, 0);
      *a3 = sub_ECD6B0(*(_QWORD *)(a1 + 48));
      sub_EABFE0(a1);
      if ( v36 == *(void **)v143 )
      {
        if ( *(_QWORD *)&v143[8] )
        {
          v81 = (void **)(*(_QWORD *)&v143[8] + 24LL * *(_QWORD *)(*(_QWORD *)&v143[8] - 8LL));
          while ( *(void ***)&v143[8] != v81 )
          {
            v81 -= 3;
            if ( v36 == *v81 )
              sub_969EE0((__int64)v81);
            else
              sub_C338F0((__int64)v81);
          }
          j_j_j___libc_free_0_0(v81 - 1);
        }
      }
      else
      {
        sub_C338F0((__int64)v143);
      }
      return 0;
    case 12:
      v120 = v10;
      sub_EABFE0(a1);
      v12 = sub_EB6490(a1, a2, a3, a4);
      if ( v12 )
        return 1;
      *a2 = sub_E81970(3, *a2, *(_QWORD **)(a1 + 224), v120);
      return v12;
    case 13:
      v121 = v10;
      sub_EABFE0(a1);
      v12 = sub_EB6490(a1, a2, a3, a4);
      if ( v12 )
        return 1;
      *a2 = sub_E81970(1, *a2, *(_QWORD **)(a1 + 224), v121);
      return v12;
    case 14:
      v125 = v10;
      sub_EABFE0(a1);
      v12 = sub_EB6490(a1, a2, a3, a4);
      if ( v12 )
        return 1;
      *a2 = sub_E81970(2, *a2, *(_QWORD **)(a1 + 224), v125);
      return v12;
    case 17:
      sub_EABFE0(a1);
      return sub_EAD8C0(a1, a2, a3);
    case 19:
      if ( !*(_BYTE *)(*(_QWORD *)(a1 + 272) + 16LL) )
      {
        v144.m128i_i8[9] = 1;
        v14 = "brackets expression not supported on this target";
        goto LABEL_6;
      }
      sub_EABFE0(a1);
      *(_QWORD *)v143 = 0;
      if ( sub_EAC4D0(a1, a2, (__int64)v143) )
        return 1;
      v64 = sub_ECD7B0(a1);
      *a3 = sub_ECD6B0(v64);
      *(_QWORD *)v143 = "expected ']' in brackets expression";
      v144.m128i_i16[4] = 259;
      return sub_ECE210(a1, 20, v143);
    case 25:
      v12 = *(_BYTE *)(*(_QWORD *)(a1 + 240) + 22LL);
      if ( v12 )
      {
        v144.m128i_i8[9] = 1;
        v14 = "cannot use . as current PC";
LABEL_6:
        *(_QWORD *)v143 = v14;
        v144.m128i_i8[8] = 3;
        return sub_ECE0E0(a1, v143, 0, 0);
      }
      else
      {
        v65 = sub_E6C430(*(_QWORD *)(a1 + 224), (__int64)a2, v11, (unsigned int)v11, v9);
        (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a1 + 232) + 208LL))(
          *(_QWORD *)(a1 + 232),
          v65,
          0);
        *a2 = sub_E808D0(v65, 0, *(_QWORD **)(a1 + 224), 0);
        *a3 = sub_ECD6B0(*(_QWORD *)(a1 + 48));
        sub_EABFE0(a1);
      }
      return v12;
    case 35:
      v118 = v10;
      sub_EABFE0(a1);
      v12 = sub_EB6490(a1, a2, a3, a4);
      if ( v12 )
        return 1;
      *a2 = sub_E81970(0, *a2, *(_QWORD **)(a1 + 224), v118);
      return v12;
    default:
      v144.m128i_i8[9] = 1;
      v14 = "unknown token in expression";
      goto LABEL_6;
  }
}
