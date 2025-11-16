// Function: sub_2C56E90
// Address: 0x2c56e90
//
__int64 __fastcall sub_2C56E90(__int64 a1, __int64 a2)
{
  unsigned int v2; // r12d
  __int64 v5; // rdx
  __int64 v6; // r15
  __int64 v7; // rax
  __int64 v8; // r14
  unsigned int v9; // r12d
  bool v10; // al
  char v11; // al
  __int64 v12; // r13
  __int64 v13; // rsi
  __int64 v14; // rdx
  unsigned __int64 v15; // r14
  __int64 v16; // r15
  unsigned __int64 v17; // r14
  unsigned __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  unsigned __int64 v21; // rdx
  char v22; // cl
  char v23; // cl
  unsigned __int8 v24; // al
  char v25; // cl
  __int64 v26; // rax
  __int64 v27; // rax
  int v28; // edx
  __int64 v29; // rcx
  int v30; // eax
  int v31; // edx
  unsigned __int64 v32; // rax
  __int64 v33; // r10
  signed __int64 v34; // r8
  int v35; // edx
  int v36; // r14d
  int v37; // r11d
  unsigned __int64 v38; // rdx
  int v39; // r9d
  __int64 v40; // rdx
  _DWORD *v41; // rax
  bool v42; // dl
  bool v43; // al
  __int64 v44; // rsi
  __int64 (__fastcall *v45)(__int64, unsigned __int64, __int64); // rax
  unsigned __int64 v46; // r14
  _BYTE *v47; // r13
  __int64 v48; // rdx
  unsigned int v49; // esi
  _QWORD *v50; // rax
  _QWORD *v51; // r10
  __int64 v52; // rsi
  unsigned __int64 v53; // r14
  __int64 v54; // r10
  _BYTE *v55; // r13
  __int64 v56; // rdx
  unsigned int v57; // esi
  _DWORD *v58; // r15
  __int64 v59; // r14
  __int64 v60; // rax
  _BYTE *v61; // r10
  _BYTE *v62; // r11
  __int64 (__fastcall *v63)(__int64, _BYTE *, _BYTE *, __int64, __int64, __int64); // rax
  __int64 v64; // rax
  __int64 v65; // r13
  __int64 i; // r14
  __int64 *v67; // rdx
  __int64 v68; // r14
  unsigned int v69; // r12d
  bool v70; // al
  void *v71; // rdi
  __int64 v72; // rax
  int v73; // edx
  __int64 v74; // r12
  _BYTE *v75; // rax
  unsigned int v76; // r12d
  _QWORD *v77; // rax
  unsigned __int64 v78; // r15
  _BYTE *v79; // r14
  __int64 v80; // rdx
  unsigned int v81; // esi
  unsigned __int64 v82; // r14
  _BYTE *v83; // r13
  __int64 v84; // rdx
  unsigned int v85; // esi
  unsigned int v86; // r13d
  bool v87; // r12
  __int64 v88; // rax
  unsigned int v89; // r12d
  __int64 v90; // rax
  __int64 v91; // r12
  __int64 v92; // rdx
  _BYTE *v93; // rax
  unsigned int v94; // r12d
  unsigned __int64 v95; // rax
  bool v96; // r13
  unsigned int v97; // r12d
  __int64 v98; // rax
  unsigned int v99; // r13d
  __int64 v100; // [rsp+0h] [rbp-200h]
  int v101; // [rsp+8h] [rbp-1F8h]
  unsigned __int64 v102; // [rsp+10h] [rbp-1F0h]
  signed __int64 v103; // [rsp+10h] [rbp-1F0h]
  __int64 v104; // [rsp+18h] [rbp-1E8h]
  unsigned __int64 v105; // [rsp+18h] [rbp-1E8h]
  unsigned int v106; // [rsp+20h] [rbp-1E0h]
  int v107; // [rsp+24h] [rbp-1DCh]
  unsigned int v108; // [rsp+30h] [rbp-1D0h]
  int v109; // [rsp+30h] [rbp-1D0h]
  unsigned int v110; // [rsp+38h] [rbp-1C8h]
  signed __int64 v111; // [rsp+38h] [rbp-1C8h]
  char v112; // [rsp+40h] [rbp-1C0h]
  char v113; // [rsp+40h] [rbp-1C0h]
  _BYTE *v114; // [rsp+40h] [rbp-1C0h]
  __int64 v115; // [rsp+40h] [rbp-1C0h]
  _BYTE *v116; // [rsp+40h] [rbp-1C0h]
  __int64 v117; // [rsp+48h] [rbp-1B8h]
  unsigned __int64 v118; // [rsp+50h] [rbp-1B0h]
  unsigned __int64 v119; // [rsp+50h] [rbp-1B0h]
  char v120; // [rsp+50h] [rbp-1B0h]
  unsigned __int8 v121; // [rsp+50h] [rbp-1B0h]
  int v122; // [rsp+50h] [rbp-1B0h]
  int v123; // [rsp+50h] [rbp-1B0h]
  __int64 *v124; // [rsp+58h] [rbp-1A8h]
  __int64 v125; // [rsp+58h] [rbp-1A8h]
  _QWORD *v126; // [rsp+58h] [rbp-1A8h]
  __int64 v127; // [rsp+58h] [rbp-1A8h]
  _BYTE *v128; // [rsp+58h] [rbp-1A8h]
  __int64 v129; // [rsp+58h] [rbp-1A8h]
  __int64 v130; // [rsp+58h] [rbp-1A8h]
  char v131; // [rsp+60h] [rbp-1A0h]
  int v132; // [rsp+60h] [rbp-1A0h]
  int v133; // [rsp+60h] [rbp-1A0h]
  __int64 v135; // [rsp+78h] [rbp-188h]
  unsigned __int64 v136; // [rsp+80h] [rbp-180h] BYREF
  unsigned int v137; // [rsp+88h] [rbp-178h]
  _BYTE v138[32]; // [rsp+90h] [rbp-170h] BYREF
  __int16 v139; // [rsp+B0h] [rbp-150h]
  _BYTE v140[32]; // [rsp+C0h] [rbp-140h] BYREF
  __int16 v141; // [rsp+E0h] [rbp-120h]
  void *s; // [rsp+F0h] [rbp-110h] BYREF
  __int64 v143; // [rsp+F8h] [rbp-108h]
  _DWORD v144[16]; // [rsp+100h] [rbp-100h] BYREF
  _BYTE *v145; // [rsp+140h] [rbp-C0h] BYREF
  __int64 v146; // [rsp+148h] [rbp-B8h]
  _BYTE v147[32]; // [rsp+150h] [rbp-B0h] BYREF
  __int64 v148; // [rsp+170h] [rbp-90h]
  __int64 v149; // [rsp+178h] [rbp-88h]
  __int64 v150; // [rsp+180h] [rbp-80h]
  __int64 *v151; // [rsp+188h] [rbp-78h]
  void **v152; // [rsp+190h] [rbp-70h]
  void **v153; // [rsp+198h] [rbp-68h]
  __int64 v154; // [rsp+1A0h] [rbp-60h]
  int v155; // [rsp+1A8h] [rbp-58h]
  __int16 v156; // [rsp+1ACh] [rbp-54h]
  char v157; // [rsp+1AEh] [rbp-52h]
  __int64 v158; // [rsp+1B0h] [rbp-50h]
  __int64 v159; // [rsp+1B8h] [rbp-48h]
  void *v160; // [rsp+1C0h] [rbp-40h] BYREF
  void *v161; // [rsp+1C8h] [rbp-38h] BYREF

  if ( *(_BYTE *)a2 != 91 )
    return 0;
  v5 = (*(_BYTE *)(a2 + 7) & 0x40) != 0 ? *(_QWORD *)(a2 - 8) : a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  if ( **(_BYTE **)v5 != 13 )
    return 0;
  v6 = *(_QWORD *)(v5 + 32);
  v7 = *(_QWORD *)(v6 + 16);
  if ( !v7 )
    return 0;
  v102 = *(_QWORD *)(v7 + 8);
  if ( v102 )
    return 0;
  v8 = *(_QWORD *)(v5 + 64);
  if ( *(_BYTE *)v8 == 17 )
  {
    v9 = *(_DWORD *)(v8 + 32);
    if ( v9 <= 0x40 )
      v10 = *(_QWORD *)(v8 + 24) == 0;
    else
      v10 = v9 == (unsigned int)sub_C444A0(v8 + 24);
  }
  else
  {
    v91 = *(_QWORD *)(v8 + 8);
    v92 = (unsigned int)*(unsigned __int8 *)(v91 + 8) - 17;
    if ( (unsigned int)v92 > 1 || *(_BYTE *)v8 > 0x15u )
      return 0;
    v93 = sub_AD7630(v8, 0, v92);
    if ( !v93 || *v93 != 17 )
    {
      if ( *(_BYTE *)(v91 + 8) == 17 )
      {
        v133 = *(_DWORD *)(v91 + 32);
        if ( v133 )
        {
          v96 = 0;
          v97 = 0;
          while ( 1 )
          {
            v98 = sub_AD69F0((unsigned __int8 *)v8, v97);
            if ( !v98 )
              break;
            if ( *(_BYTE *)v98 != 13 )
            {
              if ( *(_BYTE *)v98 != 17 )
                break;
              v99 = *(_DWORD *)(v98 + 32);
              v96 = v99 <= 0x40 ? *(_QWORD *)(v98 + 24) == 0 : v99 == (unsigned int)sub_C444A0(v98 + 24);
              if ( !v96 )
                break;
            }
            if ( v133 == ++v97 )
            {
              if ( v96 )
                goto LABEL_13;
              return 0;
            }
          }
        }
      }
      return 0;
    }
    v94 = *((_DWORD *)v93 + 8);
    if ( v94 <= 0x40 )
      v10 = *((_QWORD *)v93 + 3) == 0;
    else
      v10 = v94 == (unsigned int)sub_C444A0((__int64)(v93 + 24));
  }
  if ( !v10 )
    return 0;
LABEL_13:
  v11 = *(_BYTE *)v6;
  if ( *(_BYTE *)v6 != 90 )
    goto LABEL_14;
  if ( (*(_BYTE *)(v6 + 7) & 0x40) != 0 )
  {
    v67 = *(__int64 **)(v6 - 8);
    v12 = *v67;
    if ( *v67 )
      goto LABEL_93;
  }
  else
  {
    v67 = (__int64 *)(v6 - 32LL * (*(_DWORD *)(v6 + 4) & 0x7FFFFFF));
    v12 = *v67;
    if ( *v67 )
    {
LABEL_93:
      v68 = v67[4];
      if ( *(_BYTE *)v68 == 17 )
      {
        v69 = *(_DWORD *)(v68 + 32);
        if ( v69 <= 0x40 )
          v70 = *(_QWORD *)(v68 + 24) == 0;
        else
          v70 = v69 == (unsigned int)sub_C444A0(v68 + 24);
      }
      else
      {
        v74 = *(_QWORD *)(v68 + 8);
        if ( (unsigned int)*(unsigned __int8 *)(v74 + 8) - 17 > 1 || *(_BYTE *)v68 > 0x15u )
        {
          v131 = 0;
LABEL_16:
          v12 = 0;
          goto LABEL_17;
        }
        v75 = sub_AD7630(v67[4], 0, (__int64)v67);
        if ( !v75 || *v75 != 17 )
        {
          if ( *(_BYTE *)(v74 + 8) == 17 )
          {
            v132 = *(_DWORD *)(v74 + 32);
            if ( v132 )
            {
              v130 = v12;
              v86 = 0;
              v87 = 0;
              while ( 1 )
              {
                v88 = sub_AD69F0((unsigned __int8 *)v68, v86);
                if ( !v88 )
                  break;
                if ( *(_BYTE *)v88 != 13 )
                {
                  if ( *(_BYTE *)v88 != 17 )
                    break;
                  v89 = *(_DWORD *)(v88 + 32);
                  v87 = v89 <= 0x40 ? *(_QWORD *)(v88 + 24) == 0 : v89 == (unsigned int)sub_C444A0(v88 + 24);
                  if ( !v87 )
                    break;
                }
                if ( v132 == ++v86 )
                {
                  v12 = v130;
                  if ( v87 )
                    goto LABEL_97;
                  goto LABEL_99;
                }
              }
            }
          }
          goto LABEL_99;
        }
        v76 = *((_DWORD *)v75 + 8);
        if ( v76 <= 0x40 )
        {
          if ( !*((_QWORD *)v75 + 3) )
            goto LABEL_97;
          goto LABEL_99;
        }
        v70 = v76 == (unsigned int)sub_C444A0((__int64)(v75 + 24));
      }
      if ( v70 )
      {
LABEL_97:
        v131 = 1;
        v11 = *(_BYTE *)v12;
LABEL_15:
        if ( v11 == 61 )
          goto LABEL_17;
        goto LABEL_16;
      }
LABEL_99:
      v11 = *(_BYTE *)v6;
LABEL_14:
      v131 = 0;
      v12 = v6;
      goto LABEL_15;
    }
  }
  v131 = 0;
  v12 = 0;
LABEL_17:
  v13 = *(_QWORD *)(a1 + 152);
  v2 = sub_2C4D8C0(v12, v13);
  if ( !(_BYTE)v2 )
    return v2;
  v124 = *(__int64 **)(v6 + 8);
  v145 = (_BYTE *)sub_BCAE30((__int64)v124);
  v146 = v14;
  v118 = sub_CA1930(&v145);
  v15 = (unsigned int)sub_DFB1F0(*(_QWORD *)(a1 + 152));
  BYTE4(v135) = 0;
  v16 = (__int64)sub_BD3990(*(unsigned __int8 **)(v12 - 32), v13);
  v17 = v15 / v118;
  LODWORD(v135) = v17;
  v110 = v17;
  v125 = sub_BCE1B0(v124, v135);
  _BitScanReverse64(&v18, 1LL << (*(_WORD *)(v12 + 2) >> 1));
  v112 = 63 - (v18 ^ 0x3F);
  if ( (unsigned __int8)sub_D31180(
                          v16,
                          v125,
                          0,
                          *(_QWORD *)(a1 + 184),
                          v12,
                          *(_QWORD *)(a1 + 176),
                          *(__int64 **)(a1 + 160),
                          0) )
  {
    v23 = v112;
    v106 = 0;
    v108 = 0;
    goto LABEL_33;
  }
  LODWORD(v143) = sub_AE43F0(*(_QWORD *)(a1 + 184), *(_QWORD *)(v16 + 8));
  if ( (unsigned int)v143 > 0x40 )
    sub_C43690((__int64)&s, 0, 0);
  else
    s = 0;
  v16 = (__int64)sub_BD45C0((unsigned __int8 *)v16, *(_QWORD *)(a1 + 184), (__int64)&s, 0, 0, 0, 0, 0);
  v19 = 1LL << ((unsigned __int8)v143 - 1);
  if ( (unsigned int)v143 > 0x40 )
  {
    v71 = s;
    if ( (*((_QWORD *)s + ((unsigned int)(v143 - 1) >> 6)) & v19) != 0 )
      goto LABEL_106;
  }
  else if ( ((unsigned __int64)s & v19) != 0 )
  {
    return 0;
  }
  v119 = v118 >> 3;
  if ( sub_C459C0((__int64)&s, v119) )
    goto LABEL_108;
  sub_C45850((__int64)&v145, (unsigned __int64 **)&s, v119);
  if ( (unsigned int)v146 <= 0x40 )
  {
    v108 = (unsigned int)v145;
  }
  else
  {
    v108 = *(_DWORD *)v145;
    j_j___libc_free_0_0((unsigned __int64)v145);
  }
  if ( (unsigned int)v17 <= v108
    || !(unsigned __int8)sub_D31180(
                           v16,
                           v125,
                           0,
                           *(_QWORD *)(a1 + 184),
                           v12,
                           *(_QWORD *)(a1 + 176),
                           *(__int64 **)(a1 + 160),
                           0) )
  {
LABEL_108:
    if ( (unsigned int)v143 > 0x40 )
    {
      v71 = s;
LABEL_106:
      if ( v71 )
        j_j___libc_free_0_0((unsigned __int64)v71);
      return 0;
    }
    return 0;
  }
  v20 = 1LL << v112;
  if ( (unsigned int)v143 <= 0x40 )
  {
    v23 = -1;
    v95 = -(__int64)((unsigned __int64)s | v20) & ((unsigned __int64)s | v20);
    if ( v95 )
    {
      _BitScanReverse64(&v95, v95);
      v23 = 63 - (v95 ^ 0x3F);
    }
  }
  else
  {
    v21 = (*(_QWORD *)s | v20) & -(*(_QWORD *)s | v20);
    if ( v21 )
    {
      _BitScanReverse64(&v21, v21);
      v22 = 63 - (v21 ^ 0x3F);
    }
    else
    {
      v22 = -1;
    }
    v120 = v22;
    j_j___libc_free_0_0((unsigned __int64)s);
    v23 = v120;
  }
  v106 = v108;
LABEL_33:
  v121 = v23;
  v24 = sub_BD5420((unsigned __int8 *)v16, *(_QWORD *)(a1 + 184));
  v25 = v121;
  if ( v24 >= v121 )
    v25 = v24;
  v26 = *(_QWORD *)(*(_QWORD *)(v12 - 32) + 8LL);
  v113 = v25;
  if ( (unsigned int)*(unsigned __int8 *)(v26 + 8) - 17 <= 1 )
    v26 = **(_QWORD **)(v26 + 16);
  v107 = *(_DWORD *)(v26 + 8) >> 8;
  v27 = sub_DFD4A0(*(__int64 **)(a1 + 152));
  v137 = v17;
  v104 = v27;
  v122 = v28;
  if ( v17 > 0x40 )
  {
    sub_C43690((__int64)&v136, 0, 0);
    v110 = v137;
    v102 = v136;
  }
  else
  {
    v136 = 0;
  }
  if ( v110 > 0x40 )
    *(_QWORD *)v102 |= 1uLL;
  else
    v136 = v102 | 1;
  v29 = sub_DFAAD0(*(__int64 **)(a1 + 152), v125, (__int64)&v136, 1u, v131 & 1);
  v30 = 1;
  if ( v31 != 1 )
    v30 = v122;
  v123 = v30;
  v32 = v29 + v104;
  if ( __OFADD__(v29, v104) )
  {
    v32 = 0x8000000000000000LL;
    if ( v29 > 0 )
      v32 = 0x7FFFFFFFFFFFFFFFLL;
  }
  v111 = v32;
  v33 = sub_DFD4A0(*(__int64 **)(a1 + 152));
  v34 = v33;
  v36 = v35;
  v37 = v35;
  v38 = *(unsigned int *)(*(_QWORD *)(a2 + 8) + 32LL);
  s = v144;
  v39 = v38;
  v143 = 0x1000000000LL;
  if ( (unsigned int)v38 > 0x10 )
  {
    v100 = v33;
    v103 = v33;
    v101 = v38;
    v105 = v38;
    sub_C8D5F0((__int64)&s, v144, v38, 4u, v33, v38);
    memset(s, 255, 4 * v105);
    v41 = s;
    v34 = v103;
    v37 = v36;
    LODWORD(v143) = v101;
    v33 = v100;
  }
  else
  {
    if ( v38 )
    {
      v40 = 4 * v38;
      if ( v40 )
      {
        if ( (unsigned int)v40 >= 8 )
        {
          *(_QWORD *)((char *)&v144[-2] + (unsigned int)v40) = -1;
          memset(v144, 0xFFu, 8LL * ((unsigned int)(v40 - 1) >> 3));
        }
        else if ( (v40 & 4) != 0 )
        {
          v144[0] = -1;
          *(_DWORD *)((char *)&v144[-1] + (unsigned int)v40) = -1;
        }
        else if ( (_DWORD)v40 )
        {
          LOBYTE(v144[0]) = -1;
        }
      }
    }
    LODWORD(v143) = v39;
    v41 = v144;
  }
  *v41 = v106;
  if ( v108 )
  {
    v117 = v33;
    v109 = v37;
    v72 = sub_DFBC30(
            *(__int64 **)(a1 + 152),
            7,
            v125,
            (__int64)s,
            (unsigned int)v143,
            *(unsigned int *)(a1 + 192),
            0,
            0,
            0,
            0,
            0);
    if ( v73 == 1 )
    {
      v42 = v2;
      v37 = 1;
    }
    else
    {
      v37 = v109;
      v42 = v36 != 0;
    }
    v34 = v72 + v117;
    if ( __OFADD__(v72, v117) )
    {
      v34 = 0x7FFFFFFFFFFFFFFFLL;
      if ( v72 <= 0 )
        v34 = 0x8000000000000000LL;
    }
  }
  else
  {
    v42 = v36 != 0;
  }
  if ( v37 == v123 )
    v43 = v111 < v34;
  else
    v43 = v123 < v37;
  if ( !v43 && !v42 )
  {
    v151 = (__int64 *)sub_BD5C60(v12);
    v152 = &v160;
    v153 = &v161;
    LOWORD(v150) = 0;
    v145 = v147;
    v160 = &unk_49DA100;
    v146 = 0x200000000LL;
    v154 = 0;
    v161 = &unk_49DA0B0;
    v155 = 0;
    v156 = 512;
    v157 = 7;
    v158 = 0;
    v159 = 0;
    v148 = 0;
    v149 = 0;
    sub_D5F1F0((__int64)&v145, v12);
    v139 = 257;
    v44 = sub_BCE3C0(v151, v107);
    if ( v44 != *(_QWORD *)(v16 + 8) )
    {
      if ( *(_BYTE *)v16 > 0x15u )
      {
        v141 = 257;
        v16 = sub_B52190(v16, v44, (__int64)v140, 0, 0);
        (*((void (__fastcall **)(void **, __int64, _BYTE *, __int64, __int64))*v153 + 2))(v153, v16, v138, v149, v150);
        v82 = (unsigned __int64)v145;
        v83 = &v145[16 * (unsigned int)v146];
        if ( v145 != v83 )
        {
          do
          {
            v84 = *(_QWORD *)(v82 + 8);
            v85 = *(_DWORD *)v82;
            v82 += 16LL;
            sub_B99FD0(v16, v85, v84);
          }
          while ( v83 != (_BYTE *)v82 );
        }
      }
      else
      {
        v45 = (__int64 (__fastcall *)(__int64, unsigned __int64, __int64))*((_QWORD *)*v152 + 18);
        if ( v45 == sub_B32D70 )
          v16 = sub_ADB060(v16, v44);
        else
          v16 = v45((__int64)v152, v16, v44);
        if ( *(_BYTE *)v16 > 0x1Cu )
        {
          (*((void (__fastcall **)(void **, __int64, _BYTE *, __int64, __int64))*v153 + 2))(v153, v16, v138, v149, v150);
          v46 = (unsigned __int64)v145;
          v47 = &v145[16 * (unsigned int)v146];
          if ( v145 != v47 )
          {
            do
            {
              v48 = *(_QWORD *)(v46 + 8);
              v49 = *(_DWORD *)v46;
              v46 += 16LL;
              sub_B99FD0(v16, v49, v48);
            }
            while ( v47 != (_BYTE *)v46 );
          }
        }
      }
    }
    v141 = 257;
    v139 = 257;
    v50 = sub_BD2C40(80, 1u);
    v51 = v50;
    if ( v50 )
    {
      v52 = v125;
      v126 = v50;
      sub_B4D190((__int64)v50, v52, v16, (__int64)v140, 0, v113, 0, 0);
      v51 = v126;
    }
    v127 = (__int64)v51;
    (*((void (__fastcall **)(void **, _QWORD *, _BYTE *, __int64, __int64))*v153 + 2))(v153, v51, v138, v149, v150);
    v53 = (unsigned __int64)v145;
    v54 = v127;
    v55 = &v145[16 * (unsigned int)v146];
    if ( v145 != v55 )
    {
      do
      {
        v56 = *(_QWORD *)(v53 + 8);
        v57 = *(_DWORD *)v53;
        v53 += 16LL;
        sub_B99FD0(v127, v57, v56);
      }
      while ( v55 != (_BYTE *)v53 );
      v54 = v127;
    }
    v128 = (_BYTE *)v54;
    v58 = s;
    v139 = 257;
    v59 = (unsigned int)v143;
    v60 = sub_ACADE0(*(__int64 ***)(v54 + 8));
    v61 = v128;
    v62 = (_BYTE *)v60;
    v63 = (__int64 (__fastcall *)(__int64, _BYTE *, _BYTE *, __int64, __int64, __int64))*((_QWORD *)*v152 + 14);
    if ( v63 == sub_9B6630 )
    {
      if ( *v128 > 0x15u || *v62 > 0x15u )
        goto LABEL_129;
      v114 = v62;
      v64 = sub_AD5CE0((__int64)v128, (__int64)v62, v58, v59, 0);
      v61 = v128;
      v62 = v114;
      v65 = v64;
    }
    else
    {
      v116 = v62;
      v90 = ((__int64 (__fastcall *)(void **, _BYTE *, _BYTE *, _DWORD *, __int64))v63)(v152, v128, v62, v58, v59);
      v62 = v116;
      v61 = v128;
      v65 = v90;
    }
    if ( v65 )
    {
LABEL_75:
      sub_BD84D0(a2, v65);
      if ( *(_BYTE *)v65 > 0x1Cu )
      {
        sub_BD6B90((unsigned __int8 *)v65, (unsigned __int8 *)a2);
        for ( i = *(_QWORD *)(v65 + 16); i; i = *(_QWORD *)(i + 8) )
          sub_F15FC0(a1 + 200, *(_QWORD *)(i + 24));
        if ( *(_BYTE *)v65 > 0x1Cu )
          sub_F15FC0(a1 + 200, v65);
      }
      if ( *(_BYTE *)a2 > 0x1Cu )
        sub_F15FC0(a1 + 200, a2);
      nullsub_61();
      v160 = &unk_49DA100;
      nullsub_63();
      if ( v145 != v147 )
        _libc_free((unsigned __int64)v145);
      goto LABEL_84;
    }
LABEL_129:
    v115 = (__int64)v61;
    v129 = (__int64)v62;
    v141 = 257;
    v77 = sub_BD2C40(112, unk_3F1FE60);
    v65 = (__int64)v77;
    if ( v77 )
      sub_B4E9E0((__int64)v77, v115, v129, v58, v59, (__int64)v140, 0, 0);
    (*((void (__fastcall **)(void **, __int64, _BYTE *, __int64, __int64))*v153 + 2))(v153, v65, v138, v149, v150);
    v78 = (unsigned __int64)v145;
    v79 = &v145[16 * (unsigned int)v146];
    if ( v145 != v79 )
    {
      do
      {
        v80 = *(_QWORD *)(v78 + 8);
        v81 = *(_DWORD *)v78;
        v78 += 16LL;
        sub_B99FD0(v65, v81, v80);
      }
      while ( v79 != (_BYTE *)v78 );
    }
    goto LABEL_75;
  }
  v2 = 0;
LABEL_84:
  if ( s != v144 )
    _libc_free((unsigned __int64)s);
  if ( v137 > 0x40 && v136 )
    j_j___libc_free_0_0(v136);
  return v2;
}
