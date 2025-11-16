// Function: sub_11993B0
// Address: 0x11993b0
//
__int64 __fastcall sub_11993B0(const __m128i *a1, unsigned __int8 *a2)
{
  unsigned __int8 *v4; // r14
  __int64 v5; // r12
  __int64 v6; // rax
  unsigned __int8 *v7; // r15
  unsigned __int8 v8; // al
  __int64 v9; // rax
  __int64 v11; // r15
  __int64 v12; // rbx
  int *v13; // rdx
  __int64 v14; // r14
  int v15; // edi
  __int64 v16; // r8
  __int64 *v17; // r9
  unsigned __int64 v18; // rdx
  __int64 v19; // r10
  _BYTE *v20; // r11
  __int64 *v21; // rdi
  __int64 v22; // rax
  bool v23; // al
  _QWORD *v24; // rax
  __int64 v25; // r12
  __int64 v26; // rbx
  __int64 v27; // r12
  __int64 v28; // rdx
  unsigned int v29; // esi
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rax
  unsigned __int8 *v33; // r11
  unsigned __int8 v34; // al
  __int64 v35; // rax
  __int64 v36; // rax
  __int16 v37; // ax
  unsigned __int8 *v38; // rax
  _BYTE *v39; // r12
  __int64 *v40; // rax
  __int64 v41; // rdx
  __int64 v42; // rax
  unsigned __int8 *v43; // r10
  __int64 *v44; // r13
  __int64 v45; // rdx
  __int64 v46; // rdx
  __int64 v47; // r12
  __int64 v48; // r12
  __int64 v49; // rax
  __int64 v50; // rdi
  __int64 v51; // r8
  unsigned int v52; // eax
  __int64 v53; // r8
  unsigned __int64 v54; // rdx
  bool v55; // al
  __int64 v56; // r8
  unsigned int v57; // eax
  unsigned __int64 v58; // rdx
  unsigned __int64 v59; // rsi
  unsigned __int64 v60; // rdx
  bool v61; // zf
  unsigned __int64 v62; // rax
  unsigned int v63; // eax
  unsigned int v64; // edx
  unsigned int v65; // eax
  unsigned __int64 v66; // rax
  unsigned __int64 v67; // rcx
  bool v68; // al
  unsigned int v69; // eax
  unsigned __int64 v70; // rdx
  unsigned __int64 v71; // rsi
  unsigned __int64 v72; // rax
  __int64 v73; // r12
  int v74; // edi
  __int64 v75; // rax
  _BYTE *v76; // r8
  __int64 v77; // rsi
  __int64 v78; // rax
  unsigned __int8 *v79; // r10
  __int64 *v80; // r13
  __int64 v81; // rdx
  __int64 v82; // rdx
  __int64 v83; // r12
  __int64 v84; // r12
  __int64 v85; // rax
  _BYTE *v86; // rax
  char v87; // al
  char v88; // al
  unsigned int v89; // edx
  unsigned int v90; // eax
  unsigned __int64 v91; // rsi
  unsigned __int64 v92; // rcx
  unsigned int v93; // esi
  unsigned int v94; // eax
  unsigned __int64 v95; // rdi
  unsigned __int64 v96; // rdx
  __int64 v97; // rdx
  __int64 v98; // rsi
  unsigned __int64 v99; // rdx
  __int64 v100; // rax
  int v101; // eax
  unsigned int v102; // r15d
  __int64 v103; // rax
  __int64 v104; // rbx
  __int64 *v105; // rdx
  __int64 v106; // rax
  unsigned int v107; // r15d
  _QWORD *v108; // r14
  int v109; // edi
  int v110; // eax
  char v111; // al
  unsigned __int8 *v112; // r11
  __int64 v113; // rax
  __int64 v114; // rax
  __int64 v115; // r10
  __int64 v116; // rdx
  _BYTE *v117; // rax
  __int64 v118; // rdx
  _BYTE *v119; // rax
  __int64 v120; // rdx
  unsigned __int8 *v121; // rax
  __int64 v122; // rax
  int v123; // ecx
  int v124; // edx
  _QWORD *v125; // rdi
  __int64 *v126; // rax
  __int64 v127; // rsi
  __int64 v128; // r15
  __int64 v129; // rbx
  __int64 v130; // rdx
  unsigned int v131; // esi
  __int64 v132; // [rsp+8h] [rbp-108h]
  unsigned __int64 v133; // [rsp+10h] [rbp-100h]
  unsigned __int8 *v134; // [rsp+10h] [rbp-100h]
  __int64 v135; // [rsp+18h] [rbp-F8h]
  bool v136; // [rsp+18h] [rbp-F8h]
  unsigned int v137; // [rsp+18h] [rbp-F8h]
  __int64 v138; // [rsp+18h] [rbp-F8h]
  bool v139; // [rsp+20h] [rbp-F0h]
  unsigned __int8 *v140; // [rsp+20h] [rbp-F0h]
  __int64 v141; // [rsp+20h] [rbp-F0h]
  unsigned __int8 v142; // [rsp+28h] [rbp-E8h]
  __int64 v143; // [rsp+28h] [rbp-E8h]
  unsigned int v144; // [rsp+28h] [rbp-E8h]
  __int64 v145; // [rsp+28h] [rbp-E8h]
  unsigned int v146; // [rsp+28h] [rbp-E8h]
  __int64 *v147; // [rsp+28h] [rbp-E8h]
  unsigned int v148; // [rsp+30h] [rbp-E0h]
  __int64 v149; // [rsp+30h] [rbp-E0h]
  __int64 v150; // [rsp+38h] [rbp-D8h]
  bool v151; // [rsp+38h] [rbp-D8h]
  _BYTE *v152; // [rsp+38h] [rbp-D8h]
  __int64 v153; // [rsp+38h] [rbp-D8h]
  int v154; // [rsp+38h] [rbp-D8h]
  __int64 v155; // [rsp+40h] [rbp-D0h]
  unsigned __int8 v156; // [rsp+40h] [rbp-D0h]
  __int64 v157; // [rsp+40h] [rbp-D0h]
  int v158; // [rsp+40h] [rbp-D0h]
  _BYTE *v159; // [rsp+40h] [rbp-D0h]
  __int64 v160; // [rsp+48h] [rbp-C8h]
  unsigned int **v161; // [rsp+48h] [rbp-C8h]
  int v162; // [rsp+54h] [rbp-BCh] BYREF
  __int64 v163; // [rsp+58h] [rbp-B8h] BYREF
  unsigned __int8 *v164; // [rsp+60h] [rbp-B0h] BYREF
  __int64 v165; // [rsp+68h] [rbp-A8h] BYREF
  __int64 v166; // [rsp+70h] [rbp-A0h] BYREF
  __int64 v167; // [rsp+78h] [rbp-98h]
  unsigned __int64 v168; // [rsp+80h] [rbp-90h] BYREF
  int *v169; // [rsp+88h] [rbp-88h]
  __int64 *v170; // [rsp+90h] [rbp-80h]
  __int64 *v171; // [rsp+98h] [rbp-78h]
  unsigned __int8 **v172; // [rsp+A0h] [rbp-70h]
  unsigned __int64 v173; // [rsp+B0h] [rbp-60h] BYREF
  __int64 v174; // [rsp+B8h] [rbp-58h]
  __int16 v175; // [rsp+D0h] [rbp-40h]

  v4 = (unsigned __int8 *)*((_QWORD *)a2 - 4);
  v5 = *((_QWORD *)a2 + 1);
  v160 = *((_QWORD *)a2 - 8);
  v6 = *((_QWORD *)v4 + 2);
  if ( v6 )
  {
    if ( !*(_QWORD *)(v6 + 8) && *v4 == 69 )
    {
      v11 = *((_QWORD *)v4 - 4);
      if ( v11 )
      {
        v12 = a1[2].m128i_i64[0];
        LOWORD(v172) = 261;
        v168 = (unsigned __int64)sub_BD5D20((__int64)v4);
        v169 = v13;
        if ( v5 == *(_QWORD *)(v11 + 8) )
        {
          v14 = v11;
        }
        else
        {
          v14 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(v12 + 80) + 120LL))(
                  *(_QWORD *)(v12 + 80),
                  39,
                  v11,
                  v5);
          if ( !v14 )
          {
            v175 = 257;
            v24 = sub_BD2C40(72, unk_3F10A14);
            v14 = (__int64)v24;
            if ( v24 )
              sub_B515B0((__int64)v24, v11, v5, (__int64)&v173, 0, 0);
            (*(void (__fastcall **)(_QWORD, __int64, unsigned __int64 *, _QWORD, _QWORD))(**(_QWORD **)(v12 + 88) + 16LL))(
              *(_QWORD *)(v12 + 88),
              v14,
              &v168,
              *(_QWORD *)(v12 + 56),
              *(_QWORD *)(v12 + 64));
            v25 = 16LL * *(unsigned int *)(v12 + 8);
            v26 = *(_QWORD *)v12;
            v27 = v26 + v25;
            while ( v27 != v26 )
            {
              v28 = *(_QWORD *)(v26 + 8);
              v29 = *(_DWORD *)v26;
              v26 += 16;
              sub_B99FD0(v14, v29, v28);
            }
          }
        }
        v15 = *a2 - 29;
        v175 = 257;
        return sub_B504D0(v15, v160, v14, (__int64)&v173, 0, 0);
      }
    }
  }
  v7 = a2;
  if ( (unsigned __int8)sub_11AE870(a1, a2) )
    return (__int64)v7;
  v8 = *v4;
  if ( *(_BYTE *)v160 <= 0x15u && v8 == 86 )
  {
    v9 = (__int64)sub_F26350((__int64)a1, a2, (__int64)v4, 0);
    if ( v9 )
      return v9;
    v8 = *v4;
  }
  if ( v8 <= 0x15u && v8 != 5 && !(unsigned __int8)sub_AD6CA0((__int64)v4) )
  {
    v9 = (__int64)sub_1198360((__int64)a1, (unsigned __int8 *)v160, v4, (__int64)a2, v16, v17);
    if ( v9 )
      return v9;
  }
  v9 = sub_1195590((__int64)a1, a2, a1 + 6, 0);
  if ( v9 )
    return v9;
  if ( *(_BYTE *)v160 > 0x15u )
  {
    v148 = sub_BCB060(v5);
    goto LABEL_41;
  }
  v18 = *v4;
  if ( (unsigned __int8)v18 <= 0x1Cu )
  {
    if ( (_BYTE)v18 != 5 )
      goto LABEL_38;
    v37 = *((_WORD *)v4 + 1);
    if ( (v37 & 0xFFF7) != 0x11 && (v37 & 0xFFFD) != 0xD )
      goto LABEL_38;
    if ( v37 != 13 )
      goto LABEL_38;
    goto LABEL_54;
  }
  if ( (unsigned __int8)v18 <= 0x36u )
  {
    v30 = 0x40540000000000LL;
    if ( !_bittest64(&v30, v18) || (_BYTE)v18 != 42 )
      goto LABEL_38;
LABEL_54:
    if ( (v4[1] & 2) != 0 )
    {
      v19 = *((_QWORD *)v4 - 8);
      if ( v19 )
      {
        v20 = (_BYTE *)*((_QWORD *)v4 - 4);
        if ( *v20 <= 0x15u )
          goto LABEL_28;
      }
    }
    goto LABEL_38;
  }
  if ( (_BYTE)v18 == 58 && (v4[1] & 2) != 0 )
  {
    v19 = *((_QWORD *)v4 - 8);
    if ( v19 )
    {
      v20 = (_BYTE *)*((_QWORD *)v4 - 4);
      if ( *v20 <= 0x15u )
      {
LABEL_28:
        v21 = (__int64 *)a1[2].m128i_i64[0];
        v175 = 257;
        v155 = v19;
        v22 = sub_1194380(v21, (unsigned int)*a2 - 29, v160, (__int64)v20, v168, 0, (__int64)&v173, 0);
        v175 = 257;
        v7 = (unsigned __int8 *)sub_B504D0((unsigned int)*a2 - 29, v22, v155, (__int64)&v173, 0, 0);
        if ( *a2 != 54 )
        {
          v23 = sub_B44E60((__int64)a2);
          sub_B448B0((__int64)v7, v23);
          return (__int64)v7;
        }
        v87 = sub_B44900((__int64)a2);
        sub_B44850(v7, v87);
LABEL_164:
        v88 = sub_B448F0((__int64)a2);
        sub_B447F0(v7, v88);
        return (__int64)v7;
      }
    }
  }
LABEL_38:
  v156 = *(_BYTE *)v160;
  v142 = *v4;
  v148 = sub_BCB060(v5);
  if ( v156 == 17 )
  {
    v31 = v142;
    v150 = v160 + 24;
  }
  else
  {
    if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v160 + 8) + 8LL) - 17 > 1 )
      goto LABEL_41;
    v86 = sub_AD7630(v160, 0, v31);
    if ( !v86 || *v86 != 17 )
      goto LABEL_41;
    v31 = *v4;
    v150 = (__int64)(v86 + 24);
  }
  if ( (_BYTE)v31 == 42 )
  {
    v132 = *((_QWORD *)v4 - 8);
    if ( v132 )
    {
      v50 = *((_QWORD *)v4 - 4);
      v51 = v50 + 24;
      if ( *(_BYTE *)v50 != 17 )
      {
        v31 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v50 + 8) + 8LL) - 17;
        if ( (unsigned int)v31 > 1 )
          goto LABEL_41;
        if ( *(_BYTE *)v50 > 0x15u )
          goto LABEL_41;
        v117 = sub_AD7630(v50, 0, v31);
        if ( !v117 || *v117 != 17 )
          goto LABEL_41;
        v51 = (__int64)(v117 + 24);
      }
      v52 = *(_DWORD *)(v51 + 8);
      v31 = *(_QWORD *)v51;
      if ( v52 > 0x40 )
        v31 = *(_QWORD *)(v31 + 8LL * ((v52 - 1) >> 6));
      if ( (v31 & (1LL << ((unsigned __int8)v52 - 1))) != 0 )
      {
        v157 = v51;
        sub_9865C0((__int64)&v168, v51);
        v53 = v157;
        if ( (unsigned int)v169 > 0x40 )
        {
          sub_C43D10((__int64)&v168);
          v53 = v157;
        }
        else
        {
          v54 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v169;
          if ( !(_DWORD)v169 )
            v54 = 0;
          v168 = v54 & ~v168;
        }
        v135 = v53;
        sub_C46250((__int64)&v168);
        LODWORD(v174) = (_DWORD)v169;
        v144 = (unsigned int)v169;
        v173 = v168;
        v133 = v168;
        LODWORD(v169) = 0;
        v55 = sub_986EE0((__int64)&v173, v148);
        v31 = v144;
        v56 = v135;
        if ( v144 > 0x40 )
        {
          if ( v133 )
          {
            v136 = v55;
            v145 = v56;
            j_j___libc_free_0_0(v133);
            v56 = v145;
            v55 = v136;
            if ( (unsigned int)v169 > 0x40 )
            {
              if ( v168 )
              {
                j_j___libc_free_0_0(v168);
                v55 = v136;
                v56 = v145;
              }
            }
          }
        }
        if ( v55 )
        {
          v57 = *(_DWORD *)(v56 + 8);
          LODWORD(v174) = v57;
          if ( v57 > 0x40 )
          {
            sub_C43780((__int64)&v173, (const void **)v56);
            v57 = v174;
            if ( (unsigned int)v174 > 0x40 )
            {
              sub_C43D10((__int64)&v173);
              goto LABEL_97;
            }
            v58 = v173;
          }
          else
          {
            v58 = *(_QWORD *)v56;
          }
          v59 = ~v58;
          v60 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v57;
          v61 = v57 == 0;
          v62 = 0;
          if ( !v61 )
            v62 = v60;
          v173 = v59 & v62;
LABEL_97:
          sub_C46250((__int64)&v173);
          v63 = v174;
          LODWORD(v174) = 0;
          v146 = v173;
          if ( v63 > 0x40 )
          {
            v146 = *(_DWORD *)v173;
            j_j___libc_free_0_0(v173);
            if ( (unsigned int)v174 > 0x40 )
            {
              if ( v173 )
                j_j___libc_free_0_0(v173);
            }
          }
          v31 = *a2;
          if ( (_DWORD)v31 != 55 )
          {
            if ( (_DWORD)v31 != 56 )
            {
              if ( (_DWORD)v31 != 54 || !sub_B44900((__int64)a2) && !sub_B448F0((__int64)a2) )
                goto LABEL_41;
              v64 = *(_DWORD *)(v150 + 8);
              LODWORD(v169) = v64;
              if ( v64 > 0x40 )
              {
                sub_C43780((__int64)&v168, (const void **)v150);
                v65 = (unsigned int)v169;
                v64 = (unsigned int)v169;
                if ( (unsigned int)v169 > 0x40 )
                {
                  sub_C482E0((__int64)&v168, v146);
                  v65 = (unsigned int)v169;
                  goto LABEL_110;
                }
              }
              else
              {
                v168 = *(_QWORD *)v150;
                v65 = v64;
              }
              if ( v146 == v64 )
                v168 = 0;
              else
                v168 >>= v146;
LABEL_110:
              LODWORD(v174) = v65;
              v31 = v65;
              if ( v65 > 0x40 )
              {
                sub_C43780((__int64)&v173, (const void **)&v168);
                v31 = (unsigned int)v174;
                if ( (unsigned int)v174 > 0x40 )
                {
                  sub_C47690((__int64 *)&v173, v146);
                  v31 = (unsigned int)v174;
                  goto LABEL_117;
                }
              }
              else
              {
                v173 = v168;
              }
              v66 = 0;
              if ( v146 != (_DWORD)v31 )
                v66 = v173 << v146;
              v67 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v31;
              if ( !(_DWORD)v31 )
                v67 = 0;
              v173 = v67 & v66;
LABEL_117:
              if ( *(_DWORD *)(v150 + 8) <= 0x40u )
              {
                v139 = *(_QWORD *)v150 == v173;
              }
              else
              {
                v137 = v31;
                v68 = sub_C43C50(v150, (const void **)&v173);
                v31 = v137;
                v139 = v68;
              }
              if ( (unsigned int)v31 <= 0x40 )
                goto LABEL_122;
LABEL_120:
              if ( v173 )
                j_j___libc_free_0_0(v173);
LABEL_122:
              if ( (unsigned int)v169 > 0x40 && v168 )
                j_j___libc_free_0_0(v168);
              if ( v139 )
              {
                v61 = *a2 == 54;
                v69 = *(_DWORD *)(v150 + 8);
                LODWORD(v174) = v69;
                if ( v61 )
                {
                  if ( v69 > 0x40 )
                  {
                    sub_C43780((__int64)&v173, (const void **)v150);
                    v69 = v174;
                    if ( (unsigned int)v174 > 0x40 )
                    {
                      sub_C482E0((__int64)&v173, v146);
                      goto LABEL_134;
                    }
                  }
                  else
                  {
                    v173 = *(_QWORD *)v150;
                  }
                  if ( v69 == v146 )
                    v173 = 0;
                  else
                    v173 >>= v146;
                }
                else
                {
                  if ( v69 > 0x40 )
                  {
                    sub_C43780((__int64)&v173, (const void **)v150);
                    v69 = v174;
                    if ( (unsigned int)v174 > 0x40 )
                    {
                      sub_C47690((__int64 *)&v173, v146);
                      goto LABEL_134;
                    }
                  }
                  else
                  {
                    v173 = *(_QWORD *)v150;
                  }
                  v70 = 0;
                  if ( v69 != v146 )
                    v70 = v173 << v146;
                  v71 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v69;
                  v61 = v69 == 0;
                  v72 = 0;
                  if ( !v61 )
                    v72 = v71;
                  v173 = v72 & v70;
                }
LABEL_134:
                v73 = sub_AD8D80(v5, (__int64)&v173);
                if ( (unsigned int)v174 > 0x40 && v173 )
                  j_j___libc_free_0_0(v173);
                v74 = *a2;
                v175 = 257;
                v75 = sub_B504D0(v74 - 29, v73, v132, (__int64)&v173, 0, 0);
                v7 = (unsigned __int8 *)v75;
                if ( *a2 != 54 )
                {
                  sub_B448B0(v75, 1);
                  return (__int64)v7;
                }
                goto LABEL_164;
              }
              goto LABEL_41;
            }
            if ( !sub_B44E60((__int64)a2) )
              goto LABEL_41;
            v93 = *(_DWORD *)(v150 + 8);
            LODWORD(v169) = v93;
            if ( v93 > 0x40 )
            {
              sub_C43780((__int64)&v168, (const void **)v150);
              v94 = (unsigned int)v169;
              v93 = (unsigned int)v169;
              if ( (unsigned int)v169 > 0x40 )
              {
                sub_C47690((__int64 *)&v168, v146);
                v94 = (unsigned int)v169;
LABEL_189:
                LODWORD(v174) = v94;
                if ( v94 > 0x40 )
                {
                  sub_C43780((__int64)&v173, (const void **)&v168);
                  v94 = v174;
                  if ( (unsigned int)v174 > 0x40 )
                  {
                    sub_C44B70((__int64)&v173, v146);
                    goto LABEL_177;
                  }
                }
                else
                {
                  v173 = v168;
                }
                if ( v94 )
                  v97 = (__int64)(v173 << (64 - (unsigned __int8)v94)) >> (64 - (unsigned __int8)v94);
                else
                  v97 = 0;
                v98 = v97 >> v146;
                if ( v94 == v146 )
                  v98 = v97 >> 63;
                v99 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v94;
                if ( !v94 )
                  v99 = 0;
                v31 = v98 & v99;
                v173 = v31;
LABEL_177:
                if ( *(_DWORD *)(v150 + 8) <= 0x40u )
                  v139 = *(_QWORD *)v150 == v173;
                else
                  v139 = sub_C43C50(v150, (const void **)&v173);
                if ( (unsigned int)v174 <= 0x40 )
                  goto LABEL_122;
                goto LABEL_120;
              }
            }
            else
            {
              v168 = *(_QWORD *)v150;
              v94 = v93;
            }
            if ( v93 == v146 )
              v95 = 0;
            else
              v95 = v168 << v146;
            v96 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v93;
            if ( !v93 )
              v96 = 0;
            v168 = v95 & v96;
            goto LABEL_189;
          }
          if ( !sub_B44E60((__int64)a2) )
            goto LABEL_41;
          v89 = *(_DWORD *)(v150 + 8);
          LODWORD(v169) = v89;
          if ( v89 > 0x40 )
          {
            sub_C43780((__int64)&v168, (const void **)v150);
            v90 = (unsigned int)v169;
            v89 = (unsigned int)v169;
            if ( (unsigned int)v169 > 0x40 )
            {
              sub_C47690((__int64 *)&v168, v146);
              v90 = (unsigned int)v169;
              goto LABEL_173;
            }
          }
          else
          {
            v168 = *(_QWORD *)v150;
            v90 = v89;
          }
          if ( v89 == v146 )
            v91 = 0;
          else
            v91 = v168 << v146;
          v92 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v89;
          if ( !v89 )
            v92 = 0;
          v168 = v91 & v92;
LABEL_173:
          LODWORD(v174) = v90;
          if ( v90 > 0x40 )
          {
            sub_C43780((__int64)&v173, (const void **)&v168);
            v90 = v174;
            if ( (unsigned int)v174 > 0x40 )
            {
              sub_C482E0((__int64)&v173, v146);
              goto LABEL_177;
            }
          }
          else
          {
            v31 = v168;
            v173 = v168;
          }
          if ( v90 == v146 )
            v173 = 0;
          else
            v173 >>= v146;
          goto LABEL_177;
        }
      }
    }
  }
LABEL_41:
  v32 = *((_QWORD *)v4 + 2);
  if ( v32 )
  {
    if ( !*(_QWORD *)(v32 + 8) && *v4 == 52 )
    {
      v152 = (_BYTE *)*((_QWORD *)v4 - 8);
      if ( v152 )
      {
        if ( **((_BYTE **)v4 - 4) <= 0x15u )
        {
          v143 = *((_QWORD *)v4 - 4);
          v173 = 0;
          if ( (unsigned __int8)sub_10C4250((__int64 **)&v173, v143, v31) )
          {
            v38 = (unsigned __int8 *)sub_AD64C0(v5, 1, 0);
            v39 = (_BYTE *)sub_AD57F0(v143, v38, 0, 0);
            v161 = (unsigned int **)a1[2].m128i_i64[0];
            v40 = (__int64 *)sub_BD5D20((__int64)v4);
            v175 = 261;
            v174 = v41;
            v173 = (unsigned __int64)v40;
            v42 = sub_A82350(v161, v152, v39, (__int64)&v173);
            if ( (a2[7] & 0x40) != 0 )
              v43 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
            else
              v43 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
            v44 = (__int64 *)*((_QWORD *)v43 + 4);
            if ( v44 )
            {
              v45 = *((_QWORD *)v43 + 5);
              **((_QWORD **)v43 + 6) = v45;
              if ( v45 )
                *(_QWORD *)(v45 + 16) = *((_QWORD *)v43 + 6);
            }
            *((_QWORD *)v43 + 4) = v42;
            if ( v42 )
            {
              v46 = *(_QWORD *)(v42 + 16);
              *((_QWORD *)v43 + 5) = v46;
              if ( v46 )
                *(_QWORD *)(v46 + 16) = v43 + 40;
              *((_QWORD *)v43 + 6) = v42 + 16;
              *(_QWORD *)(v42 + 16) = v43 + 32;
            }
            if ( *(_BYTE *)v44 > 0x1Cu )
            {
              v47 = a1[2].m128i_i64[1];
              v173 = (unsigned __int64)v44;
              v48 = v47 + 2096;
              sub_1196C30(v48, (__int64 *)&v173);
              v49 = v44[2];
              if ( v49 )
              {
                if ( !*(_QWORD *)(v49 + 8) )
                {
                  v173 = *(_QWORD *)(v49 + 24);
                  sub_1196C30(v48, (__int64 *)&v173);
                }
              }
            }
            return (__int64)v7;
          }
        }
      }
    }
  }
  v33 = (unsigned __int8 *)*((_QWORD *)a2 - 8);
  v34 = *v33;
  if ( *v33 > 0x1Cu && (unsigned int)v34 - 42 <= 0x11 )
  {
    v151 = (unsigned int)v34 - 57 > 2 && ((v34 - 42) & 0xFD) != 0;
    if ( !v151 )
    {
      v35 = *((_QWORD *)v33 + 2);
      if ( v35 )
      {
        if ( !*(_QWORD *)(v35 + 8) && **((_BYTE **)a2 - 4) <= 0x15u )
        {
          v164 = (unsigned __int8 *)*((_QWORD *)a2 - 4);
          v110 = *a2 - 29;
          v147 = (__int64 *)a1[2].m128i_i64[0];
          v162 = v110;
          if ( ((*v33 - 42) & 0xFD) != 0 || v110 == 25 )
          {
            v140 = v33;
            v165 = *((_QWORD *)a2 + 1);
            v168 = (unsigned __int64)&v165;
            v169 = &v162;
            v170 = &v166;
            v171 = &v163;
            v172 = &v164;
            v111 = sub_11946C0((__int64)&v168, *((unsigned __int8 **)v33 - 8), *((_BYTE **)v33 - 4));
            v112 = v140;
            if ( v111 )
            {
              v141 = *((_QWORD *)v140 - 4);
            }
            else
            {
              if ( !(unsigned __int8)sub_11946C0((__int64)&v168, *((unsigned __int8 **)v140 - 4), *((_BYTE **)v140 - 8)) )
                goto LABEL_46;
              v112 = v140;
              v151 = *v140 == 44;
              v141 = *((_QWORD *)v140 - 8);
            }
            v134 = v112;
            v113 = sub_AD57C0(v163, v164, 0, 0);
            v175 = 257;
            v138 = sub_1194380(v147, v162, v166, v113, v167, 0, (__int64)&v173, 0);
            v175 = 257;
            v114 = sub_1194380(v147, v162, v141, (__int64)v164, v167, 0, (__int64)&v173, 0);
            v115 = v138;
            v116 = v114;
            if ( v151 )
            {
              v115 = v114;
              v116 = v138;
            }
            v175 = 257;
            v9 = sub_B504D0((unsigned int)*v134 - 29, v115, v116, (__int64)&v173, 0, 0);
            if ( v9 )
              return v9;
          }
        }
      }
    }
  }
LABEL_46:
  if ( *v4 != 58 )
    goto LABEL_47;
  v76 = (_BYTE *)*((_QWORD *)v4 - 4);
  if ( !v76 )
    BUG();
  if ( *v76 != 17 )
  {
    v118 = (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v76 + 1) + 8LL) - 17;
    if ( (unsigned int)v118 > 1
      || *v76 > 0x15u
      || (v119 = sub_AD7630(*((_QWORD *)v4 - 4), 0, v118), (v76 = v119) == 0)
      || *v119 != 17 )
    {
LABEL_47:
      if ( (unsigned __int8)(*a2 - 55) > 1u )
        return 0;
      v36 = *(_QWORD *)(v160 + 16);
      if ( !v36 )
        return 0;
      if ( *(_QWORD *)(v36 + 8) )
        return 0;
      if ( *(_BYTE *)v160 != 85 )
        return 0;
      v100 = *(_QWORD *)(v160 - 32);
      if ( !v100 )
        return 0;
      if ( *(_BYTE *)v100 )
        return 0;
      if ( *(_QWORD *)(v100 + 24) != *(_QWORD *)(v160 + 80) )
        return 0;
      if ( (*(_BYTE *)(v100 + 33) & 0x20) == 0 )
        return 0;
      v101 = *(_DWORD *)(v100 + 36);
      if ( v101 != 313 && v101 != 362 )
        return 0;
      v158 = sub_BCB060(v5);
      if ( *v4 != 17 )
      {
        v120 = (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v4 + 1) + 8LL) - 17;
        if ( (unsigned int)v120 > 1 )
          return 0;
        if ( *v4 > 0x15u )
          return 0;
        v121 = sub_AD7630((__int64)v4, 0, v120);
        v4 = v121;
        if ( !v121 || *v121 != 17 )
          return 0;
      }
      v102 = *((_DWORD *)v4 + 8);
      if ( v102 > 0x40 )
      {
        if ( v102 - (unsigned int)sub_C444A0((__int64)(v4 + 24)) > 0x40 )
          return 0;
        v103 = **((_QWORD **)v4 + 3);
      }
      else
      {
        v103 = *((_QWORD *)v4 + 3);
      }
      if ( v158 - 1 == v103 )
      {
        v104 = a1[2].m128i_i64[0];
        LOWORD(v172) = 257;
        if ( (*(_BYTE *)(v160 + 7) & 0x40) != 0 )
          v105 = *(__int64 **)(v160 - 8);
        else
          v105 = (__int64 *)(v160 - 32LL * (*(_DWORD *)(v160 + 4) & 0x7FFFFFF));
        v106 = *(_QWORD *)(v160 - 32);
        v149 = v105[4];
        v153 = *v105;
        if ( !v106 || *(_BYTE *)v106 || *(_QWORD *)(v106 + 24) != *(_QWORD *)(v160 + 80) )
          BUG();
        v107 = 4 * (*(_DWORD *)(v106 + 36) == 313) + 36;
        v108 = (_QWORD *)(*(__int64 (__fastcall **)(_QWORD, _QWORD))(**(_QWORD **)(v104 + 80) + 56LL))(
                           *(_QWORD *)(v104 + 80),
                           v107);
        if ( !v108 )
        {
          v175 = 257;
          v108 = sub_BD2C40(72, unk_3F10FD0);
          if ( v108 )
          {
            v122 = *(_QWORD *)(v153 + 8);
            v123 = *(unsigned __int8 *)(v122 + 8);
            if ( (unsigned int)(v123 - 17) > 1 )
            {
              v127 = sub_BCB2A0(*(_QWORD **)v122);
            }
            else
            {
              v124 = *(_DWORD *)(v122 + 32);
              v125 = *(_QWORD **)v122;
              BYTE4(v167) = (_BYTE)v123 == 18;
              LODWORD(v167) = v124;
              v126 = (__int64 *)sub_BCB2A0(v125);
              v127 = sub_BCE1B0(v126, v167);
            }
            sub_B523C0((__int64)v108, v127, 53, v107, v153, v149, (__int64)&v173, 0, 0, 0);
          }
          (*(void (__fastcall **)(_QWORD, _QWORD *, unsigned __int64 *, _QWORD, _QWORD))(**(_QWORD **)(v104 + 88) + 16LL))(
            *(_QWORD *)(v104 + 88),
            v108,
            &v168,
            *(_QWORD *)(v104 + 56),
            *(_QWORD *)(v104 + 64));
          v128 = *(_QWORD *)v104;
          v129 = *(_QWORD *)v104 + 16LL * *(unsigned int *)(v104 + 8);
          while ( v129 != v128 )
          {
            v130 = *(_QWORD *)(v128 + 8);
            v131 = *(_DWORD *)v128;
            v128 += 16;
            sub_B99FD0((__int64)v108, v131, v130);
          }
        }
        v109 = (*a2 != 55) + 39;
        v175 = 257;
        return sub_B51D30(v109, (__int64)v108, v5, (__int64)&v173, 0, 0);
      }
      return 0;
    }
  }
  if ( *((_DWORD *)v76 + 8) > 0x40u )
  {
    v154 = *((_DWORD *)v76 + 8);
    v159 = v76;
    if ( v154 - (unsigned int)sub_C444A0((__int64)(v76 + 24)) > 0x40 )
      goto LABEL_47;
    v77 = **((_QWORD **)v159 + 3);
  }
  else
  {
    v77 = *((_QWORD *)v76 + 3);
  }
  if ( v148 - 1 != v77 )
    goto LABEL_47;
  v78 = sub_AD64C0(v5, v77, 0);
  if ( (a2[7] & 0x40) != 0 )
    v79 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
  else
    v79 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
  v80 = (__int64 *)*((_QWORD *)v79 + 4);
  if ( v80 )
  {
    v81 = *((_QWORD *)v79 + 5);
    **((_QWORD **)v79 + 6) = v81;
    if ( v81 )
      *(_QWORD *)(v81 + 16) = *((_QWORD *)v79 + 6);
  }
  *((_QWORD *)v79 + 4) = v78;
  if ( v78 )
  {
    v82 = *(_QWORD *)(v78 + 16);
    *((_QWORD *)v79 + 5) = v82;
    if ( v82 )
      *(_QWORD *)(v82 + 16) = v79 + 40;
    *((_QWORD *)v79 + 6) = v78 + 16;
    *(_QWORD *)(v78 + 16) = v79 + 32;
  }
  if ( *(_BYTE *)v80 > 0x1Cu )
  {
    v83 = a1[2].m128i_i64[1];
    v173 = (unsigned __int64)v80;
    v84 = v83 + 2096;
    sub_1196C30(v84, (__int64 *)&v173);
    v85 = v80[2];
    if ( v85 )
    {
      if ( !*(_QWORD *)(v85 + 8) )
      {
        v173 = *(_QWORD *)(v85 + 24);
        sub_1196C30(v84, (__int64 *)&v173);
      }
    }
  }
  return (__int64)v7;
}
