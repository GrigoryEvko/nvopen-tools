// Function: sub_9AE470
// Address: 0x9ae470
//
__int64 __fastcall sub_9AE470(__int64 a1, __int64 a2, unsigned int a3, __m128i *a4)
{
  __int64 v6; // r12
  __int64 v7; // rbx
  __int64 v8; // rsi
  int v9; // edx
  __int64 *v10; // rax
  __int64 v11; // rdi
  __m128i v12; // rax
  _DWORD *v13; // rax
  unsigned __int8 v14; // cl
  __int64 v15; // rax
  unsigned int v16; // r8d
  unsigned int v17; // r15d
  __int64 v18; // r14
  unsigned int v19; // r13d
  __int64 v20; // rax
  __int64 v21; // rax
  unsigned int v22; // edi
  unsigned __int64 v23; // rsi
  unsigned int v24; // ecx
  __int64 v25; // r9
  unsigned __int64 v26; // rsi
  unsigned __int64 v27; // rax
  unsigned int v28; // eax
  unsigned int v29; // ebx
  __int64 v30; // rax
  unsigned int v31; // esi
  unsigned __int64 v32; // r8
  int v33; // eax
  __int64 v35; // rax
  unsigned int v36; // eax
  unsigned __int64 v37; // rax
  __int64 v38; // rdi
  unsigned int v39; // eax
  unsigned int v40; // eax
  bool v41; // zf
  unsigned int v42; // eax
  unsigned int v43; // eax
  _QWORD *v44; // rax
  __int64 v45; // rax
  unsigned int v46; // eax
  __int64 v47; // rax
  __int64 v48; // r15
  __int64 v49; // rdi
  int v50; // edx
  unsigned int v51; // eax
  unsigned int v52; // r8d
  bool v53; // al
  int v54; // eax
  unsigned int v55; // ebx
  unsigned int v56; // eax
  unsigned __int64 v57; // rax
  _BYTE *v58; // r11
  int v59; // r10d
  __int32 v60; // edx
  __int64 v61; // rax
  __int64 v62; // rax
  unsigned int v63; // eax
  __int64 v64; // rax
  __m128i v65; // xmm1
  __int64 v66; // r15
  __m128i v67; // xmm2
  unsigned __int64 v68; // xmm3_8
  __int64 v69; // rbx
  __int64 v70; // rdx
  unsigned int v71; // r12d
  int v72; // ecx
  unsigned __int64 v73; // rax
  unsigned int v74; // eax
  __int64 v75; // rdx
  __int64 v76; // rcx
  unsigned __int64 v77; // rax
  _QWORD *v78; // rax
  unsigned int v79; // ebx
  _QWORD *v80; // rax
  __int64 v81; // rax
  _BYTE *v82; // rdi
  unsigned int v83; // eax
  unsigned int v84; // ebx
  __int64 v85; // rax
  _QWORD *v86; // rax
  _QWORD *v87; // rax
  unsigned int v88; // ebx
  __int64 v89; // rax
  int v90; // edx
  __int64 v91; // rax
  _QWORD *v92; // rax
  unsigned int v93; // eax
  __int64 v94; // rax
  char v95; // al
  __int64 v96; // rbx
  _QWORD *v97; // rax
  unsigned int v98; // eax
  _QWORD *v99; // rax
  __int64 v100; // rax
  int v101; // eax
  __int64 v102; // rax
  unsigned int v103; // ebx
  __int64 v104; // rsi
  __int64 v105; // rax
  _QWORD *v106; // rax
  int v107; // eax
  __int64 v108; // r12
  _QWORD *v109; // rax
  unsigned int v110; // eax
  __int64 v111; // rax
  char v112; // al
  __int64 v113; // r12
  unsigned int v114; // ebx
  __int64 v115; // rsi
  __int64 v116; // rax
  bool v117; // al
  __int32 v118; // r12d
  int v119; // ebx
  __int64 v120; // rax
  __int64 v121; // r15
  _BYTE *v122; // rax
  unsigned int v123; // ebx
  _QWORD *v124; // rax
  unsigned int v125; // ebx
  __int64 v126; // r15
  _QWORD *v127; // rax
  char v128; // al
  int v129; // r8d
  int v130; // r8d
  bool v131; // al
  unsigned int v132; // eax
  __int64 v133; // r15
  unsigned __int64 v134; // rax
  __int64 v135; // rbx
  __int32 v136; // edx
  int v137; // eax
  int v138; // r10d
  _BYTE *v139; // r8
  __int64 v140; // rt0
  unsigned int v141; // ebx
  unsigned int v142; // eax
  int v143; // eax
  _QWORD *v144; // rax
  _BYTE *v145; // rax
  __int64 v146; // rax
  __int64 *v147; // r9
  bool v148; // al
  unsigned int v149; // r8d
  unsigned __int8 **v150; // rax
  __int64 *v151; // r9
  bool v152; // al
  _BYTE *v153; // rax
  __int64 v154; // r8
  int v155; // eax
  int v156; // ebx
  __int64 *v157; // [rsp+0h] [rbp-D0h]
  __int64 *v158; // [rsp+0h] [rbp-D0h]
  __m128i *v159; // [rsp+8h] [rbp-C8h]
  int v160; // [rsp+8h] [rbp-C8h]
  int v161; // [rsp+10h] [rbp-C0h]
  unsigned __int8 v162; // [rsp+10h] [rbp-C0h]
  int v163; // [rsp+10h] [rbp-C0h]
  bool v164; // [rsp+10h] [rbp-C0h]
  bool v165; // [rsp+10h] [rbp-C0h]
  unsigned int v166; // [rsp+18h] [rbp-B8h]
  unsigned __int8 v167; // [rsp+18h] [rbp-B8h]
  unsigned __int8 v168; // [rsp+18h] [rbp-B8h]
  __int64 v169; // [rsp+18h] [rbp-B8h]
  unsigned int v170; // [rsp+18h] [rbp-B8h]
  unsigned int v171; // [rsp+18h] [rbp-B8h]
  int v172; // [rsp+18h] [rbp-B8h]
  __int64 v173; // [rsp+18h] [rbp-B8h]
  __int64 v174; // [rsp+18h] [rbp-B8h]
  __int64 v175; // [rsp+18h] [rbp-B8h]
  unsigned int v176; // [rsp+24h] [rbp-ACh]
  int v177; // [rsp+28h] [rbp-A8h]
  unsigned int v179; // [rsp+2Ch] [rbp-A4h]
  unsigned int v180; // [rsp+2Ch] [rbp-A4h]
  unsigned int v181; // [rsp+2Ch] [rbp-A4h]
  unsigned int v182; // [rsp+2Ch] [rbp-A4h]
  unsigned int v183; // [rsp+2Ch] [rbp-A4h]
  unsigned int v184; // [rsp+2Ch] [rbp-A4h]
  __int64 *v185; // [rsp+30h] [rbp-A0h] BYREF
  unsigned int v186; // [rsp+38h] [rbp-98h]
  __int64 *v187; // [rsp+40h] [rbp-90h] BYREF
  int v188; // [rsp+48h] [rbp-88h]
  __m128i v189; // [rsp+50h] [rbp-80h] BYREF
  __m128i v190; // [rsp+60h] [rbp-70h] BYREF
  __m128i v191; // [rsp+70h] [rbp-60h]
  __int128 v192; // [rsp+80h] [rbp-50h]
  __int64 v193; // [rsp+90h] [rbp-40h]

  v6 = a1;
  v7 = *(_QWORD *)(a1 + 8);
  v177 = 0;
  while ( 2 )
  {
    v8 = v7;
    v9 = *(unsigned __int8 *)(v7 + 8);
    if ( (unsigned int)(v9 - 17) <= 1 )
    {
      v10 = *(__int64 **)(v7 + 16);
      v8 = *v10;
      LOBYTE(v9) = *(_BYTE *)(*v10 + 8);
    }
    v11 = a4->m128i_i64[0];
    if ( (_BYTE)v9 == 14 )
    {
      v176 = sub_AE43A0(v11, v8);
    }
    else
    {
      v12.m128i_i64[0] = sub_9208B0(v11, v8);
      v189 = v12;
      v176 = sub_CA1930(&v189);
    }
    v13 = (_DWORD *)sub_C94E20(qword_4F862D0);
    if ( v13 )
    {
      if ( a3 != *v13 )
        goto LABEL_8;
LABEL_46:
      v16 = 1;
      return v16 + v177;
    }
    if ( a3 == LODWORD(qword_4F862D0[2]) )
      goto LABEL_46;
LABEL_8:
    v14 = *(_BYTE *)v6;
    if ( *(_BYTE *)v6 > 0x1Cu )
    {
      v33 = v14 - 29;
    }
    else
    {
      if ( v14 != 5 )
      {
LABEL_10:
        v166 = 1;
        goto LABEL_11;
      }
      v33 = *(unsigned __int16 *)(v6 + 2);
    }
    switch ( v33 )
    {
      case 13:
        v79 = a3 + 1;
        v80 = (_QWORD *)sub_986520(v6);
        v166 = sub_9AF7E0(*v80, a3 + 1, a4);
        if ( v166 == 1 )
          goto LABEL_92;
        v81 = sub_986520(v6);
        v82 = *(_BYTE **)(v81 + 32);
        if ( *v82 > 0x15u )
          goto LABEL_121;
        if ( !(unsigned __int8)sub_AD7930(v82) )
          goto LABEL_120;
        sub_9878D0((__int64)&v189, v176);
        v150 = (unsigned __int8 **)sub_986520(v6);
        sub_9AB8E0(*v150, a2, (unsigned __int64 *)&v189, v79, a4);
        sub_9865C0((__int64)&v185, (__int64)&v189);
        v151 = (__int64 *)&v185;
        if ( v186 <= 0x40 )
        {
          v185 = (__int64 *)((unsigned __int64)v185 | 1);
          sub_9842C0((unsigned __int64 *)&v185);
        }
        else
        {
          *v185 |= 1uLL;
        }
        v158 = v151;
        v188 = v186;
        v186 = 0;
        v187 = v185;
        v165 = sub_986760((__int64)&v187);
        sub_969240((__int64 *)&v187);
        sub_969240(v158);
        if ( v165 )
          goto LABEL_208;
        v152 = sub_986C60(v189.m128i_i64, v189.m128i_i32[2] - 1);
        v149 = v166;
        if ( v152 )
          goto LABEL_209;
        sub_969240(v190.m128i_i64);
        sub_969240(v189.m128i_i64);
LABEL_120:
        v81 = sub_986520(v6);
LABEL_121:
        v83 = sub_9AE470(*(_QWORD *)(v81 + 32), a2, v79, a4);
        if ( v83 != 1 )
          goto LABEL_122;
        goto LABEL_130;
      case 15:
        v84 = a3 + 1;
        v85 = sub_986520(v6);
        v166 = sub_9AE470(*(_QWORD *)(v85 + 32), a2, a3 + 1, a4);
        if ( v166 == 1 )
          goto LABEL_92;
        v86 = (_QWORD *)sub_986520(v6);
        if ( *(_BYTE *)*v86 > 0x15u )
          goto LABEL_129;
        if ( !(unsigned __int8)sub_AC30F0(*v86) )
          goto LABEL_128;
        sub_9878D0((__int64)&v189, v176);
        v146 = sub_986520(v6);
        sub_9AB8E0(*(unsigned __int8 **)(v146 + 32), a2, (unsigned __int64 *)&v189, v84, a4);
        sub_9865C0((__int64)&v185, (__int64)&v189);
        v147 = (__int64 *)&v185;
        if ( v186 <= 0x40 )
        {
          v185 = (__int64 *)((unsigned __int64)v185 | 1);
          sub_9842C0((unsigned __int64 *)&v185);
        }
        else
        {
          *v185 |= 1uLL;
        }
        v157 = v147;
        v188 = v186;
        v186 = 0;
        v187 = v185;
        v164 = sub_986760((__int64)&v187);
        sub_969240((__int64 *)&v187);
        sub_969240(v157);
        if ( v164 )
        {
LABEL_208:
          v149 = v176;
          goto LABEL_209;
        }
        v148 = sub_986C60(v189.m128i_i64, v189.m128i_i32[2] - 1);
        v149 = v166;
        if ( v148 )
        {
LABEL_209:
          v183 = v149;
          sub_969240(v190.m128i_i64);
          sub_969240(v189.m128i_i64);
          v16 = v183;
          return v16 + v177;
        }
        sub_969240(v190.m128i_i64);
        sub_969240(v189.m128i_i64);
LABEL_128:
        v86 = (_QWORD *)sub_986520(v6);
LABEL_129:
        v83 = sub_9AE470(*v86, a2, v84, a4);
        if ( v83 != 1 )
        {
LABEL_122:
          if ( v166 <= v83 )
            v83 = v166;
          v16 = v83 - 1;
          return v16 + v177;
        }
LABEL_130:
        v166 = 1;
        v14 = *(_BYTE *)v6;
LABEL_11:
        if ( v14 > 0x15u || (v15 = *(_QWORD *)(v6 + 8), *(_BYTE *)(v15 + 8) != 17) )
        {
LABEL_29:
          sub_9878D0((__int64)&v189, v176);
          sub_9AB8E0((unsigned __int8 *)v6, a2, (unsigned __int64 *)&v189, a3, a4);
          v29 = v189.m128i_u32[2];
          v30 = 1LL << (v189.m128i_i8[8] - 1);
          if ( v189.m128i_i32[2] > 0x40u )
          {
            if ( (*(_QWORD *)(v189.m128i_i64[0] + 8LL * ((unsigned int)(v189.m128i_i32[2] - 1) >> 6)) & v30) != 0 )
            {
              v40 = sub_C44500(&v189);
              v16 = v166;
              if ( v166 < v40 )
                v16 = v40;
              if ( v190.m128i_i32[2] <= 0x40u )
              {
LABEL_40:
                if ( v189.m128i_i64[0] )
                {
                  v180 = v16;
                  j_j___libc_free_0_0(v189.m128i_i64[0]);
                  v16 = v180;
                }
                return v16 + v177;
              }
              goto LABEL_37;
            }
            v31 = v190.m128i_u32[2];
          }
          else
          {
            v31 = v190.m128i_u32[2];
            if ( (v189.m128i_i64[0] & v30) != 0 )
            {
              if ( v189.m128i_i32[2] )
              {
                v16 = 64;
                if ( v189.m128i_i64[0] << (64 - v189.m128i_i8[8]) != -1 )
                {
                  _BitScanReverse64(&v32, ~(v189.m128i_i64[0] << (64 - v189.m128i_i8[8])));
                  v16 = v32 ^ 0x3F;
                }
                if ( v166 >= v16 )
                  v16 = v166;
                if ( v190.m128i_i32[2] <= 0x40u )
                  return v16 + v177;
              }
              else
              {
                v16 = v166;
                if ( v190.m128i_i32[2] <= 0x40u )
                  return v16 + v177;
              }
              goto LABEL_37;
            }
          }
          v35 = 1LL << ((unsigned __int8)v31 - 1);
          if ( v31 <= 0x40 )
          {
            if ( (v35 & v190.m128i_i64[0]) != 0 )
            {
              if ( !v31 )
              {
                v16 = v166;
                goto LABEL_39;
              }
              v36 = 64;
              if ( v190.m128i_i64[0] << (64 - (unsigned __int8)v31) != -1 )
              {
                _BitScanReverse64(&v37, ~(v190.m128i_i64[0] << (64 - (unsigned __int8)v31)));
                v36 = v37 ^ 0x3F;
              }
            }
            else
            {
              v36 = 1;
            }
            v16 = v166;
            if ( v166 < v36 )
              v16 = v36;
LABEL_39:
            if ( v29 <= 0x40 )
              return v16 + v177;
            goto LABEL_40;
          }
          v41 = (*(_QWORD *)(v190.m128i_i64[0] + 8LL * ((v31 - 1) >> 6)) & v35) == 0;
          v42 = 1;
          if ( !v41 )
            v42 = sub_C44500(&v190);
          v16 = v166;
          if ( v166 < v42 )
            v16 = v42;
LABEL_37:
          if ( v190.m128i_i64[0] )
          {
            v179 = v16;
            j_j___libc_free_0_0(v190.m128i_i64[0]);
            v29 = v189.m128i_u32[2];
            v16 = v179;
          }
          goto LABEL_39;
        }
        v16 = v176;
        v161 = *(_DWORD *)(v15 + 32);
        if ( v161 )
        {
          v159 = a4;
          v17 = 0;
          v18 = a2;
          v19 = v176;
          while ( 1 )
          {
            v20 = *(_QWORD *)v18;
            if ( *(_DWORD *)(v18 + 8) > 0x40u )
              v20 = *(_QWORD *)(v20 + 8LL * (v17 >> 6));
            if ( (v20 & (1LL << v17)) != 0 )
            {
              v21 = sub_AD69F0(v6, v17);
              if ( !v21 || *(_BYTE *)v21 != 17 )
              {
                a2 = v18;
                a4 = v159;
                goto LABEL_29;
              }
              v22 = *(_DWORD *)(v21 + 32);
              v23 = *(_QWORD *)(v21 + 24);
              v24 = v22 - 1;
              v25 = 1LL << ((unsigned __int8)v22 - 1);
              if ( v22 > 0x40 )
              {
                v38 = v21 + 24;
                if ( (*(_QWORD *)(v23 + 8LL * (v24 >> 6)) & v25) != 0 )
                {
                  v39 = sub_C44500(v38);
                  if ( v19 > v39 )
                    v19 = v39;
                }
                else
                {
                  v43 = sub_C444A0(v38);
                  if ( v19 > v43 )
                    v19 = v43;
                }
              }
              else if ( (v25 & v23) != 0 )
              {
                if ( v22 )
                {
                  v26 = ~(v23 << (64 - (unsigned __int8)v22));
                  if ( v26 )
                  {
                    _BitScanReverse64(&v27, v26);
                    v28 = v27 ^ 0x3F;
                    if ( v19 > v28 )
                      v19 = v28;
                  }
                  else if ( v19 > 0x40 )
                  {
                    v19 = 64;
                  }
                }
                else
                {
                  v19 = 0;
                }
              }
              else
              {
                if ( v23 )
                {
                  _BitScanReverse64(&v23, v23);
                  v22 = v22 - 64 + (v23 ^ 0x3F);
                }
                if ( v19 > v22 )
                  v19 = v22;
              }
            }
            if ( v161 == ++v17 )
            {
              v16 = v19;
              a2 = v18;
              a4 = v159;
              break;
            }
          }
        }
        if ( !v16 )
          goto LABEL_29;
        return v16 + v177;
      case 17:
        v99 = (_QWORD *)sub_986520(v6);
        v166 = sub_9AE470(*v99, a2, a3 + 1, a4);
        if ( v166 == 1 )
          goto LABEL_92;
        v100 = sub_986520(v6);
        v101 = sub_9AE470(*(_QWORD *)(v100 + 32), a2, a3 + 1, a4);
        if ( v101 == 1 )
          goto LABEL_130;
        if ( v176 < 2 * (v176 + 1) - v166 - v101 )
          goto LABEL_46;
        v16 = v101 + v176 + 1 + v166 - 2 * (v176 + 1);
        return v16 + v177;
      case 20:
        v189.m128i_i8[8] = 0;
        v189.m128i_i64[0] = (__int64)&v187;
        v102 = sub_986520(v6);
        if ( !(unsigned __int8)sub_991580((__int64)&v189, *(_QWORD *)(v102 + 32)) )
          goto LABEL_130;
        v103 = *((_DWORD *)v187 + 2);
        v104 = *v187;
        v105 = 1LL << ((unsigned __int8)v103 - 1);
        if ( v103 > 0x40 )
        {
          if ( (*(_QWORD *)(v104 + 8LL * ((v103 - 1) >> 6)) & v105) != 0 || v103 == (unsigned int)sub_C444A0(v187) )
            goto LABEL_130;
        }
        else if ( (v105 & v104) != 0 || !v104 )
        {
          goto LABEL_130;
        }
        v106 = (_QWORD *)sub_986520(v6);
        v107 = sub_9AE470(*v106, a2, a3 + 1, a4);
        v108 = (__int64)v187;
        v16 = v107 + *(_DWORD *)(v108 + 8) - 1 - sub_9871A0((__int64)v187);
        if ( v16 > v176 )
          v16 = v176;
        return v16 + v177;
      case 23:
        v109 = (_QWORD *)sub_986520(v6);
        v110 = sub_9AE470(*v109, a2, a3 + 1, a4);
        v189.m128i_i8[8] = 0;
        v181 = v110;
        v189.m128i_i64[0] = (__int64)&v187;
        v111 = sub_986520(v6);
        v112 = sub_991580((__int64)&v189, *(_QWORD *)(v111 + 32));
        v16 = v181;
        if ( !v112 )
          return v16 + v177;
        v113 = (__int64)v187;
        v114 = *((_DWORD *)v187 + 2);
        v115 = *v187;
        v116 = 1LL << ((unsigned __int8)v114 - 1);
        if ( v114 > 0x40 )
        {
          if ( (*(_QWORD *)(v115 + 8LL * ((v114 - 1) >> 6)) & v116) != 0 )
            return v16 + v177;
          v143 = sub_C444A0(v187);
          v16 = v181;
          v117 = v114 == v143;
        }
        else
        {
          if ( (v116 & v115) != 0 )
            return v16 + v177;
          v117 = v115 == 0;
        }
        if ( !v117 )
        {
          v182 = v16;
          sub_9865C0((__int64)&v189, v113);
          sub_C46E90(&v189);
          v118 = v189.m128i_i32[2];
          v119 = sub_9871A0((__int64)&v189);
          sub_969240(v189.m128i_i64);
          v16 = v182;
          if ( v182 < v119 + v176 - v118 )
            v16 = v119 + v176 - v118;
        }
        return v16 + v177;
      case 25:
        v189.m128i_i8[8] = 0;
        v189.m128i_i64[0] = (__int64)&v187;
        v120 = sub_986520(v6);
        if ( !(unsigned __int8)sub_991580((__int64)&v189, *(_QWORD *)(v120 + 32)) )
          goto LABEL_130;
        v121 = (__int64)v187;
        if ( !sub_986EE0((__int64)v187, v176) )
          goto LABEL_130;
        v122 = *(_BYTE **)sub_986520(v6);
        v123 = a3 + 1;
        if ( *v122 == 68
          && (v154 = *((_QWORD *)v122 - 4)) != 0
          && (v175 = *((_QWORD *)v122 - 4), v155 = sub_BCB060(*(_QWORD *)(v154 + 8)), !sub_986EE0(v121, v176 - v155)) )
        {
          v156 = sub_9AE470(v175, a2, v123, a4);
          v125 = v176 + v156 - sub_BCB060(*(_QWORD *)(v175 + 8));
        }
        else
        {
          v124 = (_QWORD *)sub_986520(v6);
          v125 = sub_9AE470(*v124, a2, v123, a4);
        }
        v126 = (__int64)v187;
        if ( !sub_986EE0((__int64)v187, v125) )
          goto LABEL_130;
        v127 = *(_QWORD **)v126;
        if ( *(_DWORD *)(v126 + 8) > 0x40u )
          v127 = (_QWORD *)*v127;
        v16 = v125 - (_DWORD)v127;
        return v16 + v177;
      case 27:
        v92 = (_QWORD *)sub_986520(v6);
        v93 = sub_9AE470(*v92, a2, a3 + 1, a4);
        v189.m128i_i8[8] = 0;
        v170 = v93;
        v189.m128i_i64[0] = (__int64)&v187;
        v94 = sub_986520(v6);
        v95 = sub_991580((__int64)&v189, *(_QWORD *)(v94 + 32));
        v16 = v170;
        if ( !v95 )
          return v16 + v177;
        v96 = (__int64)v187;
        if ( !sub_986EE0((__int64)v187, v176) )
          goto LABEL_130;
        v97 = *(_QWORD **)v96;
        if ( *(_DWORD *)(v96 + 8) > 0x40u )
          v97 = (_QWORD *)*v97;
        v98 = v170 + (_DWORD)v97;
        v16 = v176;
        if ( v98 <= v176 )
          v16 = v98;
        return v16 + v177;
      case 28:
      case 29:
      case 30:
        v44 = (_QWORD *)sub_986520(v6);
        v166 = sub_9AE470(*v44, a2, a3 + 1, a4);
        if ( v166 == 1 )
          goto LABEL_92;
        v45 = sub_986520(v6);
        v46 = sub_9AE470(*(_QWORD *)(v45 + 32), a2, a3 + 1, a4);
        v14 = *(_BYTE *)v6;
        if ( v166 <= v46 )
          v46 = v166;
        v166 = v46;
        goto LABEL_11;
      case 38:
        v87 = (_QWORD *)sub_986520(v6);
        v88 = sub_9AF7E0(*v87, a3 + 1, a4);
        v89 = sub_986520(v6);
        v90 = sub_BCB060(*(_QWORD *)(*(_QWORD *)v89 + 8LL));
        if ( v88 <= v90 - v176 )
          goto LABEL_46;
        v16 = v88 + v176 - v90;
        return v16 + v177;
      case 40:
        v91 = sub_986520(v6);
        v6 = *(_QWORD *)v91;
        v7 = *(_QWORD *)(*(_QWORD *)v91 + 8LL);
        ++a3;
        v177 += v176 - sub_BCB060(v7);
        continue;
      case 49:
        v167 = *(_BYTE *)v6;
        v47 = sub_986520(v6);
        v14 = v167;
        v48 = *(_QWORD *)v47;
        v49 = *(_QWORD *)(*(_QWORD *)v47 + 8LL);
        v50 = *(unsigned __int8 *)(v49 + 8);
        if ( (unsigned int)(v50 - 17) <= 1 )
          LOBYTE(v50) = *(_BYTE *)(**(_QWORD **)(v49 + 16) + 8LL);
        v166 = 1;
        if ( (_BYTE)v50 != 12 )
          goto LABEL_11;
        v162 = v14;
        v51 = sub_BCB060(v49);
        v14 = v162;
        v52 = v51;
        if ( v51 % v176 || *(_BYTE *)(v7 + 8) != 17 )
          goto LABEL_11;
        if ( (unsigned int)sub_9AF7E0(v48, a3 + 1, a4) != v52 )
          goto LABEL_92;
        v16 = v176;
        return v16 + v177;
      case 55:
        v14 = *(_BYTE *)v6;
        v166 = 1;
        v64 = *(_DWORD *)(v6 + 4) & 0x7FFFFFF;
        if ( (unsigned int)(v64 - 1) > 3 )
          goto LABEL_11;
        v65 = _mm_loadu_si128(a4 + 1);
        v66 = 0;
        v67 = _mm_loadu_si128(a4 + 2);
        v68 = _mm_loadu_si128(a4 + 3).m128i_u64[0];
        v169 = 8 * v64;
        v69 = v6;
        v70 = a4[4].m128i_i64[0];
        v189 = _mm_loadu_si128(a4);
        v71 = v176;
        v192 = v68;
        v193 = v70;
        v190 = v65;
        v191 = v67;
        while ( v71 != 1 )
        {
          v75 = *(_QWORD *)(v69 - 8);
          v76 = *(_QWORD *)(v75 + 32LL * *(unsigned int *)(v69 + 72) + v66);
          v77 = *(_QWORD *)(v76 + 48) & 0xFFFFFFFFFFFFFFF8LL;
          if ( v77 == v76 + 48 )
          {
            v73 = 0;
          }
          else
          {
            if ( !v77 )
              BUG();
            v72 = *(unsigned __int8 *)(v77 - 24);
            v73 = v77 - 24;
            if ( (unsigned int)(v72 - 30) >= 0xB )
              v73 = 0;
          }
          v191.m128i_i64[1] = v73;
          v74 = sub_9AE470(*(_QWORD *)(v75 + 4 * v66), a2, a3 + 1, &v189);
          if ( v71 > v74 )
            v71 = v74;
          v66 += 8;
          if ( v66 == v169 )
          {
            v16 = v71;
            return v16 + v177;
          }
        }
        goto LABEL_46;
      case 56:
        v168 = *(_BYTE *)v6;
        v53 = sub_988010(v6);
        v14 = v168;
        if ( !v53 )
          goto LABEL_10;
        v54 = sub_987FE0(v6);
        if ( v54 == 1 )
        {
          v144 = (_QWORD *)sub_986520(v6);
          v166 = sub_9AE470(*v144, a2, a3 + 1, a4);
          v16 = v166 - 1;
          if ( v166 != 1 )
            return v16 + v177;
          goto LABEL_92;
        }
        v14 = v168;
        if ( (unsigned int)(v54 - 329) > 1 )
          goto LABEL_10;
        if ( sub_990670(v6, &v187, &v189) )
        {
          v55 = sub_969260(v189.m128i_i64[0]);
          v56 = sub_969260((__int64)v187);
          if ( v55 <= v56 )
            v56 = v55;
          v16 = v56;
          return v16 + v177;
        }
        goto LABEL_130;
      case 57:
        v185 = 0;
        v187 = 0;
        v57 = sub_99AEC0((_BYTE *)v6, (__int64 *)&v185, (__int64 *)&v187, 0, 0);
        v58 = v185;
        v189.m128i_i64[0] = v57;
        v59 = v57;
        v189.m128i_i32[2] = v60;
        if ( (v57 & 0xFFFFFFFD) != 1 )
          goto LABEL_101;
        v133 = (__int64)(v187 + 3);
        if ( *(_BYTE *)v187 == 17 )
          goto LABEL_178;
        v160 = v57;
        v174 = (__int64)v185;
        if ( (unsigned int)*(unsigned __int8 *)(v187[1] + 8) - 17 > 1 )
          goto LABEL_101;
        if ( *(_BYTE *)v187 > 0x15u )
          goto LABEL_101;
        v145 = (_BYTE *)sub_AD7630(v187, 0);
        if ( !v145 || *v145 != 17 )
          goto LABEL_101;
        v59 = v160;
        v133 = (__int64)(v145 + 24);
        v58 = (_BYTE *)v174;
LABEL_178:
        v163 = v59;
        v185 = 0;
        v187 = 0;
        v134 = sub_99AEC0(v58, (__int64 *)&v185, (__int64 *)&v187, 0, 0);
        v135 = (__int64)v187;
        v172 = v134;
        v189.m128i_i64[0] = v134;
        v189.m128i_i32[2] = v136;
        v137 = sub_990570(v163);
        v138 = v163;
        if ( v172 != v137 )
          goto LABEL_101;
        v139 = (_BYTE *)(v135 + 24);
        if ( *(_BYTE *)v135 == 17 )
          goto LABEL_180;
        if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v135 + 8) + 8LL) - 17 > 1 )
          goto LABEL_101;
        if ( *(_BYTE *)v135 > 0x15u )
          goto LABEL_101;
        v153 = (_BYTE *)sub_AD7630(v135, 0);
        if ( !v153 || *v153 != 17 )
          goto LABEL_101;
        v138 = v163;
        v139 = v153 + 24;
LABEL_180:
        if ( v138 == 1 )
        {
          v140 = (__int64)v139;
          v139 = (_BYTE *)v133;
          v133 = v140;
        }
        v173 = (__int64)v139;
        if ( (int)sub_C4C880(v133, v139) <= 0 )
        {
          v141 = sub_969260(v173);
          v142 = sub_969260(v133);
          if ( v141 <= v142 )
            v142 = v141;
          v16 = v142;
        }
        else
        {
LABEL_101:
          v61 = sub_986520(v6);
          v166 = sub_9AE470(*(_QWORD *)(v61 + 32), a2, a3 + 1, a4);
          if ( v166 == 1 )
          {
LABEL_92:
            v14 = *(_BYTE *)v6;
            goto LABEL_11;
          }
          v62 = sub_986520(v6);
          v63 = sub_9AE470(*(_QWORD *)(v62 + 64), a2, a3 + 1, a4);
          v16 = v166;
          if ( v166 > v63 )
            v16 = v63;
        }
        return v16 + v177;
      case 61:
        v78 = (_QWORD *)sub_986520(v6);
        v16 = sub_9AF7E0(*v78, a3 + 1, a4);
        return v16 + v177;
      case 63:
        v16 = 1;
        if ( *(_BYTE *)v6 != 92 )
          return v16 + v177;
        v188 = 1;
        v187 = 0;
        v189.m128i_i32[2] = 1;
        v189.m128i_i64[0] = 0;
        v128 = sub_984B30(v6, a2, (__int64)&v187, (__int64)&v189);
        v129 = 1;
        if ( !v128 )
          goto LABEL_218;
        if ( sub_9867B0((__int64)&v187) )
        {
          v130 = -1;
        }
        else
        {
          v130 = sub_9AE470(*(_QWORD *)(v6 - 64), &v187, a3 + 1, a4);
          if ( v130 == 1 )
            goto LABEL_176;
        }
        v171 = v130;
        v131 = sub_9867B0((__int64)&v189);
        v129 = v171;
        if ( v131 )
          goto LABEL_218;
        v132 = sub_9AE470(*(_QWORD *)(v6 - 32), &v189, a3 + 1, a4);
        v129 = v171;
        if ( v171 > v132 )
          v129 = v132;
        if ( v129 != 1 )
        {
LABEL_218:
          v184 = v129;
          sub_969240(v189.m128i_i64);
          sub_969240((__int64 *)&v187);
          v16 = v184;
          return v16 + v177;
        }
LABEL_176:
        sub_969240(v189.m128i_i64);
        sub_969240((__int64 *)&v187);
        v14 = *(_BYTE *)v6;
        v166 = 1;
        goto LABEL_11;
      default:
        goto LABEL_10;
    }
  }
}
