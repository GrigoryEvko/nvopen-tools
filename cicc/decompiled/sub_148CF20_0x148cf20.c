// Function: sub_148CF20
// Address: 0x148cf20
//
__int64 __fastcall sub_148CF20(__int64 a1, __int64 a2, __m128i a3, __m128i a4)
{
  unsigned __int8 v6; // al
  __int64 v7; // r8
  __int64 v8; // rax
  __int64 v9; // rdi
  __int64 v10; // rsi
  unsigned int v11; // ecx
  __int64 *v12; // rdx
  __int64 v13; // rbx
  unsigned __int8 v14; // dl
  __int64 result; // rax
  char v16; // r13
  __int64 v17; // r12
  __int64 v18; // rax
  __int64 v19; // rsi
  unsigned int v20; // eax
  __m128i v21; // xmm4
  __m128i v22; // xmm5
  __int64 v23; // r12
  __int64 v24; // rax
  __int64 v25; // r12
  __int64 v26; // rax
  __int64 v27; // r13
  unsigned __int8 v28; // bl
  unsigned int v29; // r12d
  unsigned int v30; // r15d
  _QWORD *v31; // rdx
  __int64 v32; // rax
  __int64 v33; // r12
  __int64 v34; // r12
  __int64 v35; // rax
  __int64 v36; // rbx
  __int64 v37; // r13
  unsigned __int64 v38; // r12
  unsigned __int64 v39; // rbx
  __int64 v40; // rax
  __int64 v41; // r8
  int v42; // eax
  int v43; // eax
  __int64 *v44; // rax
  __int64 v45; // rcx
  _QWORD *v46; // rax
  __int64 v47; // r15
  __int64 v48; // rax
  __int64 v49; // rax
  __int64 v50; // r12
  __int64 v51; // rbx
  unsigned int v52; // eax
  unsigned int v53; // r12d
  __int64 v54; // r12
  int v55; // eax
  unsigned __int64 v56; // rax
  unsigned __int64 v57; // rax
  unsigned int v58; // eax
  int v59; // eax
  unsigned __int32 v60; // edx
  __int64 v61; // rax
  __int64 v62; // rax
  __int64 v63; // rbx
  __int64 v64; // rax
  __int64 v65; // r15
  __int64 *v66; // rax
  unsigned int v67; // ecx
  __int64 v68; // r12
  __int64 v69; // rax
  __int64 v70; // r12
  __int64 v71; // r15
  __int64 v72; // rax
  __int64 v73; // rax
  __int64 v74; // rax
  __int64 v75; // rax
  __int64 v76; // r12
  __int64 v77; // r13
  unsigned int v78; // ebx
  __int64 v79; // rax
  __int16 v80; // dx
  __int64 v81; // rsi
  __int64 v82; // r15
  unsigned int v83; // eax
  __int64 v84; // r12
  _QWORD *v85; // rbx
  bool v86; // al
  __int64 v87; // r13
  char v88; // al
  __int64 v89; // rdx
  __int64 v90; // rax
  __int64 v91; // rax
  int v92; // eax
  unsigned int v93; // r13d
  __int64 v94; // rax
  __int64 v95; // rdx
  __int64 v96; // rcx
  unsigned int v97; // r8d
  __int64 v98; // rax
  __int64 v99; // rax
  __int64 v100; // rsi
  __int64 v101; // rax
  __int64 *v102; // rax
  __int64 v103; // r12
  __int64 *v104; // rax
  __int64 v105; // rax
  __int64 v106; // r12
  __int64 *v107; // rax
  __int64 v108; // rax
  __int64 *v109; // rax
  __int64 v110; // r12
  __int64 *v111; // rax
  __int64 v112; // rax
  __int64 **v113; // rax
  __int64 v114; // rax
  __m128i v115; // xmm2
  __m128i v116; // xmm3
  __m128i v117; // xmm6
  __m128i v118; // xmm7
  int v119; // edx
  int v120; // r10d
  __int64 *v121; // rdi
  bool v122; // al
  __int64 v123; // rbx
  __int64 v124; // rax
  __int64 v125; // r12
  __int64 *v126; // rax
  __int64 v127; // rax
  __int64 v128; // r12
  __int64 v129; // rax
  unsigned int v130; // r15d
  __int64 v131; // rbx
  __int64 v132; // rax
  __int64 v133; // r13
  __int64 v134; // rax
  __int64 v135; // r12
  __int64 v136; // rax
  __int64 v137; // rax
  unsigned int v138; // ebx
  __int64 v139; // rax
  __int64 v140; // rax
  __int64 v141; // rax
  __int64 v142; // rax
  bool v143; // [rsp+0h] [rbp-170h]
  __int64 v144; // [rsp+0h] [rbp-170h]
  unsigned int v145; // [rsp+8h] [rbp-168h]
  __int64 v146; // [rsp+28h] [rbp-148h]
  __int64 v147; // [rsp+28h] [rbp-148h]
  unsigned int v148; // [rsp+28h] [rbp-148h]
  __int64 v149; // [rsp+30h] [rbp-140h]
  __int64 v150; // [rsp+30h] [rbp-140h]
  int v151; // [rsp+30h] [rbp-140h]
  __int64 v152; // [rsp+38h] [rbp-138h]
  unsigned int v153; // [rsp+38h] [rbp-138h]
  int v154; // [rsp+38h] [rbp-138h]
  unsigned int v155; // [rsp+40h] [rbp-130h]
  __int64 v156; // [rsp+40h] [rbp-130h]
  __int64 v157; // [rsp+40h] [rbp-130h]
  __int64 v158; // [rsp+48h] [rbp-128h]
  __int64 v159; // [rsp+48h] [rbp-128h]
  __int64 v160; // [rsp+48h] [rbp-128h]
  __int64 v161; // [rsp+48h] [rbp-128h]
  __int64 v162; // [rsp+48h] [rbp-128h]
  __int64 v163; // [rsp+48h] [rbp-128h]
  __int64 v164; // [rsp+48h] [rbp-128h]
  unsigned __int64 v165; // [rsp+50h] [rbp-120h] BYREF
  unsigned int v166; // [rsp+58h] [rbp-118h]
  __int64 v167; // [rsp+60h] [rbp-110h] BYREF
  int v168; // [rsp+68h] [rbp-108h]
  __int64 v169; // [rsp+70h] [rbp-100h] BYREF
  int v170; // [rsp+78h] [rbp-F8h]
  unsigned __int64 v171; // [rsp+80h] [rbp-F0h] BYREF
  unsigned int v172; // [rsp+88h] [rbp-E8h]
  unsigned __int64 v173; // [rsp+90h] [rbp-E0h] BYREF
  unsigned int v174; // [rsp+98h] [rbp-D8h]
  unsigned __int64 v175; // [rsp+A0h] [rbp-D0h] BYREF
  unsigned int v176; // [rsp+A8h] [rbp-C8h]
  __m128i v177; // [rsp+B0h] [rbp-C0h] BYREF
  __m128i v178; // [rsp+C0h] [rbp-B0h]
  __int64 v179; // [rsp+D0h] [rbp-A0h]
  char v180; // [rsp+D8h] [rbp-98h]
  __int64 *v181; // [rsp+E0h] [rbp-90h] BYREF
  __int64 v182; // [rsp+E8h] [rbp-88h]
  __int64 v183[4]; // [rsp+F0h] [rbp-80h] BYREF
  __m128i v184; // [rsp+110h] [rbp-60h] BYREF
  __m128i v185; // [rsp+120h] [rbp-50h] BYREF
  __int64 v186; // [rsp+130h] [rbp-40h]
  char v187; // [rsp+138h] [rbp-38h]

  v6 = *(_BYTE *)(a2 + 16);
  if ( v6 <= 0x17u )
  {
    switch ( v6 )
    {
      case 0xDu:
        return sub_145CE20(a1, a2);
      case 0xFu:
        return sub_145CF80(a1, *(_QWORD *)a2, 0, 0);
      case 1u:
        __asm { jmp     rax }
        break;
    }
    if ( v6 == 5 )
    {
      v7 = *(_QWORD *)(a1 + 56);
      goto LABEL_6;
    }
    return sub_145DC80(a1, a2);
  }
  v7 = *(_QWORD *)(a1 + 56);
  v8 = *(unsigned int *)(v7 + 48);
  if ( !(_DWORD)v8 )
    return sub_145DC80(a1, a2);
  v9 = *(_QWORD *)(a2 + 40);
  v10 = *(_QWORD *)(v7 + 32);
  v11 = (v8 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
  v12 = (__int64 *)(v10 + 16LL * v11);
  v13 = *v12;
  if ( v9 != *v12 )
  {
    v119 = 1;
    while ( v13 != -8 )
    {
      v120 = v119 + 1;
      v11 = (v8 - 1) & (v119 + v11);
      v12 = (__int64 *)(v10 + 16LL * v11);
      v13 = *v12;
      if ( v9 == *v12 )
        goto LABEL_4;
      v119 = v120;
    }
    return sub_145DC80(a1, a2);
  }
LABEL_4:
  if ( v12 == (__int64 *)(v10 + 16 * v8) || !v12[1] )
    return sub_145DC80(a1, a2);
LABEL_6:
  sub_1455040((__int64)&v177, a2, v7);
  if ( v180 )
  {
    switch ( v177.m128i_i32[0] )
    {
      case 0xB:
        v181 = v183;
        v182 = 0x400000000LL;
LABEL_93:
        v81 = v179;
        if ( v179 )
          goto LABEL_94;
        do
        {
          do
          {
            if ( v177.m128i_i32[0] == 13 )
            {
              v114 = sub_146F1B0(a1, v178.m128i_i64[0]);
              v184.m128i_i64[0] = sub_1480620(a1, v114, 0);
            }
            else
            {
              v184.m128i_i64[0] = sub_146F1B0(a1, v178.m128i_i64[0]);
            }
            sub_1458920((__int64)&v181, &v184);
            sub_1455040((__int64)&v184, v177.m128i_i64[1], *(_QWORD *)(a1 + 56));
            if ( !v187 || ((v184.m128i_i32[0] - 11) & 0xFFFFFFFD) != 0 )
            {
              v175 = sub_146F1B0(a1, v177.m128i_i64[1]);
              sub_1458920((__int64)&v181, &v175);
              goto LABEL_145;
            }
            if ( !v180 )
            {
              v115 = _mm_loadu_si128(&v184);
              v116 = _mm_loadu_si128(&v185);
              v180 = 1;
              v177 = v115;
              v179 = v186;
              v178 = v116;
              goto LABEL_93;
            }
            a3 = _mm_loadu_si128(&v184);
            a4 = _mm_loadu_si128(&v185);
            v179 = v186;
            v81 = v186;
            v177 = a3;
            v178 = a4;
          }
          while ( !v186 );
LABEL_94:
          v184.m128i_i64[0] = sub_14646A0(a1, v81);
          if ( v184.m128i_i64[0] )
            goto LABEL_169;
          v82 = sub_146F1B0(a1, v178.m128i_i64[0]);
          v83 = sub_1471280(a1, v179);
        }
        while ( !v83 );
        v138 = v83;
        v139 = sub_146F1B0(a1, v177.m128i_i64[1]);
        if ( v177.m128i_i32[0] != 13 )
        {
          v184.m128i_i64[0] = sub_13A5B00(a1, v139, v82, v138, 0);
LABEL_169:
          sub_1458920((__int64)&v181, &v184);
          goto LABEL_145;
        }
        v184.m128i_i64[0] = sub_14806B0(a1, v139, v82, v138, 0);
        sub_1458920((__int64)&v181, &v184);
LABEL_145:
        result = (__int64)sub_147DD40(a1, (__int64 *)&v181, 0, 0, a3, a4);
        v121 = v181;
        if ( v181 != v183 )
          goto LABEL_146;
        return result;
      case 0xD:
        v16 = 0;
        if ( v179 )
          v16 = sub_1471280(a1, v179);
        v17 = sub_146F1B0(a1, v178.m128i_i64[0]);
        v18 = sub_146F1B0(a1, v177.m128i_i64[1]);
        return sub_14806B0(a1, v18, v17, v16, 0);
      case 0xF:
        v181 = v183;
        v182 = 0x400000000LL;
LABEL_25:
        v19 = v179;
        if ( v179 )
          goto LABEL_26;
        while ( 1 )
        {
          do
          {
            v184.m128i_i64[0] = sub_146F1B0(a1, v178.m128i_i64[0]);
            sub_1458920((__int64)&v181, &v184);
            sub_1455040((__int64)&v184, v177.m128i_i64[1], *(_QWORD *)(a1 + 56));
            if ( !v187 || v184.m128i_i32[0] != 15 )
            {
              v175 = sub_146F1B0(a1, v177.m128i_i64[1]);
              sub_1458920((__int64)&v181, &v175);
              goto LABEL_148;
            }
            if ( !v180 )
            {
              v117 = _mm_loadu_si128(&v184);
              v118 = _mm_loadu_si128(&v185);
              v180 = 1;
              v177 = v117;
              v179 = v186;
              v178 = v118;
              goto LABEL_25;
            }
            v21 = _mm_loadu_si128(&v184);
            v22 = _mm_loadu_si128(&v185);
            v179 = v186;
            v19 = v186;
            v177 = v21;
            v178 = v22;
          }
          while ( !v186 );
LABEL_26:
          v184.m128i_i64[0] = sub_14646A0(a1, v19);
          if ( v184.m128i_i64[0] )
            break;
          v20 = sub_1471280(a1, v179);
          if ( v20 )
          {
            v130 = v20;
            v131 = sub_146F1B0(a1, v178.m128i_i64[0]);
            v132 = sub_146F1B0(a1, v177.m128i_i64[1]);
            v184.m128i_i64[0] = sub_13A5B60(a1, v132, v131, v130, 0);
            break;
          }
        }
        sub_1458920((__int64)&v181, &v184);
LABEL_148:
        result = sub_147EE30((_QWORD *)a1, &v181, 0, 0, a3, a4);
        v121 = v181;
        if ( v181 != v183 )
        {
LABEL_146:
          v163 = result;
          _libc_free((unsigned __int64)v121);
          return v163;
        }
        return result;
      case 0x11:
        v23 = sub_146F1B0(a1, v178.m128i_i64[0]);
        v24 = sub_146F1B0(a1, v177.m128i_i64[1]);
        return sub_1483CF0((_QWORD *)a1, v24, v23, a3, a4);
      case 0x14:
        v25 = sub_146F1B0(a1, v178.m128i_i64[0]);
        v26 = sub_146F1B0(a1, v177.m128i_i64[1]);
        return sub_1484870((_QWORD *)a1, v26, v25, a3, a4);
      case 0x17:
        v27 = v178.m128i_i64[0];
        v28 = *(_BYTE *)(v178.m128i_i64[0] + 16);
        if ( v28 == 13 )
        {
          v29 = *(_DWORD *)(*(_QWORD *)v178.m128i_i64[0] + 8LL) >> 8;
          if ( !sub_13D0480(v178.m128i_i64[0] + 24, v29) )
            break;
          v30 = 0;
          v158 = v179;
          if ( v179 && sub_13D0480(v27 + 24, v29 - 1) )
            v30 = sub_1471280(a1, v158);
          v31 = *(_QWORD **)(v27 + 24);
          if ( *(_DWORD *)(v27 + 32) > 0x40u )
            v31 = (_QWORD *)*v31;
          sub_1455760((__int64)&v184, v29, (unsigned int)v31);
          v32 = sub_15E0530(*(_QWORD *)(a1 + 24));
          v33 = sub_159C0E0(v32, &v184);
          sub_135E100(v184.m128i_i64);
          v34 = sub_146F1B0(a1, v33);
          v35 = sub_146F1B0(a1, v177.m128i_i64[1]);
          return sub_13A5B60(a1, v35, v34, v30, 0);
        }
        if ( *(_BYTE *)(v177.m128i_i64[1] + 16) == 13 )
        {
          v164 = v179;
          if ( *(_BYTE *)(v179 + 16) > 0x17u )
          {
            v122 = sub_1455000(v177.m128i_i64[1] + 24);
            if ( v28 > 0x17u && !v122 )
            {
              v123 = *(_QWORD *)(v27 + 8);
              if ( v123 )
              {
                while ( 1 )
                {
                  v124 = sub_1648700(v123);
                  v125 = v124;
                  if ( *(_BYTE *)(v124 + 16) == 47 )
                  {
                    v126 = (__int64 *)sub_13CF970(v124);
                    if ( v27 == v126[3] )
                    {
                      v127 = *v126;
                      if ( *(_BYTE *)(v127 + 16) == 13 && sub_1455000(v127 + 24) )
                        break;
                    }
                  }
                  v123 = *(_QWORD *)(v123 + 8);
                  if ( !v123 )
                    goto LABEL_9;
                }
                if ( (unsigned __int8)sub_15CCEE0(*(_QWORD *)(a1 + 56), v125, v164) )
                {
                  v128 = sub_146F1B0(a1, v125);
                  v129 = sub_146F1B0(a1, v177.m128i_i64[1]);
                  return sub_13A5B60(a1, v129, v128, 0, 0);
                }
              }
            }
          }
        }
        break;
      case 0x19:
        v36 = v178.m128i_i64[0];
        if ( *(_BYTE *)(v178.m128i_i64[0] + 16) != 13 )
          break;
        v37 = *(_QWORD *)v177.m128i_i64[1];
        v159 = v178.m128i_i64[0] + 24;
        v38 = sub_1456C90(a1, *(_QWORD *)v177.m128i_i64[1]);
        if ( !sub_13D0480(v36 + 24, v38) )
          break;
        if ( sub_13D01C0(v159) )
          return sub_146F1B0(a1, v177.m128i_i64[1]);
        if ( *(_DWORD *)(v36 + 32) <= 0x40u )
          v39 = *(_QWORD *)(v36 + 24);
        else
          v39 = **(_QWORD **)(v36 + 24);
        v40 = sub_15E0530(*(_QWORD *)(a1 + 24));
        v41 = sub_1644900(v40, (unsigned int)(v38 - v39));
        v42 = *(unsigned __int8 *)(v177.m128i_i64[1] + 16);
        if ( (unsigned __int8)v42 > 0x17u )
        {
          v43 = v42 - 24;
        }
        else
        {
          if ( (_BYTE)v42 != 5 )
            break;
          v43 = *(unsigned __int16 *)(v177.m128i_i64[1] + 18);
        }
        if ( v43 != 23 )
          break;
        v146 = v41;
        v149 = v177.m128i_i64[1];
        v44 = (__int64 *)sub_13CF970(v177.m128i_i64[1]);
        v152 = sub_146F1B0(a1, *v44);
        v45 = *(_QWORD *)(sub_13CF970(v149) + 24);
        if ( v45 == v178.m128i_i64[0] )
        {
          v141 = sub_14835F0((_QWORD *)a1, v152, v146, 0, a3, a4);
          return sub_147B0D0(a1, v141, v37, 0);
        }
        if ( *(_BYTE *)(v45 + 16) == 13 )
        {
          v150 = v45;
          if ( sub_13D0480(v45 + 24, v38) )
          {
            v46 = *(_QWORD **)(v150 + 24);
            if ( *(_DWORD *)(v150 + 32) > 0x40u )
              v46 = (_QWORD *)*v46;
            if ( v39 < (unsigned __int64)v46 )
            {
              sub_1455760((__int64)&v184, v38 - v39, (_DWORD)v46 - v39);
              v47 = sub_145CF40(a1, (__int64)&v184);
              v48 = sub_14835F0((_QWORD *)a1, v152, v146, 0, a3, a4);
              v49 = sub_13A5B60(a1, v48, v47, 0, 0);
              v160 = sub_147B0D0(a1, v49, v37, 0);
              sub_135E100(v184.m128i_i64);
              return v160;
            }
          }
        }
        break;
      case 0x1A:
        v50 = v178.m128i_i64[0];
        if ( *(_BYTE *)(v178.m128i_i64[0] + 16) != 13 )
          break;
        v51 = v178.m128i_i64[0] + 24;
        if ( sub_13D01C0(v178.m128i_i64[0] + 24) )
          return sub_146F1B0(a1, v50);
        if ( sub_1454FB0(v50 + 24) )
          return sub_146F1B0(a1, v177.m128i_i64[1]);
        v151 = sub_1455840(v50 + 24);
        v52 = sub_1455870((__int64 *)(v50 + 24));
        v53 = *(_DWORD *)(v50 + 32);
        v155 = v52;
        sub_135E0D0((__int64)&v181, v53, 0, 0);
        v153 = v53;
        sub_135E0D0((__int64)v183, v53, 0, 0);
        v54 = *(_QWORD *)(a1 + 48);
        v147 = *(_QWORD *)(a1 + 56);
        v55 = sub_1632FA0(*(_QWORD *)(*(_QWORD *)(a1 + 24) + 40LL));
        sub_14BB090(v177.m128i_i32[2], (unsigned int)&v181, v55, 0, v54, 0, v147, 0);
        v148 = v153 - (v155 + v151);
        sub_135E0D0((__int64)&v184, v153, 0, 0);
        if ( v148 )
        {
          if ( v148 > 0x40 )
          {
            sub_16A5260(&v184, 0, v148);
          }
          else
          {
            v56 = 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v155 + (unsigned __int8)v151 - (unsigned __int8)v153 + 64);
            if ( v184.m128i_i32[2] > 0x40u )
              *(_QWORD *)v184.m128i_i64[0] |= v56;
            else
              v184.m128i_i64[0] |= v56;
          }
        }
        sub_13A38D0((__int64)&v165, (__int64)&v184);
        if ( v166 > 0x40 )
        {
          sub_16A7DC0(&v165, v155);
        }
        else
        {
          v57 = 0;
          if ( v155 != v166 )
            v57 = (v165 << v155) & (0xFFFFFFFFFFFFFFFFLL >> -(char)v166);
          v165 = v57;
        }
        sub_135E100(v184.m128i_i64);
        if ( !(v155 | v151) )
          goto LABEL_8;
        sub_13A38D0((__int64)&v171, (__int64)&v181);
        sub_13D0570((__int64)&v171);
        v58 = v172;
        v172 = 0;
        v174 = v58;
        v173 = v171;
        sub_13A38D0((__int64)&v167, v51);
        sub_13D0570((__int64)&v167);
        v59 = v168;
        v60 = v174;
        v168 = 0;
        v170 = v59;
        v169 = v167;
        if ( v174 <= 0x40 )
        {
          v61 = v173 & v167;
          v174 = 0;
          v173 &= v167;
LABEL_73:
          v62 = v165 & v61;
          v175 = v62;
          goto LABEL_74;
        }
        sub_16A8890(&v173, &v169);
        v60 = v174;
        v61 = v173;
        v174 = 0;
        v176 = v60;
        v175 = v173;
        if ( v60 <= 0x40 )
          goto LABEL_73;
        sub_16A8890(&v175, &v165);
        v60 = v176;
        v62 = v175;
LABEL_74:
        v184.m128i_i32[2] = v60;
        v184.m128i_i64[0] = v62;
        v176 = 0;
        v143 = sub_13D01C0((__int64)&v184);
        sub_135E100(v184.m128i_i64);
        sub_135E100((__int64 *)&v175);
        sub_135E100(&v169);
        sub_135E100(&v167);
        sub_135E100((__int64 *)&v173);
        sub_135E100((__int64 *)&v171);
        if ( !v143 )
        {
LABEL_8:
          sub_135E100((__int64 *)&v165);
          sub_135E100(v183);
          sub_135E100((__int64 *)&v181);
          break;
        }
        sub_1455760((__int64)&v184, v153, v155);
        v63 = sub_145CF40(a1, (__int64)&v184);
        sub_135E100(v184.m128i_i64);
        v64 = sub_146F1B0(a1, v177.m128i_i64[1]);
        v65 = v64;
        if ( *(_WORD *)(v64 + 24) != 5 )
          goto LABEL_174;
        v66 = *(__int64 **)(v64 + 32);
        if ( *(_WORD *)(*v66 + 24) )
          goto LABEL_174;
        v144 = *v66;
        v67 = sub_1455870((__int64 *)(*(_QWORD *)(*v66 + 32) + 24LL));
        if ( v155 <= v67 )
          v67 = v155;
        v145 = v67;
        sub_1455760((__int64)&v173, v153, v155 - v67);
        v184.m128i_i64[0] = (__int64)&v185;
        v184.m128i_i64[1] = 0x400000000LL;
        sub_13A38D0((__int64)&v175, *(_QWORD *)(v144 + 32) + 24LL);
        if ( v176 > 0x40 )
        {
          sub_16A8110(&v175, v145);
        }
        else if ( v145 == v176 )
        {
          v175 = 0;
        }
        else
        {
          v175 >>= v145;
        }
        v171 = sub_145CF40(a1, (__int64)&v175);
        sub_1458920((__int64)&v184, &v171);
        sub_135E100((__int64 *)&v175);
        sub_145C5B0(
          (__int64)&v184,
          (_BYTE *)(*(_QWORD *)(v65 + 32) + 8LL),
          (_BYTE *)(*(_QWORD *)(v65 + 32) + 8LL * *(_QWORD *)(v65 + 40)));
        v68 = sub_147EE30((_QWORD *)a1, (__int64 **)&v184, *(_WORD *)(v65 + 26) & 7, 0, a3, a4);
        v69 = sub_145CF40(a1, (__int64)&v173);
        v70 = sub_1483CF0((_QWORD *)a1, v68, v69, a3, a4);
        if ( (__m128i *)v184.m128i_i64[0] != &v185 )
          _libc_free(v184.m128i_u64[0]);
        sub_135E100((__int64 *)&v173);
        if ( !v70 )
LABEL_174:
          v70 = sub_1483CF0((_QWORD *)a1, v65, v63, a3, a4);
        v71 = *(_QWORD *)v177.m128i_i64[1];
        v72 = sub_15E0530(*(_QWORD *)(a1 + 24));
        v73 = sub_1644900(v72, v148);
        v74 = sub_14835F0((_QWORD *)a1, v70, v73, 0, a3, a4);
        v75 = sub_14747F0(a1, v74, v71, 0);
        v156 = sub_13A5B60(a1, v75, v63, 0, 0);
        sub_135E100((__int64 *)&v165);
        sub_135E100(v183);
        sub_135E100((__int64 *)&v181);
        return v156;
      case 0x1B:
        v76 = v178.m128i_i64[0];
        if ( *(_BYTE *)(v178.m128i_i64[0] + 16) != 13 )
          break;
        v77 = sub_146F1B0(a1, v177.m128i_i64[1]);
        v78 = sub_14687F0(a1, v77);
        if ( v78 < *(_DWORD *)(v76 + 32) - (unsigned int)sub_1455840(v76 + 24) )
          break;
        v79 = sub_146F1B0(a1, v76);
        result = sub_13A5B00(a1, v77, v79, 0, 0);
        if ( *(_WORD *)(result + 24) == 7 )
        {
          v80 = *(_WORD *)(v77 + 26) & 7;
          if ( (*(_WORD *)(v77 + 26) & 6) != 0 )
            v80 = *(_WORD *)(v77 + 26) & 6 | 1;
          *(_WORD *)(result + 26) |= v80;
        }
        return result;
      case 0x1C:
        v84 = v178.m128i_i64[0];
        if ( *(_BYTE *)(v178.m128i_i64[0] + 16) != 13 )
          break;
        v85 = (_QWORD *)(v178.m128i_i64[0] + 24);
        v86 = sub_1454FB0(v178.m128i_i64[0] + 24);
        v87 = v177.m128i_i64[1];
        if ( v86 )
        {
          v140 = sub_146F1B0(a1, v177.m128i_i64[1]);
          return sub_1480810(a1, v140);
        }
        v88 = *(_BYTE *)(v177.m128i_i64[1] + 16);
        if ( (unsigned __int8)(v88 - 35) > 0x11u )
          break;
        v89 = *(_QWORD *)(v177.m128i_i64[1] - 24);
        if ( *(_BYTE *)(v89 + 16) != 13 )
          break;
        if ( v88 != 50 )
          break;
        if ( !sub_1455820(v89 + 24, (_QWORD *)(v84 + 24)) )
          break;
        v90 = sub_146F1B0(a1, v87);
        if ( *(_WORD *)(v90 + 24) != 2 )
          break;
        v157 = *(_QWORD *)(v90 + 32);
        v161 = *(_QWORD *)v177.m128i_i64[1];
        v91 = sub_1456040(v157);
        v92 = sub_1456C90(a1, v91);
        v93 = *(_DWORD *)(v84 + 32);
        if ( v93 <= 0x40 )
        {
          if ( *(_QWORD *)(v84 + 24) == 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v92) )
          {
LABEL_186:
            v142 = sub_1480810(a1, v157);
            return sub_14747F0(a1, v142, v161, 0);
          }
        }
        else
        {
          v154 = v92;
          if ( (unsigned int)sub_16A58F0(v84 + 24) == v92 && v93 == v154 + (unsigned int)sub_16A57B0(v84 + 24) )
            goto LABEL_186;
        }
        sub_16A5A50(&v181, v84 + 24);
        v94 = sub_1456C90(a1, v161);
        sub_16A5C50(&v184, &v181, v94);
        if ( sub_1455820((__int64)&v184, v85)
          && (unsigned __int8)sub_13CFF40((__int64 *)&v181, (__int64)v85, v95, v96, v97) )
        {
          sub_135E100(v184.m128i_i64);
          v98 = sub_145CF40(a1, (__int64)&v181);
          v99 = sub_13A5B00(a1, v157, v98, 0, 0);
          v162 = sub_14747F0(a1, v99, v161, 0);
          sub_135E100((__int64 *)&v181);
          return v162;
        }
        sub_135E100(v184.m128i_i64);
        sub_135E100((__int64 *)&v181);
        break;
      default:
        break;
    }
  }
LABEL_9:
  v14 = *(_BYTE *)(a2 + 16);
  if ( v14 > 0x17u )
  {
    switch ( v14 )
    {
      case 0x1Du:
      case 0x4Eu:
        v101 = a2 | 4;
        if ( v14 != 78 )
        {
          v101 = a2 & 0xFFFFFFFFFFFFFFFBLL;
          if ( v14 != 29 )
            goto LABEL_119;
        }
        goto LABEL_120;
      case 0x38u:
        return sub_14876A0((_QWORD *)a1, a2, a3, a4);
      case 0x3Cu:
        goto LABEL_125;
      case 0x3Du:
        goto LABEL_124;
      case 0x3Eu:
        goto LABEL_127;
      case 0x45u:
      case 0x46u:
        goto LABEL_115;
      case 0x47u:
        goto LABEL_131;
      case 0x4Du:
        goto LABEL_130;
      case 0x4Fu:
        goto LABEL_122;
      default:
        return sub_145DC80(a1, a2);
    }
  }
  switch ( *(_WORD *)(a2 + 18) )
  {
    case 5:
    case 0x36:
LABEL_119:
      v101 = 0;
LABEL_120:
      v184.m128i_i64[0] = v101;
      v100 = sub_145C750(v184.m128i_i64);
      if ( !v100 )
        return sub_145DC80(a1, a2);
      goto LABEL_19;
    case 0x20:
      return sub_14876A0((_QWORD *)a1, a2, a3, a4);
    case 0x24:
LABEL_125:
      v106 = *(_QWORD *)a2;
      v107 = (__int64 *)sub_13CF970(a2);
      v108 = sub_146F1B0(a1, *v107);
      return sub_14835F0((_QWORD *)a1, v108, v106, 0, a3, a4);
    case 0x25:
LABEL_124:
      v103 = *(_QWORD *)a2;
      v104 = (__int64 *)sub_13CF970(a2);
      v105 = sub_146F1B0(a1, *v104);
      return sub_14747F0(a1, v105, v103, 0);
    case 0x26:
LABEL_127:
      v109 = (__int64 *)sub_13CF970(a2);
      sub_1455040((__int64)&v184, *v109, *(_QWORD *)(a1 + 56));
      if ( v187 && v184.m128i_i32[0] == 13 && v185.m128i_i8[8] )
      {
        v133 = *(_QWORD *)a2;
        v134 = sub_146F1B0(a1, v184.m128i_i64[1]);
        v135 = sub_147B0D0(a1, v134, v133, 0);
        v136 = sub_146F1B0(a1, v185.m128i_i64[0]);
        v137 = sub_147B0D0(a1, v136, v133, 0);
        return sub_14806B0(a1, v135, v137, 4, 0);
      }
      else
      {
        v110 = *(_QWORD *)a2;
        v111 = (__int64 *)sub_13CF970(a2);
        v112 = sub_146F1B0(a1, *v111);
        return sub_147B0D0(a1, v112, v110, 0);
      }
    case 0x2D:
    case 0x2E:
LABEL_115:
      if ( !*(_BYTE *)(a1 + 20) )
        return sub_145DC80(a1, a2);
      goto LABEL_116;
    case 0x2F:
LABEL_131:
      if ( !sub_1456C80(a1, *(_QWORD *)a2) )
        return sub_145DC80(a1, a2);
      v113 = (__int64 **)sub_13CF970(a2);
      if ( !sub_1456C80(a1, **v113) )
        return sub_145DC80(a1, a2);
LABEL_116:
      v100 = *(_QWORD *)sub_13CF970(a2);
LABEL_19:
      result = sub_146F1B0(a1, v100);
      break;
    case 0x35:
LABEL_130:
      result = sub_148CD40((_QWORD *)a1, a2, a3, a4);
      break;
    case 0x37:
LABEL_122:
      if ( v14 <= 0x17u )
        return sub_145DC80(a1, a2);
      v102 = (__int64 *)sub_13CF970(a2);
      result = sub_1482570((_QWORD *)a1, (__int64 *)a2, *v102, v102[3], v102[6], a3, a4);
      break;
    default:
      return sub_145DC80(a1, a2);
  }
  return result;
}
