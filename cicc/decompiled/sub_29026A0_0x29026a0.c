// Function: sub_29026A0
// Address: 0x29026a0
//
__int64 __fastcall sub_29026A0(__int64 a1, __int64 a2, __int64 *a3, unsigned __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rdi
  int v9; // r11d
  _QWORD *v10; // rdx
  __int64 v11; // r9
  unsigned int v12; // ecx
  __int64 *v13; // rax
  __int64 v14; // r8
  _QWORD *v15; // rax
  __int64 v16; // rax
  unsigned __int64 v17; // rdx
  __int64 *v18; // r14
  unsigned int v19; // r13d
  __int64 v20; // rdi
  char v21; // dh
  __int64 v22; // r15
  char v23; // al
  __int64 v24; // rdx
  _QWORD *v25; // rax
  _QWORD *v26; // rbx
  int v27; // esi
  int v28; // ecx
  _QWORD *v29; // rbx
  _QWORD *v30; // r13
  int v31; // r8d
  unsigned int v32; // edx
  __int64 v33; // rcx
  __int64 v34; // rax
  __int64 *v35; // r15
  unsigned int v36; // r10d
  __int64 v37; // rdi
  char v38; // dh
  __int64 v39; // r8
  char v40; // al
  __int64 v41; // rdx
  _QWORD *v42; // rax
  _QWORD *v43; // r12
  int v44; // esi
  __int64 v45; // rdi
  int v46; // r15d
  __int64 *v47; // rdx
  __int64 v48; // r9
  unsigned int v49; // ecx
  __int64 *v50; // rax
  __int64 v51; // r8
  _QWORD *v52; // rax
  __int64 v53; // rax
  unsigned __int64 v54; // rdx
  __int64 v55; // rbx
  __int64 v56; // r9
  _QWORD *v57; // rbx
  int v58; // r11d
  __int64 *v59; // rdi
  unsigned int v60; // ecx
  __int64 *v61; // rdx
  __int64 v62; // r8
  __int64 v63; // r15
  __int64 v64; // r12
  _QWORD *v65; // rdi
  __int64 v66; // rax
  int v67; // esi
  __int64 v68; // r13
  int v69; // ecx
  int v71; // ecx
  __int64 v72; // rsi
  char v73; // cl
  unsigned __int8 v74; // dl
  __int64 *v75; // rax
  __int64 *v76; // r15
  __int64 *v77; // r13
  __int64 v78; // r8
  __int64 *v79; // rdx
  __int64 v80; // rax
  __int64 v81; // r12
  unsigned int v82; // edx
  unsigned __int64 v83; // rdx
  __int64 *v84; // rax
  __int64 *v85; // rdx
  __int64 v86; // r15
  __int64 *v87; // r14
  unsigned int v88; // eax
  __int64 v89; // r8
  __int64 v90; // r9
  __int64 v91; // rbx
  unsigned __int64 i; // rax
  _BYTE *v93; // r12
  __int64 *v94; // r13
  __int64 v95; // rsi
  __int64 *v96; // r8
  __int64 *v97; // rax
  __int64 v98; // rsi
  __int64 *v99; // rdx
  __int64 v100; // r12
  _QWORD *v101; // rax
  __int64 v102; // rbx
  _BYTE *v103; // r14
  int v104; // edx
  __int64 v105; // rbx
  __int64 v106; // r12
  __int64 v107; // rax
  __int64 v108; // r11
  __int64 v109; // rdx
  unsigned __int64 v110; // rax
  unsigned __int64 v111; // r8
  _QWORD *v112; // rax
  _QWORD *v113; // r13
  __int64 v114; // r12
  __int64 v115; // rdx
  __int64 v116; // rdx
  char v117; // bl
  _QWORD *v118; // rax
  __int64 v119; // r9
  _QWORD *v120; // r12
  char v121; // dh
  __int64 v122; // rsi
  char v123; // al
  __int64 v124; // rbx
  __int64 *v125; // rcx
  __int64 v126; // rax
  __int64 v127; // rax
  unsigned __int8 v128; // dl
  char v129; // dh
  __int64 v130; // rsi
  char v131; // cl
  char v132; // dh
  __int64 *v133; // r11
  __int64 v134; // r12
  __int64 v135; // rax
  __int64 v136; // [rsp+8h] [rbp-968h]
  __int64 v137; // [rsp+10h] [rbp-960h]
  __int64 v138; // [rsp+28h] [rbp-948h]
  __int64 *v139; // [rsp+30h] [rbp-940h]
  __int64 v140; // [rsp+38h] [rbp-938h]
  __int64 *v141; // [rsp+38h] [rbp-938h]
  __int64 v143; // [rsp+48h] [rbp-928h]
  __int64 *v144; // [rsp+48h] [rbp-928h]
  __int64 v145; // [rsp+50h] [rbp-920h]
  __int64 v147; // [rsp+58h] [rbp-918h]
  __int64 v148; // [rsp+60h] [rbp-910h]
  __int64 *v149; // [rsp+60h] [rbp-910h]
  unsigned int v151; // [rsp+68h] [rbp-908h]
  __int64 v152; // [rsp+68h] [rbp-908h]
  __int64 *v153; // [rsp+70h] [rbp-900h]
  __int64 v154; // [rsp+70h] [rbp-900h]
  __int64 v155; // [rsp+70h] [rbp-900h]
  __int64 v156; // [rsp+78h] [rbp-8F8h]
  __int64 v157; // [rsp+78h] [rbp-8F8h]
  _QWORD *v158; // [rsp+78h] [rbp-8F8h]
  __int64 v159; // [rsp+78h] [rbp-8F8h]
  __int64 v160; // [rsp+88h] [rbp-8E8h] BYREF
  __int64 v161; // [rsp+90h] [rbp-8E0h] BYREF
  __int64 *v162; // [rsp+98h] [rbp-8D8h]
  __int64 v163; // [rsp+A0h] [rbp-8D0h]
  unsigned int v164; // [rsp+A8h] [rbp-8C8h]
  __int64 v165[4]; // [rsp+B0h] [rbp-8C0h] BYREF
  __int16 v166; // [rsp+D0h] [rbp-8A0h]
  void *base; // [rsp+E0h] [rbp-890h] BYREF
  __int64 v168; // [rsp+E8h] [rbp-888h]
  __int64 v169; // [rsp+F0h] [rbp-880h] BYREF
  unsigned __int64 v170; // [rsp+F8h] [rbp-878h] BYREF
  __int64 v171; // [rsp+100h] [rbp-870h]
  __int64 v172; // [rsp+108h] [rbp-868h]
  void *src; // [rsp+2F0h] [rbp-680h] BYREF
  __int64 v174; // [rsp+2F8h] [rbp-678h]
  _BYTE v175[1648]; // [rsp+300h] [rbp-670h] BYREF

  src = v175;
  v148 = a5;
  v161 = 0;
  v162 = 0;
  v163 = 0;
  v164 = 0;
  v174 = 0xC800000000LL;
  if ( a4 > 0xC8 )
    sub_C8D5F0((__int64)&src, v175, a4, 8u, a5, a6);
  v145 = sub_B2BEC0(a1);
  v153 = &a3[a4];
  if ( a3 != v153 )
  {
    while ( 1 )
    {
      v165[0] = *a3;
      v18 = *(__int64 **)(v165[0] + 8);
      v19 = *(_DWORD *)(v145 + 4);
      LOWORD(v171) = 257;
      v20 = *(_QWORD *)(a1 + 80);
      if ( v20 )
        v20 -= 24;
      v22 = sub_AA4FF0(v20);
      v23 = 0;
      if ( v22 )
        v23 = v21;
      v24 = 1;
      BYTE1(v24) = v23;
      v156 = v24;
      v25 = sub_BD2C40(80, 1u);
      v26 = v25;
      if ( v25 )
        sub_B4CE50((__int64)v25, v18, v19, (__int64)&base, v22, v156);
      v27 = v164;
      if ( !v164 )
        break;
      v8 = v165[0];
      v9 = 1;
      v10 = 0;
      v11 = (__int64)v162;
      v12 = (v164 - 1) & ((LODWORD(v165[0]) >> 9) ^ (LODWORD(v165[0]) >> 4));
      v13 = &v162[2 * v12];
      v14 = *v13;
      if ( v165[0] != *v13 )
      {
        while ( v14 != -4096 )
        {
          if ( !v10 && v14 == -8192 )
            v10 = v13;
          v12 = (v164 - 1) & (v9 + v12);
          v13 = &v162[2 * v12];
          v14 = *v13;
          if ( v165[0] == *v13 )
            goto LABEL_6;
          ++v9;
        }
        if ( !v10 )
          v10 = v13;
        ++v161;
        v28 = v163 + 1;
        base = v10;
        if ( 4 * ((int)v163 + 1) < 3 * v164 )
        {
          v14 = v164 >> 3;
          if ( v164 - HIDWORD(v163) - v28 > (unsigned int)v14 )
            goto LABEL_132;
          goto LABEL_19;
        }
LABEL_18:
        v27 = 2 * v164;
LABEL_19:
        sub_29022E0((__int64)&v161, v27);
        sub_2901330((__int64)&v161, v165, &base);
        v8 = v165[0];
        v10 = base;
        v28 = v163 + 1;
LABEL_132:
        LODWORD(v163) = v28;
        if ( *v10 != -4096 )
          --HIDWORD(v163);
        *v10 = v8;
        v15 = v10 + 1;
        v10[1] = 0;
        goto LABEL_7;
      }
LABEL_6:
      v15 = v13 + 1;
LABEL_7:
      *v15 = v26;
      v16 = (unsigned int)v174;
      v17 = (unsigned int)v174 + 1LL;
      if ( v17 > HIDWORD(v174) )
      {
        sub_C8D5F0((__int64)&src, v175, v17, 8u, v14, v11);
        v16 = (unsigned int)v174;
      }
      ++a3;
      *((_QWORD *)src + v16) = v26;
      LODWORD(v174) = v174 + 1;
      if ( v153 == a3 )
        goto LABEL_20;
    }
    ++v161;
    base = 0;
    goto LABEL_18;
  }
LABEL_20:
  v140 = a5 + 112 * a6;
  if ( v140 != a5 )
  {
    v143 = a5;
    do
    {
      v29 = *(_QWORD **)(v143 + 96);
      v30 = &v29[6 * *(unsigned int *)(v143 + 104)];
      if ( v29 != v30 )
      {
        while ( 1 )
        {
          base = 0;
          v168 = 0;
          v169 = v29[2];
          if ( v169 != 0 && v169 != -4096 && v169 != -8192 )
            sub_BD6050((unsigned __int64 *)&base, *v29 & 0xFFFFFFFFFFFFFFF8LL);
          v170 = 0;
          v171 = 0;
          v172 = v29[5];
          v34 = v172;
          if ( v172 != 0 && v172 != -4096 && v172 != -8192 )
          {
            sub_BD6050(&v170, v29[3] & 0xFFFFFFFFFFFFFFF8LL);
            v34 = v172;
          }
          if ( v164 )
          {
            v31 = 1;
            v32 = (v164 - 1) & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
            v33 = v162[2 * v32];
            if ( v33 == v34 )
            {
LABEL_25:
              if ( v34 == 0 || v34 == -4096 )
                goto LABEL_28;
LABEL_26:
              if ( v34 != -8192 )
                sub_BD60C0(&v170);
              goto LABEL_28;
            }
            while ( v33 != -4096 )
            {
              v32 = (v164 - 1) & (v31 + v32);
              v33 = v162[2 * v32];
              if ( v33 == v34 )
                goto LABEL_25;
              ++v31;
            }
          }
          v160 = v34;
          v35 = *(__int64 **)(v34 + 8);
          v36 = *(_DWORD *)(v145 + 4);
          v166 = 257;
          v37 = *(_QWORD *)(a1 + 80);
          v151 = v36;
          if ( v37 )
            v37 -= 24;
          v39 = sub_AA4FF0(v37);
          v40 = 0;
          v154 = v39;
          if ( v39 )
            v40 = v38;
          v41 = 1;
          BYTE1(v41) = v40;
          v157 = v41;
          v42 = sub_BD2C40(80, 1u);
          v43 = v42;
          if ( v42 )
            sub_B4CE50((__int64)v42, v35, v151, (__int64)v165, v154, v157);
          v44 = v164;
          if ( !v164 )
            break;
          v45 = v160;
          v46 = 1;
          v47 = 0;
          v48 = (__int64)v162;
          v49 = (v164 - 1) & (((unsigned int)v160 >> 9) ^ ((unsigned int)v160 >> 4));
          v50 = &v162[2 * v49];
          v51 = *v50;
          if ( *v50 == v160 )
            goto LABEL_47;
          while ( 1 )
          {
            if ( v51 == -4096 )
            {
              if ( !v47 )
                v47 = v50;
              ++v161;
              v71 = v163 + 1;
              v165[0] = (__int64)v47;
              if ( 4 * ((int)v163 + 1) < 3 * v164 )
              {
                v51 = v164 >> 3;
                if ( v164 - HIDWORD(v163) - v71 > (unsigned int)v51 )
                {
LABEL_110:
                  LODWORD(v163) = v71;
                  if ( *v47 != -4096 )
                    --HIDWORD(v163);
                  *v47 = v45;
                  v52 = v47 + 1;
                  v47[1] = 0;
                  goto LABEL_48;
                }
LABEL_115:
                sub_29022E0((__int64)&v161, v44);
                sub_2901330((__int64)&v161, &v160, v165);
                v45 = v160;
                v47 = (__int64 *)v165[0];
                v71 = v163 + 1;
                goto LABEL_110;
              }
LABEL_114:
              v44 = 2 * v164;
              goto LABEL_115;
            }
            if ( v47 || v51 != -8192 )
              v50 = v47;
            v49 = (v164 - 1) & (v46 + v49);
            v133 = &v162[2 * v49];
            v51 = *v133;
            if ( v160 == *v133 )
              break;
            ++v46;
            v47 = v50;
            v50 = &v162[2 * v49];
          }
          v50 = &v162[2 * v49];
LABEL_47:
          v52 = v50 + 1;
LABEL_48:
          *v52 = v43;
          v53 = (unsigned int)v174;
          v54 = (unsigned int)v174 + 1LL;
          if ( v54 > HIDWORD(v174) )
          {
            sub_C8D5F0((__int64)&src, v175, v54, 8u, v51, v48);
            v53 = (unsigned int)v174;
          }
          *((_QWORD *)src + v53) = v43;
          v34 = v172;
          LODWORD(v174) = v174 + 1;
          if ( v172 != -4096 && v172 != 0 )
            goto LABEL_26;
LABEL_28:
          if ( v169 != -4096 && v169 != 0 && v169 != -8192 )
            sub_BD60C0(&base);
          v29 += 6;
          if ( v30 == v29 )
            goto LABEL_56;
        }
        ++v161;
        v165[0] = 0;
        goto LABEL_114;
      }
LABEL_56:
      v143 += 112;
    }
    while ( v143 != v140 );
    while ( 1 )
    {
      v55 = *(_QWORD *)(v148 + 48);
      v152 = v55;
      sub_29024C0(*(_QWORD *)(v55 + 16), 0, (__int64)&v161);
      if ( *(_BYTE *)v55 == 34 )
        sub_29024C0(*(_QWORD *)(*(_QWORD *)(v148 + 56) + 16LL), 0, (__int64)&v161);
      v57 = *(_QWORD **)(v148 + 96);
      v158 = &v57[6 * *(unsigned int *)(v148 + 104)];
      if ( v57 != v158 )
        break;
LABEL_96:
      if ( byte_5005008 )
      {
        base = &v169;
        v168 = 0x4000000000LL;
        if ( (_DWORD)v163 )
        {
          v75 = v162;
          v76 = &v162[2 * v164];
          if ( v162 != v76 )
          {
            while ( 1 )
            {
              v77 = v75;
              if ( *v75 != -8192 && *v75 != -4096 )
                break;
              v75 += 2;
              if ( v76 == v75 )
                goto LABEL_117;
            }
            if ( v75 != v76 )
            {
              v78 = v75[1];
              v79 = &v169;
              v80 = 0;
              v81 = v78;
              while ( 1 )
              {
                v79[v80] = v81;
                v77 += 2;
                v82 = v168 + 1;
                LODWORD(v168) = v168 + 1;
                if ( v77 == v76 )
                  break;
                while ( *v77 == -8192 || *v77 == -4096 )
                {
                  v77 += 2;
                  if ( v76 == v77 )
                    goto LABEL_117;
                }
                if ( v76 == v77 )
                  break;
                v80 = v82;
                v81 = v77[1];
                v83 = v82 + 1LL;
                if ( v83 > HIDWORD(v168) )
                {
                  sub_C8D5F0((__int64)&base, &v169, v83, 8u, v78, v56);
                  v80 = (unsigned int)v168;
                }
                v79 = (__int64 *)base;
              }
            }
          }
        }
LABEL_117:
        if ( *(_BYTE *)v152 == 34 )
        {
          v130 = sub_AA5190(*(_QWORD *)(v152 - 96));
          if ( v130 )
          {
            v131 = v129;
          }
          else
          {
            v131 = 0;
            v128 = 0;
          }
          sub_28FF080((__int64)&base, v130, v128, v131);
          v72 = sub_AA5190(*(_QWORD *)(v152 - 64));
          if ( v72 )
          {
            v73 = v132;
LABEL_120:
            sub_28FF080((__int64)&base, v72, v74, v73);
            if ( base != &v169 )
              _libc_free((unsigned __int64)base);
            goto LABEL_97;
          }
        }
        else
        {
          v72 = *(_QWORD *)(v152 + 32);
        }
        v73 = 0;
        v74 = 0;
        goto LABEL_120;
      }
LABEL_97:
      sub_C7D6A0(0, 0, 8);
      v148 += 112;
      if ( v148 == v140 )
        goto LABEL_98;
    }
    while ( 1 )
    {
      base = 0;
      v168 = 0;
      v169 = v57[2];
      if ( v169 != 0 && v169 != -4096 && v169 != -8192 )
        sub_BD6050((unsigned __int64 *)&base, *v57 & 0xFFFFFFFFFFFFFFF8LL);
      v170 = 0;
      v171 = 0;
      v172 = v57[5];
      v66 = v172;
      if ( v172 != -4096 && v172 != 0 && v172 != -8192 )
      {
        sub_BD6050(&v170, v57[3] & 0xFFFFFFFFFFFFFFF8LL);
        v66 = v172;
      }
      v67 = v164;
      v160 = v66;
      v68 = v169;
      if ( !v164 )
        break;
      v58 = 1;
      v59 = 0;
      v60 = (v164 - 1) & (((unsigned int)v66 >> 9) ^ ((unsigned int)v66 >> 4));
      v61 = &v162[2 * v60];
      v62 = *v61;
      if ( v66 != *v61 )
      {
        while ( v62 != -4096 )
        {
          if ( !v59 && v62 == -8192 )
            v59 = v61;
          v60 = (v164 - 1) & (v58 + v60);
          v61 = &v162[2 * v60];
          v62 = *v61;
          if ( v66 == *v61 )
            goto LABEL_62;
          ++v58;
        }
        if ( !v59 )
          v59 = v61;
        ++v161;
        v69 = v163 + 1;
        v165[0] = (__int64)v59;
        if ( 4 * ((int)v163 + 1) < 3 * v164 )
        {
          if ( v164 - HIDWORD(v163) - v69 <= v164 >> 3 )
          {
LABEL_81:
            sub_29022E0((__int64)&v161, v67);
            sub_2901330((__int64)&v161, &v160, v165);
            v66 = v160;
            v69 = v163 + 1;
            v59 = (__int64 *)v165[0];
          }
          LODWORD(v163) = v69;
          if ( *v59 != -4096 )
            --HIDWORD(v163);
          *v59 = v66;
          v63 = 0;
          v59[1] = 0;
          goto LABEL_63;
        }
LABEL_80:
        v67 = 2 * v164;
        goto LABEL_81;
      }
LABEL_62:
      v63 = v61[1];
LABEL_63:
      v64 = *(_QWORD *)(v68 + 32);
      v65 = sub_BD2C40(80, unk_3F10A10);
      if ( v65 )
        sub_B4D460((__int64)v65, v68, v63, v64, 0);
      if ( v172 != -4096 && v172 != 0 && v172 != -8192 )
        sub_BD60C0(&v170);
      if ( v169 != -4096 && v169 != 0 && v169 != -8192 )
        sub_BD60C0(&base);
      v57 += 6;
      if ( v158 == v57 )
        goto LABEL_96;
    }
    ++v161;
    v165[0] = 0;
    goto LABEL_80;
  }
LABEL_98:
  if ( (_DWORD)v163 )
  {
    v84 = v162;
    v85 = &v162[2 * v164];
    v141 = v85;
    if ( v162 != v85 )
    {
      while ( 1 )
      {
        v86 = *v84;
        v87 = v84;
        if ( *v84 != -8192 && v86 != -4096 )
          break;
        v84 += 2;
        if ( v85 == v84 )
          goto LABEL_99;
      }
      if ( v85 != v84 )
      {
        while ( 1 )
        {
          v159 = v87[1];
          base = &v169;
          v168 = 0x1400000000LL;
          v88 = sub_BD3960(v86);
          if ( HIDWORD(v168) < v88 )
            sub_C8D5F0((__int64)&base, &v169, v88, 8u, v89, v90);
          v91 = *(_QWORD *)(v86 + 16);
          for ( i = (unsigned int)v168; v91; v91 = *(_QWORD *)(v91 + 8) )
          {
            v93 = *(_BYTE **)(v91 + 24);
            if ( *v93 != 5 )
            {
              if ( i + 1 > HIDWORD(v168) )
              {
                sub_C8D5F0((__int64)&base, &v169, i + 1, 8u, v89, v90);
                i = (unsigned int)v168;
              }
              *((_QWORD *)base + i) = v93;
              i = (unsigned int)(v168 + 1);
              LODWORD(v168) = v168 + 1;
            }
          }
          v94 = (__int64 *)base;
          v95 = i;
          if ( i > 1 )
          {
            qsort(base, (v95 * 8) >> 3, 8u, (__compar_fn_t)sub_28FEA50);
            v94 = (__int64 *)base;
            v95 = (unsigned int)v168;
          }
          v96 = &v94[v95];
          if ( &v94[v95] == v94 )
          {
            LODWORD(v168) = 0;
            goto LABEL_197;
          }
          v97 = v94;
          while ( 1 )
          {
            v99 = v97++;
            if ( v96 == v97 )
              break;
            v98 = *(v97 - 1);
            if ( v98 == *v97 )
            {
              if ( v96 == v99 )
              {
                v97 = v96;
              }
              else
              {
                v125 = v99 + 2;
                if ( v99 + 2 != v96 )
                {
                  while ( 1 )
                  {
                    if ( v98 != *v125 )
                    {
                      v99[1] = *v125;
                      ++v99;
                    }
                    if ( v96 == ++v125 )
                      break;
                    v98 = *v99;
                  }
                  v97 = v99 + 1;
                }
              }
              break;
            }
          }
          LODWORD(v168) = v97 - v94;
          v149 = &v94[(unsigned int)v168];
          if ( v149 != v94 )
            break;
LABEL_197:
          v117 = sub_AE5020(v145, *(_QWORD *)(v86 + 8));
          v118 = sub_BD2C40(80, unk_3F10A10);
          v120 = v118;
          if ( v118 )
            sub_B4D3C0((__int64)v118, v86, v159, 0, v117, v119, 0, 0);
          if ( *(_BYTE *)v86 <= 0x1Cu )
          {
            v127 = v136;
            LOWORD(v127) = 0;
            v136 = v127;
            sub_B43E90((__int64)v120, v159 + 24);
          }
          else if ( *(_BYTE *)v86 == 34 )
          {
            v124 = v138;
            v122 = sub_AA4FF0(*(_QWORD *)(v86 - 96));
            v123 = 0;
            LOBYTE(v124) = 1;
            if ( v122 )
              v123 = v121;
            BYTE1(v124) = v123;
            v138 = v124;
            sub_B44220(v120, v122, v124);
          }
          else
          {
            v126 = v137;
            LOWORD(v126) = 0;
            v137 = v126;
            sub_B43E90((__int64)v120, v86 + 24);
          }
          if ( base != &v169 )
            _libc_free((unsigned __int64)base);
          v87 += 2;
          if ( v87 != v141 )
          {
            while ( 1 )
            {
              v86 = *v87;
              if ( *v87 != -4096 && v86 != -8192 )
                break;
              v87 += 2;
              if ( v141 == v87 )
                goto LABEL_99;
            }
            if ( v87 != v141 )
              continue;
          }
          goto LABEL_99;
        }
        v139 = v87;
LABEL_177:
        while ( 2 )
        {
          v103 = (_BYTE *)*v94;
          if ( *(_BYTE *)*v94 == 84 )
          {
            v104 = *((_DWORD *)v103 + 1);
            v105 = 0;
            if ( (v104 & 0x7FFFFFF) != 0 )
            {
              v144 = v94;
              while ( 1 )
              {
                while ( 1 )
                {
                  v106 = 32 * v105;
                  v107 = *(_QWORD *)(*((_QWORD *)v103 - 1) + 32 * v105);
                  if ( v86 == v107 )
                  {
                    if ( v107 )
                      break;
                  }
LABEL_180:
                  if ( (v104 & 0x7FFFFFFu) <= (unsigned int)++v105 )
                    goto LABEL_195;
                }
                v108 = *(_QWORD *)(v159 + 72);
                v166 = 257;
                v109 = *(_QWORD *)(*((_QWORD *)v103 - 1) + 32LL * *((unsigned int *)v103 + 18) + 8 * v105);
                v110 = *(_QWORD *)(v109 + 48) & 0xFFFFFFFFFFFFFFF8LL;
                if ( v110 == v109 + 48 )
                {
                  v111 = 0;
                }
                else
                {
                  if ( !v110 )
                    BUG();
                  v111 = v110 - 24;
                  if ( (unsigned int)*(unsigned __int8 *)(v110 - 24) - 30 >= 0xB )
                    v111 = 0;
                }
                v147 = v108;
                v155 = v111 + 24;
                v112 = sub_BD2C40(80, 1u);
                v113 = v112;
                if ( v112 )
                {
                  sub_B4D230((__int64)v112, v147, v159, (__int64)v165, v155, 0);
                  v114 = *((_QWORD *)v103 - 1) + v106;
                  if ( *(_QWORD *)v114 )
                  {
                    v115 = *(_QWORD *)(v114 + 8);
                    **(_QWORD **)(v114 + 16) = v115;
                    if ( v115 )
                      *(_QWORD *)(v115 + 16) = *(_QWORD *)(v114 + 16);
                  }
                  *(_QWORD *)v114 = v113;
                  v116 = v113[2];
                  *(_QWORD *)(v114 + 8) = v116;
                  if ( v116 )
                    *(_QWORD *)(v116 + 16) = v114 + 8;
                  *(_QWORD *)(v114 + 16) = v113 + 2;
                  v113[2] = v114;
                  goto LABEL_194;
                }
                v134 = *((_QWORD *)v103 - 1) + v106;
                if ( *(_QWORD *)v134 )
                {
                  v135 = *(_QWORD *)(v134 + 8);
                  **(_QWORD **)(v134 + 16) = v135;
                  if ( v135 )
                    *(_QWORD *)(v135 + 16) = *(_QWORD *)(v134 + 16);
                  *(_QWORD *)v134 = 0;
                  v104 = *((_DWORD *)v103 + 1);
                  goto LABEL_180;
                }
LABEL_194:
                v104 = *((_DWORD *)v103 + 1);
                if ( (v104 & 0x7FFFFFFu) <= (unsigned int)++v105 )
                {
LABEL_195:
                  v94 = v144 + 1;
                  if ( v149 == v144 + 1 )
                  {
LABEL_196:
                    v87 = v139;
                    goto LABEL_197;
                  }
                  goto LABEL_177;
                }
              }
            }
          }
          else
          {
            v100 = *(_QWORD *)(v159 + 72);
            v166 = 257;
            v101 = sub_BD2C40(80, 1u);
            v102 = (__int64)v101;
            if ( v101 )
              sub_B4D230((__int64)v101, v100, v159, (__int64)v165, (__int64)(v103 + 24), 0);
            sub_BD2ED0((__int64)v103, v86, v102);
          }
          if ( v149 == ++v94 )
            goto LABEL_196;
          continue;
        }
      }
    }
  }
LABEL_99:
  if ( (_DWORD)v174 )
    sub_2A57B70(src);
  if ( src != v175 )
    _libc_free((unsigned __int64)src);
  return sub_C7D6A0((__int64)v162, 16LL * v164, 8);
}
