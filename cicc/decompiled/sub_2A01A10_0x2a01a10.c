// Function: sub_2A01A10
// Address: 0x2a01a10
//
unsigned __int64 __fastcall sub_2A01A10(__int64 *a1, __int64 a2, _BYTE *a3)
{
  __int64 *v3; // r14
  __int64 v4; // rdi
  __int64 *v5; // r13
  __int64 v6; // r15
  __int64 v8; // rbx
  __int64 v9; // rcx
  __int64 v10; // rax
  _BYTE *v11; // rsi
  __int64 v12; // r12
  unsigned int v13; // esi
  int v14; // r11d
  __int64 v15; // rax
  _QWORD *v16; // rbx
  int v17; // edx
  __int64 v18; // rdx
  unsigned __int64 *v19; // rdi
  __int64 v20; // rdx
  _QWORD *v21; // rbx
  __int64 v22; // r8
  unsigned int v23; // ecx
  _QWORD *v24; // rdx
  __int64 v25; // rdi
  __int64 v26; // rax
  __int64 v27; // rdx
  unsigned int v28; // esi
  __int64 v29; // rdi
  unsigned int v30; // ecx
  __int64 v31; // rax
  __int64 v32; // r10
  unsigned __int64 v33; // rax
  int v34; // edx
  __int64 v35; // r12
  __int64 v36; // rax
  __int64 v37; // rax
  unsigned int v38; // esi
  __int64 v39; // rdx
  unsigned int v40; // eax
  unsigned int v41; // ecx
  __int64 v42; // rdi
  __int64 v43; // r9
  __int64 v44; // rcx
  __int64 v45; // r8
  unsigned int v46; // r9d
  __int64 v47; // rdi
  __int64 v48; // r11
  __int64 v49; // r15
  unsigned int v50; // r9d
  __int64 v51; // rdi
  __int64 v52; // r11
  __int64 v53; // r13
  unsigned int v54; // r9d
  __int64 v55; // rdi
  __int64 v56; // r10
  __int64 v57; // r9
  unsigned int v58; // ebx
  __int64 v59; // rdi
  __int64 v60; // r10
  __int64 v61; // r10
  unsigned int v62; // r12d
  __int64 v63; // rdi
  __int64 v64; // r11
  __int64 v65; // r11
  unsigned int v66; // r12d
  __int64 v67; // rdi
  __int64 v68; // rbx
  __int64 v69; // rdi
  unsigned int v70; // r12d
  __int64 v71; // rsi
  __int64 v72; // rbx
  char v73; // dl
  char v74; // al
  __int64 v75; // rcx
  unsigned __int64 result; // rax
  __int64 v77; // rdx
  __int64 v78; // rsi
  __int64 v79; // r15
  __int64 v80; // r14
  unsigned __int8 *v81; // r15
  __int64 v82; // rdx
  __int64 v83; // rcx
  __int64 v84; // r8
  __int64 v85; // r9
  __int64 v86; // rdi
  unsigned int v87; // r13d
  __int64 v88; // rax
  __int64 v89; // rdi
  __int64 v90; // r12
  unsigned __int64 v91; // rdx
  __int64 v92; // rdx
  __int64 v93; // rbx
  __int64 v94; // rdi
  __int64 v95; // rdx
  unsigned int v96; // r9d
  int v97; // eax
  __int64 v98; // rdx
  __int64 v99; // r12
  __int64 v100; // rdx
  __int64 v101; // r8
  unsigned int v102; // esi
  __int64 v103; // rcx
  __int64 v104; // r10
  int v105; // eax
  __int64 v106; // rdi
  __int64 v107; // rax
  __int64 v108; // rax
  int v109; // ecx
  int v110; // r9d
  int v111; // edi
  int v112; // r11d
  int v113; // r11d
  __int64 v114; // r10
  int v115; // esi
  _QWORD *v116; // rcx
  __int64 v117; // rdx
  __int64 v118; // rdi
  int v119; // r11d
  __int64 v120; // r10
  int v121; // esi
  __int64 v122; // rdx
  __int64 v123; // rdi
  int v124; // edi
  int v125; // r10d
  int v126; // edi
  int v127; // r11d
  __int64 v128; // rdi
  int v129; // edi
  int v130; // r11d
  int v131; // edi
  int v132; // ebx
  int v133; // edi
  int v134; // esi
  int v135; // edi
  int v136; // edi
  int v137; // r10d
  int v138; // eax
  int v139; // r9d
  int v140; // r8d
  int v141; // [rsp+Ch] [rbp-B4h]
  __int64 v142; // [rsp+10h] [rbp-B0h]
  __int64 v143; // [rsp+18h] [rbp-A8h]
  unsigned int v144; // [rsp+20h] [rbp-A0h]
  __int64 v146; // [rsp+30h] [rbp-90h]
  int v147; // [rsp+30h] [rbp-90h]
  int v148; // [rsp+30h] [rbp-90h]
  __int64 *v149; // [rsp+38h] [rbp-88h]
  int v150; // [rsp+38h] [rbp-88h]
  __int64 *v151; // [rsp+38h] [rbp-88h]
  int v152; // [rsp+38h] [rbp-88h]
  __int64 *v153; // [rsp+40h] [rbp-80h]
  __int64 v154; // [rsp+40h] [rbp-80h]
  __int64 v155; // [rsp+40h] [rbp-80h]
  __int64 v157; // [rsp+48h] [rbp-78h]
  __int64 v158; // [rsp+48h] [rbp-78h]
  __int64 v159; // [rsp+58h] [rbp-68h] BYREF
  char *v160; // [rsp+60h] [rbp-60h] BYREF
  __int64 v161; // [rsp+68h] [rbp-58h] BYREF
  _BYTE *v162; // [rsp+70h] [rbp-50h]
  __int64 v163; // [rsp+78h] [rbp-48h]
  __int64 v164; // [rsp+80h] [rbp-40h]

  v3 = a1;
  v4 = a1[7];
  v5 = *(__int64 **)(v4 + 32);
  v149 = *(__int64 **)(v4 + 40);
  if ( v149 != v5 )
  {
    v153 = v3;
    v6 = a2 + 24;
    while ( 1 )
    {
      v8 = *v5;
      v9 = *v153;
      if ( *a3 )
      {
        v162 = a3;
        v160 = ".";
        LOWORD(v164) = 771;
      }
      else
      {
        v160 = ".";
        LOWORD(v164) = 259;
      }
      v10 = sub_F4B360(v8, v6, (__int64 *)&v160, v9, 0);
      v11 = *(_BYTE **)(a2 + 8);
      v159 = v10;
      v12 = v10;
      if ( v11 == *(_BYTE **)(a2 + 16) )
      {
        sub_9319A0(a2, v11, &v159);
        v12 = v159;
      }
      else
      {
        if ( v11 )
        {
          *(_QWORD *)v11 = v10;
          v11 = *(_BYTE **)(a2 + 8);
          v12 = v159;
        }
        *(_QWORD *)(a2 + 8) = v11 + 8;
      }
      v161 = 2;
      v162 = 0;
      v163 = v8;
      if ( v8 != 0 && v8 != -4096 && v8 != -8192 )
        sub_BD73F0((__int64)&v161);
      v13 = *(_DWORD *)(a2 + 48);
      v164 = v6;
      v160 = (char *)&unk_49DD7B0;
      if ( !v13 )
        break;
      v15 = v163;
      v22 = *(_QWORD *)(a2 + 32);
      v23 = (v13 - 1) & (((unsigned int)v163 >> 9) ^ ((unsigned int)v163 >> 4));
      v24 = (_QWORD *)(v22 + ((unsigned __int64)v23 << 6));
      v25 = v24[3];
      if ( v163 != v25 )
      {
        v110 = 1;
        v16 = 0;
        while ( v25 != -4096 )
        {
          if ( v25 == -8192 && !v16 )
            v16 = v24;
          v23 = (v13 - 1) & (v110 + v23);
          v24 = (_QWORD *)(v22 + ((unsigned __int64)v23 << 6));
          v25 = v24[3];
          if ( v163 == v25 )
            goto LABEL_29;
          ++v110;
        }
        v111 = *(_DWORD *)(a2 + 40);
        if ( !v16 )
          v16 = v24;
        ++*(_QWORD *)(a2 + 24);
        v17 = v111 + 1;
        if ( 4 * (v111 + 1) < 3 * v13 )
        {
          if ( v13 - *(_DWORD *)(a2 + 44) - v17 > v13 >> 3 )
            goto LABEL_17;
          sub_CF32C0(v6, v13);
          v112 = *(_DWORD *)(a2 + 48);
          if ( v112 )
          {
            v15 = v163;
            v113 = v112 - 1;
            v114 = *(_QWORD *)(a2 + 32);
            v115 = 1;
            v116 = 0;
            LODWORD(v117) = v113 & (((unsigned int)v163 >> 9) ^ ((unsigned int)v163 >> 4));
            v16 = (_QWORD *)(v114 + ((unsigned __int64)(unsigned int)v117 << 6));
            v118 = v16[3];
            if ( v163 != v118 )
            {
              while ( v118 != -4096 )
              {
                if ( !v116 && v118 == -8192 )
                  v116 = v16;
                v117 = v113 & (unsigned int)(v117 + v115);
                v16 = (_QWORD *)(v114 + (v117 << 6));
                v118 = v16[3];
                if ( v163 == v118 )
                  goto LABEL_16;
                ++v115;
              }
LABEL_149:
              if ( v116 )
                v16 = v116;
            }
LABEL_16:
            v17 = *(_DWORD *)(a2 + 40) + 1;
LABEL_17:
            *(_DWORD *)(a2 + 40) = v17;
            if ( v16[3] == -4096 )
            {
              v19 = v16 + 1;
              if ( v15 != -4096 )
                goto LABEL_22;
            }
            else
            {
              --*(_DWORD *)(a2 + 44);
              v18 = v16[3];
              if ( v15 != v18 )
              {
                v19 = v16 + 1;
                if ( v18 != -4096 && v18 != 0 && v18 != -8192 )
                {
                  sub_BD60C0(v19);
                  v15 = v163;
                  v19 = v16 + 1;
                }
LABEL_22:
                v16[3] = v15;
                if ( v15 != 0 && v15 != -4096 && v15 != -8192 )
                  sub_BD6050(v19, v161 & 0xFFFFFFFFFFFFFFF8LL);
                v15 = v163;
              }
            }
            v20 = v164;
            v21 = v16 + 5;
            *v21 = 6;
            v21[1] = 0;
            *(v21 - 1) = v20;
            v21[2] = 0;
            goto LABEL_30;
          }
LABEL_15:
          v15 = v163;
          v16 = 0;
          goto LABEL_16;
        }
LABEL_14:
        sub_CF32C0(v6, 2 * v13);
        v14 = *(_DWORD *)(a2 + 48);
        if ( v14 )
        {
          v15 = v163;
          v119 = v14 - 1;
          v120 = *(_QWORD *)(a2 + 32);
          v121 = 1;
          v116 = 0;
          LODWORD(v122) = v119 & (((unsigned int)v163 >> 9) ^ ((unsigned int)v163 >> 4));
          v16 = (_QWORD *)(v120 + ((unsigned __int64)(unsigned int)v122 << 6));
          v123 = v16[3];
          if ( v163 != v123 )
          {
            while ( v123 != -4096 )
            {
              if ( v123 == -8192 && !v116 )
                v116 = v16;
              v122 = v119 & (unsigned int)(v122 + v121);
              v16 = (_QWORD *)(v120 + (v122 << 6));
              v123 = v16[3];
              if ( v163 == v123 )
                goto LABEL_16;
              ++v121;
            }
            goto LABEL_149;
          }
          goto LABEL_16;
        }
        goto LABEL_15;
      }
LABEL_29:
      v21 = v24 + 5;
LABEL_30:
      v160 = (char *)&unk_49DB368;
      if ( v15 != 0 && v15 != -4096 && v15 != -8192 )
        sub_BD60C0(&v161);
      v26 = v21[2];
      if ( v26 != v12 )
      {
        if ( v26 != 0 && v26 != -4096 && v26 != -8192 )
          sub_BD60C0(v21);
        v21[2] = v12;
        if ( v12 != 0 && v12 != -4096 && v12 != -8192 )
          sub_BD73F0((__int64)v21);
      }
      if ( v149 == ++v5 )
      {
        v3 = v153;
        v4 = v153[7];
        goto LABEL_42;
      }
    }
    ++*(_QWORD *)(a2 + 24);
    goto LABEL_14;
  }
LABEL_42:
  v27 = sub_D47930(v4);
  v28 = *(_DWORD *)(a2 + 48);
  if ( v28 )
  {
    v29 = *(_QWORD *)(a2 + 32);
    v30 = (v28 - 1) & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
    v31 = v29 + ((unsigned __int64)v30 << 6);
    v32 = *(_QWORD *)(v31 + 24);
    if ( v27 == v32 )
    {
LABEL_44:
      if ( v31 != v29 + ((unsigned __int64)v28 << 6) )
        v27 = *(_QWORD *)(v31 + 56);
    }
    else
    {
      v138 = 1;
      while ( v32 != -4096 )
      {
        v139 = v138 + 1;
        v30 = (v28 - 1) & (v138 + v30);
        v31 = v29 + ((unsigned __int64)v30 << 6);
        v32 = *(_QWORD *)(v31 + 24);
        if ( v27 == v32 )
          goto LABEL_44;
        v138 = v139;
      }
    }
  }
  v33 = *(_QWORD *)(v27 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v33 == v27 + 48 )
  {
    v35 = 0;
  }
  else
  {
    if ( !v33 )
LABEL_214:
      BUG();
    v34 = *(unsigned __int8 *)(v33 - 24);
    v35 = 0;
    v36 = v33 - 24;
    if ( (unsigned int)(v34 - 30) < 0xB )
      v35 = v36;
  }
  v37 = sub_B9C770((__int64 *)v3[1], 0, 0, 0, 1);
  sub_B9A090(v35, "loop_constrainer.loop.clone", 0x1Bu, v37);
  v154 = v3[12];
  v38 = *(_DWORD *)(a2 + 48);
  v39 = *(_QWORD *)(a2 + 32);
  if ( v38 )
  {
    v40 = v38 - 1;
    v41 = (v38 - 1) & (((unsigned int)v154 >> 9) ^ ((unsigned int)v154 >> 4));
    v42 = v39 + ((unsigned __int64)v41 << 6);
    v43 = *(_QWORD *)(v42 + 24);
    if ( v154 == v43 )
    {
LABEL_52:
      v44 = v39 + ((unsigned __int64)v38 << 6);
      if ( v44 != v42 )
        v154 = *(_QWORD *)(v42 + 56);
      v45 = v3[13];
    }
    else
    {
      v135 = 1;
      while ( v43 != -4096 )
      {
        v140 = v135 + 1;
        v41 = v40 & (v135 + v41);
        v42 = v39 + ((unsigned __int64)v41 << 6);
        v43 = *(_QWORD *)(v42 + 24);
        if ( v154 == v43 )
          goto LABEL_52;
        v135 = v140;
      }
      v45 = v3[13];
      v44 = v39 + ((unsigned __int64)v38 << 6);
    }
    v46 = v40 & (((unsigned int)v45 >> 9) ^ ((unsigned int)v45 >> 4));
    v47 = v39 + ((unsigned __int64)v46 << 6);
    v48 = *(_QWORD *)(v47 + 24);
    if ( v48 == v45 )
    {
LABEL_56:
      if ( v44 == v47 )
      {
        v49 = v3[14];
        goto LABEL_59;
      }
      v45 = *(_QWORD *)(v47 + 56);
    }
    else
    {
      v136 = 1;
      while ( v48 != -4096 )
      {
        v137 = v136 + 1;
        v46 = v40 & (v136 + v46);
        v47 = v39 + ((unsigned __int64)v46 << 6);
        v48 = *(_QWORD *)(v47 + 24);
        if ( v48 == v45 )
          goto LABEL_56;
        v136 = v137;
      }
    }
  }
  else
  {
    v45 = v3[13];
    v44 = *(_QWORD *)(a2 + 32);
  }
  v49 = v3[14];
  v40 = v38 - 1;
  if ( !v38 )
    goto LABEL_62;
LABEL_59:
  v50 = v40 & (((unsigned int)v49 >> 9) ^ ((unsigned int)v49 >> 4));
  v51 = v39 + ((unsigned __int64)v50 << 6);
  v52 = *(_QWORD *)(v51 + 24);
  if ( v52 != v49 )
  {
    v124 = 1;
    while ( v52 != -4096 )
    {
      v125 = v124 + 1;
      v50 = v40 & (v124 + v50);
      v51 = v39 + ((unsigned __int64)v50 << 6);
      v52 = *(_QWORD *)(v51 + 24);
      if ( v52 == v49 )
        goto LABEL_60;
      v124 = v125;
    }
LABEL_62:
    v53 = v3[15];
    v40 = v38 - 1;
    if ( !v38 )
      goto LABEL_66;
    goto LABEL_63;
  }
LABEL_60:
  if ( v51 != v44 )
  {
    v49 = *(_QWORD *)(v51 + 56);
    goto LABEL_62;
  }
  v53 = v3[15];
LABEL_63:
  v54 = v40 & (((unsigned int)v53 >> 9) ^ ((unsigned int)v53 >> 4));
  v55 = v39 + ((unsigned __int64)v54 << 6);
  v56 = *(_QWORD *)(v55 + 24);
  if ( v56 == v53 )
  {
LABEL_64:
    if ( v55 == v44 )
    {
      v57 = v3[17];
      v150 = *((_DWORD *)v3 + 32);
      goto LABEL_67;
    }
    v53 = *(_QWORD *)(v55 + 56);
  }
  else
  {
    v126 = 1;
    while ( v56 != -4096 )
    {
      v127 = v126 + 1;
      v128 = v40 & (v54 + v126);
      v54 = v128;
      v55 = v39 + (v128 << 6);
      v56 = *(_QWORD *)(v55 + 24);
      if ( v56 == v53 )
        goto LABEL_64;
      v126 = v127;
    }
  }
LABEL_66:
  v57 = v3[17];
  v150 = *((_DWORD *)v3 + 32);
  v40 = v38 - 1;
  if ( !v38 )
    goto LABEL_70;
LABEL_67:
  v58 = v40 & (((unsigned int)v57 >> 9) ^ ((unsigned int)v57 >> 4));
  v59 = v39 + ((unsigned __int64)v58 << 6);
  v60 = *(_QWORD *)(v59 + 24);
  if ( v60 != v57 )
  {
    v129 = 1;
    while ( v60 != -4096 )
    {
      v130 = v129 + 1;
      v58 = v40 & (v129 + v58);
      v59 = v39 + ((unsigned __int64)v58 << 6);
      v60 = *(_QWORD *)(v59 + 24);
      if ( v60 == v57 )
        goto LABEL_68;
      v129 = v130;
    }
LABEL_70:
    v61 = v3[18];
    v40 = v38 - 1;
    if ( !v38 )
      goto LABEL_74;
    goto LABEL_71;
  }
LABEL_68:
  if ( v44 != v59 )
  {
    v57 = *(_QWORD *)(v59 + 56);
    goto LABEL_70;
  }
  v61 = v3[18];
LABEL_71:
  v62 = v40 & (((unsigned int)v61 >> 9) ^ ((unsigned int)v61 >> 4));
  v63 = v39 + ((unsigned __int64)v62 << 6);
  v64 = *(_QWORD *)(v63 + 24);
  if ( v64 == v61 )
  {
LABEL_72:
    if ( v63 == v44 )
    {
      v65 = v3[19];
      goto LABEL_75;
    }
    v61 = *(_QWORD *)(v63 + 56);
  }
  else
  {
    v131 = 1;
    while ( v64 != -4096 )
    {
      v132 = v131 + 1;
      v62 = v40 & (v131 + v62);
      v63 = v39 + ((unsigned __int64)v62 << 6);
      v64 = *(_QWORD *)(v63 + 24);
      if ( v64 == v61 )
        goto LABEL_72;
      v131 = v132;
    }
  }
LABEL_74:
  v65 = v3[19];
  v40 = v38 - 1;
  if ( !v38 )
    goto LABEL_78;
LABEL_75:
  v66 = v40 & (((unsigned int)v65 >> 9) ^ ((unsigned int)v65 >> 4));
  v67 = v39 + ((unsigned __int64)v66 << 6);
  v68 = *(_QWORD *)(v67 + 24);
  if ( v65 != v68 )
  {
    v133 = 1;
    while ( v68 != -4096 )
    {
      v66 = v40 & (v133 + v66);
      v147 = v133 + 1;
      v67 = v39 + ((unsigned __int64)v66 << 6);
      v68 = *(_QWORD *)(v67 + 24);
      if ( v68 == v65 )
        goto LABEL_76;
      v133 = v147;
    }
LABEL_78:
    v69 = v3[20];
    v40 = v38 - 1;
    if ( !v38 )
      goto LABEL_82;
    goto LABEL_79;
  }
LABEL_76:
  if ( v67 != v44 )
  {
    v65 = *(_QWORD *)(v67 + 56);
    goto LABEL_78;
  }
  v69 = v3[20];
LABEL_79:
  v70 = v40 & (((unsigned int)v69 >> 9) ^ ((unsigned int)v69 >> 4));
  v71 = v39 + ((unsigned __int64)v70 << 6);
  v72 = *(_QWORD *)(v71 + 24);
  if ( v72 == v69 )
  {
LABEL_80:
    if ( v44 != v71 )
      v69 = *(_QWORD *)(v71 + 56);
  }
  else
  {
    v134 = 1;
    while ( v72 != -4096 )
    {
      v70 = v40 & (v134 + v70);
      v148 = v134 + 1;
      v71 = v39 + ((unsigned __int64)v70 << 6);
      v72 = *(_QWORD *)(v71 + 24);
      if ( v72 == v69 )
        goto LABEL_80;
      v134 = v148;
    }
  }
LABEL_82:
  v73 = *((_BYTE *)v3 + 168);
  v74 = *((_BYTE *)v3 + 169);
  v75 = v3[22];
  *(_QWORD *)(a2 + 112) = v154;
  *(_BYTE *)(a2 + 184) = v73;
  *(_BYTE *)(a2 + 185) = v74;
  *(_QWORD *)(a2 + 120) = v45;
  *(_QWORD *)(a2 + 128) = v49;
  *(_QWORD *)(a2 + 136) = v53;
  *(_DWORD *)(a2 + 144) = v150;
  *(_QWORD *)(a2 + 152) = v57;
  *(_QWORD *)(a2 + 160) = v61;
  *(_QWORD *)(a2 + 168) = v65;
  *(_QWORD *)(a2 + 176) = v69;
  *(_QWORD *)(a2 + 192) = v75;
  *(_QWORD *)(a2 + 104) = a3;
  result = *(_QWORD *)a2;
  v77 = (__int64)(*(_QWORD *)(a2 + 8) - *(_QWORD *)a2) >> 3;
  if ( (_DWORD)v77 )
  {
    v143 = 0;
    v142 = 8LL * (unsigned int)(v77 - 1);
    while ( 1 )
    {
      v78 = *(_QWORD *)(result + v143);
      v155 = v78;
      v79 = *(_QWORD *)(*(_QWORD *)(v3[7] + 32) + v143);
      if ( v78 + 48 != *(_QWORD *)(v78 + 56) )
      {
        v157 = *(_QWORD *)(*(_QWORD *)(v3[7] + 32) + v143);
        v151 = v3;
        v80 = *(_QWORD *)(v78 + 56);
        do
        {
          v81 = 0;
          if ( v80 )
            v81 = (unsigned __int8 *)(v80 - 24);
          sub_FC75A0((__int64 *)&v160, a2 + 24, 3, 0, 0, 0);
          sub_FCD280((__int64 *)&v160, v81, v82, v83, v84, v85);
          sub_FC7680((__int64 *)&v160, (__int64)v81);
          v80 = *(_QWORD *)(v80 + 8);
        }
        while ( v78 + 48 != v80 );
        v79 = v157;
        v3 = v151;
      }
      result = *(_QWORD *)(v79 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( result != v79 + 48 )
      {
        if ( !result )
          BUG();
        v86 = result - 24;
        v146 = result - 24;
        result = (unsigned int)*(unsigned __int8 *)(result - 24) - 30;
        if ( (unsigned int)result <= 0xA )
        {
          result = sub_B46E30(v86);
          v152 = result;
          if ( (_DWORD)result )
            break;
        }
      }
LABEL_101:
      if ( v142 == v143 )
        return result;
      v143 += 8;
      result = *(_QWORD *)a2;
    }
    v87 = 0;
    while ( 1 )
    {
      v88 = sub_B46EC0(v146, v87);
      v89 = v3[7];
      v90 = v88;
      if ( *(_BYTE *)(v89 + 84) )
      {
        result = *(_QWORD *)(v89 + 64);
        v91 = result + 8LL * *(unsigned int *)(v89 + 76);
        if ( result != v91 )
        {
          while ( v90 != *(_QWORD *)result )
          {
            result += 8LL;
            if ( v91 == result )
              goto LABEL_104;
          }
          goto LABEL_100;
        }
LABEL_104:
        result = sub_AA5930(v90);
        v158 = v92;
        v93 = result;
        if ( result == v92 )
          goto LABEL_100;
        v144 = v87;
        do
        {
          v94 = *(_QWORD *)(v93 - 8);
          v95 = 0x1FFFFFFFE0LL;
          v96 = *(_DWORD *)(v93 + 72);
          v97 = *(_DWORD *)(v93 + 4) & 0x7FFFFFF;
          if ( v97 )
          {
            v98 = 0;
            do
            {
              if ( v79 == *(_QWORD *)(v94 + 32LL * v96 + 8 * v98) )
              {
                v95 = 32 * v98;
                goto LABEL_111;
              }
              ++v98;
            }
            while ( v97 != (_DWORD)v98 );
            v95 = 0x1FFFFFFFE0LL;
          }
LABEL_111:
          v99 = *(_QWORD *)(v94 + v95);
          v100 = *(unsigned int *)(a2 + 48);
          if ( (_DWORD)v100 )
          {
            v101 = *(_QWORD *)(a2 + 32);
            v102 = (v100 - 1) & (((unsigned int)v99 >> 9) ^ ((unsigned int)v99 >> 4));
            v103 = v101 + ((unsigned __int64)v102 << 6);
            v104 = *(_QWORD *)(v103 + 24);
            if ( v104 == v99 )
            {
LABEL_113:
              if ( v103 != v101 + (v100 << 6) )
                v99 = *(_QWORD *)(v103 + 56);
            }
            else
            {
              v109 = 1;
              while ( v104 != -4096 )
              {
                v102 = (v100 - 1) & (v109 + v102);
                v141 = v109 + 1;
                v103 = v101 + ((unsigned __int64)v102 << 6);
                v104 = *(_QWORD *)(v103 + 24);
                if ( v99 == v104 )
                  goto LABEL_113;
                v109 = v141;
              }
            }
          }
          if ( v96 == v97 )
          {
            sub_B48D90(v93);
            v94 = *(_QWORD *)(v93 - 8);
            v97 = *(_DWORD *)(v93 + 4) & 0x7FFFFFF;
          }
          v105 = (v97 + 1) & 0x7FFFFFF;
          *(_DWORD *)(v93 + 4) = v105 | *(_DWORD *)(v93 + 4) & 0xF8000000;
          v106 = 32LL * (unsigned int)(v105 - 1) + v94;
          if ( *(_QWORD *)v106 )
          {
            v107 = *(_QWORD *)(v106 + 8);
            **(_QWORD **)(v106 + 16) = v107;
            if ( v107 )
              *(_QWORD *)(v107 + 16) = *(_QWORD *)(v106 + 16);
          }
          *(_QWORD *)v106 = v99;
          if ( v99 )
          {
            v108 = *(_QWORD *)(v99 + 16);
            *(_QWORD *)(v106 + 8) = v108;
            if ( v108 )
              *(_QWORD *)(v108 + 16) = v106 + 8;
            *(_QWORD *)(v106 + 16) = v99 + 16;
            *(_QWORD *)(v99 + 16) = v106;
          }
          *(_QWORD *)(*(_QWORD *)(v93 - 8)
                    + 32LL * *(unsigned int *)(v93 + 72)
                    + 8LL * ((*(_DWORD *)(v93 + 4) & 0x7FFFFFFu) - 1)) = v155;
          sub_DACA20(v3[2], v3[7], v93);
          result = *(_QWORD *)(v93 + 32);
          if ( !result )
            goto LABEL_214;
          v93 = 0;
          if ( *(_BYTE *)(result - 24) == 84 )
            v93 = result - 24;
        }
        while ( v158 != v93 );
        ++v87;
        if ( v152 == v144 + 1 )
          goto LABEL_101;
      }
      else
      {
        result = (unsigned __int64)sub_C8CA60(v89 + 56, v88);
        if ( !result )
          goto LABEL_104;
LABEL_100:
        if ( v152 == ++v87 )
          goto LABEL_101;
      }
    }
  }
  return result;
}
