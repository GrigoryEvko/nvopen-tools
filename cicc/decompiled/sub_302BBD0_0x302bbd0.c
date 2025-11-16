// Function: sub_302BBD0
// Address: 0x302bbd0
//
void __fastcall sub_302BBD0(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // rax
  __int64 v4; // r14
  signed __int64 v5; // r13
  __int64 v6; // rdx
  int v7; // r12d
  int v8; // r10d
  __int64 v9; // r8
  unsigned int v10; // edi
  __int64 *v11; // r13
  unsigned __int64 *v12; // rax
  unsigned __int64 v13; // rcx
  unsigned int v14; // esi
  __int64 v15; // r15
  __int64 v16; // r10
  int v17; // eax
  int v18; // r11d
  unsigned int *v19; // rdi
  unsigned int v20; // edx
  unsigned int v21; // esi
  unsigned int v22; // r14d
  unsigned __int64 v23; // r15
  int v24; // edx
  int v25; // edx
  __int64 v26; // r8
  __int64 v27; // rsi
  int v28; // ecx
  unsigned __int64 v29; // rdi
  int v30; // edx
  int v31; // edx
  __int64 v32; // rdi
  __int64 v33; // rcx
  unsigned int *v34; // r8
  unsigned int v35; // esi
  int v36; // r10d
  unsigned int *v37; // r13
  int v38; // ecx
  int v39; // esi
  int v40; // esi
  unsigned __int64 *v41; // r9
  __int64 v42; // r8
  int v43; // r10d
  __int64 v44; // rdx
  unsigned __int64 v45; // rdi
  _QWORD *v46; // r15
  __int64 v47; // r14
  int v48; // r11d
  __int64 v49; // r9
  unsigned int v50; // r8d
  __int64 *v51; // rdx
  __int64 *v52; // rax
  __int64 v53; // rdi
  int v54; // r13d
  size_t v55; // rdx
  __int64 v56; // rax
  size_t v57; // r9
  unsigned __int8 *v58; // r12
  size_t v59; // rdx
  size_t v60; // rbx
  _DWORD *v61; // rdx
  _QWORD *v62; // r8
  _WORD *v63; // rdi
  unsigned __int64 v64; // rax
  _BYTE *v65; // rdi
  _BYTE *v66; // rax
  __int64 v67; // rax
  __int64 v68; // rdx
  unsigned int v69; // esi
  __int64 v70; // r12
  int v71; // esi
  int v72; // esi
  __int64 v73; // r9
  unsigned int v74; // edi
  int v75; // edx
  __int64 v76; // r8
  __int64 *v77; // rdi
  unsigned __int64 v78; // rdx
  unsigned __int64 v79; // rax
  int v80; // ecx
  int v81; // esi
  int v82; // esi
  __int64 v83; // r8
  __int64 *v84; // r9
  unsigned int v85; // ebx
  int v86; // r10d
  __int64 v87; // rdi
  __int64 v88; // rax
  __int64 v89; // rax
  __int64 v90; // rax
  __int64 v91; // rax
  _BYTE *v92; // rbx
  __int64 v93; // rax
  __int64 v94; // rax
  __int64 v95; // rax
  __int64 v96; // r14
  unsigned int v97; // eax
  __int64 v98; // rax
  __int64 v99; // rax
  __int64 v100; // rax
  __int64 v101; // rax
  int v102; // edx
  int v103; // edx
  __int64 v104; // rdi
  int v105; // r10d
  __int64 v106; // rcx
  int v107; // eax
  unsigned int v108; // esi
  __int64 v109; // rax
  int v110; // r10d
  int v111; // r11d
  __int64 *v112; // r10
  _QWORD *v113; // [rsp+8h] [rbp-178h]
  int v114; // [rsp+8h] [rbp-178h]
  int v115; // [rsp+8h] [rbp-178h]
  __int64 v116; // [rsp+10h] [rbp-170h]
  __int64 v117; // [rsp+10h] [rbp-170h]
  _QWORD *src; // [rsp+28h] [rbp-158h]
  unsigned __int8 *srca; // [rsp+28h] [rbp-158h]
  unsigned int n; // [rsp+30h] [rbp-150h]
  int na; // [rsp+30h] [rbp-150h]
  unsigned int nb; // [rsp+30h] [rbp-150h]
  size_t nc; // [rsp+30h] [rbp-150h]
  size_t ne; // [rsp+30h] [rbp-150h]
  size_t nf; // [rsp+30h] [rbp-150h]
  int nd; // [rsp+30h] [rbp-150h]
  int v127; // [rsp+38h] [rbp-148h]
  _QWORD *v128; // [rsp+38h] [rbp-148h]
  _QWORD v129[4]; // [rsp+40h] [rbp-140h] BYREF
  __int16 v130; // [rsp+60h] [rbp-120h]
  _QWORD v131[3]; // [rsp+70h] [rbp-110h] BYREF
  __int64 v132; // [rsp+88h] [rbp-F8h]
  _DWORD *v133; // [rsp+90h] [rbp-F0h]
  __int64 v134; // [rsp+98h] [rbp-E8h]
  unsigned __int64 *v135; // [rsp+A0h] [rbp-E0h]
  unsigned __int64 v136[3]; // [rsp+B0h] [rbp-D0h] BYREF
  _BYTE v137[184]; // [rsp+C8h] [rbp-B8h] BYREF

  v2 = a1;
  v136[0] = (unsigned __int64)v137;
  v135 = v136;
  v134 = 0x100000000LL;
  v131[0] = &unk_49DD288;
  v136[1] = 0;
  v136[2] = 128;
  v131[1] = 2;
  v131[2] = 0;
  v132 = 0;
  v133 = 0;
  sub_CB5980((__int64)v131, 0, 0, 0);
  v3 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a2 + 16) + 200LL))(*(_QWORD *)(a2 + 16));
  v4 = *(_QWORD *)(a2 + 48);
  src = (_QWORD *)v3;
  v5 = *(_QWORD *)(v4 + 48);
  if ( v5 )
  {
    v93 = sub_904010((__int64)v131, "\t.local .align ");
    v94 = sub_CB59D0(v93, 1LL << *(_BYTE *)(v4 + 64));
    v95 = sub_904010(v94, " .b8 \t");
    v96 = sub_904010(v95, "__local_depot");
    v97 = sub_31DA6A0(a1);
    v98 = sub_CB59D0(v96, v97);
    v99 = sub_904010(v98, "[");
    v100 = sub_CB59F0(v99, v5);
    sub_904010(v100, "];\n");
    if ( *(_BYTE *)(*(_QWORD *)(a2 + 8) + 1264LL) )
    {
      v101 = sub_904010((__int64)v131, "\t.reg .b64 \t%SP;\n");
      sub_904010(v101, "\t.reg .b64 \t%SPL;\n");
    }
    else
    {
      v109 = sub_904010((__int64)v131, "\t.reg .b32 \t%SP;\n");
      sub_904010(v109, "\t.reg .b32 \t%SPL;\n");
    }
  }
  v6 = *(_QWORD *)(a1 + 1104);
  v127 = *(_DWORD *)(v6 + 64);
  if ( v127 )
  {
    v116 = a1 + 1112;
    v7 = 0;
    while ( 1 )
    {
      v21 = *(_DWORD *)(v2 + 1136);
      v22 = v7 | 0x80000000;
      v23 = *(_QWORD *)(*(_QWORD *)(v6 + 56) + 16LL * (v7 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL;
      if ( !v21 )
        break;
      v8 = 1;
      v9 = *(_QWORD *)(v2 + 1120);
      v10 = (v21 - 1) & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
      v11 = (__int64 *)(v9 + 40LL * v10);
      v12 = 0;
      v13 = *v11;
      if ( v23 != *v11 )
      {
        while ( v13 != -4096 )
        {
          if ( !v12 && v13 == -8192 )
            v12 = (unsigned __int64 *)v11;
          v10 = (v21 - 1) & (v8 + v10);
          v11 = (__int64 *)(v9 + 40LL * v10);
          v13 = *v11;
          if ( v23 == *v11 )
            goto LABEL_5;
          ++v8;
        }
        v38 = *(_DWORD *)(v2 + 1128);
        if ( !v12 )
          v12 = (unsigned __int64 *)v11;
        ++*(_QWORD *)(v2 + 1112);
        v28 = v38 + 1;
        if ( 4 * v28 < 3 * v21 )
        {
          if ( v21 - *(_DWORD *)(v2 + 1132) - v28 <= v21 >> 3 )
          {
            nb = ((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4);
            sub_302AEB0(v116, v21);
            v39 = *(_DWORD *)(v2 + 1136);
            if ( !v39 )
            {
LABEL_149:
              ++*(_DWORD *)(v2 + 1128);
              BUG();
            }
            v40 = v39 - 1;
            v41 = 0;
            v42 = *(_QWORD *)(v2 + 1120);
            v43 = 1;
            LODWORD(v44) = v40 & nb;
            v28 = *(_DWORD *)(v2 + 1128) + 1;
            v12 = (unsigned __int64 *)(v42 + 40LL * (v40 & nb));
            v45 = *v12;
            if ( v23 != *v12 )
            {
              while ( v45 != -4096 )
              {
                if ( v45 == -8192 && !v41 )
                  v41 = v12;
                v44 = v40 & (unsigned int)(v44 + v43);
                v12 = (unsigned __int64 *)(v42 + 40 * v44);
                v45 = *v12;
                if ( v23 == *v12 )
                  goto LABEL_13;
                ++v43;
              }
LABEL_37:
              if ( v41 )
                v12 = v41;
            }
          }
LABEL_13:
          *(_DWORD *)(v2 + 1128) = v28;
          if ( *v12 != -4096 )
            --*(_DWORD *)(v2 + 1132);
          v12[1] = 0;
          v18 = 1;
          v12[2] = 0;
          v12[3] = 0;
          *((_DWORD *)v12 + 8) = 0;
          *v12 = v23;
          v15 = (__int64)(v12 + 1);
LABEL_16:
          ++*(_QWORD *)v15;
          v14 = 0;
          goto LABEL_17;
        }
LABEL_11:
        sub_302AEB0(v116, 2 * v21);
        v24 = *(_DWORD *)(v2 + 1136);
        if ( !v24 )
          goto LABEL_149;
        v25 = v24 - 1;
        v26 = *(_QWORD *)(v2 + 1120);
        LODWORD(v27) = v25 & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
        v28 = *(_DWORD *)(v2 + 1128) + 1;
        v12 = (unsigned __int64 *)(v26 + 40LL * (unsigned int)v27);
        v29 = *v12;
        if ( v23 != *v12 )
        {
          v110 = 1;
          v41 = 0;
          while ( v29 != -4096 )
          {
            if ( !v41 && v29 == -8192 )
              v41 = v12;
            v27 = v25 & (unsigned int)(v27 + v110);
            v12 = (unsigned __int64 *)(v26 + 40 * v27);
            v29 = *v12;
            if ( v23 == *v12 )
              goto LABEL_13;
            ++v110;
          }
          goto LABEL_37;
        }
        goto LABEL_13;
      }
LABEL_5:
      v14 = *((_DWORD *)v11 + 8);
      v15 = (__int64)(v11 + 1);
      v16 = v11[2];
      v17 = *((_DWORD *)v11 + 6) + 1;
      v18 = v17;
      if ( !v14 )
        goto LABEL_16;
      n = (v14 - 1) & (37 * v22);
      v19 = (unsigned int *)(v16 + 8LL * n);
      v20 = *v19;
      if ( v22 == *v19 )
        goto LABEL_7;
      v114 = 1;
      v34 = 0;
      while ( v20 != -1 )
      {
        if ( v34 || v20 != -2 )
          v19 = v34;
        n = (v14 - 1) & (n + v114);
        v20 = *(_DWORD *)(v16 + 8LL * n);
        if ( v22 == v20 )
          goto LABEL_7;
        ++v114;
        v34 = v19;
        v19 = (unsigned int *)(v16 + 8LL * n);
      }
      if ( !v34 )
        v34 = v19;
      ++v11[1];
      if ( 4 * v17 < 3 * v14 )
      {
        if ( v14 - (v17 + *((_DWORD *)v11 + 7)) > v14 >> 3 )
          goto LABEL_98;
        v115 = 37 * v22;
        nd = v17;
        sub_A09770((__int64)(v11 + 1), v14);
        v102 = *((_DWORD *)v11 + 8);
        if ( !v102 )
        {
LABEL_151:
          ++*(_DWORD *)(v15 + 16);
          BUG();
        }
        v103 = v102 - 1;
        v104 = v11[2];
        v105 = 1;
        v18 = nd;
        LODWORD(v106) = v103 & v115;
        v34 = (unsigned int *)(v104 + 8LL * (v103 & (unsigned int)v115));
        v107 = *((_DWORD *)v11 + 6);
        v37 = 0;
        v108 = *v34;
        v17 = v107 + 1;
        if ( v22 == *v34 )
          goto LABEL_98;
        while ( v108 != -1 )
        {
          if ( v108 == -2 && !v37 )
            v37 = v34;
          v106 = v103 & (unsigned int)(v106 + v105);
          v34 = (unsigned int *)(v104 + 8 * v106);
          v108 = *v34;
          if ( v22 == *v34 )
            goto LABEL_98;
          ++v105;
        }
        goto LABEL_21;
      }
LABEL_17:
      na = v18;
      sub_A09770(v15, 2 * v14);
      v30 = *(_DWORD *)(v15 + 24);
      if ( !v30 )
        goto LABEL_151;
      v31 = v30 - 1;
      v32 = *(_QWORD *)(v15 + 8);
      v18 = na;
      LODWORD(v33) = v31 & (37 * v22);
      v34 = (unsigned int *)(v32 + 8LL * (unsigned int)v33);
      v35 = *v34;
      v17 = *(_DWORD *)(v15 + 16) + 1;
      if ( v22 == *v34 )
        goto LABEL_98;
      v36 = 1;
      v37 = 0;
      while ( v35 != -1 )
      {
        if ( v35 == -2 && !v37 )
          v37 = v34;
        v33 = v31 & (unsigned int)(v33 + v36);
        v34 = (unsigned int *)(v32 + 8 * v33);
        v35 = *v34;
        if ( v22 == *v34 )
          goto LABEL_98;
        ++v36;
      }
LABEL_21:
      if ( v37 )
        v34 = v37;
LABEL_98:
      *(_DWORD *)(v15 + 16) = v17;
      if ( *v34 != -1 )
        --*(_DWORD *)(v15 + 20);
      *v34 = v22;
      v34[1] = v18;
LABEL_7:
      if ( v127 == ++v7 )
        goto LABEL_40;
      v6 = *(_QWORD *)(v2 + 1104);
    }
    ++*(_QWORD *)(v2 + 1112);
    goto LABEL_11;
  }
LABEL_40:
  v117 = v2 + 1112;
  v46 = (_QWORD *)src[35];
  if ( (_QWORD *)src[36] == v46 )
    goto LABEL_67;
  v128 = (_QWORD *)src[36];
  v47 = v2;
  do
  {
    while ( 1 )
    {
      v69 = *(_DWORD *)(v47 + 1136);
      v70 = *v46;
      if ( !v69 )
      {
        ++*(_QWORD *)(v47 + 1112);
        goto LABEL_61;
      }
      v48 = 1;
      v49 = *(_QWORD *)(v47 + 1120);
      v50 = (v69 - 1) & (((unsigned int)v70 >> 9) ^ ((unsigned int)v70 >> 4));
      v51 = (__int64 *)(v49 + 40LL * v50);
      v52 = 0;
      v53 = *v51;
      if ( v70 == *v51 )
        break;
      while ( v53 != -4096 )
      {
        if ( v53 == -8192 && !v52 )
          v52 = v51;
        v50 = (v69 - 1) & (v48 + v50);
        v51 = (__int64 *)(v49 + 40LL * v50);
        v53 = *v51;
        if ( v70 == *v51 )
          goto LABEL_43;
        ++v48;
      }
      v80 = *(_DWORD *)(v47 + 1128);
      if ( !v52 )
        v52 = v51;
      ++*(_QWORD *)(v47 + 1112);
      v75 = v80 + 1;
      if ( 4 * (v80 + 1) < 3 * v69 )
      {
        if ( v69 - *(_DWORD *)(v47 + 1132) - v75 <= v69 >> 3 )
        {
          sub_302AEB0(v117, v69);
          v81 = *(_DWORD *)(v47 + 1136);
          if ( !v81 )
          {
LABEL_150:
            ++*(_DWORD *)(v47 + 1128);
            BUG();
          }
          v82 = v81 - 1;
          v83 = *(_QWORD *)(v47 + 1120);
          v84 = 0;
          v85 = v82 & (((unsigned int)v70 >> 9) ^ ((unsigned int)v70 >> 4));
          v86 = 1;
          v75 = *(_DWORD *)(v47 + 1128) + 1;
          v52 = (__int64 *)(v83 + 40LL * v85);
          v87 = *v52;
          if ( v70 != *v52 )
          {
            while ( v87 != -4096 )
            {
              if ( !v84 && v87 == -8192 )
                v84 = v52;
              v85 = v82 & (v86 + v85);
              v52 = (__int64 *)(v83 + 40LL * v85);
              v87 = *v52;
              if ( v70 == *v52 )
                goto LABEL_63;
              ++v86;
            }
            if ( v84 )
              v52 = v84;
          }
        }
        goto LABEL_63;
      }
LABEL_61:
      sub_302AEB0(v117, 2 * v69);
      v71 = *(_DWORD *)(v47 + 1136);
      if ( !v71 )
        goto LABEL_150;
      v72 = v71 - 1;
      v73 = *(_QWORD *)(v47 + 1120);
      v74 = v72 & (((unsigned int)v70 >> 9) ^ ((unsigned int)v70 >> 4));
      v75 = *(_DWORD *)(v47 + 1128) + 1;
      v52 = (__int64 *)(v73 + 40LL * v74);
      v76 = *v52;
      if ( v70 != *v52 )
      {
        v111 = 1;
        v112 = 0;
        while ( v76 != -4096 )
        {
          if ( v76 == -8192 && !v112 )
            v112 = v52;
          v74 = v72 & (v111 + v74);
          v52 = (__int64 *)(v73 + 40LL * v74);
          v76 = *v52;
          if ( v70 == *v52 )
            goto LABEL_63;
          ++v111;
        }
        if ( v112 )
          v52 = v112;
      }
LABEL_63:
      *(_DWORD *)(v47 + 1128) = v75;
      if ( *v52 != -4096 )
        --*(_DWORD *)(v47 + 1132);
      *v52 = v70;
      ++v46;
      v52[1] = 0;
      v52[2] = 0;
      v52[3] = 0;
      *((_DWORD *)v52 + 8) = 0;
      if ( v128 == v46 )
        goto LABEL_66;
    }
LABEL_43:
    v54 = *((_DWORD *)v51 + 6);
    if ( v54 )
    {
      srca = (unsigned __int8 *)sub_3058F20(*v46);
      nc = v55;
      v56 = sub_3058FE0(v70);
      v57 = nc;
      v58 = (unsigned __int8 *)v56;
      v60 = v59;
      v61 = v133;
      if ( (unsigned __int64)(v132 - (_QWORD)v133) <= 5 )
      {
        v90 = sub_CB6200((__int64)v131, "\t.reg ", 6u);
        v57 = nc;
        v63 = *(_WORD **)(v90 + 32);
        v62 = (_QWORD *)v90;
      }
      else
      {
        *v133 = 1701981705;
        v62 = v131;
        *((_WORD *)v61 + 2) = 8295;
        v63 = (_WORD *)v133 + 3;
        v133 = (_DWORD *)((char *)v133 + 6);
      }
      v64 = v62[3] - (_QWORD)v63;
      if ( v57 > v64 )
      {
        v89 = sub_CB6200((__int64)v62, srca, v57);
        v63 = *(_WORD **)(v89 + 32);
        v62 = (_QWORD *)v89;
        v64 = *(_QWORD *)(v89 + 24) - (_QWORD)v63;
      }
      else if ( v57 )
      {
        v113 = v62;
        ne = v57;
        memcpy(v63, srca, v57);
        v62 = v113;
        v91 = v113[3];
        v63 = (_WORD *)(v113[4] + ne);
        v113[4] = v63;
        v64 = v91 - (_QWORD)v63;
      }
      if ( v64 <= 1 )
      {
        v88 = sub_CB6200((__int64)v62, (unsigned __int8 *)" \t", 2u);
        v65 = *(_BYTE **)(v88 + 32);
        v62 = (_QWORD *)v88;
      }
      else
      {
        *v63 = 2336;
        v65 = (_BYTE *)(v62[4] + 2LL);
        v62[4] = v65;
      }
      v66 = (_BYTE *)v62[3];
      if ( v60 > v66 - v65 )
      {
        v62 = (_QWORD *)sub_CB6200((__int64)v62, v58, v60);
        v66 = (_BYTE *)v62[3];
        v65 = (_BYTE *)v62[4];
      }
      else if ( v60 )
      {
        nf = (size_t)v62;
        memcpy(v65, v58, v60);
        v62 = (_QWORD *)nf;
        v92 = (_BYTE *)(*(_QWORD *)(nf + 32) + v60);
        v66 = *(_BYTE **)(nf + 24);
        *(_QWORD *)(nf + 32) = v92;
        v65 = v92;
      }
      if ( v65 == v66 )
      {
        v62 = (_QWORD *)sub_CB6200((__int64)v62, "<", 1u);
      }
      else
      {
        *v65 = 60;
        ++v62[4];
      }
      v67 = sub_CB59D0((__int64)v62, (unsigned int)(v54 + 1));
      v68 = *(_QWORD *)(v67 + 32);
      if ( (unsigned __int64)(*(_QWORD *)(v67 + 24) - v68) <= 2 )
      {
        sub_CB6200(v67, ">;\n", 3u);
      }
      else
      {
        *(_BYTE *)(v68 + 2) = 10;
        *(_WORD *)v68 = 15166;
        *(_QWORD *)(v67 + 32) += 3LL;
      }
    }
    ++v46;
  }
  while ( v128 != v46 );
LABEL_66:
  v2 = v47;
LABEL_67:
  v77 = *(__int64 **)(v2 + 224);
  v78 = v135[1];
  v79 = *v135;
  v130 = 261;
  v129[0] = v79;
  v129[1] = v78;
  sub_E99A90(v77, (__int64)v129);
  v131[0] = &unk_49DD388;
  sub_CB5840((__int64)v131);
  if ( (_BYTE *)v136[0] != v137 )
    _libc_free(v136[0]);
}
