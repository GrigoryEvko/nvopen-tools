// Function: sub_2E8BCA0
// Address: 0x2e8bca0
//
void __fastcall sub_2E8BCA0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        unsigned __int8 a4,
        char a5,
        char a6,
        char a7,
        __int64 a8)
{
  unsigned __int64 v8; // r15
  __int64 v10; // rax
  __int64 v11; // rbx
  __int64 (*v12)(void); // rax
  __int64 v13; // r12
  unsigned int v14; // ebx
  int v15; // ecx
  int v16; // eax
  __int64 v17; // rdx
  _BYTE *v18; // r13
  char v19; // al
  _WORD *v20; // rdx
  int v21; // eax
  char *v22; // rsi
  size_t v23; // rax
  void *v24; // rdi
  size_t v25; // r12
  char v26; // r13
  char i; // dl
  _BYTE *v28; // rax
  __int64 v29; // r13
  __int64 v30; // r12
  __int16 v31; // ax
  __int64 v32; // rcx
  int v33; // eax
  __int64 v34; // r8
  __int16 v35; // dx
  __int64 v36; // rax
  int *v37; // rdx
  unsigned __int64 v38; // r12
  int v39; // ecx
  int v40; // ecx
  int *v41; // r12
  unsigned __int8 *v42; // rcx
  const char *v43; // r12
  unsigned __int8 *v44; // rcx
  const char *v45; // r12
  unsigned __int8 *v46; // rdx
  const char *v47; // r12
  unsigned __int8 *v48; // rdx
  unsigned int v49; // ebx
  __int64 v50; // rax
  __int64 v51; // rax
  _QWORD *v52; // r12
  const char *v53; // rax
  __int64 v54; // rdx
  __int64 v55; // rax
  __int64 *v56; // rax
  __int64 v57; // r13
  __int64 *v58; // r12
  __int64 *v59; // rax
  __int64 v60; // rdx
  __int64 *v61; // rbx
  __int64 v62; // r15
  _BYTE *v63; // rax
  _BYTE *v64; // rax
  __int16 v65; // ax
  _BYTE *v66; // rcx
  unsigned __int8 v67; // al
  _BYTE *v68; // rdx
  __int64 v69; // rdi
  __int64 v70; // rdx
  __int64 v71; // r12
  unsigned __int8 v72; // al
  _BYTE *v73; // rdx
  __int64 v74; // rdi
  unsigned __int8 *v75; // rax
  size_t v76; // rdx
  _BYTE *v77; // rdi
  size_t v78; // r13
  _BYTE *v79; // rcx
  unsigned __int8 v80; // al
  _BYTE *v81; // rdx
  __int64 v82; // rdi
  __int64 v83; // rdx
  unsigned __int8 v84; // al
  _BYTE *v85; // rdx
  __int64 v86; // rdi
  unsigned __int8 *v87; // rax
  size_t v88; // rdx
  size_t v89; // r13
  __int64 v90; // r8
  _BYTE *v91; // rax
  __int64 v92; // rdi
  __int64 v93; // r13
  int v94; // ecx
  __int64 v95; // rdx
  unsigned __int64 v96; // r12
  int v97; // r12d
  int v98; // eax
  __int64 v99; // rdi
  __int64 v100; // rbx
  _BYTE *v101; // rax
  _BYTE *v102; // rax
  _BYTE *v103; // rax
  _BYTE *v104; // rax
  _BYTE *v105; // rax
  _BYTE *v106; // rax
  _BYTE *v107; // rax
  __int64 v108; // rax
  __int64 *v109; // rax
  __int64 v110; // rdx
  _BYTE *v111; // rax
  __int64 v112; // rbx
  __int64 v113; // rax
  _BYTE *v114; // rax
  size_t v115; // rdx
  char *v116; // rsi
  _QWORD *v117; // rdi
  unsigned int v118; // r8d
  unsigned __int64 v119; // r12
  _BYTE *v120; // rax
  __int64 v121; // rdi
  _BYTE *v122; // rax
  __int64 v123; // rax
  __int64 v124; // r12
  unsigned __int8 *v125; // rax
  size_t v126; // rdx
  void *v127; // rdi
  unsigned __int64 v128; // r9
  char *v129; // rdi
  char *v130; // rsi
  unsigned int v131; // eax
  unsigned int v132; // edi
  __int64 v133; // r8
  __int64 v134; // rax
  __int64 v135; // rax
  __int64 v136; // rdx
  __int64 v137; // [rsp+0h] [rbp-B0h]
  __int64 v138; // [rsp+0h] [rbp-B0h]
  int v139; // [rsp+0h] [rbp-B0h]
  int v140; // [rsp+0h] [rbp-B0h]
  int v141; // [rsp+0h] [rbp-B0h]
  unsigned int v142; // [rsp+8h] [rbp-A8h]
  int v143; // [rsp+8h] [rbp-A8h]
  size_t v144; // [rsp+8h] [rbp-A8h]
  _BYTE *v146; // [rsp+18h] [rbp-98h]
  _BYTE *v147; // [rsp+18h] [rbp-98h]
  int v148; // [rsp+18h] [rbp-98h]
  unsigned int v149; // [rsp+18h] [rbp-98h]
  __int64 v150; // [rsp+18h] [rbp-98h]
  int v153; // [rsp+28h] [rbp-88h]
  int v154; // [rsp+28h] [rbp-88h]
  char v155; // [rsp+33h] [rbp-7Dh]
  unsigned int v156; // [rsp+34h] [rbp-7Ch]
  int v157; // [rsp+34h] [rbp-7Ch]
  unsigned __int64 v159; // [rsp+38h] [rbp-78h]
  __int64 v160; // [rsp+40h] [rbp-70h]
  __int64 *v161; // [rsp+40h] [rbp-70h]
  __int64 v162; // [rsp+48h] [rbp-68h]
  unsigned __int8 v164; // [rsp+58h] [rbp-58h]
  __int64 *v165; // [rsp+58h] [rbp-58h]
  unsigned __int64 v166; // [rsp+68h] [rbp-48h] BYREF
  unsigned __int64 v167[2]; // [rsp+70h] [rbp-40h] BYREF
  _BYTE v168[48]; // [rsp+80h] [rbp-30h] BYREF

  v8 = a1;
  v10 = *(_QWORD *)(a1 + 24);
  v155 = a5;
  v160 = v10;
  if ( v10 )
  {
    v11 = *(_QWORD *)(v10 + 32);
    if ( v11 )
    {
      a8 = 0;
      v160 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(v11 + 16) + 200LL))(*(_QWORD *)(v11 + 16));
      v162 = *(_QWORD *)(v11 + 32);
      v12 = *(__int64 (**)(void))(**(_QWORD **)(v11 + 16) + 128LL);
      if ( v12 != sub_2DAC790 )
        a8 = v12();
    }
    else
    {
      v162 = 0;
      v160 = 0;
    }
  }
  else
  {
    v162 = 0;
  }
  v166 = 0x2000000000000001LL;
  v164 = a4;
  if ( !a4 )
    v164 = sub_2E8B990(a1);
  v13 = 0;
  v14 = 0;
  v156 = *(_DWORD *)(a1 + 40) & 0xFFFFFF;
  if ( v156 )
  {
    while ( 1 )
    {
      v18 = (_BYTE *)(v13 + *(_QWORD *)(a1 + 32));
      if ( *v18 )
        break;
      v19 = v18[3];
      if ( (v19 & 0x10) == 0 || (v19 & 0x20) != 0 )
        break;
      if ( v14 )
      {
        v20 = *(_WORD **)(a2 + 32);
        if ( *(_QWORD *)(a2 + 24) - (_QWORD)v20 <= 1u )
        {
          sub_CB6200(a2, (unsigned __int8 *)", ", 2u);
        }
        else
        {
          *v20 = 8236;
          *(_QWORD *)(a2 + 32) += 2LL;
        }
      }
      v15 = 0;
      if ( v162 )
        v15 = sub_2E8BA90(a1, v14, &v166, v162);
      v16 = 0;
      if ( v164 )
      {
        v17 = v13 + *(_QWORD *)(a1 + 32);
        if ( !*(_BYTE *)v17 && (*(_WORD *)(v17 + 2) & 0xFF0) != 0 && (*(_BYTE *)(v17 + 3) & 0x10) == 0 )
        {
          v154 = v15;
          v16 = sub_2E89F40(a1, v14);
          v15 = v154;
        }
      }
      v13 += 40;
      LODWORD(v167[0]) = v14++;
      BYTE4(v167[0]) = 1;
      sub_2EAE5A0((_DWORD)v18, a2, a3, v15, v167[0], 0, a4, v164, v16, v160);
      if ( v14 == v156 )
        goto LABEL_200;
    }
    if ( !v14 )
      goto LABEL_25;
LABEL_200:
    sub_904010(a2, " = ");
    v21 = *(_DWORD *)(a1 + 44) & 0xFFFFFF;
    if ( (*(_DWORD *)(a1 + 44) & 1) == 0 )
    {
LABEL_26:
      if ( (v21 & 2) == 0 )
        goto LABEL_27;
      goto LABEL_202;
    }
  }
  else
  {
LABEL_25:
    v14 = 0;
    v21 = *(_DWORD *)(a1 + 44) & 0xFFFFFF;
    if ( (*(_DWORD *)(a1 + 44) & 1) == 0 )
      goto LABEL_26;
  }
  sub_904010(a2, "frame-setup ");
  v21 = *(_DWORD *)(a1 + 44) & 0xFFFFFF;
  if ( (*(_BYTE *)(a1 + 44) & 2) == 0 )
  {
LABEL_27:
    if ( (v21 & 0x10) == 0 )
      goto LABEL_28;
    goto LABEL_203;
  }
LABEL_202:
  sub_904010(a2, "frame-destroy ");
  v21 = *(_DWORD *)(a1 + 44) & 0xFFFFFF;
  if ( (*(_BYTE *)(a1 + 44) & 0x10) == 0 )
  {
LABEL_28:
    if ( (v21 & 0x20) == 0 )
      goto LABEL_29;
    goto LABEL_204;
  }
LABEL_203:
  sub_904010(a2, "nnan ");
  v21 = *(_DWORD *)(a1 + 44) & 0xFFFFFF;
  if ( (*(_BYTE *)(a1 + 44) & 0x20) == 0 )
  {
LABEL_29:
    if ( (v21 & 0x40) == 0 )
      goto LABEL_30;
    goto LABEL_205;
  }
LABEL_204:
  sub_904010(a2, "ninf ");
  v21 = *(_DWORD *)(a1 + 44) & 0xFFFFFF;
  if ( (*(_BYTE *)(a1 + 44) & 0x40) == 0 )
  {
LABEL_30:
    if ( (v21 & 0x80u) == 0 )
      goto LABEL_31;
    goto LABEL_206;
  }
LABEL_205:
  sub_904010(a2, "nsz ");
  v21 = *(_DWORD *)(a1 + 44) & 0xFFFFFF;
  if ( *(char *)(a1 + 44) >= 0 )
  {
LABEL_31:
    if ( (v21 & 0x100) == 0 )
      goto LABEL_32;
    goto LABEL_207;
  }
LABEL_206:
  sub_904010(a2, "arcp ");
  v21 = *(_DWORD *)(a1 + 44) & 0xFFFFFF;
  if ( (v21 & 0x100) == 0 )
  {
LABEL_32:
    if ( (v21 & 0x200) == 0 )
      goto LABEL_33;
    goto LABEL_208;
  }
LABEL_207:
  sub_904010(a2, "contract ");
  v21 = *(_DWORD *)(a1 + 44) & 0xFFFFFF;
  if ( (v21 & 0x200) == 0 )
  {
LABEL_33:
    if ( (v21 & 0x400) == 0 )
      goto LABEL_34;
    goto LABEL_209;
  }
LABEL_208:
  sub_904010(a2, "afn ");
  v21 = *(_DWORD *)(a1 + 44) & 0xFFFFFF;
  if ( (v21 & 0x400) == 0 )
  {
LABEL_34:
    if ( (v21 & 0x800) == 0 )
      goto LABEL_35;
    goto LABEL_210;
  }
LABEL_209:
  sub_904010(a2, "reassoc ");
  v21 = *(_DWORD *)(a1 + 44) & 0xFFFFFF;
  if ( (v21 & 0x800) == 0 )
  {
LABEL_35:
    if ( (v21 & 0x1000) == 0 )
      goto LABEL_36;
    goto LABEL_211;
  }
LABEL_210:
  sub_904010(a2, "nuw ");
  v21 = *(_DWORD *)(a1 + 44) & 0xFFFFFF;
  if ( (v21 & 0x1000) == 0 )
  {
LABEL_36:
    if ( (v21 & 0x2000) == 0 )
      goto LABEL_37;
    goto LABEL_212;
  }
LABEL_211:
  sub_904010(a2, "nsw ");
  v21 = *(_DWORD *)(a1 + 44) & 0xFFFFFF;
  if ( (v21 & 0x2000) == 0 )
  {
LABEL_37:
    if ( (v21 & 0x4000) == 0 )
      goto LABEL_38;
    goto LABEL_213;
  }
LABEL_212:
  sub_904010(a2, "exact ");
  v21 = *(_DWORD *)(a1 + 44) & 0xFFFFFF;
  if ( (v21 & 0x4000) == 0 )
  {
LABEL_38:
    if ( (v21 & 0x8000) == 0 )
      goto LABEL_39;
    goto LABEL_214;
  }
LABEL_213:
  sub_904010(a2, "nofpexcept ");
  v21 = *(_DWORD *)(a1 + 44) & 0xFFFFFF;
  if ( (v21 & 0x8000) == 0 )
  {
LABEL_39:
    if ( (v21 & 0x40000) == 0 )
      goto LABEL_40;
    goto LABEL_215;
  }
LABEL_214:
  sub_904010(a2, "nomerge ");
  v21 = *(_DWORD *)(a1 + 44) & 0xFFFFFF;
  if ( (*(_DWORD *)(a1 + 44) & 0x40000) == 0 )
  {
LABEL_40:
    if ( (v21 & 0x80000) == 0 )
      goto LABEL_41;
    goto LABEL_216;
  }
LABEL_215:
  sub_904010(a2, "nneg ");
  v21 = *(_DWORD *)(a1 + 44) & 0xFFFFFF;
  if ( (*(_DWORD *)(a1 + 44) & 0x80000) == 0 )
  {
LABEL_41:
    if ( (v21 & 0x200000) == 0 )
      goto LABEL_42;
    goto LABEL_217;
  }
LABEL_216:
  sub_904010(a2, "disjoint ");
  if ( (*(_DWORD *)(a1 + 44) & 0x200000) == 0 )
  {
LABEL_42:
    if ( a8 )
      goto LABEL_43;
    goto LABEL_218;
  }
LABEL_217:
  sub_904010(a2, "samesign ");
  if ( a8 )
  {
LABEL_43:
    v22 = (char *)(*(_QWORD *)(a8 + 24) + *(unsigned int *)(*(_QWORD *)(a8 + 16) + 4LL * *(unsigned __int16 *)(a1 + 68)));
    if ( v22 )
    {
      v23 = strlen(v22);
      v24 = *(void **)(a2 + 32);
      v25 = v23;
      if ( *(_QWORD *)(a2 + 24) - (_QWORD)v24 < v23 )
      {
        sub_CB6200(a2, (unsigned __int8 *)v22, v23);
      }
      else if ( v23 )
      {
        memcpy(v24, v22, v23);
        *(_QWORD *)(a2 + 32) += v25;
      }
    }
    goto LABEL_47;
  }
LABEL_218:
  sub_904010(a2, "UNKNOWN");
LABEL_47:
  if ( a5 )
    goto LABEL_178;
  v26 = 1;
  v153 = -1;
  if ( (unsigned int)*(unsigned __int16 *)(v8 + 68) - 1 <= 1 && v156 > 1 )
  {
    v97 = 0;
    sub_904010(a2, " ");
    if ( v162 )
      v97 = sub_2E8BA90(v8, 0, &v166, v162);
    v98 = 0;
    v99 = *(_QWORD *)(v8 + 32);
    if ( v164 && !*(_BYTE *)v99 && (*(_WORD *)(v99 + 2) & 0xFF0) != 0 && (*(_BYTE *)(v99 + 3) & 0x10) == 0 )
    {
      v98 = sub_2E89F40(v8, 0);
      v99 = *(_QWORD *)(v8 + 32);
    }
    v167[0] = 0x100000000LL;
    sub_2EAE5A0(v99, a2, a3, v97, 0, 1, a4, v164, v98, v160);
    v100 = *(_QWORD *)(*(_QWORD *)(v8 + 32) + 64LL);
    if ( (v100 & 1) != 0 )
      sub_904010(a2, " [sideeffect]");
    if ( (v100 & 8) != 0 )
      sub_904010(a2, " [mayload]");
    if ( (v100 & 0x10) != 0 )
      sub_904010(a2, " [maystore]");
    if ( (v100 & 0x20) != 0 )
      sub_904010(a2, " [isconvergent]");
    if ( (v100 & 2) != 0 )
      sub_904010(a2, " [alignstack]");
    if ( !(unsigned int)sub_2E89090(v8) )
      sub_904010(a2, " [attdialect]");
    if ( (unsigned int)sub_2E89090(v8) == 1 )
      sub_904010(a2, " [inteldialect]");
    v153 = 2;
    v26 = 0;
    v14 = 2;
  }
  v157 = *(_DWORD *)(v8 + 40) & 0xFFFFFF;
  if ( v157 != v14 )
  {
    v142 = 0;
    for ( i = v26; ; i = v155 )
    {
      v28 = *(_BYTE **)(a2 + 32);
      v29 = 40LL * v14;
      v30 = v29 + *(_QWORD *)(v8 + 32);
      if ( !i )
      {
        if ( v28 == *(_BYTE **)(a2 + 24) )
        {
          sub_CB6200(a2, (unsigned __int8 *)",", 1u);
          v28 = *(_BYTE **)(a2 + 32);
        }
        else
        {
          *v28 = 44;
          v28 = (_BYTE *)(*(_QWORD *)(a2 + 32) + 1LL);
          *(_QWORD *)(a2 + 32) = v28;
        }
      }
      if ( *(_BYTE **)(a2 + 24) == v28 )
      {
        sub_CB6200(a2, (unsigned __int8 *)" ", 1u);
        v31 = *(_WORD *)(v8 + 68);
        if ( (unsigned __int16)(v31 - 14) > 2u )
        {
LABEL_135:
          if ( v31 == 18 && *(_BYTE *)v30 == 14 )
          {
            v66 = *(_BYTE **)(v30 + 24);
            if ( *v66 == 27 )
            {
              v146 = v66 - 16;
              v67 = *(v66 - 16);
              v68 = (v67 & 2) != 0 ? (_BYTE *)*((_QWORD *)v66 - 4) : &v146[-8 * ((v67 >> 2) & 0xF)];
              v69 = *((_QWORD *)v68 + 1);
              v137 = *(_QWORD *)(v30 + 24);
              if ( v69 )
              {
                sub_B91420(v69);
                if ( v70 )
                {
                  v71 = sub_904010(a2, "\"");
                  v72 = *(_BYTE *)(v137 - 16);
                  if ( (v72 & 2) != 0 )
                    v73 = *(_BYTE **)(v137 - 32);
                  else
                    v73 = &v146[-8 * ((v72 >> 2) & 0xF)];
                  v74 = *((_QWORD *)v73 + 1);
                  if ( v74 )
                  {
                    v75 = (unsigned __int8 *)sub_B91420(v74);
                    v77 = *(_BYTE **)(v71 + 32);
                    v78 = v76;
                    if ( v76 > *(_QWORD *)(v71 + 24) - (_QWORD)v77 )
                    {
                      v71 = sub_CB6200(v71, v75, v76);
                      goto LABEL_147;
                    }
                    if ( v76 )
                    {
                      memcpy(v77, v75, v76);
                      v77 = (_BYTE *)(v78 + *(_QWORD *)(v71 + 32));
                      *(_QWORD *)(v71 + 32) = v77;
                    }
                  }
                  else
                  {
LABEL_147:
                    v77 = *(_BYTE **)(v71 + 32);
                  }
                  if ( (unsigned __int64)v77 >= *(_QWORD *)(v71 + 24) )
                  {
LABEL_149:
                    sub_CB5D20(v71, 34);
                    goto LABEL_69;
                  }
LABEL_165:
                  *(_QWORD *)(v71 + 32) = v77 + 1;
                  *v77 = 34;
LABEL_69:
                  if ( ++v14 == v157 )
                    goto LABEL_73;
                  continue;
                }
              }
            }
LABEL_166:
            if ( v162 )
              LODWORD(v32) = sub_2E8BA90(v8, v14, &v166, v162);
            else
              LODWORD(v32) = 0;
            v33 = 0;
            if ( v164 )
            {
              v90 = v29 + *(_QWORD *)(v8 + 32);
              if ( !*(_BYTE *)v90 && (*(_WORD *)(v90 + 2) & 0xFF0) != 0 && (*(_BYTE *)(v90 + 3) & 0x10) == 0 )
              {
                v148 = v32;
                v33 = sub_2E89F40(v8, v14);
                LODWORD(v32) = v148;
              }
            }
LABEL_68:
            LODWORD(v167[0]) = v14;
            BYTE4(v167[0]) = 1;
            sub_2EAE5A0(v30, a2, a3, v32, v14, 1, a4, v164, v33, v160);
            goto LABEL_69;
          }
          goto LABEL_56;
        }
      }
      else
      {
        *v28 = 32;
        ++*(_QWORD *)(a2 + 32);
        v31 = *(_WORD *)(v8 + 68);
        if ( (unsigned __int16)(v31 - 14) > 2u )
          goto LABEL_135;
      }
      if ( *(_BYTE *)v30 == 14 )
      {
        v79 = *(_BYTE **)(v30 + 24);
        if ( *v79 != 26 )
          goto LABEL_166;
        v147 = v79 - 16;
        v80 = *(v79 - 16);
        v81 = (v80 & 2) != 0 ? (_BYTE *)*((_QWORD *)v79 - 4) : &v147[-8 * ((v80 >> 2) & 0xF)];
        v82 = *((_QWORD *)v81 + 1);
        v138 = *(_QWORD *)(v30 + 24);
        if ( !v82 )
          goto LABEL_166;
        sub_B91420(v82);
        if ( !v83 )
          goto LABEL_166;
        v71 = sub_904010(a2, "!\"");
        v84 = *(_BYTE *)(v138 - 16);
        if ( (v84 & 2) != 0 )
          v85 = *(_BYTE **)(v138 - 32);
        else
          v85 = &v147[-8 * ((v84 >> 2) & 0xF)];
        v86 = *((_QWORD *)v85 + 1);
        if ( v86 )
        {
          v87 = (unsigned __int8 *)sub_B91420(v86);
          v77 = *(_BYTE **)(v71 + 32);
          v89 = v88;
          if ( *(_QWORD *)(v71 + 24) - (_QWORD)v77 < v88 )
          {
            v71 = sub_CB6200(v71, v87, v88);
            goto LABEL_163;
          }
          if ( v88 )
          {
            memcpy(v77, v87, v88);
            v77 = (_BYTE *)(v89 + *(_QWORD *)(v71 + 32));
            *(_QWORD *)(v71 + 32) = v77;
          }
        }
        else
        {
LABEL_163:
          v77 = *(_BYTE **)(v71 + 32);
        }
        if ( *(_QWORD *)(v71 + 24) <= (unsigned __int64)v77 )
          goto LABEL_149;
        goto LABEL_165;
      }
LABEL_56:
      if ( v153 == v14 && *(_BYTE *)v30 == 1 )
      {
        v91 = *(_BYTE **)(a2 + 32);
        if ( (unsigned __int64)v91 >= *(_QWORD *)(a2 + 24) )
        {
          v92 = sub_CB5D20(a2, 36);
        }
        else
        {
          v92 = a2;
          *(_QWORD *)(a2 + 32) = v91 + 1;
          *v91 = 36;
        }
        v149 = v142 + 1;
        sub_CB59D0(v92, v142);
        v93 = *(_QWORD *)(v30 + 24);
        sub_904010(a2, ":[");
        v94 = v93 & 7;
        switch ( (char)v94 )
        {
          case 0:
            BUG();
          case 1:
            v115 = 6;
            v116 = "reguse";
            break;
          case 2:
            v115 = 6;
            v116 = "regdef";
            break;
          case 3:
            v115 = 9;
            v116 = "regdef-ec";
            break;
          case 4:
            v115 = 7;
            v116 = "clobber";
            break;
          case 5:
            v115 = 3;
            v116 = "imm";
            break;
          case 6:
          case 7:
            v115 = 3;
            v116 = "mem";
            break;
        }
        v117 = *(_QWORD **)(a2 + 32);
        if ( *(_QWORD *)(a2 + 24) - (_QWORD)v117 < v115 )
        {
          sub_CB6200(a2, (unsigned __int8 *)v116, v115);
          v94 = v93 & 7;
        }
        else
        {
          if ( (unsigned int)v115 >= 8 )
          {
            v128 = (unsigned __int64)(v117 + 1) & 0xFFFFFFFFFFFFFFF8LL;
            *v117 = *(_QWORD *)v116;
            *(_QWORD *)((char *)v117 + v115 - 8) = *(_QWORD *)&v116[v115 - 8];
            v129 = (char *)v117 - v128;
            v130 = (char *)(v116 - v129);
            if ( (((_DWORD)v115 + (_DWORD)v129) & 0xFFFFFFF8) >= 8 )
            {
              v131 = (v115 + (_DWORD)v129) & 0xFFFFFFF8;
              v132 = 0;
              do
              {
                v133 = v132;
                v132 += 8;
                *(_QWORD *)(v128 + v133) = *(_QWORD *)&v130[v133];
              }
              while ( v132 < v131 );
            }
          }
          else if ( (v115 & 4) != 0 )
          {
            *(_DWORD *)v117 = *(_DWORD *)v116;
            *(_DWORD *)((char *)v117 + (unsigned int)v115 - 4) = *(_DWORD *)&v116[(unsigned int)v115 - 4];
          }
          else
          {
            *(_BYTE *)v117 = *v116;
            *(_WORD *)((char *)v117 + (unsigned int)v115 - 2) = *(_WORD *)&v116[(unsigned int)v115 - 2];
          }
          *(_QWORD *)(a2 + 32) += v115;
        }
        v118 = (unsigned int)v93 >> 31;
        if ( v94 != 5 )
        {
          if ( v94 != 6 )
          {
            if ( (int)v93 >= 0 )
            {
              if ( (v93 & 0x3FFF0000) != 0 )
              {
                v119 = (WORD1(v93) & 0x3FFFu) - 1;
                if ( v160 )
                {
                  v120 = *(_BYTE **)(a2 + 32);
                  if ( (unsigned __int64)v120 >= *(_QWORD *)(a2 + 24) )
                  {
                    v141 = v94;
                    v135 = sub_CB5D20(a2, 58);
                    v94 = v141;
                    v121 = v135;
                  }
                  else
                  {
                    v121 = a2;
                    *(_QWORD *)(a2 + 32) = v120 + 1;
                    *v120 = 58;
                  }
                  v139 = v94;
                  sub_904010(
                    v121,
                    (const char *)(*(_QWORD *)(v160 + 80)
                                 + *(unsigned int *)(**(_QWORD **)(*(_QWORD *)(v160 + 280) + 8 * v119) + 16LL)));
                  v94 = v139;
                }
                else
                {
                  v140 = v94;
                  v134 = sub_904010(a2, ":RC");
                  sub_CB59D0(v134, v119);
                  v94 = v140;
                }
              }
LABEL_307:
              if ( ((v94 & 0xFD) == 1 || v94 == 2) && (v93 & 0x40000000) != 0 )
                sub_904010(a2, " foldable");
LABEL_311:
              v122 = *(_BYTE **)(a2 + 32);
              if ( (unsigned __int64)v122 >= *(_QWORD *)(a2 + 24) )
              {
                sub_CB5D20(a2, 93);
              }
              else
              {
                *(_QWORD *)(a2 + 32) = v122 + 1;
                *v122 = 93;
              }
              v153 += ((unsigned __int16)v93 >> 3) + 1;
              v142 = v149;
              goto LABEL_69;
            }
LABEL_324:
            v143 = v94;
            v123 = sub_904010(a2, " tiedto:$");
            sub_CB59D0(v123, WORD1(v93) & 0x7FFF);
            v94 = v143;
            goto LABEL_307;
          }
          v124 = sub_904010(a2, ":");
          v125 = (unsigned __int8 *)sub_2E862C0(WORD1(v93) & 0x7FFF);
          v127 = *(void **)(v124 + 32);
          v94 = 6;
          v118 = (unsigned int)v93 >> 31;
          if ( *(_QWORD *)(v124 + 24) - (_QWORD)v127 < v126 )
          {
            sub_CB6200(v124, v125, v126);
            v94 = 6;
            v118 = (unsigned int)v93 >> 31;
          }
          else if ( v126 )
          {
            v144 = v126;
            memcpy(v127, v125, v126);
            *(_QWORD *)(v124 + 32) += v144;
            v118 = (unsigned int)v93 >> 31;
            v94 = 6;
          }
        }
        if ( !v118 )
          goto LABEL_311;
        goto LABEL_324;
      }
      if ( v162 )
        v32 = sub_2E8BA90(v8, v14, &v166, v162);
      else
        v32 = 0;
      v33 = 0;
      if ( v164 )
      {
        v34 = v29 + *(_QWORD *)(v8 + 32);
        if ( !*(_BYTE *)v34 && (*(_WORD *)(v34 + 2) & 0xFF0) != 0 && (*(_BYTE *)(v34 + 3) & 0x10) == 0 )
        {
          v150 = v32;
          v33 = sub_2E89F40(v8, v14);
          v32 = v150;
        }
      }
      if ( *(_BYTE *)v30 != 1 )
        goto LABEL_68;
      v35 = *(_WORD *)(v8 + 68);
      switch ( v35 )
      {
        case 8:
          if ( v14 != 2 )
            goto LABEL_68;
          break;
        case 9:
          if ( v14 != 3 )
            goto LABEL_68;
          break;
        case 19:
          if ( v14 <= 1 || (v14 & 1) != 0 )
            goto LABEL_68;
          break;
        default:
          if ( v14 != 3 || v35 != 12 )
            goto LABEL_68;
          break;
      }
      ++v14;
      sub_2EAB970(a2, *(_QWORD *)(v30 + 24), v160, v32);
      if ( v14 == v157 )
      {
LABEL_73:
        v26 = 0;
        break;
      }
    }
  }
  v36 = *(_QWORD *)(v8 + 48);
  v37 = (int *)(v36 & 0xFFFFFFFFFFFFFFF8LL);
  if ( (v36 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    goto LABEL_102;
  v38 = *(_QWORD *)(v8 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  v39 = v36 & 7;
  if ( v39 == 1 )
    goto LABEL_76;
  if ( v39 != 3 )
  {
    if ( v39 == 2 )
    {
LABEL_79:
      v41 = v37;
      goto LABEL_80;
    }
LABEL_196:
    v42 = (unsigned __int8 *)(v36 & 0xFFFFFFFFFFFFFFF8LL);
    if ( (v36 & 7) != 3 )
      goto LABEL_197;
    goto LABEL_83;
  }
  if ( *((_BYTE *)v37 + 4) )
  {
    v38 = *(_QWORD *)&v37[2 * *v37 + 4];
    if ( v38 )
    {
LABEL_76:
      if ( !v26 )
      {
        v103 = *(_BYTE **)(a2 + 32);
        if ( (unsigned __int64)v103 >= *(_QWORD *)(a2 + 24) )
        {
          sub_CB5D20(a2, 44);
        }
        else
        {
          *(_QWORD *)(a2 + 32) = v103 + 1;
          *v103 = 44;
        }
      }
      sub_904010(a2, " pre-instr-symbol ");
      sub_2EABE30(a2, v38);
      v36 = *(_QWORD *)(v8 + 48);
      v37 = (int *)(v36 & 0xFFFFFFFFFFFFFFF8LL);
      if ( (v36 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
        goto LABEL_102;
      v40 = v36 & 7;
      if ( v40 == 2 )
        goto LABEL_79;
      if ( v40 != 3 )
        goto LABEL_196;
    }
  }
  if ( !*((_BYTE *)v37 + 5) )
    goto LABEL_196;
  v41 = *(int **)&v37[2 * *((unsigned __int8 *)v37 + 4) + 4 + 2 * (__int64)*v37];
  if ( !v41 )
    goto LABEL_196;
LABEL_80:
  if ( !v26 )
  {
    v106 = *(_BYTE **)(a2 + 32);
    if ( (unsigned __int64)v106 >= *(_QWORD *)(a2 + 24) )
    {
      sub_CB5D20(a2, 44);
    }
    else
    {
      *(_QWORD *)(a2 + 32) = v106 + 1;
      *v106 = 44;
    }
  }
  sub_904010(a2, " post-instr-symbol ");
  sub_2EABE30(a2, v41);
  v36 = *(_QWORD *)(v8 + 48);
  v42 = (unsigned __int8 *)(v36 & 0xFFFFFFFFFFFFFFF8LL);
  if ( (v36 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    goto LABEL_102;
  if ( (v36 & 7) != 3 )
  {
LABEL_197:
    v44 = (unsigned __int8 *)(v36 & 0xFFFFFFFFFFFFFFF8LL);
    if ( (v36 & 7) != 3 )
      goto LABEL_198;
    goto LABEL_88;
  }
LABEL_83:
  if ( !v42[6] )
    goto LABEL_197;
  v43 = *(const char **)&v42[8 * *(int *)v42 + 16 + 8 * (__int64)(v42[5] + v42[4])];
  if ( !v43 )
    goto LABEL_197;
  if ( !v26 )
  {
    v104 = *(_BYTE **)(a2 + 32);
    if ( (unsigned __int64)v104 >= *(_QWORD *)(a2 + 24) )
    {
      sub_CB5D20(a2, 44);
    }
    else
    {
      *(_QWORD *)(a2 + 32) = v104 + 1;
      *v104 = 44;
    }
  }
  sub_904010(a2, " heap-alloc-marker ");
  sub_A61DC0(v43, a2, a3, 0);
  v36 = *(_QWORD *)(v8 + 48);
  v44 = (unsigned __int8 *)(v36 & 0xFFFFFFFFFFFFFFF8LL);
  if ( (v36 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    goto LABEL_102;
  if ( (v36 & 7) != 3 )
  {
LABEL_198:
    v46 = (unsigned __int8 *)(v36 & 0xFFFFFFFFFFFFFFF8LL);
    if ( (v36 & 7) == 3 )
      goto LABEL_93;
LABEL_199:
    v48 = (unsigned __int8 *)(v36 & 0xFFFFFFFFFFFFFFF8LL);
LABEL_97:
    if ( (v36 & 7) == 3 )
    {
      if ( v48[8] )
      {
        v49 = *(_DWORD *)&v48[8 * *(int *)v48 + 16 + 8 * v48[7] + 8 * v48[6] + 8 * (__int64)(v48[5] + v48[4])];
        if ( v49 )
        {
          if ( !v26 )
          {
            v111 = *(_BYTE **)(a2 + 32);
            if ( (unsigned __int64)v111 >= *(_QWORD *)(a2 + 24) )
            {
              sub_CB5D20(a2, 44);
            }
            else
            {
              *(_QWORD *)(a2 + 32) = v111 + 1;
              *v111 = 44;
            }
          }
          v50 = sub_904010(a2, " cfi-type ");
          sub_CB59D0(v50, v49);
        }
      }
    }
    goto LABEL_102;
  }
LABEL_88:
  if ( !v44[7] )
    goto LABEL_198;
  v45 = *(const char **)&v44[8 * v44[6] + 16 + 8 * *(int *)v44 + 8 * (__int64)(v44[5] + v44[4])];
  if ( !v45 )
    goto LABEL_198;
  if ( !v26 )
  {
    v107 = *(_BYTE **)(a2 + 32);
    if ( (unsigned __int64)v107 >= *(_QWORD *)(a2 + 24) )
    {
      sub_CB5D20(a2, 44);
    }
    else
    {
      *(_QWORD *)(a2 + 32) = v107 + 1;
      *v107 = 44;
    }
  }
  sub_904010(a2, " pcsections ");
  sub_A61DC0(v45, a2, a3, 0);
  v36 = *(_QWORD *)(v8 + 48);
  v46 = (unsigned __int8 *)(v36 & 0xFFFFFFFFFFFFFFF8LL);
  if ( (v36 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    goto LABEL_102;
  if ( (v36 & 7) != 3 )
    goto LABEL_199;
LABEL_93:
  if ( !v46[9] )
    goto LABEL_199;
  v47 = *(const char **)&v46[8 * v46[7] + 16 + 8 * v46[6] + 8 * v46[5] + 8 * v46[4] + 8 * (__int64)*(int *)v46];
  if ( !v47 )
    goto LABEL_199;
  if ( !v26 )
  {
    v105 = *(_BYTE **)(a2 + 32);
    if ( (unsigned __int64)v105 >= *(_QWORD *)(a2 + 24) )
    {
      sub_CB5D20(a2, 44);
    }
    else
    {
      *(_QWORD *)(a2 + 32) = v105 + 1;
      *v105 = 44;
    }
  }
  sub_904010(a2, " mmra ");
  sub_A61DC0(v47, a2, a3, 0);
  v36 = *(_QWORD *)(v8 + 48);
  v48 = (unsigned __int8 *)(v36 & 0xFFFFFFFFFFFFFFF8LL);
  if ( (v36 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
    goto LABEL_97;
LABEL_102:
  if ( *(_DWORD *)(v8 + 64) )
  {
    if ( !v26 )
      sub_904010(a2, ",");
    v51 = sub_904010(a2, " debug-instr-number ");
    sub_CB59D0(v51, *(unsigned int *)(v8 + 64));
  }
  if ( a6 )
  {
    sub_2E864A0(v8);
    if ( !v95 )
      goto LABEL_178;
  }
  else
  {
    v52 = (_QWORD *)(v8 + 56);
    if ( *(_QWORD *)(v8 + 56) )
    {
      if ( !v26 )
      {
        v102 = *(_BYTE **)(a2 + 32);
        if ( (unsigned __int64)v102 >= *(_QWORD *)(a2 + 24) )
        {
          sub_CB5D20(a2, 44);
        }
        else
        {
          *(_QWORD *)(a2 + 32) = v102 + 1;
          *v102 = 44;
        }
      }
      sub_904010(a2, " debug-location ");
      v53 = (const char *)sub_B10CD0(v8 + 56);
      sub_A61DC0(v53, a2, a3, 0);
    }
    sub_2E864A0(v8);
    if ( !v54 )
    {
LABEL_123:
      if ( *(_QWORD *)(v8 + 56) )
      {
        v63 = *(_BYTE **)(a2 + 32);
        if ( (unsigned __int64)v63 >= *(_QWORD *)(a2 + 24) )
        {
          sub_CB5D20(a2, 59);
        }
        else
        {
          *(_QWORD *)(a2 + 32) = v63 + 1;
          *v63 = 59;
        }
        v64 = *(_BYTE **)(a2 + 32);
        if ( (unsigned __int64)v64 >= *(_QWORD *)(a2 + 24) )
        {
          sub_CB5D20(a2, 32);
        }
        else
        {
          *(_QWORD *)(a2 + 32) = v64 + 1;
          *v64 = 32;
        }
        sub_B10EE0(v52, a2);
        v155 = 1;
      }
      v65 = *(_WORD *)(v8 + 68);
      if ( v65 == 14 )
      {
        if ( (*(_DWORD *)(v8 + 40) & 0xFFFFFFu) > 3 )
          goto LABEL_243;
      }
      else if ( v65 == 15 )
      {
        if ( (*(_DWORD *)(v8 + 40) & 0xFFFFFFu) > 1 )
          goto LABEL_243;
      }
      else if ( v65 == 16 && (*(_DWORD *)(v8 + 40) & 0xFFFFFFu) > 2 )
      {
LABEL_243:
        if ( *(_BYTE *)sub_2E89130(v8) == 14 )
        {
          if ( !v155 )
            sub_904010(a2, ";");
          v112 = sub_2E89170(v8);
          v113 = sub_904010(a2, " line no:");
          sub_CB59D0(v113, *(unsigned int *)(v112 + 16));
          if ( *(_WORD *)(v8 + 68) == 14 )
          {
            v114 = *(_BYTE **)(v8 + 32);
            if ( v114[40] == 1 && !*v114 )
              sub_904010(a2, " indirect");
          }
        }
      }
      if ( (_BYTE)qword_5020188 )
      {
        v108 = sub_904010(a2, " ; ");
        sub_CB5A80(v108, v8);
      }
      if ( a7 )
      {
        v101 = *(_BYTE **)(a2 + 32);
        if ( (unsigned __int64)v101 >= *(_QWORD *)(a2 + 24) )
        {
          sub_CB5D20(a2, 10);
        }
        else
        {
          *(_QWORD *)(a2 + 32) = v101 + 1;
          *v101 = 10;
        }
      }
      goto LABEL_178;
    }
  }
  v167[1] = 0;
  v167[0] = (unsigned __int64)v168;
  v55 = *(_QWORD *)(v8 + 24);
  if ( v55 && (v56 = *(__int64 **)(v55 + 32)) != 0 )
  {
    v57 = v56[6];
    LODWORD(v58) = sub_B2BE50(*v56);
    sub_904010(a2, " :: ");
    v59 = (__int64 *)sub_2E864A0(v8);
    v161 = 0;
    v165 = &v59[v60];
    if ( v59 == v165 )
      goto LABEL_119;
  }
  else
  {
    v109 = (__int64 *)sub_22077B0(8u);
    v58 = v109;
    if ( v109 )
    {
      sub_B6EEA0(v109);
      sub_904010(a2, " :: ");
      v59 = (__int64 *)sub_2E864A0(v8);
      v161 = v58;
      v165 = &v59[v110];
      if ( v59 == v165 )
      {
LABEL_118:
        sub_B6E710(v161);
        j_j___libc_free_0((unsigned __int64)v161);
        goto LABEL_119;
      }
      LODWORD(v57) = 0;
    }
    else
    {
      sub_904010(a2, " :: ");
      v59 = (__int64 *)sub_2E864A0(v8);
      v165 = &v59[v136];
      if ( v165 == v59 )
        goto LABEL_119;
      v161 = 0;
      LODWORD(v57) = 0;
    }
  }
  v61 = v59 + 1;
  v159 = v8;
  sub_2EAC530(*v59, a2, a3, (unsigned int)v167, (_DWORD)v58, v57, a8);
  while ( v61 != v165 )
  {
    v62 = *v61++;
    sub_904010(a2, ", ");
    sub_2EAC530(v62, a2, a3, (unsigned int)v167, (_DWORD)v58, v57, a8);
  }
  v8 = v159;
  if ( v161 )
    goto LABEL_118;
LABEL_119:
  if ( (_BYTE *)v167[0] != v168 )
    _libc_free(v167[0]);
  if ( !a6 )
  {
    v52 = (_QWORD *)(v8 + 56);
    goto LABEL_123;
  }
LABEL_178:
  v96 = v166;
  if ( (v166 & 1) == 0 && v166 )
  {
    if ( *(_QWORD *)v166 != v166 + 16 )
      _libc_free(*(_QWORD *)v166);
    j_j___libc_free_0(v96);
  }
}
