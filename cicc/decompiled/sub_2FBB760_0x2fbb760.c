// Function: sub_2FBB760
// Address: 0x2fbb760
//
void __fastcall sub_2FBB760(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  __int64 v8; // rax
  int **v9; // r14
  int **i; // rbx
  int *v11; // r13
  unsigned int v12; // eax
  __int64 v13; // r9
  __int64 v14; // rdi
  int **v15; // rax
  unsigned __int8 v16; // al
  __int64 v17; // rsi
  unsigned __int8 v18; // bl
  __int64 v19; // rdx
  __int64 v20; // rcx
  unsigned __int64 v21; // r8
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // rcx
  __int64 v25; // rdx
  __int64 v26; // r13
  int *j; // rbx
  __int64 v28; // rsi
  __int64 *v29; // r12
  char *v30; // rdx
  unsigned __int64 v31; // rcx
  __int64 v32; // r8
  __int64 v33; // r9
  __int64 v34; // r9
  __int64 v35; // r14
  unsigned int v36; // eax
  unsigned __int64 v37; // rdx
  __int64 v38; // r15
  unsigned int v39; // eax
  __int64 v40; // rcx
  __int64 *v41; // r15
  __int64 v42; // rax
  unsigned int v43; // r12d
  __int64 v44; // rax
  __int64 v45; // rdx
  __int64 k; // rax
  __int64 v47; // r9
  __int64 v48; // rdi
  __int64 v49; // rdx
  int v50; // eax
  int v51; // r14d
  __int64 v52; // r12
  unsigned int v53; // eax
  unsigned __int64 v54; // rcx
  __int64 v55; // r13
  __int64 v56; // r15
  unsigned __int64 v57; // r12
  __int64 v58; // rbx
  __int64 v59; // rax
  __int64 v60; // r8
  unsigned int v61; // r15d
  unsigned int v62; // esi
  __int64 v63; // rcx
  unsigned int v64; // edi
  int v65; // r10d
  __int64 v66; // rdx
  unsigned int v67; // r11d
  unsigned int *v68; // r13
  int v69; // eax
  __int64 v70; // r9
  __int64 v71; // rax
  __int64 v72; // r9
  int v73; // ecx
  int v74; // ecx
  __int64 v75; // rdi
  int v76; // edx
  unsigned int *v77; // rax
  unsigned int v78; // esi
  void **v79; // r13
  unsigned int *v80; // rbx
  unsigned __int64 v81; // rdi
  unsigned __int64 v82; // rdx
  unsigned __int64 v83; // r8
  unsigned __int64 v84; // rdi
  int v85; // r11d
  unsigned int v86; // r10d
  unsigned int *v87; // rdx
  int v88; // ecx
  int v89; // ecx
  int v90; // ecx
  __int64 v91; // rdi
  int v92; // r10d
  unsigned int v93; // r13d
  unsigned int v94; // esi
  int v95; // r9d
  size_t v96; // rdx
  int v97; // r9d
  size_t v98; // rdx
  unsigned int v99; // eax
  __int64 v100; // rdx
  __int64 *v101; // rbx
  __int64 v102; // rax
  __int64 v103; // rbx
  _DWORD *v104; // rax
  _DWORD *v105; // rcx
  __int64 v106; // r8
  unsigned __int64 v107; // r15
  __int64 *v108; // rax
  __int64 *v109; // rsi
  __int64 v110; // r12
  unsigned __int64 v111; // r10
  _QWORD *v112; // rax
  _QWORD *v113; // rsi
  int v114; // r11d
  __int64 v115; // rdx
  __int64 v116; // rcx
  __int64 v117; // r8
  __int64 v118; // r9
  int v119; // r11d
  unsigned int *v120; // r10
  int v121; // [rsp+Ch] [rbp-154h]
  __int64 v122; // [rsp+10h] [rbp-150h]
  int v123; // [rsp+10h] [rbp-150h]
  int v124; // [rsp+18h] [rbp-148h]
  __int64 v125; // [rsp+18h] [rbp-148h]
  int v126; // [rsp+18h] [rbp-148h]
  __int64 v127; // [rsp+28h] [rbp-138h]
  int v128; // [rsp+38h] [rbp-128h]
  _BYTE *v129; // [rsp+40h] [rbp-120h]
  __int64 v130; // [rsp+40h] [rbp-120h]
  int v131; // [rsp+48h] [rbp-118h]
  __int64 v133; // [rsp+58h] [rbp-108h]
  unsigned __int64 v134; // [rsp+58h] [rbp-108h]
  unsigned __int64 v135[2]; // [rsp+68h] [rbp-F8h] BYREF
  _BYTE v136[32]; // [rsp+78h] [rbp-E8h] BYREF
  int v137; // [rsp+98h] [rbp-C8h]
  __int64 v138; // [rsp+A0h] [rbp-C0h]
  __int64 v139; // [rsp+A8h] [rbp-B8h]
  __int64 v140; // [rsp+B0h] [rbp-B0h]
  __int64 v141; // [rsp+B8h] [rbp-A8h]
  void *p_dest; // [rsp+C0h] [rbp-A0h] BYREF
  __int64 v143; // [rsp+C8h] [rbp-98h]
  void *dest; // [rsp+D0h] [rbp-90h] BYREF
  __int64 v145; // [rsp+D8h] [rbp-88h]
  _BYTE *v146; // [rsp+E0h] [rbp-80h] BYREF
  __int64 v147; // [rsp+E8h] [rbp-78h]
  _BYTE v148[112]; // [rsp+F0h] [rbp-70h] BYREF

  v6 = a1 + 192;
  v127 = a2;
  v8 = *(_QWORD *)(*(_QWORD *)(a1 + 72) + 8LL);
  v9 = *(int ***)(v8 + 64);
  for ( i = &v9[*(unsigned int *)(v8 + 72)]; i != v9; ++v9 )
  {
    v11 = *v9;
    a4 = *((_QWORD *)*v9 + 1);
    if ( (a4 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
    {
      v133 = *((_QWORD *)*v9 + 1);
      v12 = sub_2FB3BF0(v6, v133, 0);
      a2 = v12;
      sub_2FB7970(a1, v12, v11, v133, 1u, v13);
      v14 = *(_QWORD *)(a1 + 72);
      if ( *(_BYTE *)(v14 + 172) )
      {
        v15 = *(int ***)(v14 + 152);
        a3 = (__int64)&v15[*(unsigned int *)(v14 + 164)];
        if ( v15 == (int **)a3 )
          continue;
        while ( v11 != *v15 )
        {
          if ( (int **)a3 == ++v15 )
            goto LABEL_9;
        }
      }
      else
      {
        a2 = (__int64)v11;
        if ( !sub_C8CA60(v14 + 144, (__int64)v11) )
          continue;
      }
      a2 = (__int64)v11;
      sub_2FB8080(a1, (__int64)v11);
    }
LABEL_9:
    ;
  }
  if ( (unsigned int)(*(_DWORD *)(a1 + 84) - 1) <= 1 )
    sub_2FBAAA0(a1, a2, a3, a4, a5, a6);
  v16 = sub_2FB5D60(a1);
  v17 = v16;
  v18 = v16;
  sub_2FB3D30(a1, v16, v19, v20, v21);
  if ( v18 )
  {
    sub_2FB4A90(a1);
    sub_2FB2980(a1, v17, v115, v116, v117, v118);
  }
  v24 = *(_QWORD *)(a1 + 72);
  v25 = *(_QWORD *)(v24 + 16);
  v26 = *(_QWORD *)v25 + 4LL * *(unsigned int *)(v25 + 8);
  for ( j = (int *)(*(_QWORD *)v25 + 4LL * *(unsigned int *)(v24 + 64)); (int *)v26 != j; ++j )
  {
    v34 = (unsigned int)*j;
    v35 = *(_QWORD *)(a1 + 8);
    v36 = *j & 0x7FFFFFFF;
    v37 = *(unsigned int *)(v35 + 160);
    v38 = 8LL * v36;
    if ( v36 < (unsigned int)v37 )
    {
      v28 = *(_QWORD *)(v35 + 152);
      v29 = *(__int64 **)(v28 + 8LL * v36);
      if ( v29 )
        goto LABEL_17;
    }
    v39 = v36 + 1;
    if ( (unsigned int)v37 < v39 && v39 != v37 )
    {
      if ( v39 >= v37 )
      {
        v110 = *(_QWORD *)(v35 + 168);
        v111 = v39 - v37;
        if ( v39 > (unsigned __int64)*(unsigned int *)(v35 + 164) )
        {
          v131 = *j;
          v134 = v39 - v37;
          sub_C8D5F0(v35 + 152, (const void *)(v35 + 168), v39, 8u, v22, v34);
          v37 = *(unsigned int *)(v35 + 160);
          LODWORD(v34) = v131;
          v111 = v134;
        }
        v40 = *(_QWORD *)(v35 + 152);
        v112 = (_QWORD *)(v40 + 8 * v37);
        v113 = &v112[v111];
        if ( v112 != v113 )
        {
          do
            *v112++ = v110;
          while ( v113 != v112 );
          LODWORD(v37) = *(_DWORD *)(v35 + 160);
          v40 = *(_QWORD *)(v35 + 152);
        }
        *(_DWORD *)(v35 + 160) = v111 + v37;
        goto LABEL_21;
      }
      *(_DWORD *)(v35 + 160) = v39;
    }
    v40 = *(_QWORD *)(v35 + 152);
LABEL_21:
    v41 = (__int64 *)(v40 + v38);
    v42 = sub_2E10F30(v34);
    *v41 = v42;
    v28 = v42;
    v29 = (__int64 *)v42;
    sub_2E11E80((_QWORD *)v35, v42);
LABEL_17:
    sub_2E0AF60((__int64)v29);
    sub_2E0A330(v29, v28, v30, v31, v32, v33);
  }
  if ( !v127 )
    goto LABEL_34;
  v43 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 72) + 16LL) + 8LL) - *(_DWORD *)(*(_QWORD *)(a1 + 72) + 64LL);
  LODWORD(v44) = 0;
  *(_DWORD *)(v127 + 8) = 0;
  if ( *(_DWORD *)(v127 + 12) < v43 )
  {
    sub_C8D5F0(v127, (const void *)(v127 + 16), v43, 4u, v22, v23);
    v44 = *(unsigned int *)(v127 + 8);
    v45 = *(_QWORD *)v127 + 4 * v44;
    if ( v43 )
    {
LABEL_30:
      for ( k = 0; k != v43; ++k )
        *(_DWORD *)(v45 + 4 * k) = k;
      LODWORD(v44) = *(_DWORD *)(v127 + 8);
    }
  }
  else
  {
    v45 = *(_QWORD *)v127;
    if ( v43 )
      goto LABEL_30;
  }
  *(_DWORD *)(v127 + 8) = v44 + v43;
LABEL_34:
  v137 = 0;
  v135[0] = (unsigned __int64)v136;
  v135[1] = 0x800000000LL;
  sub_3157150(v135, 0);
  v48 = *(_QWORD *)(a1 + 72);
  v49 = *(_QWORD *)(v48 + 16);
  v50 = *(_DWORD *)(v48 + 64);
  v121 = *(_DWORD *)(v49 + 8) - v50;
  if ( v121 )
  {
    v128 = 0;
    while ( 1 )
    {
      v51 = *(_DWORD *)(*(_QWORD *)v49 + 4LL * (unsigned int)(v128 + v50));
      v52 = *(_QWORD *)(a1 + 8);
      v53 = v51 & 0x7FFFFFFF;
      v54 = *(unsigned int *)(v52 + 160);
      v55 = v51 & 0x7FFFFFFF;
      if ( (v51 & 0x7FFFFFFFu) >= (unsigned int)v54 )
        break;
      v56 = *(_QWORD *)(*(_QWORD *)(v52 + 152) + 8LL * v53);
      if ( !v56 )
        break;
LABEL_38:
      v146 = v148;
      v147 = 0x800000000LL;
      sub_2E15100(v52, v56, (__int64)&v146);
      v57 = (unsigned __int64)v146;
      v58 = *(_QWORD *)(a1 + 16);
      v59 = *(_QWORD *)(v58 + 80);
      if ( *(_DWORD *)(v59 + 4 * v55) )
        v51 = *(_DWORD *)(v59 + 4 * v55);
      v129 = &v146[8 * (unsigned int)v147];
      if ( v129 != v146 )
      {
        while ( 1 )
        {
          v60 = v58 + 104;
          v61 = *(_DWORD *)(*(_QWORD *)v57 + 112LL);
          *(_DWORD *)(v59 + 4LL * (v61 & 0x7FFFFFFF)) = v51;
          v62 = *(_DWORD *)(v58 + 128);
          v63 = *(_QWORD *)(v58 + 112);
          if ( !v62 )
            goto LABEL_50;
          v64 = v62 - 1;
          v65 = 1;
          LODWORD(v66) = (v62 - 1) & (37 * v51);
          v67 = v66;
          v68 = (unsigned int *)(v63 + 72LL * (unsigned int)v66);
          v69 = *v68;
          v47 = *v68;
          if ( v51 == *v68 )
          {
LABEL_54:
            v70 = v68[12];
            v138 = *((_QWORD *)v68 + 1);
            v139 = *((_QWORD *)v68 + 2);
            v140 = *((_QWORD *)v68 + 3);
            v71 = *((_QWORD *)v68 + 4);
            v143 = 0;
            v141 = v71;
            p_dest = &dest;
            if ( (_DWORD)v70 && &p_dest != (void **)(v68 + 10) )
            {
              v126 = v70;
              sub_C8D5F0((__int64)&p_dest, &dest, (unsigned int)v70, 8u, v60, v70);
              v97 = v126;
              v60 = v58 + 104;
              v98 = 8LL * v68[12];
              if ( v98 )
              {
                memcpy(p_dest, *((const void **)v68 + 5), v98);
                v97 = v126;
                v60 = v58 + 104;
              }
              LODWORD(v143) = v97;
            }
            v72 = v68[16];
            v145 = 0;
            dest = &v146;
            if ( (_DWORD)v72 && &dest != (void **)(v68 + 14) )
            {
              v122 = v60;
              v124 = v72;
              sub_C8D5F0((__int64)&dest, &v146, (unsigned int)v72, 8u, v60, v72);
              v95 = v124;
              v60 = v122;
              v96 = 8LL * v68[16];
              if ( v96 )
              {
                v123 = v124;
                v125 = v60;
                memcpy(dest, *((const void **)v68 + 7), v96);
                v95 = v123;
                v60 = v125;
              }
              LODWORD(v145) = v95;
            }
            v62 = *(_DWORD *)(v58 + 128);
            if ( v62 )
            {
              v63 = *(_QWORD *)(v58 + 112);
              v64 = v62 - 1;
              goto LABEL_84;
            }
            ++*(_QWORD *)(v58 + 104);
LABEL_58:
            sub_2FB7300(v60, 2 * v62);
            v73 = *(_DWORD *)(v58 + 128);
            if ( !v73 )
              goto LABEL_159;
            v74 = v73 - 1;
            v75 = *(_QWORD *)(v58 + 112);
            v47 = v74 & (37 * v61);
            v76 = *(_DWORD *)(v58 + 120) + 1;
            v77 = (unsigned int *)(v75 + 72 * v47);
            v78 = *v77;
            if ( v61 != *v77 )
            {
              v119 = 1;
              v120 = 0;
              while ( v78 != -1 )
              {
                if ( !v120 && v78 == -2 )
                  v120 = v77;
                v47 = v74 & (unsigned int)(v47 + v119);
                v77 = (unsigned int *)(v75 + 72 * v47);
                v78 = *v77;
                if ( v61 == *v77 )
                  goto LABEL_60;
                ++v119;
              }
              if ( v120 )
                v77 = v120;
            }
          }
          else
          {
            while ( 1 )
            {
              if ( (_DWORD)v47 == -1 )
              {
                v57 += 8LL;
                if ( v129 != (_BYTE *)v57 )
                  goto LABEL_51;
                goto LABEL_72;
              }
              v67 = v64 & (v65 + v67);
              v47 = *(unsigned int *)(v63 + 72LL * v67);
              if ( v51 == (_DWORD)v47 )
                break;
              ++v65;
            }
            v114 = 1;
            while ( v69 != -1 )
            {
              v66 = v64 & ((_DWORD)v66 + v114);
              v68 = (unsigned int *)(v63 + 72 * v66);
              v69 = *v68;
              if ( (_DWORD)v47 == *v68 )
                goto LABEL_54;
              ++v114;
            }
            v138 = 0;
            v139 = 0;
            p_dest = &dest;
            v140 = -1;
            v141 = -1;
            v143 = 0;
            dest = &v146;
            v145 = 0;
LABEL_84:
            v85 = 1;
            v86 = v64 & (37 * v61);
            v87 = (unsigned int *)(v63 + 72LL * v86);
            v77 = 0;
            v47 = *v87;
            if ( v61 == (_DWORD)v47 )
            {
LABEL_85:
              v80 = v87 + 2;
              v79 = (void **)(v87 + 14);
              goto LABEL_63;
            }
            while ( (_DWORD)v47 != -1 )
            {
              if ( !v77 && (_DWORD)v47 == -2 )
                v77 = v87;
              v86 = v64 & (v85 + v86);
              v87 = (unsigned int *)(v63 + 72LL * v86);
              v47 = *v87;
              if ( v61 == (_DWORD)v47 )
                goto LABEL_85;
              ++v85;
            }
            v88 = *(_DWORD *)(v58 + 120);
            if ( !v77 )
              v77 = v87;
            ++*(_QWORD *)(v58 + 104);
            v76 = v88 + 1;
            if ( 4 * (v88 + 1) >= 3 * v62 )
              goto LABEL_58;
            if ( v62 - (v76 + *(_DWORD *)(v58 + 124)) <= v62 >> 3 )
            {
              sub_2FB7300(v60, v62);
              v89 = *(_DWORD *)(v58 + 128);
              if ( v89 )
              {
                v90 = v89 - 1;
                v91 = *(_QWORD *)(v58 + 112);
                v92 = 1;
                v93 = v90 & (37 * v61);
                v47 = 0;
                v76 = *(_DWORD *)(v58 + 120) + 1;
                v77 = (unsigned int *)(v91 + 72LL * v93);
                v94 = *v77;
                if ( v61 != *v77 )
                {
                  while ( v94 != -1 )
                  {
                    if ( !v47 && v94 == -2 )
                      v47 = (__int64)v77;
                    v93 = v90 & (v93 + v92);
                    v77 = (unsigned int *)(v91 + 72LL * v93);
                    v94 = *v77;
                    if ( v61 == *v77 )
                      goto LABEL_60;
                    ++v92;
                  }
                  if ( v47 )
                    v77 = (unsigned int *)v47;
                }
                goto LABEL_60;
              }
LABEL_159:
              ++*(_DWORD *)(v58 + 120);
              BUG();
            }
          }
LABEL_60:
          *(_DWORD *)(v58 + 120) = v76;
          if ( *v77 != -1 )
            --*(_DWORD *)(v58 + 124);
          v79 = (void **)(v77 + 14);
          *v77 = v61;
          *((_QWORD *)v77 + 1) = 0;
          v80 = v77 + 2;
          *((_QWORD *)v77 + 2) = 0;
          *((_QWORD *)v77 + 3) = -1;
          *((_QWORD *)v77 + 4) = -1;
          *((_QWORD *)v77 + 5) = v77 + 14;
          *((_QWORD *)v77 + 6) = 0;
          *((_QWORD *)v77 + 7) = v77 + 18;
          *((_QWORD *)v77 + 8) = 0;
LABEL_63:
          *(_QWORD *)v80 = v138;
          *((_QWORD *)v80 + 1) = v139;
          *((_QWORD *)v80 + 2) = v140;
          *((_QWORD *)v80 + 3) = v141;
          if ( &p_dest != (void **)(v80 + 8) )
          {
            if ( (_DWORD)v143 )
            {
              v81 = *((_QWORD *)v80 + 4);
              if ( v79 != (void **)v81 )
                _libc_free(v81);
              *((_QWORD *)v80 + 4) = p_dest;
              *((_QWORD *)v80 + 5) = v143;
              v143 = 0;
              p_dest = &dest;
            }
            else
            {
              v80[10] = 0;
            }
          }
          if ( v79 != &dest )
          {
            if ( (_DWORD)v145 )
            {
              v84 = *((_QWORD *)v80 + 6);
              if ( (unsigned int *)v84 != v80 + 16 )
                _libc_free(v84);
              *((_QWORD *)v80 + 6) = dest;
              *((_QWORD *)v80 + 7) = v145;
              goto LABEL_48;
            }
            v80[14] = 0;
          }
          if ( dest != &v146 )
            _libc_free((unsigned __int64)dest);
LABEL_48:
          if ( p_dest != &dest )
            _libc_free((unsigned __int64)p_dest);
LABEL_50:
          v57 += 8LL;
          if ( v129 == (_BYTE *)v57 )
            break;
LABEL_51:
          v58 = *(_QWORD *)(a1 + 16);
          v59 = *(_QWORD *)(v58 + 80);
        }
      }
LABEL_72:
      if ( v127 )
      {
        v82 = *(unsigned int *)(v127 + 8);
        v83 = (unsigned int)(*(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 72) + 16LL) + 8LL)
                           - *(_DWORD *)(*(_QWORD *)(a1 + 72) + 64LL));
        if ( v83 != v82 )
        {
          if ( v83 >= v82 )
          {
            v103 = v83 - v82;
            if ( v83 > *(unsigned int *)(v127 + 12) )
            {
              sub_C8D5F0(
                v127,
                (const void *)(v127 + 16),
                (unsigned int)(*(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 72) + 16LL) + 8LL)
                             - *(_DWORD *)(*(_QWORD *)(a1 + 72) + 64LL)),
                4u,
                v83,
                v47);
              v82 = *(unsigned int *)(v127 + 8);
            }
            v104 = (_DWORD *)(*(_QWORD *)v127 + 4 * v82);
            v105 = &v104[v103];
            if ( v104 != v105 )
            {
              do
                *v104++ = v128;
              while ( v105 != v104 );
              LODWORD(v82) = *(_DWORD *)(v127 + 8);
            }
            *(_DWORD *)(v127 + 8) = v103 + v82;
          }
          else
          {
            *(_DWORD *)(v127 + 8) = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 72) + 16LL) + 8LL)
                                  - *(_DWORD *)(*(_QWORD *)(a1 + 72) + 64LL);
          }
        }
      }
      if ( v146 != v148 )
        _libc_free((unsigned __int64)v146);
      ++v128;
      v48 = *(_QWORD *)(a1 + 72);
      if ( v128 == v121 )
        goto LABEL_122;
      v49 = *(_QWORD *)(v48 + 16);
      v50 = *(_DWORD *)(v48 + 64);
    }
    v99 = v53 + 1;
    if ( (unsigned int)v54 < v99 && v99 != v54 )
    {
      if ( v99 >= v54 )
      {
        v106 = *(_QWORD *)(v52 + 168);
        v107 = v99 - v54;
        if ( v99 > (unsigned __int64)*(unsigned int *)(v52 + 164) )
        {
          v130 = *(_QWORD *)(v52 + 168);
          sub_C8D5F0(v52 + 152, (const void *)(v52 + 168), v99, 8u, v106, v47);
          v54 = *(unsigned int *)(v52 + 160);
          v106 = v130;
        }
        v100 = *(_QWORD *)(v52 + 152);
        v108 = (__int64 *)(v100 + 8 * v54);
        v109 = &v108[v107];
        if ( v108 != v109 )
        {
          do
            *v108++ = v106;
          while ( v109 != v108 );
          LODWORD(v54) = *(_DWORD *)(v52 + 160);
          v100 = *(_QWORD *)(v52 + 152);
        }
        *(_DWORD *)(v52 + 160) = v107 + v54;
        goto LABEL_112;
      }
      *(_DWORD *)(v52 + 160) = v99;
    }
    v100 = *(_QWORD *)(v52 + 152);
LABEL_112:
    v101 = (__int64 *)(v100 + 8LL * (v51 & 0x7FFFFFFF));
    v102 = sub_2E10F30(v51);
    *v101 = v102;
    v56 = v102;
    sub_2E11E80((_QWORD *)v52, v102);
    v52 = *(_QWORD *)(a1 + 8);
    goto LABEL_38;
  }
LABEL_122:
  sub_350A960(v48, *(_QWORD *)(*(_QWORD *)(a1 + 16) + 24LL), *(_QWORD *)(a1 + 64));
  if ( (_BYTE *)v135[0] != v136 )
    _libc_free(v135[0]);
}
