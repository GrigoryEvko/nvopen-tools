// Function: sub_1906720
// Address: 0x1906720
//
__int64 __fastcall sub_1906720(__int64 a1, __int64 a2)
{
  _QWORD *v3; // r13
  __int64 v4; // rax
  _QWORD *v5; // r14
  __int64 v6; // r12
  unsigned __int64 v7; // rbx
  _QWORD *v8; // rdx
  unsigned __int64 v9; // rcx
  _QWORD *v10; // rax
  __int64 v11; // rdi
  __int64 v12; // rax
  char *v13; // rbx
  char *v14; // r12
  unsigned __int64 *v15; // rdx
  char *v16; // rsi
  _QWORD *v17; // rax
  __int64 v18; // rcx
  unsigned __int64 *v19; // rdx
  __int64 v20; // r13
  __int64 result; // rax
  unsigned __int64 v22; // r12
  __int64 v23; // rax
  __int64 v24; // rsi
  unsigned int v25; // ecx
  __int64 *v26; // rdx
  __int64 v27; // r8
  __int64 v28; // rax
  unsigned __int64 v29; // rdx
  unsigned __int64 v30; // rbx
  unsigned __int64 v31; // r15
  unsigned __int8 v32; // al
  _QWORD *v33; // rdx
  _QWORD *i; // r9
  unsigned __int64 v35; // rcx
  _QWORD *v36; // rax
  _QWORD *v37; // r10
  unsigned __int64 v38; // rcx
  _QWORD *v39; // rax
  _QWORD *v40; // rsi
  unsigned __int64 v41; // rdx
  unsigned __int64 v42; // r8
  __int64 **v43; // r9
  __int64 **v44; // rcx
  __int64 ***v45; // r8
  __int64 **v46; // r9
  __int64 v47; // rax
  __int64 v48; // rcx
  unsigned int v49; // esi
  __int64 *v50; // rdx
  __int64 v51; // rdi
  __int64 v52; // rdx
  bool v53; // al
  __int64 **v54; // rdx
  unsigned int v55; // eax
  char *j; // r12
  __int64 v57; // rdi
  __int64 v58; // rax
  _BOOL4 v59; // r11d
  _QWORD *v60; // rax
  _QWORD *v61; // rdx
  __int64 v62; // rax
  _BOOL4 v63; // r11d
  _QWORD *v64; // rax
  _QWORD *v65; // rdx
  int v66; // edx
  __int64 v67; // rdx
  unsigned __int64 *v68; // rdi
  int v69; // edx
  int v70; // r9d
  __int64 v71; // rdx
  __int64 v72; // rax
  void *v73; // rcx
  _QWORD *v74; // rcx
  __int64 v75; // rax
  __int64 v76; // rax
  int v77; // r8d
  char *v78; // rax
  size_t v79; // rdx
  __int64 *v80; // rax
  __int64 *v81; // rcx
  __int64 *v82; // r11
  __int64 *v83; // rdi
  _BYTE *v84; // rax
  __int64 *v85; // rax
  __int64 *v86; // rsi
  __int64 *v87; // r11
  __int64 *v88; // rdi
  _BYTE *v89; // rax
  _QWORD *v90; // [rsp+0h] [rbp-140h]
  _QWORD *v91; // [rsp+0h] [rbp-140h]
  __int64 *v92; // [rsp+10h] [rbp-130h]
  __int64 **v93; // [rsp+10h] [rbp-130h]
  __int64 **v94; // [rsp+18h] [rbp-128h]
  __int64 ***v95; // [rsp+18h] [rbp-128h]
  unsigned __int64 v96; // [rsp+20h] [rbp-120h]
  __int64 **v97; // [rsp+20h] [rbp-120h]
  _QWORD *v98; // [rsp+28h] [rbp-118h]
  _QWORD *v99; // [rsp+28h] [rbp-118h]
  _BYTE *v100; // [rsp+30h] [rbp-110h]
  _BYTE *v101; // [rsp+30h] [rbp-110h]
  __int64 v102; // [rsp+38h] [rbp-108h]
  __int64 v103; // [rsp+40h] [rbp-100h]
  unsigned __int64 v104; // [rsp+48h] [rbp-F8h]
  char *src; // [rsp+50h] [rbp-F0h]
  _QWORD *v106; // [rsp+58h] [rbp-E8h]
  unsigned __int64 *v107; // [rsp+68h] [rbp-D8h]
  char *v108; // [rsp+70h] [rbp-D0h]
  _BOOL4 v109; // [rsp+78h] [rbp-C8h]
  __int64 v110; // [rsp+78h] [rbp-C8h]
  _QWORD *v111; // [rsp+78h] [rbp-C8h]
  __int64 *v112; // [rsp+78h] [rbp-C8h]
  __int64 *v113; // [rsp+78h] [rbp-C8h]
  _QWORD *v114; // [rsp+80h] [rbp-C0h]
  _BOOL4 v115; // [rsp+80h] [rbp-C0h]
  _QWORD *v116; // [rsp+80h] [rbp-C0h]
  _QWORD *v117; // [rsp+80h] [rbp-C0h]
  __int64 v118; // [rsp+80h] [rbp-C0h]
  __int64 *v119; // [rsp+80h] [rbp-C0h]
  __int64 *v120; // [rsp+80h] [rbp-C0h]
  __int64 v121; // [rsp+88h] [rbp-B8h]
  bool v122; // [rsp+88h] [rbp-B8h]
  bool v123; // [rsp+88h] [rbp-B8h]
  _QWORD *v124; // [rsp+88h] [rbp-B8h]
  _QWORD *v125; // [rsp+88h] [rbp-B8h]
  _QWORD *v126; // [rsp+88h] [rbp-B8h]
  _QWORD *v127; // [rsp+88h] [rbp-B8h]
  _QWORD *v128; // [rsp+88h] [rbp-B8h]
  _QWORD *v129; // [rsp+88h] [rbp-B8h]
  __int64 v130; // [rsp+88h] [rbp-B8h]
  __int64 *v131; // [rsp+88h] [rbp-B8h]
  __int64 *v132; // [rsp+88h] [rbp-B8h]
  unsigned __int64 *v133; // [rsp+90h] [rbp-B0h]
  unsigned __int64 v134; // [rsp+90h] [rbp-B0h]
  _QWORD *v135; // [rsp+90h] [rbp-B0h]
  unsigned __int64 v136; // [rsp+98h] [rbp-A8h]
  unsigned __int64 *v137; // [rsp+A0h] [rbp-A0h]
  __int64 v138; // [rsp+A8h] [rbp-98h]
  _QWORD *v139; // [rsp+A8h] [rbp-98h]
  __int64 v140; // [rsp+B0h] [rbp-90h] BYREF
  unsigned int v141; // [rsp+B8h] [rbp-88h]
  __int64 v142; // [rsp+C0h] [rbp-80h]
  unsigned int v143; // [rsp+C8h] [rbp-78h]
  __int64 v144; // [rsp+D0h] [rbp-70h] BYREF
  unsigned int v145; // [rsp+D8h] [rbp-68h]
  __int64 v146; // [rsp+E0h] [rbp-60h]
  unsigned int v147; // [rsp+E8h] [rbp-58h]
  const void *v148; // [rsp+F0h] [rbp-50h] BYREF
  __int64 v149; // [rsp+F8h] [rbp-48h]
  unsigned __int64 v150; // [rsp+100h] [rbp-40h] BYREF
  unsigned int v151; // [rsp+108h] [rbp-38h]

  v3 = *(_QWORD **)(a2 + 16);
  if ( v3 == *(_QWORD **)(a2 + 8) )
    v4 = *(unsigned int *)(a2 + 28);
  else
    v4 = *(unsigned int *)(a2 + 24);
  v5 = &v3[v4];
  if ( v3 == v5 )
  {
LABEL_6:
    v3 = v5;
LABEL_7:
    v138 = 0;
    v6 = 8;
    v7 = 24;
LABEL_17:
    v104 = 8;
    v11 = 64;
    goto LABEL_18;
  }
  while ( *v3 >= 0xFFFFFFFFFFFFFFFELL )
  {
    if ( ++v3 == v5 )
      goto LABEL_6;
  }
  if ( v5 == v3 )
    goto LABEL_7;
  v8 = v3;
  v9 = 0;
  while ( 1 )
  {
    v10 = v8 + 1;
    if ( v8 + 1 == v5 )
      break;
    while ( 1 )
    {
      v8 = v10;
      if ( *v10 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v5 == ++v10 )
        goto LABEL_13;
    }
    ++v9;
    if ( v10 == v5 )
      goto LABEL_14;
  }
LABEL_13:
  ++v9;
LABEL_14:
  if ( (__int64)v9 > 0xFFFFFFFFFFFFFFFLL )
LABEL_224:
    sub_4262D8((__int64)"cannot create std::deque larger than max_size()");
  v138 = v9 & 0x3F;
  v104 = (v9 >> 6) + 3;
  v6 = 8 * ((v9 >> 6) + 1);
  if ( v104 <= 8 )
  {
    v7 = 8 * ((8 - ((v9 >> 6) + 1)) >> 1);
    goto LABEL_17;
  }
  v11 = v6 + 16;
  v7 = 8;
LABEL_18:
  v12 = sub_22077B0(v11);
  v13 = (char *)(v12 + v7);
  v103 = v12;
  v14 = &v13[v6];
  for ( src = v13; v14 > v13; *((_QWORD *)v13 - 1) = sub_22077B0(512) )
    v13 += 8;
  v15 = *(unsigned __int64 **)src;
  v133 = (unsigned __int64 *)*((_QWORD *)v14 - 1);
  v106 = v133 + 64;
  v107 = *(unsigned __int64 **)src;
  v102 = *(_QWORD *)src + 512LL;
  v108 = v14 - 8;
  v137 = &v133[v138];
  if ( src >= v14 - 8 )
  {
    v17 = v3;
  }
  else
  {
    v16 = src;
    while ( 1 )
    {
      v17 = v3;
      v18 = 64;
      do
      {
        do
          ++v17;
        while ( v17 != v5 && *v17 >= 0xFFFFFFFFFFFFFFFELL );
        --v18;
      }
      while ( v18 );
      while ( v3 != v17 )
      {
        *v15++ = *v3;
        do
          ++v3;
        while ( v3 != v5 && *v3 >= 0xFFFFFFFFFFFFFFFELL );
      }
      v16 += 8;
      if ( v16 >= v14 - 8 )
        break;
      v15 = *(unsigned __int64 **)v16;
      v3 = v17;
    }
  }
  v19 = v133;
  if ( v5 != v17 )
  {
LABEL_37:
    *v19++ = *v17;
    while ( ++v17 != v5 )
    {
      if ( *v17 < 0xFFFFFFFFFFFFFFFELL )
      {
        if ( v17 != v5 )
          goto LABEL_37;
        break;
      }
    }
  }
  v20 = a1;
  v139 = (_QWORD *)(a1 + 168);
  while ( 1 )
  {
    result = (__int64)v137;
    if ( v107 == v137 )
      break;
LABEL_42:
    if ( v133 == (unsigned __int64 *)result )
    {
      v22 = *(_QWORD *)(*((_QWORD *)v108 - 1) + 504LL);
      j_j___libc_free_0(v133, 512);
      v133 = (unsigned __int64 *)*((_QWORD *)v108 - 1);
      v106 = v133 + 64;
      v137 = v133 + 63;
      v108 -= 8;
    }
    else
    {
      v22 = *--v137;
    }
    v23 = *(unsigned int *)(v20 + 24);
    if ( !(_DWORD)v23 )
      goto LABEL_48;
    v24 = *(_QWORD *)(v20 + 8);
    v25 = (v23 - 1) & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
    v26 = (__int64 *)(v24 + 16LL * v25);
    v27 = *v26;
    if ( *v26 != v22 )
    {
      v69 = 1;
      while ( v27 != -8 )
      {
        v70 = v69 + 1;
        v25 = (v23 - 1) & (v69 + v25);
        v26 = (__int64 *)(v24 + 16LL * v25);
        v27 = *v26;
        if ( *v26 == v22 )
          goto LABEL_46;
        v69 = v70;
      }
LABEL_48:
      v136 = v22;
      switch ( *(_BYTE *)(v22 + 16) )
      {
        case '$':
        case '&':
        case '(':
        case '?':
        case '@':
        case 'L':
          sub_1904880((__int64)&v148);
          goto LABEL_50;
        case 'A':
        case 'B':
          if ( (*(_BYTE *)(v22 + 23) & 0x40) != 0 )
            v54 = *(__int64 ***)(v22 - 8);
          else
            v54 = (__int64 **)(v22 - 24LL * (*(_DWORD *)(v22 + 20) & 0xFFFFFFF));
          v55 = sub_1643030(**v54);
          sub_15897D0((__int64)&v140, v55, 1);
          sub_158DBC0((__int64)&v144, (__int64)&v140, *(unsigned __int8 *)(v22 + 16) - 24, dword_4FAE5E0 + 1);
          sub_19048B0((__int64)&v148, v20, &v144);
          sub_19058A0(v20, v22, (__int64 *)&v148);
          if ( v151 > 0x40 && v150 )
            j_j___libc_free_0_0(v150);
          if ( (unsigned int)v149 > 0x40 && v148 )
            j_j___libc_free_0_0(v148);
          if ( v147 > 0x40 && v146 )
            j_j___libc_free_0_0(v146);
          if ( v145 > 0x40 && v144 )
            j_j___libc_free_0_0(v144);
          if ( v143 > 0x40 && v142 )
            j_j___libc_free_0_0(v142);
          if ( v141 <= 0x40 || !v140 )
            continue;
          j_j___libc_free_0_0(v140);
          result = (__int64)v137;
          if ( v107 == v137 )
            goto LABEL_135;
          goto LABEL_42;
        default:
          sub_1904850((__int64)&v148);
LABEL_50:
          sub_19058A0(v20, v22, (__int64 *)&v148);
          if ( v151 > 0x40 && v150 )
            j_j___libc_free_0_0(v150);
          if ( (unsigned int)v149 > 0x40 && v148 )
            j_j___libc_free_0_0(v148);
          v28 = 24LL * (*(_DWORD *)(v22 + 20) & 0xFFFFFFF);
          v29 = v22 - v28;
          if ( (*(_BYTE *)(v22 + 23) & 0x40) != 0 )
          {
            v29 = *(_QWORD *)(v22 - 8);
            v136 = v29 + v28;
          }
          if ( v29 == v136 )
            continue;
          v30 = v29;
          v31 = *(_QWORD *)v29;
          v32 = *(_BYTE *)(*(_QWORD *)v29 + 16LL);
          if ( v32 > 0x17u )
            goto LABEL_60;
          break;
      }
LABEL_107:
      if ( v32 != 14 )
      {
        sub_1904850((__int64)&v148);
        sub_19058A0(v20, v22, (__int64 *)&v148);
        if ( v151 > 0x40 && v150 )
          j_j___libc_free_0_0(v150);
        if ( (unsigned int)v149 > 0x40 )
        {
          if ( v148 )
            j_j___libc_free_0_0(v148);
        }
      }
LABEL_105:
      v30 += 24LL;
      if ( v136 == v30 )
        continue;
      v31 = *(_QWORD *)v30;
      v32 = *(_BYTE *)(*(_QWORD *)v30 + 16LL);
      if ( v32 <= 0x17u )
        goto LABEL_107;
LABEL_60:
      v33 = *(_QWORD **)(v20 + 176);
      v148 = &v148;
      v149 = 1;
      v150 = v22;
      if ( v33 )
      {
        for ( i = v33; ; i = v36 )
        {
          v35 = i[6];
          v36 = (_QWORD *)i[3];
          if ( v35 > v22 )
            v36 = (_QWORD *)i[2];
          if ( !v36 )
            break;
        }
        if ( v22 >= v35 )
        {
          if ( v35 >= v22 )
          {
            v148 = &v148;
            v37 = i;
            v149 = 1;
            v150 = v31;
            goto LABEL_70;
          }
LABEL_144:
          v59 = 1;
          if ( v139 != i )
            v59 = i[6] > v22;
LABEL_146:
          v125 = i;
          v115 = v59;
          v60 = (_QWORD *)sub_22077B0(56);
          v60[4] = v60 + 4;
          v61 = v125;
          v60[5] = 1;
          v60[6] = v22;
          v126 = v60;
          sub_220F040(v115, v60, v61, v139);
          ++*(_QWORD *)(v20 + 200);
          v33 = *(_QWORD **)(v20 + 176);
          v37 = v126;
LABEL_147:
          v148 = &v148;
          v149 = 1;
          v150 = v31;
          if ( !v33 )
          {
            v33 = *(_QWORD **)(v20 + 184);
            if ( v139 != v33 )
            {
              v33 = v139;
              goto LABEL_150;
            }
LABEL_153:
            v63 = 1;
            goto LABEL_154;
          }
          while ( 1 )
          {
LABEL_70:
            v38 = v33[6];
            v39 = (_QWORD *)v33[3];
            if ( v31 < v38 )
              v39 = (_QWORD *)v33[2];
            if ( !v39 )
              break;
            v33 = v39;
          }
          if ( v31 < v38 )
          {
            if ( v33 == *(_QWORD **)(v20 + 184) )
              goto LABEL_152;
LABEL_150:
            v127 = v37;
            v116 = v33;
            v62 = sub_220EF80(v33);
            v37 = v127;
            if ( v31 <= *(_QWORD *)(v62 + 48) )
            {
              v33 = (_QWORD *)v62;
LABEL_75:
              v40 = v33;
              if ( v139 == v33 )
                goto LABEL_155;
            }
            else
            {
              v33 = v116;
              if ( v116 )
                goto LABEL_152;
              v40 = 0;
            }
          }
          else
          {
            if ( v31 <= v38 )
              goto LABEL_75;
LABEL_152:
            if ( v139 == v33 )
              goto LABEL_153;
            v63 = v31 < v33[6];
LABEL_154:
            v117 = v37;
            v109 = v63;
            v128 = v33;
            v64 = (_QWORD *)sub_22077B0(56);
            v65 = v128;
            v64[4] = v64 + 4;
            v64[5] = 1;
            v64[6] = v31;
            v129 = v64;
            sub_220F040(v109, v64, v65, v139);
            ++*(_QWORD *)(v20 + 200);
            v40 = v129;
            v37 = v117;
            if ( v139 == v129 )
            {
LABEL_155:
              v41 = 0;
              if ( v37 == v139 )
              {
LABEL_90:
                sub_1904850((__int64)&v148);
                v47 = *(unsigned int *)(v20 + 24);
                if ( (_DWORD)v47 )
                {
                  v48 = *(_QWORD *)(v20 + 8);
                  v49 = (v47 - 1) & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
                  v50 = (__int64 *)(v48 + 16LL * v49);
                  v51 = *v50;
                  if ( *v50 == v22 )
                  {
LABEL_92:
                    if ( v50 != (__int64 *)(v48 + 16 * v47) )
                    {
                      v52 = *(_QWORD *)(v20 + 32) + 40LL * *((unsigned int *)v50 + 2);
                      goto LABEL_94;
                    }
                  }
                  else
                  {
                    v66 = 1;
                    while ( v51 != -8 )
                    {
                      v77 = v66 + 1;
                      v49 = (v47 - 1) & (v49 + v66);
                      v50 = (__int64 *)(v48 + 16LL * v49);
                      v51 = *v50;
                      if ( *v50 == v22 )
                        goto LABEL_92;
                      v66 = v77;
                    }
                  }
                }
                v52 = *(_QWORD *)(v20 + 40);
LABEL_94:
                if ( *(_DWORD *)(v52 + 16) <= 0x40u )
                {
                  if ( *(const void **)(v52 + 8) == v148 )
                    goto LABEL_96;
                  v53 = 0;
                }
                else
                {
                  v121 = v52;
                  v53 = sub_16A5220(v52 + 8, &v148);
                  v52 = v121;
                  if ( v53 )
                  {
LABEL_96:
                    if ( *(_DWORD *)(v52 + 32) <= 0x40u )
                      v53 = *(_QWORD *)(v52 + 24) == v150;
                    else
                      v53 = sub_16A5220(v52 + 24, (const void **)&v150);
                  }
                }
                if ( v151 > 0x40 && v150 )
                {
                  v122 = v53;
                  j_j___libc_free_0_0(v150);
                  v53 = v122;
                }
                if ( (unsigned int)v149 > 0x40 && v148 )
                {
                  v123 = v53;
                  j_j___libc_free_0_0(v148);
                  v53 = v123;
                }
                if ( !v53 )
                {
                  if ( v137 == v106 - 1 )
                  {
                    v130 = v108 - src;
                    v67 = (v108 - src) >> 3;
                    if ( ((v102 - (__int64)v107) >> 3) + v137 - v133 + ((v67 - 1) << 6) == 0xFFFFFFFFFFFFFFFLL )
                      goto LABEL_224;
                    if ( v104 - ((__int64)&v108[-v103] >> 3) <= 1 )
                    {
                      v71 = v67 + 2;
                      if ( v104 > 2 * v71 )
                      {
                        v74 = (_QWORD *)(v103 + 8 * ((v104 - v71) >> 1));
                        v78 = v108 + 8;
                        v79 = v108 + 8 - src;
                        if ( src <= (char *)v74 )
                        {
                          if ( src != v78 )
                          {
                            v135 = v74;
                            memmove(v74, src, v79);
                            v74 = v135;
                          }
                        }
                        else if ( src != v78 )
                        {
                          v74 = memmove(v74, src, v79);
                        }
                      }
                      else
                      {
                        v72 = 1;
                        if ( v104 )
                          v72 = v104;
                        v134 = v104 + v72 + 2;
                        if ( v134 > 0xFFFFFFFFFFFFFFFLL )
                          sub_4261EA(v104, 0xFFFFFFFFFFFFFFFLL, v71);
                        v110 = v71;
                        v118 = sub_22077B0(8 * v134);
                        v73 = (void *)(v118 + 8 * ((v134 - v110) >> 1));
                        if ( src != v108 + 8 )
                          v73 = memmove(v73, src, v108 + 8 - src);
                        v111 = v73;
                        j_j___libc_free_0(v103, 8 * v104);
                        v74 = v111;
                        v104 = v134;
                        v103 = v118;
                      }
                      src = (char *)v74;
                      v102 = *v74 + 512LL;
                      v108 = (char *)v74 + v130;
                    }
                    *((_QWORD *)v108 + 1) = sub_22077B0(512);
                    *v137 = v31;
                    v68 = (unsigned __int64 *)*((_QWORD *)v108 + 1);
                    v137 = v68;
                    v106 = v68 + 64;
                    v108 += 8;
                    v133 = v68;
                  }
                  else
                  {
                    if ( v137 )
                      *v137 = v31;
                    ++v137;
                  }
                }
                goto LABEL_105;
              }
LABEL_82:
              v44 = (__int64 **)(v37 + 4);
              if ( (v37[5] & 1) == 0 )
              {
                v44 = (__int64 **)v37[4];
                if ( ((_BYTE)v44[1] & 1) == 0 )
                {
                  v45 = (__int64 ***)*v44;
                  if ( ((*v44)[1] & 1) != 0 )
                  {
                    v44 = (__int64 **)*v44;
                  }
                  else
                  {
                    v46 = *v45;
                    if ( ((_BYTE)(*v45)[1] & 1) == 0 )
                    {
                      v113 = *v46;
                      if ( ((*v46)[1] & 1) != 0 )
                      {
                        v46 = (__int64 **)*v46;
                      }
                      else
                      {
                        v76 = **v46;
                        v132 = (__int64 *)v76;
                        if ( (*(_BYTE *)(v76 + 8) & 1) == 0 )
                        {
                          v85 = *(__int64 **)v76;
                          v120 = v85;
                          if ( (v85[1] & 1) == 0 )
                          {
                            v86 = (__int64 *)*v85;
                            if ( (*(_BYTE *)(*v85 + 8) & 1) == 0 )
                            {
                              v87 = (__int64 *)*v86;
                              if ( (*(_BYTE *)(*v86 + 8) & 1) == 0 )
                              {
                                v88 = (_BYTE *)*v87;
                                v91 = (_QWORD *)*v86;
                                if ( (*(_BYTE *)(*v87 + 8) & 1) == 0 )
                                {
                                  v93 = *v45;
                                  v95 = (__int64 ***)*v44;
                                  v97 = (__int64 **)v37[4];
                                  v99 = v37;
                                  v101 = (_BYTE *)v41;
                                  v89 = sub_19053B0((__int64 *)v88);
                                  v46 = v93;
                                  v45 = v95;
                                  v88 = v89;
                                  *v91 = v89;
                                  v44 = v97;
                                  v37 = v99;
                                  v41 = (unsigned __int64)v101;
                                }
                                *v86 = (__int64)v88;
                                v87 = (__int64 *)v88;
                              }
                              v86 = v87;
                              *v120 = (__int64)v87;
                            }
                            v120 = v86;
                            *v132 = (__int64)v86;
                          }
                          v132 = v120;
                          *v113 = (__int64)v120;
                        }
                        *v46 = v132;
                        v46 = (__int64 **)v132;
                      }
                      *v45 = v46;
                    }
                    *v44 = (__int64 *)v46;
                    v44 = v46;
                  }
                  v37[4] = v44;
                }
              }
LABEL_88:
              if ( v44 == (__int64 **)v41 )
                goto LABEL_90;
              goto LABEL_89;
            }
          }
          v41 = (unsigned __int64)(v40 + 4);
          if ( (v40[5] & 1) == 0 )
          {
            v41 = v40[4];
            if ( (*(_BYTE *)(v41 + 8) & 1) == 0 )
            {
              v42 = *(_QWORD *)v41;
              if ( (*(_BYTE *)(*(_QWORD *)v41 + 8LL) & 1) != 0 )
              {
                v41 = *(_QWORD *)v41;
              }
              else
              {
                v43 = *(__int64 ***)v42;
                if ( (*(_BYTE *)(*(_QWORD *)v42 + 8LL) & 1) == 0 )
                {
                  v112 = *v43;
                  if ( ((*v43)[1] & 1) != 0 )
                  {
                    v43 = (__int64 **)*v43;
                  }
                  else
                  {
                    v75 = **v43;
                    v131 = (__int64 *)v75;
                    if ( (*(_BYTE *)(v75 + 8) & 1) == 0 )
                    {
                      v80 = *(__int64 **)v75;
                      v119 = v80;
                      if ( (v80[1] & 1) == 0 )
                      {
                        v81 = (__int64 *)*v80;
                        if ( (*(_BYTE *)(*v80 + 8) & 1) == 0 )
                        {
                          v82 = (__int64 *)*v81;
                          if ( (*(_BYTE *)(*v81 + 8) & 1) == 0 )
                          {
                            v83 = (_BYTE *)*v82;
                            v90 = (_QWORD *)*v81;
                            if ( (*(_BYTE *)(*v82 + 8) & 1) == 0 )
                            {
                              v92 = (__int64 *)*v80;
                              v94 = *(__int64 ***)v42;
                              v96 = *(_QWORD *)v41;
                              v98 = v37;
                              v100 = (_BYTE *)v40[4];
                              v84 = sub_19053B0((__int64 *)v83);
                              v81 = v92;
                              v43 = v94;
                              v83 = v84;
                              *v90 = v84;
                              v42 = v96;
                              v37 = v98;
                              v41 = (unsigned __int64)v100;
                            }
                            *v81 = (__int64)v83;
                            v82 = (__int64 *)v83;
                          }
                          v81 = v82;
                          *v119 = (__int64)v82;
                        }
                        v119 = v81;
                        *v131 = (__int64)v81;
                      }
                      v131 = v119;
                      *v112 = (__int64)v119;
                    }
                    *v43 = v131;
                    v43 = (__int64 **)v131;
                  }
                  *(_QWORD *)v42 = v43;
                }
                *(_QWORD *)v41 = v43;
                v41 = (unsigned __int64)v43;
              }
              v40[4] = v41;
              v44 = 0;
              if ( v37 != v139 )
                goto LABEL_82;
              goto LABEL_88;
            }
          }
          if ( v37 != v139 )
            goto LABEL_82;
          v44 = 0;
LABEL_89:
          (*v44)[1] = v41 | (*v44)[1] & 1;
          *v44 = *(__int64 **)v41;
          *(_QWORD *)(v41 + 8) &= ~1uLL;
          *(_QWORD *)v41 = v44;
          goto LABEL_90;
        }
        if ( i == *(_QWORD **)(v20 + 184) )
          goto LABEL_144;
      }
      else
      {
        i = v139;
        if ( *(_QWORD **)(v20 + 184) == v139 )
        {
          i = v139;
          v59 = 1;
          goto LABEL_146;
        }
      }
      v124 = v33;
      v114 = i;
      v58 = sub_220EF80(i);
      v33 = v124;
      v37 = (_QWORD *)v58;
      if ( *(_QWORD *)(v58 + 48) < v22 )
      {
        i = v114;
        v37 = 0;
        if ( v114 )
          goto LABEL_144;
      }
      goto LABEL_147;
    }
LABEL_46:
    if ( v26 == (__int64 *)(v24 + 16 * v23)
      || *(_QWORD *)(v20 + 40) == *(_QWORD *)(v20 + 32) + 40LL * *((unsigned int *)v26 + 2) )
    {
      goto LABEL_48;
    }
  }
LABEL_135:
  if ( v103 )
  {
    for ( j = src; v108 + 8 > j; j += 8 )
    {
      v57 = *(_QWORD *)j;
      j_j___libc_free_0(v57, 512);
    }
    return j_j___libc_free_0(v103, 8 * v104);
  }
  return result;
}
