// Function: sub_1CA8CD0
// Address: 0x1ca8cd0
//
__int64 __fastcall sub_1CA8CD0(_QWORD *a1, __int64 a2, int a3, __int64 a4, __int64 a5, _BYTE *a6)
{
  __int64 v6; // rbx
  __int64 v9; // r12
  int v10; // edx
  __int64 v11; // rax
  __int64 v12; // rcx
  __int64 v13; // rdx
  unsigned int v14; // esi
  __int64 v15; // rdi
  unsigned int v16; // edx
  __int64 v17; // rax
  __int64 v18; // rcx
  _BYTE *v20; // rsi
  unsigned int v21; // eax
  unsigned int v22; // esi
  __int64 v23; // rdx
  __int64 v24; // rdi
  unsigned int v25; // esi
  __int64 *v26; // rcx
  __int64 v27; // r10
  unsigned int v28; // r9d
  unsigned int v29; // edx
  __int64 *v30; // rax
  __int64 v31; // rsi
  _BYTE *i; // r14
  __int64 v33; // rax
  _BYTE *v34; // r14
  unsigned int v35; // eax
  unsigned int v36; // eax
  int v37; // r14d
  unsigned int v38; // eax
  unsigned int v39; // r11d
  unsigned __int64 v40; // rcx
  _BYTE **v41; // rax
  _BYTE *v42; // rsi
  __int64 v43; // r14
  _QWORD *v44; // rax
  unsigned int v45; // r15d
  __int64 v46; // rbx
  __int64 v47; // r11
  __int64 v48; // rax
  _BYTE *v49; // rsi
  unsigned __int8 v50; // dl
  __int64 v51; // rax
  __int64 v52; // r8
  unsigned int v53; // edi
  __int64 v54; // rcx
  _BYTE *v55; // r9
  unsigned __int8 v56; // al
  unsigned int v57; // edx
  __int64 *v58; // rax
  __int64 v59; // rcx
  __int64 *v60; // rax
  __int64 v61; // rdx
  unsigned int v62; // eax
  unsigned int v63; // esi
  __int64 v64; // rdi
  unsigned int v65; // r10d
  __int64 v66; // rax
  __int64 v67; // rcx
  bool v68; // zf
  int v69; // eax
  __int64 v70; // r10
  _QWORD *v71; // rdx
  _QWORD *v72; // r9
  _QWORD *v73; // r8
  _QWORD *v74; // rax
  int v75; // ecx
  _QWORD *v76; // rax
  __int64 *v77; // rax
  _QWORD *v78; // r14
  int v79; // ecx
  __int64 v80; // rax
  _QWORD *v81; // rsi
  _QWORD *v82; // rax
  __int64 v83; // rdx
  _BOOL8 v84; // rdi
  int v85; // ecx
  int v86; // r10d
  int v87; // r11d
  __int64 v88; // r10
  int v89; // edi
  _BYTE *v90; // rax
  int v91; // r11d
  __int64 *v92; // rcx
  int v93; // edx
  __int64 v94; // rsi
  int v95; // esi
  int v96; // r9d
  __int64 v97; // r11
  int v98; // edi
  int v99; // edi
  int v100; // r11d
  __int64 *v101; // r8
  int v102; // ecx
  __int64 v103; // rsi
  int v104; // esi
  __int64 *v105; // rdx
  __int64 *v106; // r8
  __int64 v107; // [rsp+0h] [rbp-A0h]
  __int64 v108; // [rsp+8h] [rbp-98h]
  _QWORD *v109; // [rsp+8h] [rbp-98h]
  unsigned int v110; // [rsp+10h] [rbp-90h]
  int v111; // [rsp+10h] [rbp-90h]
  int v112; // [rsp+10h] [rbp-90h]
  unsigned __int64 v113; // [rsp+10h] [rbp-90h]
  _BYTE *v114; // [rsp+20h] [rbp-80h]
  int v115; // [rsp+20h] [rbp-80h]
  __int64 v116; // [rsp+20h] [rbp-80h]
  _QWORD *v117; // [rsp+20h] [rbp-80h]
  __int64 v119; // [rsp+30h] [rbp-70h]
  __int64 v121; // [rsp+40h] [rbp-60h]
  unsigned __int8 v123; // [rsp+5Fh] [rbp-41h] BYREF
  __int64 v124; // [rsp+60h] [rbp-40h] BYREF
  __int64 v125[7]; // [rsp+68h] [rbp-38h] BYREF

  v6 = *(_QWORD *)(a2 + 48);
  v119 = a5;
  v123 = 0;
  v121 = a2 + 40;
  if ( v6 != a2 + 40 )
  {
    while ( 2 )
    {
      v9 = v6 - 24;
      if ( !v6 )
        v9 = 0;
      if ( a3 == 1 )
      {
        sub_1C99380(a1, v9);
        v10 = *(unsigned __int8 *)(v9 + 16);
        if ( (_BYTE)v10 != 78 )
          goto LABEL_6;
      }
      else
      {
        v10 = *(unsigned __int8 *)(v9 + 16);
        if ( (_BYTE)v10 != 78 )
        {
LABEL_6:
          v11 = *(_QWORD *)v9;
          v12 = *(unsigned __int8 *)(*(_QWORD *)v9 + 8LL);
          if ( (_BYTE)v12 == 15 )
          {
            v13 = (unsigned int)(v10 - 53);
            switch ( (int)v13 )
            {
              case 0:
                sub_1CA7B50(v9, 8, a4, &v123);
                goto LABEL_30;
              case 1:
                goto LABEL_28;
              case 3:
              case 18:
                if ( (*(_BYTE *)(v9 + 23) & 0x40) != 0 )
                {
                  v20 = **(_BYTE ***)(v9 - 8);
                  v21 = *(_DWORD *)(*(_QWORD *)v20 + 8LL) >> 8;
                  if ( v21 )
                    goto LABEL_14;
                }
                else
                {
                  v20 = *(_BYTE **)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF));
                  v21 = *(_DWORD *)(*(_QWORD *)v20 + 8LL) >> 8;
                  if ( v21 )
                    goto LABEL_14;
                }
                v22 = sub_1CA8350((__int64)a1, v20, a4, *(_QWORD *)(a2 + 56));
                goto LABEL_29;
              case 17:
                v23 = *(unsigned int *)(a4 + 24);
                if ( !(_DWORD)v23 )
                  goto LABEL_23;
                v24 = *(_QWORD *)(a4 + 8);
                v25 = (v23 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
                v26 = (__int64 *)(v24 + 16LL * v25);
                a5 = *v26;
                if ( *v26 == v9 )
                  goto LABEL_22;
                v85 = 1;
                while ( 2 )
                {
                  if ( a5 != -8 )
                  {
                    v86 = v85 + 1;
                    v25 = (v23 - 1) & (v85 + v25);
                    v26 = (__int64 *)(v24 + 16LL * v25);
                    a5 = *v26;
                    if ( v9 != *v26 )
                    {
                      v85 = v86;
                      continue;
                    }
LABEL_22:
                    if ( v26 != (__int64 *)(v24 + 16 * v23) )
                      goto LABEL_30;
                  }
                  break;
                }
LABEL_23:
                if ( !byte_4FBDC80 || !unk_4FBE1ED )
                  goto LABEL_25;
                if ( !(unsigned __int8)sub_1C2F070(*(_QWORD *)(a2 + 56))
                  || ((*(_BYTE *)(v9 + 23) & 0x40) == 0
                    ? (v105 = (__int64 *)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)))
                    : (v105 = *(__int64 **)(v9 - 8)),
                      v22 = 1,
                      !(unsigned __int8)sub_1C9F350(*v105)) )
                {
                  v11 = *(_QWORD *)v9;
                  if ( *(_BYTE *)(*(_QWORD *)v9 + 8LL) == 16 )
                    v11 = **(_QWORD **)(v11 + 16);
LABEL_25:
                  v21 = *(_DWORD *)(v11 + 8) >> 8;
LABEL_14:
                  if ( v21 == 4 )
                  {
                    v22 = 4;
                  }
                  else if ( v21 > 4 )
                  {
                    v22 = 8;
                    if ( v21 != 5 )
                      v22 = (v21 == 101) + 15;
                  }
                  else
                  {
                    v22 = 1;
                    if ( v21 != 1 )
                    {
                      v22 = 15;
                      if ( v21 == 3 )
                        v22 = 2;
                    }
                  }
                }
                goto LABEL_29;
              case 19:
                v36 = *(_DWORD *)(v11 + 8) >> 8;
                if ( v36 )
                {
                  if ( v36 == 4 )
                  {
                    v37 = 4;
                  }
                  else if ( v36 > 4 )
                  {
                    v37 = 8;
                    if ( v36 != 5 )
                      v37 = (v36 == 101) + 15;
                  }
                  else
                  {
                    v37 = 1;
                    if ( v36 != 1 )
                    {
                      v37 = 15;
                      if ( v36 == 3 )
                        v37 = 2;
                    }
                  }
LABEL_50:
                  sub_1CA7B50(v9, v37, a4, &v123);
                  goto LABEL_30;
                }
                if ( (*(_BYTE *)(v9 + 23) & 0x40) != 0 )
                  v60 = *(__int64 **)(v9 - 8);
                else
                  v60 = (__int64 *)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF));
                v61 = *v60;
                v62 = *(_DWORD *)(*(_QWORD *)*v60 + 8LL) >> 8;
                if ( v62 == 4 )
                {
                  v37 = 4;
                }
                else if ( v62 > 4 )
                {
                  v37 = 8;
                  if ( v62 != 5 )
                    v37 = (v62 == 101) + 15;
                }
                else
                {
                  v37 = 1;
                  if ( v62 != 1 )
                  {
                    v37 = 15;
                    if ( v62 == 3 )
                      v37 = 2;
                  }
                }
                v63 = *(_DWORD *)(a4 + 24);
                v124 = v61;
                if ( v63 )
                {
                  v64 = *(_QWORD *)(a4 + 8);
                  v65 = (v63 - 1) & (((unsigned int)v61 >> 9) ^ ((unsigned int)v61 >> 4));
                  v66 = v64 + 16LL * v65;
                  v67 = *(_QWORD *)v66;
                  if ( v61 == *(_QWORD *)v66 )
                  {
LABEL_105:
                    *(_DWORD *)(v66 + 8) = v37;
                    goto LABEL_50;
                  }
                  v96 = 1;
                  v97 = 0;
                  while ( v67 != -8 )
                  {
                    if ( v67 == -16 && !v97 )
                      v97 = v66;
                    v65 = (v63 - 1) & (v96 + v65);
                    v66 = v64 + 16LL * v65;
                    v67 = *(_QWORD *)v66;
                    if ( v61 == *(_QWORD *)v66 )
                      goto LABEL_105;
                    ++v96;
                  }
                  v98 = *(_DWORD *)(a4 + 16);
                  if ( v97 )
                    v66 = v97;
                  ++*(_QWORD *)a4;
                  v99 = v98 + 1;
                  if ( 4 * v99 < 3 * v63 )
                  {
                    if ( v63 - *(_DWORD *)(a4 + 20) - v99 > v63 >> 3 )
                    {
LABEL_220:
                      *(_DWORD *)(a4 + 16) = v99;
                      if ( *(_QWORD *)v66 != -8 )
                        --*(_DWORD *)(a4 + 20);
                      *(_QWORD *)v66 = v61;
                      *(_DWORD *)(v66 + 8) = 0;
                      goto LABEL_105;
                    }
LABEL_225:
                    sub_177C7D0(a4, v63);
                    sub_190E590(a4, &v124, v125);
                    v66 = v125[0];
                    v61 = v124;
                    v99 = *(_DWORD *)(a4 + 16) + 1;
                    goto LABEL_220;
                  }
                }
                else
                {
                  ++*(_QWORD *)a4;
                }
                v63 *= 2;
                goto LABEL_225;
              case 24:
                v38 = *(_DWORD *)(v11 + 8) >> 8;
                if ( !v38 )
                  goto LABEL_75;
                if ( v38 == 4 )
                {
                  v39 = 4;
                }
                else if ( v38 <= 4 )
                {
                  v39 = 1;
                  if ( v38 != 1 )
                  {
                    v68 = v38 == 3;
                    v69 = 2;
                    if ( !v68 )
                      v69 = 15;
                    v39 = v69;
                  }
                }
                else
                {
                  v39 = 8;
                  if ( v38 != 5 )
                    v39 = (v38 == 101) + 15;
                }
                goto LABEL_56;
              case 25:
                v40 = *(_QWORD *)(v9 - 24);
                if ( *(_BYTE *)(v40 + 16) )
                  goto LABEL_115;
                goto LABEL_62;
              case 26:
                if ( (*(_BYTE *)(v9 + 23) & 0x40) != 0 )
                  v33 = *(_QWORD *)(v9 - 8);
                else
                  v33 = v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF);
                v34 = *(_BYTE **)(v33 + 48);
                v114 = *(_BYTE **)(v33 + 24);
                v110 = sub_1CA8350((__int64)a1, v114, a4, *(_QWORD *)(a2 + 56));
                v35 = sub_1CA8350((__int64)a1, v34, a4, *(_QWORD *)(a2 + 56));
                v22 = v35;
                if ( v114[16] != 15 )
                {
                  if ( v34[16] == 15 )
                    v22 = v110;
                  else
                    v22 = v110 | v35;
                }
                goto LABEL_29;
              case 33:
                v124 = v9;
                v27 = *(_QWORD *)(v119 + 8);
                v28 = *(_DWORD *)(v119 + 24);
                if ( a3 == 1 )
                {
                  for ( i = *(_BYTE **)(v9 - 24); ; i = (_BYTE *)*((_QWORD *)i - 3) )
                  {
                    v56 = i[16];
                    if ( v56 <= 0x17u )
                    {
                      if ( v56 != 17 )
                        goto LABEL_130;
                      goto LABEL_93;
                    }
                    if ( v56 != 86 )
                      break;
                  }
                  if ( v56 == 77 )
                  {
                    v13 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)i + 8LL) - 13;
                    if ( (unsigned int)v13 > 1 )
                    {
                      while ( 1 )
                      {
LABEL_130:
                        if ( v56 != 87 )
                        {
                          i = 0;
                          goto LABEL_93;
                        }
                        if ( (unsigned __int8)sub_1C957E0((__int64)i, v9, v13, v12, a5) )
                          break;
                        i = (_BYTE *)*((_QWORD *)i - 6);
                        if ( i )
                          v56 = i[16];
                        else
                          v56 = MEMORY[0x10];
                      }
                      i = (_BYTE *)*((_QWORD *)i - 3);
                    }
                    else
                    {
                      if ( (i[23] & 0x40) != 0 )
                        v90 = (_BYTE *)*((_QWORD *)i - 1);
                      else
                        v90 = &i[-24 * (*((_DWORD *)i + 5) & 0xFFFFFFF)];
                      if ( *(_BYTE *)(*(_QWORD *)v90 + 16LL) != 54 )
                        i = 0;
                    }
                  }
                  else if ( v56 != 54 )
                  {
                    goto LABEL_130;
                  }
LABEL_93:
                  if ( v28 )
                  {
                    v57 = (v28 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
                    v58 = (__int64 *)(v27 + 16LL * v57);
                    v59 = *v58;
                    if ( v9 == *v58 )
                    {
LABEL_95:
                      v58[1] = (__int64)i;
                      goto LABEL_37;
                    }
                    v100 = 1;
                    v101 = 0;
                    while ( v59 != -8 )
                    {
                      if ( !v101 && v59 == -16 )
                        v101 = v58;
                      v57 = (v28 - 1) & (v100 + v57);
                      v58 = (__int64 *)(v27 + 16LL * v57);
                      v59 = *v58;
                      if ( v9 == *v58 )
                        goto LABEL_95;
                      ++v100;
                    }
                    if ( v101 )
                      v58 = v101;
                    ++*(_QWORD *)v119;
                    v102 = *(_DWORD *)(v119 + 16) + 1;
                    if ( 4 * v102 < 3 * v28 )
                    {
                      v103 = v9;
                      if ( v28 - *(_DWORD *)(v119 + 20) - v102 > v28 >> 3 )
                      {
LABEL_232:
                        *(_DWORD *)(v119 + 16) = v102;
                        if ( *v58 != -8 )
                          --*(_DWORD *)(v119 + 20);
                        *v58 = v103;
                        v58[1] = 0;
                        goto LABEL_95;
                      }
                      v104 = v28;
LABEL_237:
                      sub_1CA8B10(v119, v104);
                      sub_1C9EDD0(v119, &v124, v125);
                      v58 = (__int64 *)v125[0];
                      v103 = v124;
                      v102 = *(_DWORD *)(v119 + 16) + 1;
                      goto LABEL_232;
                    }
                  }
                  else
                  {
                    ++*(_QWORD *)v119;
                  }
                  v104 = 2 * v28;
                  goto LABEL_237;
                }
                if ( !v28 )
                {
                  ++*(_QWORD *)v119;
                  goto LABEL_207;
                }
                v29 = (v28 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
                v30 = (__int64 *)(v27 + 16LL * v29);
                v31 = *v30;
                if ( v9 == *v30 )
                {
                  i = (_BYTE *)v30[1];
                  goto LABEL_37;
                }
                v91 = 1;
                v92 = 0;
                while ( 1 )
                {
                  if ( v31 == -8 )
                  {
                    if ( !v92 )
                      v92 = v30;
                    ++*(_QWORD *)v119;
                    v93 = *(_DWORD *)(v119 + 16) + 1;
                    if ( 4 * v93 < 3 * v28 )
                    {
                      v94 = v9;
                      if ( v28 - *(_DWORD *)(v119 + 20) - v93 > v28 >> 3 )
                      {
LABEL_203:
                        *(_DWORD *)(v119 + 16) = v93;
                        if ( *v92 != -8 )
                          --*(_DWORD *)(v119 + 20);
                        *v92 = v94;
                        v22 = 15;
                        v92[1] = 0;
                        goto LABEL_29;
                      }
                      v95 = v28;
LABEL_208:
                      sub_1CA8B10(v119, v95);
                      sub_1C9EDD0(v119, &v124, v125);
                      v92 = (__int64 *)v125[0];
                      v94 = v124;
                      v93 = *(_DWORD *)(v119 + 16) + 1;
                      goto LABEL_203;
                    }
LABEL_207:
                    v95 = 2 * v28;
                    goto LABEL_208;
                  }
                  if ( v92 || v31 != -16 )
                    v30 = v92;
                  v29 = (v28 - 1) & (v91 + v29);
                  v106 = (__int64 *)(v27 + 16LL * v29);
                  v31 = *v106;
                  if ( v9 == *v106 )
                    break;
                  ++v91;
                  v92 = v30;
                  v30 = (__int64 *)(v27 + 16LL * v29);
                }
                i = (_BYTE *)v106[1];
LABEL_37:
                if ( !i )
                  goto LABEL_136;
                v22 = sub_1CA8350((__int64)a1, i, a4, *(_QWORD *)(a2 + 56));
LABEL_29:
                sub_1CA7B50(v9, v22, a4, &v123);
LABEL_30:
                v6 = *(_QWORD *)(v6 + 8);
                if ( v121 == v6 )
                  return v123;
                continue;
              default:
                v14 = *(_DWORD *)(a4 + 24);
                v124 = v9;
                if ( !v14 )
                {
                  ++*(_QWORD *)a4;
                  goto LABEL_150;
                }
                LODWORD(a5) = v14 - 1;
                v15 = *(_QWORD *)(a4 + 8);
                v16 = (v14 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
                v17 = v15 + 16LL * v16;
                v18 = *(_QWORD *)v17;
                if ( v9 == *(_QWORD *)v17 )
                  goto LABEL_10;
                v87 = 1;
                v88 = 0;
                while ( 1 )
                {
                  if ( v18 == -8 )
                  {
                    v89 = *(_DWORD *)(a4 + 16);
                    if ( v88 )
                      v17 = v88;
                    ++*(_QWORD *)a4;
                    v79 = v89 + 1;
                    if ( 4 * (v89 + 1) < 3 * v14 )
                    {
                      if ( v14 - *(_DWORD *)(a4 + 20) - v79 <= v14 >> 3 )
                      {
LABEL_151:
                        sub_177C7D0(a4, v14);
                        sub_190E590(a4, &v124, v125);
                        v17 = v125[0];
                        v9 = v124;
                        v79 = *(_DWORD *)(a4 + 16) + 1;
                      }
                      *(_DWORD *)(a4 + 16) = v79;
                      if ( *(_QWORD *)v17 != -8 )
                        --*(_DWORD *)(a4 + 20);
                      *(_QWORD *)v17 = v9;
                      *(_DWORD *)(v17 + 8) = 0;
                      break;
                    }
LABEL_150:
                    v14 *= 2;
                    goto LABEL_151;
                  }
                  if ( !v88 && v18 == -16 )
                    v88 = v17;
                  v16 = a5 & (v87 + v16);
                  v17 = v15 + 16LL * v16;
                  v18 = *(_QWORD *)v17;
                  if ( v9 == *(_QWORD *)v17 )
                    break;
                  ++v87;
                }
LABEL_10:
                *(_DWORD *)(v17 + 8) = 15;
                v6 = *(_QWORD *)(v6 + 8);
                if ( v121 == v6 )
                  return v123;
                continue;
            }
          }
          if ( (_BYTE)v10 == 54 )
          {
            if ( (unsigned int)(v12 - 13) <= 1 )
            {
LABEL_28:
              v22 = sub_1CA8520(a1, v9, a4, *(_QWORD *)(a2 + 56), a6);
              goto LABEL_29;
            }
            goto LABEL_30;
          }
          if ( (_BYTE)v10 != 77 || (unsigned int)(v12 - 13) > 1 )
            goto LABEL_30;
LABEL_75:
          v39 = *(_DWORD *)(v9 + 20) & 0xFFFFFFF;
          if ( !v39 )
          {
LABEL_56:
            sub_1CA7B50(v9, v39, a4, &v123);
            goto LABEL_30;
          }
          v116 = v6;
          v43 = 0;
          v44 = a1;
          v45 = 0;
          v46 = 24LL * v39;
          v47 = (__int64)v44;
          while ( 1 )
          {
            while ( 1 )
            {
              if ( (*(_BYTE *)(v9 + 23) & 0x40) != 0 )
                v48 = *(_QWORD *)(v9 - 8);
              else
                v48 = v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF);
              v49 = *(_BYTE **)(v48 + v43);
              v125[0] = (__int64)v49;
              v50 = v49[16];
              if ( v50 == 15 || v50 == 9 )
                goto LABEL_84;
              v51 = *(unsigned int *)(a4 + 24);
              if ( (_DWORD)v51 )
                break;
LABEL_124:
              if ( v50 <= 0x17u )
              {
                v108 = v47;
                v111 = sub_1CA7E20(v47, v49, a4, *(_QWORD *)(a2 + 56));
                v77 = sub_1CA7910(a4, v125);
                v47 = v108;
                *((_DWORD *)v77 + 2) = v111;
                v45 |= v111;
                goto LABEL_84;
              }
              v43 += 24;
              *a6 = 1;
              if ( v43 == v46 )
              {
LABEL_126:
                v76 = (_QWORD *)v47;
                v6 = v116;
                v39 = v45;
                a1 = v76;
                goto LABEL_56;
              }
            }
            v52 = *(_QWORD *)(a4 + 8);
            v53 = (v51 - 1) & (((unsigned int)v49 >> 9) ^ ((unsigned int)v49 >> 4));
            v54 = v52 + 16LL * v53;
            v55 = *(_BYTE **)v54;
            if ( v49 != *(_BYTE **)v54 )
            {
              v75 = 1;
              while ( v55 != (_BYTE *)-8LL )
              {
                v53 = (v51 - 1) & (v75 + v53);
                v112 = v75 + 1;
                v54 = v52 + 16LL * v53;
                v55 = *(_BYTE **)v54;
                if ( v49 == *(_BYTE **)v54 )
                  goto LABEL_82;
                v75 = v112;
              }
              goto LABEL_124;
            }
LABEL_82:
            if ( v54 == v52 + 16 * v51 )
              goto LABEL_124;
            v45 |= *(_DWORD *)(v54 + 8);
LABEL_84:
            v43 += 24;
            if ( v43 == v46 )
              goto LABEL_126;
          }
        }
      }
      break;
    }
    v40 = *(_QWORD *)(v9 - 24);
    if ( *(_BYTE *)(v40 + 16) )
    {
      if ( *(_BYTE *)(*(_QWORD *)v9 + 8LL) != 15 )
        goto LABEL_30;
LABEL_115:
      v70 = a1[2];
      if ( v70 )
      {
        v71 = *(_QWORD **)(v70 + 16);
        v72 = (_QWORD *)(v70 + 8);
        if ( v71 )
        {
          v73 = (_QWORD *)(v70 + 8);
          v74 = *(_QWORD **)(v70 + 16);
          do
          {
            if ( v74[4] < v40 )
            {
              v74 = (_QWORD *)v74[3];
            }
            else
            {
              v73 = v74;
              v74 = (_QWORD *)v74[2];
            }
          }
          while ( v74 );
          if ( v72 != v73 )
          {
            v78 = (_QWORD *)(v70 + 8);
            if ( v73[4] <= v40 )
            {
              do
              {
                if ( v71[4] < v40 )
                {
                  v71 = (_QWORD *)v71[3];
                }
                else
                {
                  v78 = v71;
                  v71 = (_QWORD *)v71[2];
                }
              }
              while ( v71 );
              if ( v72 == v78 || v78[4] > v40 )
              {
                v109 = (_QWORD *)a1[2];
                v113 = v40;
                v107 = v70 + 8;
                v80 = sub_22077B0(48);
                *(_DWORD *)(v80 + 40) = 0;
                v81 = v78;
                v78 = (_QWORD *)v80;
                *(_QWORD *)(v80 + 32) = v113;
                v82 = sub_1C70810(v109, v81, (unsigned __int64 *)(v80 + 32));
                if ( v83 )
                {
                  v84 = v82 || v107 == v83 || *(_QWORD *)(v83 + 32) > v113;
                  sub_220F040(v84, v78, v83, v107);
                  ++v109[5];
                }
                else
                {
                  v117 = v82;
                  j_j___libc_free_0(v78, 48);
                  v78 = v117;
                }
              }
              v22 = *((_DWORD *)v78 + 10);
              if ( v22 == 4 )
                goto LABEL_29;
              if ( v22 <= 4 )
              {
                if ( v22 != 1 )
                {
                  if ( v22 == 3 )
                    v22 = 2;
                  else
                    v22 = 15;
                }
                goto LABEL_29;
              }
              if ( v22 == 5 )
              {
                v22 = 8;
                goto LABEL_29;
              }
              if ( v22 == 101 )
              {
                v22 = 16;
                goto LABEL_29;
              }
            }
          }
        }
      }
LABEL_136:
      v22 = 15;
      goto LABEL_29;
    }
    if ( (*(_BYTE *)(v40 + 33) & 0x20) == 0 || a3 != 1 || (unsigned int)(*(_DWORD *)(v40 + 36) - 4048) > 5 )
    {
      if ( *(_BYTE *)(*(_QWORD *)v9 + 8LL) != 15 )
        goto LABEL_30;
LABEL_62:
      if ( (*(_BYTE *)(v40 + 33) & 0x20) != 0 )
      {
        v22 = 15;
        if ( *(_DWORD *)(v40 + 36) == 3660 )
        {
          if ( (*(_BYTE *)(v9 + 23) & 0x40) != 0 )
            v41 = *(_BYTE ***)(v9 - 8);
          else
            v41 = (_BYTE **)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF));
          v115 = sub_1CA8350((__int64)a1, *v41, a4, *(_QWORD *)(a2 + 56));
          sub_1CA7B50(v9, v115, a4, &v123);
          v22 = v115;
        }
        goto LABEL_29;
      }
      goto LABEL_115;
    }
    v125[0] = v9;
    v42 = (_BYTE *)a1[13];
    if ( v42 == (_BYTE *)a1[14] )
    {
      sub_1C9B1F0((__int64)(a1 + 12), v42, v125);
    }
    else
    {
      if ( v42 )
      {
        *(_QWORD *)v42 = v9;
        v42 = (_BYTE *)a1[13];
      }
      a1[13] = v42 + 8;
    }
    v10 = *(unsigned __int8 *)(v9 + 16);
    goto LABEL_6;
  }
  return 0;
}
