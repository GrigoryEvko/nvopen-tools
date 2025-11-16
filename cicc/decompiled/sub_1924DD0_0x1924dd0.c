// Function: sub_1924DD0
// Address: 0x1924dd0
//
__int64 __fastcall sub_1924DD0(__int64 a1, __int64 a2, __int64 a3, _DWORD *a4)
{
  unsigned int v5; // ebx
  __int64 v6; // rcx
  __int64 v7; // r9
  __int64 v8; // rsi
  __int64 v9; // rax
  __int64 v11; // rdx
  __int64 v12; // r15
  __int64 v13; // rdi
  __int64 v14; // r12
  __int64 v16; // rax
  __int64 v17; // rsi
  unsigned int v18; // ecx
  __int64 *v19; // rdx
  __int64 v20; // r10
  int v21; // eax
  int v22; // edx
  __int64 v23; // rsi
  unsigned int v24; // eax
  __int64 v25; // rcx
  int v26; // edi
  __int64 v27; // rdx
  __int64 v28; // rax
  __int64 v29; // rsi
  unsigned int v30; // ecx
  __int64 *v31; // rdx
  __int64 v32; // r9
  __int64 v33; // r13
  __int64 v34; // rax
  __int64 v35; // rbx
  __int64 v36; // rax
  int v37; // edx
  int v38; // edx
  __int64 v39; // rsi
  unsigned int v40; // r8d
  __int64 *v41; // rdi
  __int64 v42; // r10
  unsigned int v43; // r8d
  unsigned int v44; // edi
  __int64 *v45; // rax
  __int64 v46; // r10
  int v47; // edx
  int v48; // edx
  __int64 v49; // r8
  unsigned int v50; // edi
  __int64 *v51; // rsi
  __int64 v52; // r10
  unsigned int v53; // r10d
  unsigned int v54; // edi
  __int64 *v55; // rsi
  __int64 v56; // r9
  __int64 v57; // r10
  char v58; // al
  char v59; // r8
  bool v60; // al
  __int64 v61; // rax
  int v62; // edx
  unsigned __int64 v63; // rax
  __int64 v64; // rcx
  unsigned int v65; // esi
  __int64 v66; // r15
  __int64 v67; // rdi
  __int64 v68; // r9
  unsigned int v69; // edx
  __int64 *v70; // rax
  __int64 v71; // rcx
  int v72; // esi
  int v73; // ecx
  int v74; // esi
  int v75; // eax
  int v76; // ecx
  int v77; // edi
  int v78; // edx
  int v79; // r8d
  unsigned __int64 v80; // rax
  unsigned int v81; // esi
  __int64 v82; // rcx
  __int64 v83; // r11
  unsigned int v84; // edx
  __int64 *v85; // rax
  __int64 v86; // r10
  __int64 v87; // r8
  __int64 v88; // r11
  __int64 v89; // rcx
  int v90; // edx
  __int64 *v91; // r10
  int v92; // ecx
  int v93; // edx
  int v94; // ecx
  int v95; // ecx
  int v96; // r8d
  int v97; // r8d
  __int64 *v98; // r11
  int v99; // ecx
  int v100; // r10d
  int v101; // r10d
  __int64 v102; // r11
  int v103; // edx
  unsigned int v104; // ecx
  __int64 v105; // r9
  int v106; // edx
  int v107; // r9d
  __int64 *v108; // rdi
  int v109; // edi
  int v110; // r8d
  __int64 *v111; // rsi
  __int64 v112; // [rsp+0h] [rbp-190h]
  int v113; // [rsp+Ch] [rbp-184h]
  __int64 v114; // [rsp+18h] [rbp-178h]
  __int64 v115; // [rsp+20h] [rbp-170h]
  unsigned int v118; // [rsp+38h] [rbp-158h]
  char v119; // [rsp+3Fh] [rbp-151h]
  __int64 v120; // [rsp+40h] [rbp-150h]
  __int64 v121; // [rsp+48h] [rbp-148h]
  __int64 v122; // [rsp+50h] [rbp-140h] BYREF
  __int64 *v123; // [rsp+58h] [rbp-138h] BYREF
  __int64 v124; // [rsp+60h] [rbp-130h] BYREF
  __int64 v125; // [rsp+68h] [rbp-128h]
  unsigned __int64 v126; // [rsp+70h] [rbp-120h]
  __int64 v127; // [rsp+C8h] [rbp-C8h]
  __int64 v128; // [rsp+D0h] [rbp-C0h]
  __int64 v129; // [rsp+D8h] [rbp-B8h]
  __int64 v130; // [rsp+E0h] [rbp-B0h] BYREF
  __int64 v131; // [rsp+E8h] [rbp-A8h]
  unsigned __int64 v132; // [rsp+F0h] [rbp-A0h]
  __int64 v133; // [rsp+148h] [rbp-48h]
  __int64 v134; // [rsp+150h] [rbp-40h]
  __int64 v135; // [rsp+158h] [rbp-38h]

  v5 = a2;
  v114 = *(_QWORD *)(a2 + 40);
  v112 = *(_QWORD *)(a3 + 64);
  sub_1923B40(&v124, v112);
  sub_1920340(&v130);
  v6 = v127;
  v7 = v133;
  v8 = v134;
  v9 = a3;
  v113 = (v5 >> 9) ^ (v5 >> 4);
  v11 = v128;
  v12 = v9;
  while ( 1 )
  {
    while ( 1 )
    {
      v13 = v7;
      if ( v11 - v6 == v8 - v7 )
      {
        if ( v11 == v6 )
        {
LABEL_59:
          if ( v7 )
            j_j___libc_free_0(v7, v135 - v7);
          if ( v132 != v131 )
            _libc_free(v132);
          if ( v127 )
            j_j___libc_free_0(v127, v129 - v127);
          if ( v126 != v125 )
            _libc_free(v126);
          return 0;
        }
        v57 = v7;
        while ( *(_QWORD *)v6 == *(_QWORD *)v57 )
        {
          v58 = *(_BYTE *)(v6 + 16);
          v59 = *(_BYTE *)(v57 + 16);
          if ( v58 && v59 )
            v60 = *(_QWORD *)(v6 + 8) == *(_QWORD *)(v57 + 8);
          else
            v60 = v58 == v59;
          if ( !v60 )
            break;
          v6 += 24;
          v57 += 24;
          if ( v6 == v11 )
            goto LABEL_59;
        }
      }
      v14 = *(_QWORD *)(v11 - 24);
      if ( v114 != v14 )
        break;
      v61 = v11 - 24;
      v11 = v127;
      v128 = v61;
      v6 = v127;
      if ( v61 != v127 )
        goto LABEL_50;
    }
    if ( !*a4 )
      goto LABEL_5;
    v16 = *(unsigned int *)(a1 + 320);
    v122 = *(_QWORD *)(v11 - 24);
    if ( (_DWORD)v16 )
    {
      v17 = *(_QWORD *)(a1 + 304);
      v18 = (v16 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
      v19 = (__int64 *)(v17 + 16LL * v18);
      v20 = *v19;
      if ( v14 == *v19 )
      {
LABEL_16:
        if ( v19 != (__int64 *)(v17 + 16 * v16) )
        {
          if ( *((_BYTE *)v19 + 8) )
            goto LABEL_5;
          goto LABEL_18;
        }
      }
      else
      {
        v62 = 1;
        while ( v20 != -8 )
        {
          v96 = v62 + 1;
          v18 = (v16 - 1) & (v62 + v18);
          v19 = (__int64 *)(v17 + 16LL * v18);
          v20 = *v19;
          if ( v14 == *v19 )
            goto LABEL_16;
          v62 = v96;
        }
      }
    }
    v63 = (unsigned int)*(unsigned __int8 *)(sub_157ED20(v14) + 16) - 34;
    if ( (unsigned int)v63 <= 0x36 )
    {
      v64 = 0x40018000000001LL;
      if ( _bittest64(&v64, v63) )
        break;
    }
    if ( *(_WORD *)(v122 + 18) )
      break;
    v80 = sub_157EBA0(v122);
    if ( sub_15F3330(v80) )
    {
      v65 = *(_DWORD *)(a1 + 320);
      v66 = a1;
      if ( v65 )
      {
        v67 = v122;
        v87 = *(_QWORD *)(a1 + 304);
        v88 = (v65 - 1) & (((unsigned int)v122 >> 9) ^ ((unsigned int)v122 >> 4));
        v70 = (__int64 *)(v87 + 16 * v88);
        v89 = *v70;
        if ( *v70 == v122 )
          goto LABEL_77;
        v90 = 1;
        v91 = 0;
        while ( v89 != -8 )
        {
          if ( v89 == -16 && !v91 )
            v91 = v70;
          LODWORD(v88) = (v65 - 1) & (v90 + v88);
          v70 = (__int64 *)(v87 + 16LL * (unsigned int)v88);
          v89 = *v70;
          if ( v122 == *v70 )
            goto LABEL_77;
          ++v90;
        }
        v92 = *(_DWORD *)(a1 + 312);
        if ( v91 )
          v70 = v91;
        ++*(_QWORD *)(a1 + 296);
        v93 = v92 + 1;
        if ( 4 * (v92 + 1) < 3 * v65 )
        {
LABEL_108:
          if ( v65 - *(_DWORD *)(v66 + 316) - v93 > v65 >> 3 )
            goto LABEL_109;
          goto LABEL_130;
        }
      }
      else
      {
        ++*(_QWORD *)(a1 + 296);
      }
      v65 *= 2;
LABEL_130:
      sub_1378900(a1 + 296, v65);
      sub_1923450(a1 + 296, &v122, &v123);
      v70 = v123;
      v67 = v122;
      v93 = *(_DWORD *)(v66 + 312) + 1;
      goto LABEL_109;
    }
    v81 = *(_DWORD *)(a1 + 320);
    if ( !v81 )
    {
      ++*(_QWORD *)(a1 + 296);
      goto LABEL_133;
    }
    v82 = v122;
    v83 = *(_QWORD *)(a1 + 304);
    v84 = (v81 - 1) & (((unsigned int)v122 >> 9) ^ ((unsigned int)v122 >> 4));
    v85 = (__int64 *)(v83 + 16LL * v84);
    v86 = *v85;
    if ( v122 != *v85 )
    {
      v107 = 1;
      v108 = 0;
      while ( v86 != -8 )
      {
        if ( !v108 && v86 == -16 )
          v108 = v85;
        v84 = (v81 - 1) & (v107 + v84);
        v85 = (__int64 *)(v83 + 16LL * v84);
        v86 = *v85;
        if ( v122 == *v85 )
          goto LABEL_100;
        ++v107;
      }
      if ( v108 )
        v85 = v108;
      v109 = *(_DWORD *)(a1 + 312);
      ++*(_QWORD *)(a1 + 296);
      v106 = v109 + 1;
      if ( 4 * (v109 + 1) < 3 * v81 )
      {
        if ( v81 - *(_DWORD *)(a1 + 316) - v106 <= v81 >> 3 )
        {
LABEL_134:
          sub_1378900(a1 + 296, v81);
          sub_1923450(a1 + 296, &v122, &v123);
          v85 = v123;
          v82 = v122;
          v106 = *(_DWORD *)(a1 + 312) + 1;
        }
        *(_DWORD *)(a1 + 312) = v106;
        if ( *v85 != -8 )
          --*(_DWORD *)(a1 + 316);
        *v85 = v82;
        *((_BYTE *)v85 + 8) = 0;
        goto LABEL_100;
      }
LABEL_133:
      v81 *= 2;
      goto LABEL_134;
    }
LABEL_100:
    *((_BYTE *)v85 + 8) = 0;
LABEL_18:
    if ( v112 != v14 )
    {
      v21 = *(_DWORD *)(a1 + 352);
      if ( v21 )
      {
        v22 = v21 - 1;
        v23 = *(_QWORD *)(a1 + 336);
        v24 = (v21 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
        v25 = *(_QWORD *)(v23 + 8LL * v24);
        if ( v14 == v25 )
        {
LABEL_21:
          v13 = v133;
          goto LABEL_5;
        }
        v26 = 1;
        while ( v25 != -8 )
        {
          v24 = v22 & (v26 + v24);
          v25 = *(_QWORD *)(v23 + 8LL * v24);
          if ( v14 == v25 )
            goto LABEL_21;
          ++v26;
        }
      }
    }
    v27 = *(_QWORD *)(a1 + 248);
    v28 = *(unsigned int *)(v27 + 80);
    if ( (_DWORD)v28 )
    {
      v29 = *(_QWORD *)(v27 + 64);
      v30 = (v28 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
      v31 = (__int64 *)(v29 + 16LL * v30);
      v32 = *v31;
      if ( v14 == *v31 )
      {
LABEL_26:
        if ( v31 != (__int64 *)(v29 + 16 * v28) )
        {
          v33 = v31[1];
          if ( v33 )
          {
            v34 = *(_QWORD *)(v12 + 72);
            v35 = *(_QWORD *)(v33 + 8);
            v115 = v34;
            v121 = *(_QWORD *)(v34 + 40);
            v120 = *(_QWORD *)(a2 + 40);
            if ( v35 != v33 )
            {
              v119 = 0;
              v118 = ((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4);
              do
              {
                if ( !v35 )
                  BUG();
                if ( *(_BYTE *)(v35 - 16) == 21 )
                {
                  v36 = *(_QWORD *)(v35 + 40);
                  if ( v14 == v121 )
                  {
                    v47 = *(_DWORD *)(a1 + 288);
                    if ( v47 )
                    {
                      v48 = v47 - 1;
                      v49 = *(_QWORD *)(a1 + 272);
                      v50 = v48 & v118;
                      v51 = (__int64 *)(v49 + 16LL * (v48 & v118));
                      v52 = *v51;
                      if ( v115 == *v51 )
                      {
LABEL_45:
                        v53 = *((_DWORD *)v51 + 2);
                      }
                      else
                      {
                        v74 = 1;
                        while ( v52 != -8 )
                        {
                          v94 = v74 + 1;
                          v50 = v48 & (v74 + v50);
                          v51 = (__int64 *)(v49 + 16LL * v50);
                          v52 = *v51;
                          if ( v115 == *v51 )
                            goto LABEL_45;
                          v74 = v94;
                        }
                        v53 = 0;
                      }
                      v54 = v48 & (((unsigned int)v36 >> 9) ^ ((unsigned int)v36 >> 4));
                      v55 = (__int64 *)(v49 + 16LL * v54);
                      v56 = *v55;
                      if ( *v55 == v36 )
                      {
LABEL_47:
                        if ( *((_DWORD *)v55 + 2) > v53 )
                          break;
                      }
                      else
                      {
                        v72 = 1;
                        while ( v56 != -8 )
                        {
                          v73 = v72 + 1;
                          v54 = v48 & (v72 + v54);
                          v55 = (__int64 *)(v49 + 16LL * v54);
                          v56 = *v55;
                          if ( v36 == *v55 )
                            goto LABEL_47;
                          v72 = v73;
                        }
                      }
                    }
                  }
                  if ( v14 != v120 || v119 )
                    goto LABEL_30;
                  v37 = *(_DWORD *)(a1 + 288);
                  if ( !v37 )
                    goto LABEL_42;
                  v38 = v37 - 1;
                  v39 = *(_QWORD *)(a1 + 272);
                  v40 = v38 & (((unsigned int)v36 >> 9) ^ ((unsigned int)v36 >> 4));
                  v41 = (__int64 *)(v39 + 16LL * v40);
                  v42 = *v41;
                  if ( v36 == *v41 )
                  {
LABEL_39:
                    v43 = *((_DWORD *)v41 + 2);
                  }
                  else
                  {
                    v77 = 1;
                    while ( v42 != -8 )
                    {
                      v95 = v77 + 1;
                      v40 = v38 & (v77 + v40);
                      v41 = (__int64 *)(v39 + 16LL * v40);
                      v42 = *v41;
                      if ( v36 == *v41 )
                        goto LABEL_39;
                      v77 = v95;
                    }
                    v43 = 0;
                  }
                  v44 = v38 & v113;
                  v45 = (__int64 *)(v39 + 16LL * (v38 & (unsigned int)v113));
                  v46 = *v45;
                  if ( a2 != *v45 )
                  {
                    v75 = 1;
                    while ( v46 != -8 )
                    {
                      v76 = v75 + 1;
                      v44 = v38 & (v75 + v44);
                      v45 = (__int64 *)(v39 + 16LL * v44);
                      v46 = *v45;
                      if ( a2 == *v45 )
                        goto LABEL_41;
                      v75 = v76;
                    }
LABEL_42:
                    v119 = 1;
LABEL_30:
                    if ( (unsigned __int8)sub_1420ED0(v12, v35 - 32, *(_QWORD **)(a1 + 232)) )
                      goto LABEL_21;
                    goto LABEL_31;
                  }
LABEL_41:
                  if ( *((_DWORD *)v45 + 2) <= v43 )
                    goto LABEL_42;
                }
LABEL_31:
                v35 = *(_QWORD *)(v35 + 8);
              }
              while ( v33 != v35 );
            }
          }
        }
      }
      else
      {
        v78 = 1;
        while ( v32 != -8 )
        {
          v79 = v78 + 1;
          v30 = (v28 - 1) & (v78 + v30);
          v31 = (__int64 *)(v29 + 16LL * v30);
          v32 = *v31;
          if ( v14 == *v31 )
            goto LABEL_26;
          v78 = v79;
        }
      }
    }
    if ( *a4 != -1 )
      --*a4;
LABEL_50:
    sub_1923CE0((__int64)&v124);
    v6 = v127;
    v11 = v128;
    v7 = v133;
    v8 = v134;
  }
  v65 = *(_DWORD *)(a1 + 320);
  v66 = a1;
  if ( !v65 )
  {
    ++*(_QWORD *)(a1 + 296);
    goto LABEL_125;
  }
  v67 = v122;
  v68 = *(_QWORD *)(a1 + 304);
  v69 = (v65 - 1) & (((unsigned int)v122 >> 9) ^ ((unsigned int)v122 >> 4));
  v70 = (__int64 *)(v68 + 16LL * v69);
  v71 = *v70;
  if ( *v70 == v122 )
    goto LABEL_77;
  v97 = 1;
  v98 = 0;
  while ( v71 != -8 )
  {
    if ( v71 == -16 && !v98 )
      v98 = v70;
    v69 = (v65 - 1) & (v97 + v69);
    v70 = (__int64 *)(v68 + 16LL * v69);
    v71 = *v70;
    if ( v122 == *v70 )
      goto LABEL_77;
    ++v97;
  }
  v99 = *(_DWORD *)(a1 + 312);
  if ( v98 )
    v70 = v98;
  ++*(_QWORD *)(a1 + 296);
  v93 = v99 + 1;
  if ( 4 * (v99 + 1) < 3 * v65 )
    goto LABEL_108;
LABEL_125:
  sub_1378900(a1 + 296, 2 * v65);
  v100 = *(_DWORD *)(a1 + 320);
  if ( !v100 )
  {
    ++*(_DWORD *)(a1 + 312);
    BUG();
  }
  v67 = v122;
  v101 = v100 - 1;
  v102 = *(_QWORD *)(a1 + 304);
  v103 = *(_DWORD *)(a1 + 312);
  v104 = v101 & (((unsigned int)v122 >> 9) ^ ((unsigned int)v122 >> 4));
  v70 = (__int64 *)(v102 + 16LL * v104);
  v105 = *v70;
  if ( *v70 == v122 )
  {
LABEL_127:
    v93 = v103 + 1;
  }
  else
  {
    v110 = 1;
    v111 = 0;
    while ( v105 != -8 )
    {
      if ( !v111 && v105 == -16 )
        v111 = v70;
      v104 = v101 & (v110 + v104);
      v70 = (__int64 *)(v102 + 16LL * v104);
      v105 = *v70;
      if ( v122 == *v70 )
        goto LABEL_127;
      ++v110;
    }
    v93 = v103 + 1;
    if ( v111 )
      v70 = v111;
  }
LABEL_109:
  *(_DWORD *)(v66 + 312) = v93;
  if ( *v70 != -8 )
    --*(_DWORD *)(v66 + 316);
  *v70 = v67;
  *((_BYTE *)v70 + 8) = 0;
LABEL_77:
  *((_BYTE *)v70 + 8) = 1;
  v13 = v133;
LABEL_5:
  if ( v13 )
    j_j___libc_free_0(v13, v135 - v13);
  if ( v132 != v131 )
    _libc_free(v132);
  if ( v127 )
    j_j___libc_free_0(v127, v129 - v127);
  if ( v126 != v125 )
    _libc_free(v126);
  return 1;
}
