// Function: sub_2622810
// Address: 0x2622810
//
void __fastcall sub_2622810(char *a1, char *a2, __int64 a3)
{
  char *v4; // r14
  unsigned int v5; // ecx
  __int64 v6; // rdx
  int v7; // r11d
  unsigned int v8; // r9d
  __int64 *v9; // rsi
  __int64 *v10; // rax
  __int64 v11; // rdi
  unsigned int v12; // r10d
  int v13; // r13d
  unsigned int v14; // r9d
  __int64 *v15; // rsi
  __int64 *v16; // rax
  __int64 v17; // rdi
  __int64 v18; // r11
  char *v19; // rbx
  unsigned int v20; // r8d
  __int64 v21; // r12
  __int64 v22; // rbx
  int v23; // esi
  int v24; // esi
  __int64 v25; // r8
  __int64 v26; // rcx
  int v27; // edx
  __int64 v28; // rdi
  int v29; // eax
  int v30; // esi
  __int64 v31; // rdi
  unsigned int v32; // ecx
  int v33; // edx
  __int64 v34; // r8
  char *v35; // rbx
  unsigned int v36; // ecx
  __int64 v37; // rax
  int v38; // r13d
  unsigned int v39; // r10d
  __int64 *v40; // rdi
  __int64 *v41; // rdx
  __int64 v42; // r9
  unsigned int v43; // edi
  unsigned int v44; // r13d
  unsigned int v45; // r9d
  __int64 *v46; // rdx
  __int64 v47; // r8
  __int64 v48; // r12
  unsigned int v49; // esi
  int v50; // ecx
  int v51; // ecx
  __int64 v52; // r8
  unsigned int v53; // edi
  __int64 v54; // rsi
  int v55; // eax
  int v56; // esi
  int v57; // esi
  __int64 v58; // r8
  unsigned int v59; // ecx
  int v60; // edx
  _QWORD *v61; // rax
  __int64 v62; // rdi
  int v63; // eax
  int v64; // ecx
  int v65; // ecx
  __int64 v66; // r8
  int v67; // r13d
  __int64 *v68; // r10
  __int64 v69; // rdi
  __int64 v70; // rsi
  _QWORD *v71; // r10
  int v72; // ebx
  int v73; // ecx
  int v74; // ecx
  __int64 v75; // rdi
  _QWORD *v76; // r9
  __int64 v77; // r13
  int v78; // r10d
  __int64 v79; // rsi
  int v80; // edi
  int v81; // eax
  int v82; // ecx
  __int64 v83; // rdi
  __int64 *v84; // r8
  unsigned int v85; // r12d
  int v86; // r9d
  __int64 v87; // rsi
  int v88; // edi
  int v89; // ecx
  int v90; // ecx
  __int64 v91; // rdi
  __int64 *v92; // r8
  __int64 v93; // r13
  int v94; // r9d
  __int64 v95; // rsi
  int v96; // r10d
  __int64 *v97; // r9
  int v98; // r10d
  int v99; // r9d
  unsigned int v100; // r10d
  _QWORD *v101; // rbx
  int v102; // r10d
  __int64 *v103; // r9
  int v104; // r10d
  __int64 *v105; // r9
  __int64 v106; // [rsp+8h] [rbp-58h]
  __int64 v107; // [rsp+8h] [rbp-58h]
  int v108; // [rsp+8h] [rbp-58h]
  unsigned int v111; // [rsp+20h] [rbp-40h]
  __int64 v112; // [rsp+20h] [rbp-40h]
  __int64 v113; // [rsp+20h] [rbp-40h]
  __int64 v114; // [rsp+28h] [rbp-38h]
  char *v115; // [rsp+28h] [rbp-38h]

  if ( a1 == a2 || a1 + 8 == a2 )
    return;
  v4 = a1 + 8;
  do
  {
    while ( 1 )
    {
      v20 = *(_DWORD *)(a3 + 24);
      v21 = *(_QWORD *)v4;
      v22 = *(_QWORD *)a1;
      if ( v20 )
      {
        v5 = v20 - 1;
        v6 = *(_QWORD *)(a3 + 8);
        v7 = 1;
        v8 = (v20 - 1) & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
        v9 = (__int64 *)(v6 + 40LL * v8);
        v10 = 0;
        v11 = *v9;
        if ( v21 == *v9 )
        {
LABEL_5:
          v12 = *((_DWORD *)v9 + 2);
          goto LABEL_6;
        }
        while ( v11 != -4096 )
        {
          if ( v11 == -8192 && !v10 )
            v10 = v9;
          v8 = v5 & (v7 + v8);
          v9 = (__int64 *)(v6 + 40LL * v8);
          v11 = *v9;
          if ( v21 == *v9 )
            goto LABEL_5;
          ++v7;
        }
        v88 = *(_DWORD *)(a3 + 16);
        if ( !v10 )
          v10 = v9;
        ++*(_QWORD *)a3;
        v27 = v88 + 1;
        if ( 4 * (v88 + 1) < 3 * v20 )
        {
          if ( v20 - *(_DWORD *)(a3 + 20) - v27 <= v20 >> 3 )
          {
            sub_261D190(a3, v20);
            v89 = *(_DWORD *)(a3 + 24);
            if ( !v89 )
            {
LABEL_173:
              ++*(_DWORD *)(a3 + 16);
              BUG();
            }
            v90 = v89 - 1;
            v91 = *(_QWORD *)(a3 + 8);
            v92 = 0;
            LODWORD(v93) = v90 & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
            v94 = 1;
            v27 = *(_DWORD *)(a3 + 16) + 1;
            v10 = (__int64 *)(v91 + 40LL * (unsigned int)v93);
            v95 = *v10;
            if ( v21 != *v10 )
            {
              while ( v95 != -4096 )
              {
                if ( !v92 && v95 == -8192 )
                  v92 = v10;
                v93 = v90 & (unsigned int)(v93 + v94);
                v10 = (__int64 *)(v91 + 40 * v93);
                v95 = *v10;
                if ( v21 == *v10 )
                  goto LABEL_15;
                ++v94;
              }
              if ( v92 )
                v10 = v92;
            }
          }
          goto LABEL_15;
        }
      }
      else
      {
        ++*(_QWORD *)a3;
      }
      sub_261D190(a3, 2 * v20);
      v23 = *(_DWORD *)(a3 + 24);
      if ( !v23 )
        goto LABEL_173;
      v24 = v23 - 1;
      v25 = *(_QWORD *)(a3 + 8);
      LODWORD(v26) = v24 & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
      v27 = *(_DWORD *)(a3 + 16) + 1;
      v10 = (__int64 *)(v25 + 40LL * (unsigned int)v26);
      v28 = *v10;
      if ( v21 != *v10 )
      {
        v104 = 1;
        v105 = 0;
        while ( v28 != -4096 )
        {
          if ( !v105 && v28 == -8192 )
            v105 = v10;
          v26 = v24 & (unsigned int)(v26 + v104);
          v10 = (__int64 *)(v25 + 40 * v26);
          v28 = *v10;
          if ( v21 == *v10 )
            goto LABEL_15;
          ++v104;
        }
        if ( v105 )
          v10 = v105;
      }
LABEL_15:
      *(_DWORD *)(a3 + 16) = v27;
      if ( *v10 != -4096 )
        --*(_DWORD *)(a3 + 20);
      *v10 = v21;
      *(_OWORD *)(v10 + 1) = 0;
      *(_OWORD *)(v10 + 3) = 0;
      v20 = *(_DWORD *)(a3 + 24);
      if ( !v20 )
      {
        ++*(_QWORD *)a3;
        goto LABEL_19;
      }
      v6 = *(_QWORD *)(a3 + 8);
      v5 = v20 - 1;
      v12 = 0;
LABEL_6:
      v13 = 1;
      v14 = v5 & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
      v15 = (__int64 *)(v6 + 40LL * v14);
      v16 = 0;
      v17 = *v15;
      if ( v22 != *v15 )
      {
        while ( v17 != -4096 )
        {
          if ( !v16 && v17 == -8192 )
            v16 = v15;
          v14 = v5 & (v13 + v14);
          v15 = (__int64 *)(v6 + 40LL * v14);
          v17 = *v15;
          if ( v22 == *v15 )
            goto LABEL_7;
          ++v13;
        }
        v80 = *(_DWORD *)(a3 + 16);
        if ( !v16 )
          v16 = v15;
        ++*(_QWORD *)a3;
        v33 = v80 + 1;
        if ( 4 * (v80 + 1) < 3 * v20 )
        {
          if ( v20 - (v33 + *(_DWORD *)(a3 + 20)) <= v20 >> 3 )
          {
            sub_261D190(a3, v20);
            v81 = *(_DWORD *)(a3 + 24);
            if ( !v81 )
            {
LABEL_171:
              ++*(_DWORD *)(a3 + 16);
              BUG();
            }
            v82 = v81 - 1;
            v83 = *(_QWORD *)(a3 + 8);
            v84 = 0;
            v85 = (v81 - 1) & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
            v86 = 1;
            v33 = *(_DWORD *)(a3 + 16) + 1;
            v16 = (__int64 *)(v83 + 40LL * v85);
            v87 = *v16;
            if ( v22 != *v16 )
            {
              while ( v87 != -4096 )
              {
                if ( v87 == -8192 && !v84 )
                  v84 = v16;
                v85 = v82 & (v86 + v85);
                v16 = (__int64 *)(v83 + 40LL * v85);
                v87 = *v16;
                if ( v22 == *v16 )
                  goto LABEL_21;
                ++v86;
              }
              if ( v84 )
                v16 = v84;
            }
          }
          goto LABEL_21;
        }
LABEL_19:
        sub_261D190(a3, 2 * v20);
        v29 = *(_DWORD *)(a3 + 24);
        if ( !v29 )
          goto LABEL_171;
        v30 = v29 - 1;
        v31 = *(_QWORD *)(a3 + 8);
        v32 = (v29 - 1) & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
        v33 = *(_DWORD *)(a3 + 16) + 1;
        v16 = (__int64 *)(v31 + 40LL * v32);
        v34 = *v16;
        if ( v22 != *v16 )
        {
          v102 = 1;
          v103 = 0;
          while ( v34 != -4096 )
          {
            if ( !v103 && v34 == -8192 )
              v103 = v16;
            v32 = v30 & (v102 + v32);
            v16 = (__int64 *)(v31 + 40LL * v32);
            v34 = *v16;
            if ( v22 == *v16 )
              goto LABEL_21;
            ++v102;
          }
          if ( v103 )
            v16 = v103;
        }
LABEL_21:
        *(_DWORD *)(a3 + 16) = v33;
        if ( *v16 != -4096 )
          --*(_DWORD *)(a3 + 20);
        *v16 = v22;
        *(_OWORD *)(v16 + 1) = 0;
        *(_OWORD *)(v16 + 3) = 0;
        v18 = *(_QWORD *)v4;
        v20 = *(_DWORD *)(a3 + 24);
        break;
      }
LABEL_7:
      v18 = *(_QWORD *)v4;
      if ( *((_DWORD *)v15 + 2) <= v12 )
        break;
      v19 = v4 + 8;
      if ( a1 != v4 )
      {
        v114 = *(_QWORD *)v4;
        memmove(a1 + 8, a1, v4 - a1);
        v18 = v114;
      }
      v4 += 8;
      *(_QWORD *)a1 = v18;
      if ( a2 == v19 )
        return;
    }
    v35 = v4;
    v111 = ((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4);
    while ( 1 )
    {
      v115 = v35;
      v48 = *((_QWORD *)v35 - 1);
      v49 = v20;
      if ( v20 )
      {
        v36 = v20 - 1;
        v37 = *(_QWORD *)(a3 + 8);
        v38 = 1;
        v39 = (v20 - 1) & v111;
        v40 = (__int64 *)(v37 + 40LL * v39);
        v41 = 0;
        v42 = *v40;
        if ( v18 == *v40 )
        {
LABEL_26:
          v43 = *((_DWORD *)v40 + 2);
          goto LABEL_27;
        }
        while ( v42 != -4096 )
        {
          if ( v42 == -8192 && !v41 )
            v41 = v40;
          v39 = v36 & (v38 + v39);
          v40 = (__int64 *)(v37 + 40LL * v39);
          v42 = *v40;
          if ( *v40 == v18 )
            goto LABEL_26;
          ++v38;
        }
        v63 = *(_DWORD *)(a3 + 16);
        if ( !v41 )
          v41 = v40;
        ++*(_QWORD *)a3;
        v55 = v63 + 1;
        if ( 4 * v55 < 3 * v20 )
        {
          if ( v20 - *(_DWORD *)(a3 + 20) - v55 <= v20 >> 3 )
          {
            v107 = v18;
            sub_261D190(a3, v20);
            v64 = *(_DWORD *)(a3 + 24);
            if ( !v64 )
            {
LABEL_174:
              ++*(_DWORD *)(a3 + 16);
              BUG();
            }
            v65 = v64 - 1;
            v66 = *(_QWORD *)(a3 + 8);
            v67 = 1;
            v18 = v107;
            v68 = 0;
            LODWORD(v69) = v65 & v111;
            v41 = (__int64 *)(v66 + 40LL * (v65 & v111));
            v70 = *v41;
            v55 = *(_DWORD *)(a3 + 16) + 1;
            if ( *v41 != v107 )
            {
              while ( v70 != -4096 )
              {
                if ( v70 == -8192 && !v68 )
                  v68 = v41;
                v69 = v65 & (unsigned int)(v69 + v67);
                v41 = (__int64 *)(v66 + 40 * v69);
                v70 = *v41;
                if ( v107 == *v41 )
                  goto LABEL_34;
                ++v67;
              }
              if ( v68 )
                v41 = v68;
            }
          }
          goto LABEL_34;
        }
      }
      else
      {
        ++*(_QWORD *)a3;
      }
      v106 = v18;
      sub_261D190(a3, 2 * v20);
      v50 = *(_DWORD *)(a3 + 24);
      if ( !v50 )
        goto LABEL_174;
      v51 = v50 - 1;
      v52 = *(_QWORD *)(a3 + 8);
      v18 = v106;
      v53 = v51 & v111;
      v41 = (__int64 *)(v52 + 40LL * (v51 & v111));
      v54 = *v41;
      v55 = *(_DWORD *)(a3 + 16) + 1;
      if ( *v41 != v106 )
      {
        v96 = 1;
        v97 = 0;
        while ( v54 != -4096 )
        {
          if ( v97 || v54 != -8192 )
            v41 = v97;
          v53 = v51 & (v96 + v53);
          v54 = *(_QWORD *)(v52 + 40LL * v53);
          if ( v54 == v106 )
          {
            v41 = (__int64 *)(v52 + 40LL * v53);
            goto LABEL_34;
          }
          ++v96;
          v97 = v41;
          v41 = (__int64 *)(v52 + 40LL * v53);
        }
        if ( v97 )
          v41 = v97;
      }
LABEL_34:
      *(_DWORD *)(a3 + 16) = v55;
      if ( *v41 != -4096 )
        --*(_DWORD *)(a3 + 20);
      *v41 = v18;
      *(_OWORD *)(v41 + 1) = 0;
      *(_OWORD *)(v41 + 3) = 0;
      v49 = *(_DWORD *)(a3 + 24);
      if ( !v49 )
      {
        ++*(_QWORD *)a3;
        goto LABEL_38;
      }
      v37 = *(_QWORD *)(a3 + 8);
      v36 = v49 - 1;
      v43 = 0;
LABEL_27:
      v44 = ((unsigned int)v48 >> 9) ^ ((unsigned int)v48 >> 4);
      v45 = v44 & v36;
      v46 = (__int64 *)(v37 + 40LL * (v44 & v36));
      v47 = *v46;
      if ( *v46 != v48 )
        break;
LABEL_28:
      v35 -= 8;
      if ( *((_DWORD *)v46 + 2) <= v43 )
        goto LABEL_43;
      *((_QWORD *)v35 + 1) = *(_QWORD *)v35;
      v20 = *(_DWORD *)(a3 + 24);
    }
    v108 = 1;
    v71 = 0;
    while ( v47 != -4096 )
    {
      if ( !v71 && v47 == -8192 )
        v71 = v46;
      v45 = v36 & (v108 + v45);
      v46 = (__int64 *)(v37 + 40LL * v45);
      v47 = *v46;
      if ( v48 == *v46 )
        goto LABEL_28;
      ++v108;
    }
    v72 = *(_DWORD *)(a3 + 16);
    v61 = v71;
    if ( !v71 )
      v61 = v46;
    ++*(_QWORD *)a3;
    v60 = v72 + 1;
    if ( 4 * (v72 + 1) < 3 * v49 )
    {
      if ( v49 - (v60 + *(_DWORD *)(a3 + 20)) > v49 >> 3 )
        goto LABEL_40;
      v113 = v18;
      sub_261D190(a3, v49);
      v73 = *(_DWORD *)(a3 + 24);
      if ( v73 )
      {
        v74 = v73 - 1;
        v75 = *(_QWORD *)(a3 + 8);
        v76 = 0;
        LODWORD(v77) = v74 & v44;
        v18 = v113;
        v78 = 1;
        v60 = *(_DWORD *)(a3 + 16) + 1;
        v61 = (_QWORD *)(v75 + 40LL * (unsigned int)v77);
        v79 = *v61;
        if ( *v61 != v48 )
        {
          while ( v79 != -4096 )
          {
            if ( !v76 && v79 == -8192 )
              v76 = v61;
            v77 = v74 & (unsigned int)(v77 + v78);
            v61 = (_QWORD *)(v75 + 40 * v77);
            v79 = *v61;
            if ( v48 == *v61 )
              goto LABEL_40;
            ++v78;
          }
LABEL_71:
          if ( v76 )
            v61 = v76;
        }
        goto LABEL_40;
      }
LABEL_172:
      ++*(_DWORD *)(a3 + 16);
      BUG();
    }
LABEL_38:
    v112 = v18;
    sub_261D190(a3, 2 * v49);
    v56 = *(_DWORD *)(a3 + 24);
    if ( !v56 )
      goto LABEL_172;
    v57 = v56 - 1;
    v58 = *(_QWORD *)(a3 + 8);
    v18 = v112;
    v59 = v57 & (((unsigned int)v48 >> 9) ^ ((unsigned int)v48 >> 4));
    v60 = *(_DWORD *)(a3 + 16) + 1;
    v61 = (_QWORD *)(v58 + 40LL * v59);
    v62 = *v61;
    if ( *v61 != v48 )
    {
      v98 = 1;
      v76 = 0;
      while ( v62 != -4096 )
      {
        if ( v76 || v62 != -8192 )
          v61 = v76;
        v99 = v98 + 1;
        v100 = v59 + v98;
        v59 = v57 & v100;
        v101 = (_QWORD *)(v58 + 40LL * (v57 & v100));
        v62 = *v101;
        if ( v48 == *v101 )
        {
          v61 = (_QWORD *)(v58 + 40LL * (v57 & v100));
          goto LABEL_40;
        }
        v98 = v99;
        v76 = v61;
        v61 = v101;
      }
      goto LABEL_71;
    }
LABEL_40:
    *(_DWORD *)(a3 + 16) = v60;
    if ( *v61 != -4096 )
      --*(_DWORD *)(a3 + 20);
    *v61 = v48;
    *(_OWORD *)(v61 + 1) = 0;
    *(_OWORD *)(v61 + 3) = 0;
LABEL_43:
    v4 += 8;
    *(_QWORD *)v115 = v18;
  }
  while ( a2 != v4 );
}
