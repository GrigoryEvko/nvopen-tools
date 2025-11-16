// Function: sub_C185F0
// Address: 0xc185f0
//
_BOOL8 __fastcall sub_C185F0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rsi
  __int64 v7; // r12
  __int64 v8; // rcx
  unsigned int *v9; // r13
  unsigned int v10; // esi
  unsigned int v11; // r14d
  unsigned int v12; // r15d
  unsigned int v13; // ecx
  __int64 v14; // rdx
  int v15; // r11d
  unsigned int v16; // r10d
  int *v17; // rdi
  unsigned int *v18; // rax
  int v19; // r9d
  unsigned __int64 v20; // r8
  unsigned int v21; // r11d
  int *v22; // rdi
  int v23; // r10d
  unsigned __int64 v24; // rax
  bool v25; // al
  unsigned int v26; // esi
  unsigned int v27; // r14d
  unsigned int v28; // r15d
  unsigned int v29; // edx
  __int64 v30; // rcx
  int v31; // r11d
  unsigned int v32; // r10d
  int *v33; // rdi
  unsigned int *v34; // rax
  int v35; // r9d
  unsigned __int64 v36; // r8
  __int64 v37; // r11
  int *v38; // rdi
  int v39; // r10d
  unsigned __int64 v40; // rax
  int v41; // edx
  int v42; // edx
  int v43; // ecx
  int v44; // ecx
  __int64 v45; // r9
  unsigned int v46; // esi
  int v47; // edx
  unsigned int *v48; // rax
  unsigned int v49; // edi
  int v50; // r10d
  unsigned int *v51; // r11
  int v52; // edx
  int v53; // edx
  int v54; // ecx
  int v55; // ecx
  __int64 v56; // rdi
  unsigned int v57; // r10d
  int v58; // edx
  unsigned int *v59; // rax
  unsigned int v60; // esi
  int v61; // r9d
  unsigned int *v62; // r11
  int v64; // edx
  int v65; // edx
  int v66; // ecx
  int v67; // ecx
  __int64 v68; // r8
  __int64 v69; // rsi
  unsigned int v70; // edi
  int v71; // r11d
  unsigned int *v72; // r10
  int v73; // ecx
  int v74; // ecx
  __int64 v75; // r8
  __int64 v76; // rsi
  unsigned int v77; // edi
  int v78; // r11d
  unsigned int *v79; // r10
  int v80; // ecx
  int v81; // ecx
  __int64 v82; // rdi
  int v83; // r11d
  __int64 v84; // r8
  unsigned int v85; // esi
  int v86; // ecx
  int v87; // ecx
  __int64 v88; // rdi
  int v89; // r10d
  unsigned int v90; // r9d
  unsigned int v91; // esi
  int v92; // ecx
  int v93; // ecx
  int v94; // r11d
  __int64 v95; // rdi
  unsigned int *v96; // r9
  __int64 v97; // r10
  unsigned int v98; // esi
  int v99; // ecx
  int v100; // ecx
  __int64 v101; // rdi
  int v102; // r11d
  __int64 v103; // r8
  unsigned int v104; // esi
  unsigned __int64 v105; // [rsp+8h] [rbp-58h]
  int v106; // [rsp+8h] [rbp-58h]
  int v107; // [rsp+8h] [rbp-58h]
  unsigned __int64 v108; // [rsp+8h] [rbp-58h]
  int v109; // [rsp+8h] [rbp-58h]
  unsigned int *v110; // [rsp+10h] [rbp-50h]
  __int64 v111; // [rsp+18h] [rbp-48h]
  unsigned int *v112; // [rsp+20h] [rbp-40h]
  bool v113; // [rsp+28h] [rbp-38h]
  unsigned __int64 v114; // [rsp+28h] [rbp-38h]
  int v115; // [rsp+28h] [rbp-38h]
  int v116; // [rsp+28h] [rbp-38h]
  unsigned __int64 v117; // [rsp+28h] [rbp-38h]

  v4 = 4LL * *(unsigned int *)(a3 + 16);
  v110 = *(unsigned int **)(a3 + 8);
  v5 = 4LL * *(unsigned int *)(a2 + 16);
  v6 = *(_QWORD *)(a2 + 8);
  v7 = v6 + v5;
  v112 = &v110[(unsigned __int64)v4 / 4];
  v8 = v6 + v5 - v4;
  if ( v5 <= v4 )
    v8 = v6;
  v111 = v8;
  if ( v7 != v8 )
  {
    v9 = &v110[(unsigned __int64)v4 / 4 - 1];
    do
    {
      v26 = *(_DWORD *)(a1 + 24);
      v27 = *v9;
      v112 = v9;
      v28 = *(_DWORD *)(v7 - 4);
      if ( v26 )
      {
        v29 = v26 - 1;
        v30 = *(_QWORD *)(a1 + 8);
        v31 = 1;
        v32 = (v26 - 1) & (37 * v28);
        v33 = (int *)(v30 + 24LL * v32);
        v34 = 0;
        v35 = *v33;
        if ( v28 == *v33 )
        {
LABEL_18:
          v36 = *((_QWORD *)v33 + 1);
          goto LABEL_19;
        }
        while ( v35 != -1 )
        {
          if ( !v34 && v35 == -2 )
            v34 = (unsigned int *)v33;
          v32 = v29 & (v31 + v32);
          v33 = (int *)(v30 + 24LL * v32);
          v35 = *v33;
          if ( v28 == *v33 )
            goto LABEL_18;
          ++v31;
        }
        v41 = *(_DWORD *)(a1 + 16);
        if ( !v34 )
          v34 = (unsigned int *)v33;
        ++*(_QWORD *)a1;
        v42 = v41 + 1;
        if ( 4 * v42 < 3 * v26 )
        {
          if ( v26 - *(_DWORD *)(a1 + 20) - v42 > v26 >> 3 )
            goto LABEL_33;
          v116 = 37 * v28;
          sub_C18120(a1, v26);
          v80 = *(_DWORD *)(a1 + 24);
          if ( !v80 )
          {
LABEL_172:
            ++*(_DWORD *)(a1 + 16);
            BUG();
          }
          v81 = v80 - 1;
          v82 = *(_QWORD *)(a1 + 8);
          v72 = 0;
          v83 = 1;
          LODWORD(v84) = v81 & v116;
          v42 = *(_DWORD *)(a1 + 16) + 1;
          v34 = (unsigned int *)(v82 + 24LL * (v81 & (unsigned int)v116));
          v85 = *v34;
          if ( v28 == *v34 )
            goto LABEL_33;
          while ( v85 != -1 )
          {
            if ( v85 == -2 && !v72 )
              v72 = v34;
            v84 = v81 & (unsigned int)(v84 + v83);
            v34 = (unsigned int *)(v82 + 24 * v84);
            v85 = *v34;
            if ( v28 == *v34 )
              goto LABEL_33;
            ++v83;
          }
          goto LABEL_93;
        }
      }
      else
      {
        ++*(_QWORD *)a1;
      }
      sub_C18120(a1, 2 * v26);
      v66 = *(_DWORD *)(a1 + 24);
      if ( !v66 )
        goto LABEL_172;
      v67 = v66 - 1;
      v68 = *(_QWORD *)(a1 + 8);
      LODWORD(v69) = v67 & (37 * v28);
      v42 = *(_DWORD *)(a1 + 16) + 1;
      v34 = (unsigned int *)(v68 + 24LL * (unsigned int)v69);
      v70 = *v34;
      if ( v28 == *v34 )
        goto LABEL_33;
      v71 = 1;
      v72 = 0;
      while ( v70 != -1 )
      {
        if ( !v72 && v70 == -2 )
          v72 = v34;
        v69 = v67 & (unsigned int)(v69 + v71);
        v34 = (unsigned int *)(v68 + 24 * v69);
        v70 = *v34;
        if ( v28 == *v34 )
          goto LABEL_33;
        ++v71;
      }
LABEL_93:
      if ( v72 )
        v34 = v72;
LABEL_33:
      *(_DWORD *)(a1 + 16) = v42;
      if ( *v34 != -1 )
        --*(_DWORD *)(a1 + 20);
      *v34 = v28;
      *((_QWORD *)v34 + 1) = 0;
      *((_QWORD *)v34 + 2) = 0;
      v26 = *(_DWORD *)(a1 + 24);
      if ( !v26 )
      {
        ++*(_QWORD *)a1;
        v36 = 0;
        goto LABEL_37;
      }
      v30 = *(_QWORD *)(a1 + 8);
      v29 = v26 - 1;
      v36 = 0;
LABEL_19:
      v37 = (37 * v27) & v29;
      v38 = (int *)(v30 + 24 * v37);
      v39 = *v38;
      if ( v27 == *v38 )
      {
LABEL_20:
        v40 = *((_QWORD *)v38 + 1);
        goto LABEL_21;
      }
      v115 = 1;
      v48 = 0;
      while ( v39 != -1 )
      {
        if ( !v48 && v39 == -2 )
          v48 = (unsigned int *)v38;
        LODWORD(v37) = v29 & (v115 + v37);
        v38 = (int *)(v30 + 24LL * (unsigned int)v37);
        v39 = *v38;
        if ( v27 == *v38 )
          goto LABEL_20;
        ++v115;
      }
      v64 = *(_DWORD *)(a1 + 16);
      if ( !v48 )
        v48 = (unsigned int *)v38;
      ++*(_QWORD *)a1;
      v47 = v64 + 1;
      if ( 4 * v47 < 3 * v26 )
      {
        if ( v26 - (v47 + *(_DWORD *)(a1 + 20)) > v26 >> 3 )
          goto LABEL_76;
        v107 = 37 * v27;
        v117 = v36;
        sub_C18120(a1, v26);
        v86 = *(_DWORD *)(a1 + 24);
        if ( !v86 )
        {
LABEL_171:
          ++*(_DWORD *)(a1 + 16);
          BUG();
        }
        v87 = v86 - 1;
        v88 = *(_QWORD *)(a1 + 8);
        v51 = 0;
        v36 = v117;
        v89 = 1;
        v90 = v87 & v107;
        v47 = *(_DWORD *)(a1 + 16) + 1;
        v48 = (unsigned int *)(v88 + 24LL * (v87 & (unsigned int)v107));
        v91 = *v48;
        if ( v27 == *v48 )
          goto LABEL_76;
        while ( v91 != -1 )
        {
          if ( v91 == -2 && !v51 )
            v51 = v48;
          v90 = v87 & (v89 + v90);
          v48 = (unsigned int *)(v88 + 24LL * v90);
          v91 = *v48;
          if ( v27 == *v48 )
            goto LABEL_76;
          ++v89;
        }
        goto LABEL_41;
      }
LABEL_37:
      v114 = v36;
      sub_C18120(a1, 2 * v26);
      v43 = *(_DWORD *)(a1 + 24);
      if ( !v43 )
        goto LABEL_171;
      v44 = v43 - 1;
      v45 = *(_QWORD *)(a1 + 8);
      v36 = v114;
      v46 = v44 & (37 * v27);
      v47 = *(_DWORD *)(a1 + 16) + 1;
      v48 = (unsigned int *)(v45 + 24LL * v46);
      v49 = *v48;
      if ( v27 == *v48 )
        goto LABEL_76;
      v50 = 1;
      v51 = 0;
      while ( v49 != -1 )
      {
        if ( !v51 && v49 == -2 )
          v51 = v48;
        v46 = v44 & (v50 + v46);
        v48 = (unsigned int *)(v45 + 24LL * v46);
        v49 = *v48;
        if ( v27 == *v48 )
          goto LABEL_76;
        ++v50;
      }
LABEL_41:
      if ( v51 )
        v48 = v51;
LABEL_76:
      *(_DWORD *)(a1 + 16) = v47;
      if ( *v48 != -1 )
        --*(_DWORD *)(a1 + 20);
      *v48 = v27;
      *((_QWORD *)v48 + 1) = 0;
      *((_QWORD *)v48 + 2) = 0;
      v40 = 0;
LABEL_21:
      if ( v36 == v40 )
        v113 = v27 > v28;
      else
        v113 = v36 < v40;
      if ( v113 )
        return v113;
      v10 = *(_DWORD *)(a1 + 24);
      v11 = *(_DWORD *)(v7 - 4);
      v12 = *v9;
      if ( v10 )
      {
        v13 = v10 - 1;
        v14 = *(_QWORD *)(a1 + 8);
        v15 = 1;
        v16 = (v10 - 1) & (37 * v12);
        v17 = (int *)(v14 + 24LL * v16);
        v18 = 0;
        v19 = *v17;
        if ( v12 == *v17 )
        {
LABEL_9:
          v20 = *((_QWORD *)v17 + 1);
          goto LABEL_10;
        }
        while ( v19 != -1 )
        {
          if ( v19 == -2 && !v18 )
            v18 = (unsigned int *)v17;
          v16 = v13 & (v15 + v16);
          v17 = (int *)(v14 + 24LL * v16);
          v19 = *v17;
          if ( v12 == *v17 )
            goto LABEL_9;
          ++v15;
        }
        v52 = *(_DWORD *)(a1 + 16);
        if ( !v18 )
          v18 = (unsigned int *)v17;
        ++*(_QWORD *)a1;
        v53 = v52 + 1;
        if ( 4 * v53 < 3 * v10 )
        {
          if ( v10 - *(_DWORD *)(a1 + 20) - v53 > v10 >> 3 )
            goto LABEL_54;
          v109 = 37 * v12;
          sub_C18120(a1, v10);
          v99 = *(_DWORD *)(a1 + 24);
          if ( !v99 )
          {
LABEL_173:
            ++*(_DWORD *)(a1 + 16);
            BUG();
          }
          v100 = v99 - 1;
          v101 = *(_QWORD *)(a1 + 8);
          v79 = 0;
          v102 = 1;
          LODWORD(v103) = v100 & v109;
          v53 = *(_DWORD *)(a1 + 16) + 1;
          v18 = (unsigned int *)(v101 + 24LL * (v100 & (unsigned int)v109));
          v104 = *v18;
          if ( v12 == *v18 )
            goto LABEL_54;
          while ( v104 != -1 )
          {
            if ( !v79 && v104 == -2 )
              v79 = v18;
            v103 = v100 & (unsigned int)(v103 + v102);
            v18 = (unsigned int *)(v101 + 24 * v103);
            v104 = *v18;
            if ( v12 == *v18 )
              goto LABEL_54;
            ++v102;
          }
          goto LABEL_101;
        }
      }
      else
      {
        ++*(_QWORD *)a1;
      }
      sub_C18120(a1, 2 * v10);
      v73 = *(_DWORD *)(a1 + 24);
      if ( !v73 )
        goto LABEL_173;
      v74 = v73 - 1;
      v75 = *(_QWORD *)(a1 + 8);
      LODWORD(v76) = v74 & (37 * v12);
      v53 = *(_DWORD *)(a1 + 16) + 1;
      v18 = (unsigned int *)(v75 + 24LL * (unsigned int)v76);
      v77 = *v18;
      if ( v12 == *v18 )
        goto LABEL_54;
      v78 = 1;
      v79 = 0;
      while ( v77 != -1 )
      {
        if ( !v79 && v77 == -2 )
          v79 = v18;
        v76 = v74 & (unsigned int)(v76 + v78);
        v18 = (unsigned int *)(v75 + 24 * v76);
        v77 = *v18;
        if ( v12 == *v18 )
          goto LABEL_54;
        ++v78;
      }
LABEL_101:
      if ( v79 )
        v18 = v79;
LABEL_54:
      *(_DWORD *)(a1 + 16) = v53;
      if ( *v18 != -1 )
        --*(_DWORD *)(a1 + 20);
      *v18 = v12;
      *((_QWORD *)v18 + 1) = 0;
      *((_QWORD *)v18 + 2) = 0;
      v10 = *(_DWORD *)(a1 + 24);
      if ( !v10 )
      {
        ++*(_QWORD *)a1;
        v20 = 0;
        goto LABEL_58;
      }
      v14 = *(_QWORD *)(a1 + 8);
      v13 = v10 - 1;
      v20 = 0;
LABEL_10:
      v21 = v13 & (37 * v11);
      v22 = (int *)(v14 + 24LL * v21);
      v23 = *v22;
      if ( v11 == *v22 )
      {
LABEL_11:
        v24 = *((_QWORD *)v22 + 1);
        goto LABEL_12;
      }
      v106 = 1;
      v59 = 0;
      while ( v23 != -1 )
      {
        if ( !v59 && v23 == -2 )
          v59 = (unsigned int *)v22;
        v21 = v13 & (v106 + v21);
        v22 = (int *)(v14 + 24LL * v21);
        v23 = *v22;
        if ( v11 == *v22 )
          goto LABEL_11;
        ++v106;
      }
      v65 = *(_DWORD *)(a1 + 16);
      if ( !v59 )
        v59 = (unsigned int *)v22;
      ++*(_QWORD *)a1;
      v58 = v65 + 1;
      if ( 4 * v58 < 3 * v10 )
      {
        if ( v10 - (v58 + *(_DWORD *)(a1 + 20)) <= v10 >> 3 )
        {
          v108 = v20;
          sub_C18120(a1, v10);
          v92 = *(_DWORD *)(a1 + 24);
          if ( !v92 )
          {
LABEL_170:
            ++*(_DWORD *)(a1 + 16);
            BUG();
          }
          v93 = v92 - 1;
          v94 = 1;
          v95 = *(_QWORD *)(a1 + 8);
          v20 = v108;
          v58 = *(_DWORD *)(a1 + 16) + 1;
          v96 = 0;
          v97 = v93 & (37 * v11);
          v59 = (unsigned int *)(v95 + 24 * v97);
          v98 = *v59;
          if ( v11 != *v59 )
          {
            while ( v98 != -1 )
            {
              if ( v98 != -2 || v96 )
                v59 = v96;
              LODWORD(v97) = v93 & (v94 + v97);
              v98 = *(_DWORD *)(v95 + 24LL * (unsigned int)v97);
              if ( v11 == v98 )
              {
                v59 = (unsigned int *)(v95 + 24LL * (unsigned int)v97);
                goto LABEL_85;
              }
              ++v94;
              v96 = v59;
              v59 = (unsigned int *)(v95 + 24LL * (unsigned int)v97);
            }
            if ( v96 )
              v59 = v96;
          }
        }
        goto LABEL_85;
      }
LABEL_58:
      v105 = v20;
      sub_C18120(a1, 2 * v10);
      v54 = *(_DWORD *)(a1 + 24);
      if ( !v54 )
        goto LABEL_170;
      v55 = v54 - 1;
      v56 = *(_QWORD *)(a1 + 8);
      v20 = v105;
      v57 = v55 & (37 * v11);
      v58 = *(_DWORD *)(a1 + 16) + 1;
      v59 = (unsigned int *)(v56 + 24LL * v57);
      v60 = *v59;
      if ( v11 != *v59 )
      {
        v61 = 1;
        v62 = 0;
        while ( v60 != -1 )
        {
          if ( !v62 && v60 == -2 )
            v62 = v59;
          v57 = v55 & (v61 + v57);
          v59 = (unsigned int *)(v56 + 24LL * v57);
          v60 = *v59;
          if ( v11 == *v59 )
            goto LABEL_85;
          ++v61;
        }
        if ( v62 )
          v59 = v62;
      }
LABEL_85:
      *(_DWORD *)(a1 + 16) = v58;
      if ( *v59 != -1 )
        --*(_DWORD *)(a1 + 20);
      *v59 = v11;
      *((_QWORD *)v59 + 1) = 0;
      *((_QWORD *)v59 + 2) = 0;
      v24 = 0;
LABEL_12:
      if ( v20 == v24 )
        v25 = v11 > v12;
      else
        v25 = v20 < v24;
      if ( v25 )
        return v113;
      v7 -= 4;
      --v9;
    }
    while ( v111 != v7 );
  }
  return v110 != v112;
}
