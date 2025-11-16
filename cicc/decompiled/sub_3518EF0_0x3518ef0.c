// Function: sub_3518EF0
// Address: 0x3518ef0
//
void __fastcall sub_3518EF0(unsigned __int64 *a1, __int64 a2)
{
  unsigned __int64 v2; // rcx
  unsigned __int64 *i; // rdx
  unsigned __int64 *v5; // rax
  unsigned __int64 *v6; // rax
  unsigned __int64 *v7; // r11
  unsigned __int64 *v8; // r13
  unsigned __int64 *v9; // r15
  unsigned int v10; // esi
  unsigned int v11; // edi
  __int64 v12; // rdx
  int v13; // r10d
  unsigned __int64 **v14; // rcx
  unsigned int v15; // r9d
  unsigned __int64 **v16; // rax
  unsigned __int64 *v17; // r8
  unsigned __int64 v18; // r10
  int v19; // r14d
  unsigned __int64 **v20; // rcx
  unsigned int v21; // r9d
  unsigned __int64 **v22; // rax
  unsigned __int64 *v23; // r8
  unsigned __int64 *v24; // r12
  unsigned int v25; // edi
  int v26; // r10d
  unsigned __int64 **v27; // rcx
  unsigned int v28; // r9d
  unsigned __int64 **v29; // rax
  unsigned __int64 *v30; // r8
  unsigned __int64 v31; // r8
  int v32; // r14d
  unsigned __int64 **v33; // rcx
  unsigned int v34; // r10d
  unsigned __int64 **v35; // rax
  unsigned __int64 *v36; // r9
  int v37; // eax
  int v38; // esi
  __int64 v39; // rdi
  unsigned int v40; // eax
  int v41; // edx
  unsigned __int64 *v42; // r8
  int v43; // edx
  int v44; // edx
  __int64 v45; // r8
  unsigned int v46; // edi
  int v47; // eax
  unsigned __int64 *v48; // rsi
  unsigned __int64 v49; // rcx
  unsigned __int64 v50; // rax
  unsigned __int64 v51; // rsi
  unsigned __int64 v52; // rcx
  unsigned __int64 v53; // rdx
  unsigned __int64 v54; // rax
  int v55; // eax
  int v56; // edx
  int v57; // edx
  __int64 v58; // r8
  unsigned __int64 **v59; // r9
  int v60; // r14d
  unsigned int v61; // edi
  unsigned __int64 *v62; // rsi
  int v63; // eax
  int v64; // eax
  int v65; // eax
  __int64 v66; // rdi
  unsigned __int64 **v67; // r8
  unsigned int v68; // r14d
  int v69; // r10d
  unsigned __int64 *v70; // rsi
  int v71; // eax
  int v72; // eax
  int v73; // eax
  int v74; // edx
  int v75; // eax
  int v76; // edi
  __int64 v77; // rsi
  __int64 v78; // rdx
  unsigned __int64 *v79; // r8
  int v80; // r10d
  unsigned __int64 **v81; // r9
  int v82; // eax
  int v83; // edi
  __int64 v84; // rsi
  __int64 v85; // rax
  unsigned __int64 *v86; // r8
  int v87; // r10d
  unsigned __int64 **v88; // r9
  int v89; // eax
  int v90; // eax
  __int64 v91; // rdi
  unsigned __int64 **v92; // r8
  __int64 v93; // r12
  int v94; // r9d
  unsigned __int64 *v95; // rsi
  int v96; // eax
  int v97; // edx
  __int64 v98; // rdi
  unsigned __int64 **v99; // r8
  __int64 v100; // r12
  int v101; // r9d
  unsigned __int64 *v102; // rsi
  int v103; // r10d
  int v104; // r10d
  unsigned __int64 **v105; // r9
  unsigned __int64 *v106; // [rsp+0h] [rbp-60h]
  unsigned __int64 *v107; // [rsp+0h] [rbp-60h]
  unsigned __int64 *v108; // [rsp+0h] [rbp-60h]
  unsigned __int64 *v109; // [rsp+0h] [rbp-60h]
  unsigned int v111; // [rsp+18h] [rbp-48h]
  unsigned __int64 *v112; // [rsp+18h] [rbp-48h]
  unsigned __int64 *v113; // [rsp+18h] [rbp-48h]
  unsigned __int64 *v114; // [rsp+18h] [rbp-48h]
  unsigned __int64 *v115; // [rsp+18h] [rbp-48h]
  unsigned __int64 v116; // [rsp+20h] [rbp-40h] BYREF
  unsigned __int64 *v117; // [rsp+28h] [rbp-38h]

  v2 = *a1 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (unsigned __int64 *)v2 != a1 )
  {
    i = (unsigned __int64 *)a1[1];
    v5 = (unsigned __int64 *)i[1];
    if ( a1 != v5 )
    {
      if ( a1 != i )
      {
        for ( i = (unsigned __int64 *)i[1]; ; i = (unsigned __int64 *)i[1] )
        {
          v6 = (unsigned __int64 *)v5[1];
          if ( a1 == v6 )
            break;
          v5 = (unsigned __int64 *)v6[1];
          if ( a1 == v5 )
            break;
        }
      }
      v117 = &v116;
      v116 = (unsigned __int64)&v116 + 4;
      if ( a1 != i )
      {
        *(_QWORD *)((*i & 0xFFFFFFFFFFFFFFF8LL) + 8) = a1;
        *a1 = *a1 & 7 | *i & 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v2 + 8) = &v116;
        *i = (unsigned __int64)&v116 | *i & 7;
        v117 = i;
        v116 = v116 & 7 | v2;
      }
      sub_3518EF0(a1, a2);
      sub_3518EF0(&v116, a2);
      v7 = &v116;
      if ( &v116 != (unsigned __int64 *)(v116 & 0xFFFFFFFFFFFFFFF8LL) )
      {
        v8 = (unsigned __int64 *)a1[1];
        v9 = v117;
        while ( a1 != v8 )
        {
          v10 = *(_DWORD *)(a2 + 24);
          if ( v10 )
          {
            v11 = v10 - 1;
            v12 = *(_QWORD *)(a2 + 8);
            v13 = 1;
            v14 = 0;
            v15 = (v10 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
            v16 = (unsigned __int64 **)(v12 + 16LL * v15);
            v17 = *v16;
            if ( *v16 == v9 )
            {
LABEL_14:
              v18 = (unsigned __int64)v16[1];
              goto LABEL_15;
            }
            while ( v17 != (unsigned __int64 *)-4096LL )
            {
              if ( !v14 && v17 == (unsigned __int64 *)-8192LL )
                v14 = v16;
              v15 = v11 & (v13 + v15);
              v16 = (unsigned __int64 **)(v12 + 16LL * v15);
              v17 = *v16;
              if ( *v16 == v9 )
                goto LABEL_14;
              ++v13;
            }
            if ( !v14 )
              v14 = v16;
            v73 = *(_DWORD *)(a2 + 16);
            ++*(_QWORD *)a2;
            v74 = v73 + 1;
            if ( 4 * (v73 + 1) < 3 * v10 )
            {
              if ( v10 - *(_DWORD *)(a2 + 20) - v74 <= v10 >> 3 )
              {
                v114 = v7;
                sub_2E3E470(a2, v10);
                v89 = *(_DWORD *)(a2 + 24);
                if ( !v89 )
                {
LABEL_184:
                  ++*(_DWORD *)(a2 + 16);
                  BUG();
                }
                v90 = v89 - 1;
                v91 = *(_QWORD *)(a2 + 8);
                v92 = 0;
                LODWORD(v93) = v90 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
                v7 = v114;
                v94 = 1;
                v74 = *(_DWORD *)(a2 + 16) + 1;
                v14 = (unsigned __int64 **)(v91 + 16LL * (unsigned int)v93);
                v95 = *v14;
                if ( *v14 != v9 )
                {
                  while ( v95 != (unsigned __int64 *)-4096LL )
                  {
                    if ( v95 == (unsigned __int64 *)-8192LL && !v92 )
                      v92 = v14;
                    v93 = v90 & (unsigned int)(v93 + v94);
                    v14 = (unsigned __int64 **)(v91 + 16 * v93);
                    v95 = *v14;
                    if ( *v14 == v9 )
                      goto LABEL_100;
                    ++v94;
                  }
                  if ( v92 )
                    v14 = v92;
                }
              }
              goto LABEL_100;
            }
          }
          else
          {
            ++*(_QWORD *)a2;
          }
          v113 = v7;
          sub_2E3E470(a2, 2 * v10);
          v82 = *(_DWORD *)(a2 + 24);
          if ( !v82 )
            goto LABEL_184;
          v83 = v82 - 1;
          v84 = *(_QWORD *)(a2 + 8);
          v7 = v113;
          LODWORD(v85) = (v82 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
          v74 = *(_DWORD *)(a2 + 16) + 1;
          v14 = (unsigned __int64 **)(v84 + 16LL * (unsigned int)v85);
          v86 = *v14;
          if ( *v14 != v9 )
          {
            v87 = 1;
            v88 = 0;
            while ( v86 != (unsigned __int64 *)-4096LL )
            {
              if ( !v88 && v86 == (unsigned __int64 *)-8192LL )
                v88 = v14;
              v85 = v83 & (unsigned int)(v85 + v87);
              v14 = (unsigned __int64 **)(v84 + 16 * v85);
              v86 = *v14;
              if ( *v14 == v9 )
                goto LABEL_100;
              ++v87;
            }
            if ( v88 )
              v14 = v88;
          }
LABEL_100:
          *(_DWORD *)(a2 + 16) = v74;
          if ( *v14 != (unsigned __int64 *)-4096LL )
            --*(_DWORD *)(a2 + 20);
          *v14 = v9;
          v14[1] = 0;
          v10 = *(_DWORD *)(a2 + 24);
          if ( !v10 )
          {
            ++*(_QWORD *)a2;
            goto LABEL_104;
          }
          v12 = *(_QWORD *)(a2 + 8);
          v11 = v10 - 1;
          v18 = 0;
LABEL_15:
          v19 = 1;
          v20 = 0;
          v21 = v11 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
          v22 = (unsigned __int64 **)(v12 + 16LL * v21);
          v23 = *v22;
          if ( *v22 != v8 )
          {
            while ( v23 != (unsigned __int64 *)-4096LL )
            {
              if ( !v20 && v23 == (unsigned __int64 *)-8192LL )
                v20 = v22;
              v21 = v11 & (v19 + v21);
              v22 = (unsigned __int64 **)(v12 + 16LL * v21);
              v23 = *v22;
              if ( *v22 == v8 )
                goto LABEL_16;
              ++v19;
            }
            if ( !v20 )
              v20 = v22;
            v71 = *(_DWORD *)(a2 + 16);
            ++*(_QWORD *)a2;
            v72 = v71 + 1;
            if ( 4 * v72 >= 3 * v10 )
            {
LABEL_104:
              v112 = v7;
              sub_2E3E470(a2, 2 * v10);
              v75 = *(_DWORD *)(a2 + 24);
              if ( !v75 )
                goto LABEL_183;
              v76 = v75 - 1;
              v77 = *(_QWORD *)(a2 + 8);
              v7 = v112;
              LODWORD(v78) = (v75 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
              v72 = *(_DWORD *)(a2 + 16) + 1;
              v20 = (unsigned __int64 **)(v77 + 16LL * (unsigned int)v78);
              v79 = *v20;
              if ( v8 != *v20 )
              {
                v80 = 1;
                v81 = 0;
                while ( v79 != (unsigned __int64 *)-4096LL )
                {
                  if ( !v81 && v79 == (unsigned __int64 *)-8192LL )
                    v81 = v20;
                  v78 = v76 & (unsigned int)(v78 + v80);
                  v20 = (unsigned __int64 **)(v77 + 16 * v78);
                  v79 = *v20;
                  if ( *v20 == v8 )
                    goto LABEL_87;
                  ++v80;
                }
                if ( v81 )
                  v20 = v81;
              }
            }
            else if ( v10 - (v72 + *(_DWORD *)(a2 + 20)) <= v10 >> 3 )
            {
              v115 = v7;
              sub_2E3E470(a2, v10);
              v96 = *(_DWORD *)(a2 + 24);
              if ( !v96 )
              {
LABEL_183:
                ++*(_DWORD *)(a2 + 16);
                BUG();
              }
              v97 = v96 - 1;
              v98 = *(_QWORD *)(a2 + 8);
              v99 = 0;
              LODWORD(v100) = (v96 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
              v7 = v115;
              v101 = 1;
              v72 = *(_DWORD *)(a2 + 16) + 1;
              v20 = (unsigned __int64 **)(v98 + 16LL * (unsigned int)v100);
              v102 = *v20;
              if ( *v20 != v8 )
              {
                while ( v102 != (unsigned __int64 *)-4096LL )
                {
                  if ( v102 == (unsigned __int64 *)-8192LL && !v99 )
                    v99 = v20;
                  v100 = v97 & (unsigned int)(v100 + v101);
                  v20 = (unsigned __int64 **)(v98 + 16 * v100);
                  v102 = *v20;
                  if ( *v20 == v8 )
                    goto LABEL_87;
                  ++v101;
                }
                if ( v99 )
                  v20 = v99;
              }
            }
LABEL_87:
            *(_DWORD *)(a2 + 16) = v72;
            if ( *v20 != (unsigned __int64 *)-4096LL )
              --*(_DWORD *)(a2 + 20);
            *v20 = v8;
            v20[1] = 0;
            goto LABEL_41;
          }
LABEL_16:
          if ( (unsigned __int64)v22[1] > v18 )
          {
            v24 = (unsigned __int64 *)v9[1];
            v111 = ((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4);
            if ( v24 == v7 )
            {
LABEL_112:
              v24 = v7;
LABEL_36:
              if ( v24 != v8 && v24 != v9 )
              {
                v49 = *v24 & 0xFFFFFFFFFFFFFFF8LL;
                *(_QWORD *)((*v9 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v24;
                *v24 = *v24 & 7 | *v9 & 0xFFFFFFFFFFFFFFF8LL;
                v50 = *v8;
                *(_QWORD *)(v49 + 8) = v8;
                v50 &= 0xFFFFFFFFFFFFFFF8LL;
                *v9 = v50 | *v9 & 7;
                *(_QWORD *)(v50 + 8) = v9;
                *v8 = v49 | *v8 & 7;
              }
              if ( v24 == v7 )
                return;
              v9 = v24;
              goto LABEL_41;
            }
            while ( 1 )
            {
              v25 = v10 - 1;
              v26 = 1;
              v27 = 0;
              v28 = (v10 - 1) & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
              v29 = (unsigned __int64 **)(v12 + 16LL * v28);
              v30 = *v29;
              if ( *v29 == v24 )
              {
LABEL_20:
                v31 = (unsigned __int64)v29[1];
              }
              else
              {
                while ( v30 != (unsigned __int64 *)-4096LL )
                {
                  if ( !v27 && v30 == (unsigned __int64 *)-8192LL )
                    v27 = v29;
                  v28 = v25 & (v26 + v28);
                  v29 = (unsigned __int64 **)(v12 + 16LL * v28);
                  v30 = *v29;
                  if ( *v29 == v24 )
                    goto LABEL_20;
                  ++v26;
                }
                if ( !v27 )
                  v27 = v29;
                v63 = *(_DWORD *)(a2 + 16);
                ++*(_QWORD *)a2;
                v41 = v63 + 1;
                if ( 4 * (v63 + 1) >= 3 * v10 )
                {
                  v106 = v7;
                  sub_2E3E470(a2, 2 * v10);
                  v37 = *(_DWORD *)(a2 + 24);
                  if ( !v37 )
                    goto LABEL_185;
                  v38 = v37 - 1;
                  v39 = *(_QWORD *)(a2 + 8);
                  v7 = v106;
                  v40 = (v37 - 1) & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
                  v41 = *(_DWORD *)(a2 + 16) + 1;
                  v27 = (unsigned __int64 **)(v39 + 16LL * v40);
                  v42 = *v27;
                  if ( *v27 != v24 )
                  {
                    v104 = 1;
                    v105 = 0;
                    while ( v42 != (unsigned __int64 *)-4096LL )
                    {
                      if ( v42 == (unsigned __int64 *)-8192LL && !v105 )
                        v105 = v27;
                      v40 = v38 & (v104 + v40);
                      v27 = (unsigned __int64 **)(v39 + 16LL * v40);
                      v42 = *v27;
                      if ( v24 == *v27 )
                        goto LABEL_27;
                      ++v104;
                    }
                    if ( v105 )
                      v27 = v105;
                  }
                }
                else if ( v10 - *(_DWORD *)(a2 + 20) - v41 <= v10 >> 3 )
                {
                  v109 = v7;
                  sub_2E3E470(a2, v10);
                  v64 = *(_DWORD *)(a2 + 24);
                  if ( !v64 )
                  {
LABEL_185:
                    ++*(_DWORD *)(a2 + 16);
                    BUG();
                  }
                  v65 = v64 - 1;
                  v66 = *(_QWORD *)(a2 + 8);
                  v67 = 0;
                  v68 = v65 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
                  v7 = v109;
                  v69 = 1;
                  v41 = *(_DWORD *)(a2 + 16) + 1;
                  v27 = (unsigned __int64 **)(v66 + 16LL * v68);
                  v70 = *v27;
                  if ( *v27 != v24 )
                  {
                    while ( v70 != (unsigned __int64 *)-4096LL )
                    {
                      if ( v70 != (unsigned __int64 *)-8192LL || v67 )
                        v27 = v67;
                      v68 = v65 & (v69 + v68);
                      v70 = *(unsigned __int64 **)(v66 + 16LL * v68);
                      if ( v24 == v70 )
                      {
                        v27 = (unsigned __int64 **)(v66 + 16LL * v68);
                        goto LABEL_27;
                      }
                      ++v69;
                      v67 = v27;
                      v27 = (unsigned __int64 **)(v66 + 16LL * v68);
                    }
                    if ( v67 )
                      v27 = v67;
                  }
                }
LABEL_27:
                *(_DWORD *)(a2 + 16) = v41;
                if ( *v27 != (unsigned __int64 *)-4096LL )
                  --*(_DWORD *)(a2 + 20);
                *v27 = v24;
                v27[1] = 0;
                v10 = *(_DWORD *)(a2 + 24);
                if ( !v10 )
                {
                  ++*(_QWORD *)a2;
                  goto LABEL_31;
                }
                v12 = *(_QWORD *)(a2 + 8);
                v25 = v10 - 1;
                v31 = 0;
              }
              v32 = 1;
              v33 = 0;
              v34 = v25 & v111;
              v35 = (unsigned __int64 **)(v12 + 16LL * (v25 & v111));
              v36 = *v35;
              if ( v8 != *v35 )
                break;
LABEL_22:
              if ( v31 >= (unsigned __int64)v35[1] )
                goto LABEL_36;
              v24 = (unsigned __int64 *)v24[1];
              if ( v24 == v7 )
                goto LABEL_112;
            }
            while ( v36 != (unsigned __int64 *)-4096LL )
            {
              if ( !v33 && v36 == (unsigned __int64 *)-8192LL )
                v33 = v35;
              v34 = v25 & (v32 + v34);
              v35 = (unsigned __int64 **)(v12 + 16LL * v34);
              v36 = *v35;
              if ( *v35 == v8 )
                goto LABEL_22;
              ++v32;
            }
            if ( !v33 )
              v33 = v35;
            v55 = *(_DWORD *)(a2 + 16);
            ++*(_QWORD *)a2;
            v47 = v55 + 1;
            if ( 4 * v47 >= 3 * v10 )
            {
LABEL_31:
              v107 = v7;
              sub_2E3E470(a2, 2 * v10);
              v43 = *(_DWORD *)(a2 + 24);
              if ( !v43 )
                goto LABEL_186;
              v44 = v43 - 1;
              v45 = *(_QWORD *)(a2 + 8);
              v7 = v107;
              v46 = v44 & v111;
              v47 = *(_DWORD *)(a2 + 16) + 1;
              v33 = (unsigned __int64 **)(v45 + 16LL * (v44 & v111));
              v48 = *v33;
              if ( *v33 != v8 )
              {
                v103 = 1;
                v59 = 0;
                while ( v48 != (unsigned __int64 *)-4096LL )
                {
                  if ( v48 == (unsigned __int64 *)-8192LL && !v59 )
                    v59 = v33;
                  v46 = v44 & (v103 + v46);
                  v33 = (unsigned __int64 **)(v45 + 16LL * v46);
                  v48 = *v33;
                  if ( *v33 == v8 )
                    goto LABEL_33;
                  ++v103;
                }
LABEL_58:
                if ( v59 )
                  v33 = v59;
              }
            }
            else if ( v10 - (v47 + *(_DWORD *)(a2 + 20)) <= v10 >> 3 )
            {
              v108 = v7;
              sub_2E3E470(a2, v10);
              v56 = *(_DWORD *)(a2 + 24);
              if ( !v56 )
              {
LABEL_186:
                ++*(_DWORD *)(a2 + 16);
                BUG();
              }
              v57 = v56 - 1;
              v58 = *(_QWORD *)(a2 + 8);
              v59 = 0;
              v7 = v108;
              v60 = 1;
              v61 = v57 & v111;
              v47 = *(_DWORD *)(a2 + 16) + 1;
              v33 = (unsigned __int64 **)(v58 + 16LL * (v57 & v111));
              v62 = *v33;
              if ( v8 != *v33 )
              {
                while ( v62 != (unsigned __int64 *)-4096LL )
                {
                  if ( v62 != (unsigned __int64 *)-8192LL || v59 )
                    v33 = v59;
                  v61 = v57 & (v60 + v61);
                  v62 = *(unsigned __int64 **)(v58 + 16LL * v61);
                  if ( v8 == v62 )
                  {
                    v33 = (unsigned __int64 **)(v58 + 16LL * v61);
                    goto LABEL_33;
                  }
                  ++v60;
                  v59 = v33;
                  v33 = (unsigned __int64 **)(v58 + 16LL * v61);
                }
                goto LABEL_58;
              }
            }
LABEL_33:
            *(_DWORD *)(a2 + 16) = v47;
            if ( *v33 != (unsigned __int64 *)-4096LL )
              --*(_DWORD *)(a2 + 20);
            *v33 = v8;
            v33[1] = 0;
            goto LABEL_36;
          }
LABEL_41:
          v8 = (unsigned __int64 *)v8[1];
        }
        if ( v9 != &v116 )
        {
          v51 = v116;
          v52 = v116 & 7;
          *(_QWORD *)((*v9 & 0xFFFFFFFFFFFFFFF8LL) + 8) = &v116;
          v53 = *v9;
          v51 &= 0xFFFFFFFFFFFFFFF8LL;
          v54 = *a1;
          *(_QWORD *)(v51 + 8) = a1;
          v116 = v52 | v53 & 0xFFFFFFFFFFFFFFF8LL;
          *v9 = v54 & 0xFFFFFFFFFFFFFFF8LL | *v9 & 7;
          *(_QWORD *)((v54 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v9;
          *a1 = v51 | *a1 & 7;
        }
      }
    }
  }
}
