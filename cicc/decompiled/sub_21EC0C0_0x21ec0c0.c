// Function: sub_21EC0C0
// Address: 0x21ec0c0
//
__int64 *__fastcall sub_21EC0C0(__int64 *a1, __int64 a2, int *a3, __int64 a4)
{
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 *v6; // r15
  __int64 v7; // rbx
  int *v8; // rcx
  __int64 v9; // rax
  int v10; // edi
  __int64 v11; // rdx
  unsigned int v12; // r8d
  __int64 *v13; // rsi
  __int64 v14; // r12
  __int64 v15; // r12
  __int64 v16; // r14
  unsigned int v17; // esi
  __int64 v18; // r8
  unsigned int v19; // r13d
  unsigned int v20; // r9d
  int *v21; // r10
  int v22; // r11d
  unsigned int v23; // eax
  unsigned __int64 v24; // rdx
  __int64 v25; // rax
  __int64 v26; // rax
  int v28; // edi
  __int64 v29; // r11
  unsigned int v30; // r10d
  __int64 *v31; // r9
  __int64 v32; // r12
  __int64 v33; // r12
  unsigned int v34; // r13d
  unsigned int v35; // r9d
  int *v36; // r10
  int v37; // r11d
  unsigned int v38; // eax
  unsigned __int64 v39; // rdx
  __int64 v40; // rax
  __int64 v41; // rax
  int v42; // edi
  __int64 v43; // r11
  unsigned int v44; // r10d
  __int64 *v45; // r9
  __int64 v46; // r12
  __int64 v47; // r12
  unsigned int v48; // r13d
  unsigned int v49; // r9d
  int *v50; // r10
  int v51; // r11d
  unsigned int v52; // eax
  unsigned __int64 v53; // rdx
  __int64 v54; // rax
  __int64 v55; // rax
  int v56; // ecx
  __int64 v57; // r10
  unsigned int v58; // r9d
  __int64 *v59; // rdi
  __int64 v60; // r11
  __int64 v61; // r12
  unsigned int v62; // r9d
  unsigned int v63; // eax
  int *v64; // rdx
  int v65; // edi
  unsigned int v66; // eax
  unsigned __int64 v67; // rdx
  __int64 v68; // rax
  __int64 v69; // rax
  int v70; // r10d
  unsigned int v71; // r13d
  int m; // r11d
  int v73; // esi
  int v74; // r9d
  int v75; // r9d
  int v76; // edi
  int v77; // edx
  int *v78; // rax
  int v79; // ecx
  int v80; // ecx
  int v81; // edx
  int *v82; // rax
  int v83; // ecx
  int v84; // ecx
  int v85; // edx
  int *v86; // rax
  int v87; // ecx
  int v88; // ecx
  int v89; // r11d
  int v90; // r11d
  int *v91; // r10
  int v92; // eax
  int v93; // edx
  int v94; // r13d
  int *v95; // rcx
  unsigned int v96; // eax
  int *v97; // rcx
  unsigned int v98; // eax
  int *v99; // rcx
  unsigned int v100; // eax
  int *v101; // r13
  unsigned int v102; // eax
  unsigned int v103; // [rsp+14h] [rbp-6Ch]
  unsigned int v104; // [rsp+14h] [rbp-6Ch]
  unsigned int v105; // [rsp+14h] [rbp-6Ch]
  int i; // [rsp+18h] [rbp-68h]
  int j; // [rsp+18h] [rbp-68h]
  int k; // [rsp+18h] [rbp-68h]
  int v109; // [rsp+1Ch] [rbp-64h]
  int v110; // [rsp+1Ch] [rbp-64h]
  int v111; // [rsp+1Ch] [rbp-64h]
  int v112; // [rsp+1Ch] [rbp-64h]
  int v113; // [rsp+1Ch] [rbp-64h]
  __int64 *v115; // [rsp+28h] [rbp-58h]
  int *v116; // [rsp+30h] [rbp-50h] BYREF
  __int64 v117; // [rsp+38h] [rbp-48h]
  int v118; // [rsp+44h] [rbp-3Ch] BYREF
  _QWORD v119[7]; // [rsp+48h] [rbp-38h] BYREF

  v116 = a3;
  v4 = (a2 - (__int64)a1) >> 5;
  v5 = (a2 - (__int64)a1) >> 3;
  v117 = a4;
  if ( v4 > 0 )
  {
    v6 = a1;
    v115 = &a1[4 * v4];
    while ( 1 )
    {
      v7 = v117;
      v8 = v116;
      v9 = *(unsigned int *)(v117 + 136);
      v10 = *v116;
      v11 = *(_QWORD *)(v117 + 120);
      if ( (_DWORD)v9 )
      {
        v12 = (v9 - 1) & (((unsigned int)*v6 >> 9) ^ ((unsigned int)*v6 >> 4));
        v13 = (__int64 *)(v11 + 16LL * v12);
        v14 = *v13;
        if ( *v6 == *v13 )
          goto LABEL_5;
        v73 = 1;
        while ( v14 != -8 )
        {
          v89 = v73 + 1;
          v12 = (v9 - 1) & (v73 + v12);
          v13 = (__int64 *)(v11 + 16LL * v12);
          v14 = *v13;
          if ( *v6 == *v13 )
            goto LABEL_5;
          v73 = v89;
        }
      }
      v13 = (__int64 *)(v11 + 16LL * (unsigned int)v9);
LABEL_5:
      v15 = v13[1];
      v118 = *v116;
      v16 = v117 + 56;
      v17 = *(_DWORD *)(v117 + 80);
      v18 = *(_QWORD *)(v117 + 64);
      if ( !v17 )
        goto LABEL_11;
      v19 = v17 - 1;
      v20 = (v17 - 1) & (37 * v10);
      v21 = (int *)(v18 + 8LL * v20);
      v22 = *v21;
      if ( v10 != *v21 )
      {
        v109 = *v21;
        v103 = (v17 - 1) & (37 * v10);
        for ( i = 1; ; ++i )
        {
          if ( v109 == -1 )
            goto LABEL_11;
          v103 = v19 & (i + v103);
          v109 = *(_DWORD *)(v18 + 8LL * v103);
          if ( v10 == v109 )
            break;
        }
        v77 = 1;
        v78 = 0;
        while ( v22 != -1 )
        {
          if ( v78 || v22 != -2 )
            v21 = v78;
          v20 = v19 & (v77 + v20);
          v95 = (int *)(v18 + 8LL * v20);
          v22 = *v95;
          if ( v10 == *v95 )
          {
            v96 = v95[1];
            v24 = v96 & 0x3F;
            v25 = 8LL * (v96 >> 6);
            goto LABEL_8;
          }
          ++v77;
          v78 = v21;
          v21 = (int *)(v18 + 8LL * v20);
        }
        v79 = *(_DWORD *)(v117 + 72);
        if ( !v78 )
          v78 = v21;
        ++*(_QWORD *)(v117 + 56);
        v80 = v79 + 1;
        if ( 4 * v80 >= 3 * v17 )
        {
          v17 *= 2;
        }
        else if ( v17 - *(_DWORD *)(v7 + 76) - v80 > v17 >> 3 )
        {
LABEL_63:
          *(_DWORD *)(v7 + 72) = v80;
          if ( *v78 != -1 )
            --*(_DWORD *)(v7 + 76);
          *v78 = v10;
          v24 = 0;
          v78[1] = 0;
          v25 = 0;
          goto LABEL_8;
        }
        sub_1BFDD60(v7 + 56, v17);
        sub_1BFD720(v7 + 56, &v118, v119);
        v78 = (int *)v119[0];
        v10 = v118;
        v80 = *(_DWORD *)(v7 + 72) + 1;
        goto LABEL_63;
      }
      v23 = v21[1];
      v24 = v23 & 0x3F;
      v25 = 8LL * (v23 >> 6);
LABEL_8:
      v26 = *(_QWORD *)(*(_QWORD *)(v15 + 24) + v25);
      if ( _bittest64(&v26, v24) )
        return v6;
      v7 = v117;
      v8 = v116;
      v11 = *(_QWORD *)(v117 + 120);
      v9 = *(unsigned int *)(v117 + 136);
      v16 = v117 + 56;
      v18 = *(_QWORD *)(v117 + 64);
      v17 = *(_DWORD *)(v117 + 80);
LABEL_11:
      v28 = *v8;
      if ( (_DWORD)v9 )
      {
        v29 = v6[1];
        v30 = (v9 - 1) & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
        v31 = (__int64 *)(v11 + 16LL * v30);
        v32 = *v31;
        if ( v29 == *v31 )
          goto LABEL_13;
        v74 = 1;
        while ( v32 != -8 )
        {
          v30 = (v9 - 1) & (v74 + v30);
          v112 = v74 + 1;
          v31 = (__int64 *)(v11 + 16LL * v30);
          v32 = *v31;
          if ( v29 == *v31 )
            goto LABEL_13;
          v74 = v112;
        }
      }
      v31 = (__int64 *)(v11 + 16LL * (unsigned int)v9);
LABEL_13:
      v33 = v31[1];
      v118 = *v8;
      if ( !v17 )
        goto LABEL_18;
      v34 = v17 - 1;
      v35 = (v17 - 1) & (37 * v28);
      v36 = (int *)(v18 + 8LL * v35);
      v37 = *v36;
      if ( *v36 != v28 )
      {
        v110 = *v36;
        v104 = (v17 - 1) & (37 * v28);
        for ( j = 1; ; ++j )
        {
          if ( v110 == -1 )
            goto LABEL_18;
          v104 = v34 & (v104 + j);
          v110 = *(_DWORD *)(v18 + 8LL * v104);
          if ( v28 == v110 )
            break;
        }
        v81 = 1;
        v82 = 0;
        while ( v37 != -1 )
        {
          if ( v82 || v37 != -2 )
            v36 = v82;
          v35 = v34 & (v81 + v35);
          v99 = (int *)(v18 + 8LL * v35);
          v37 = *v99;
          if ( v28 == *v99 )
          {
            v100 = v99[1];
            v39 = v100 & 0x3F;
            v40 = 8LL * (v100 >> 6);
            goto LABEL_16;
          }
          ++v81;
          v82 = v36;
          v36 = (int *)(v18 + 8LL * v35);
        }
        v83 = *(_DWORD *)(v7 + 72);
        if ( !v82 )
          v82 = v36;
        ++*(_QWORD *)(v7 + 56);
        v84 = v83 + 1;
        if ( 4 * v84 >= 3 * v17 )
        {
          v17 *= 2;
        }
        else if ( v17 - *(_DWORD *)(v7 + 76) - v84 > v17 >> 3 )
        {
LABEL_75:
          *(_DWORD *)(v7 + 72) = v84;
          if ( *v82 != -1 )
            --*(_DWORD *)(v7 + 76);
          *v82 = v28;
          v39 = 0;
          v82[1] = 0;
          v40 = 0;
          goto LABEL_16;
        }
        sub_1BFDD60(v16, v17);
        sub_1BFD720(v16, &v118, v119);
        v82 = (int *)v119[0];
        v28 = v118;
        v84 = *(_DWORD *)(v7 + 72) + 1;
        goto LABEL_75;
      }
      v38 = v36[1];
      v39 = v38 & 0x3F;
      v40 = 8LL * (v38 >> 6);
LABEL_16:
      v41 = *(_QWORD *)(*(_QWORD *)(v33 + 24) + v40);
      if ( _bittest64(&v41, v39) )
        return ++v6;
      v7 = v117;
      v8 = v116;
      v11 = *(_QWORD *)(v117 + 120);
      v9 = *(unsigned int *)(v117 + 136);
      v16 = v117 + 56;
      v18 = *(_QWORD *)(v117 + 64);
      v17 = *(_DWORD *)(v117 + 80);
LABEL_18:
      v42 = *v8;
      if ( (_DWORD)v9 )
      {
        v43 = v6[2];
        v44 = (v9 - 1) & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4));
        v45 = (__int64 *)(v11 + 16LL * v44);
        v46 = *v45;
        if ( v43 == *v45 )
          goto LABEL_20;
        v75 = 1;
        while ( v46 != -8 )
        {
          v44 = (v9 - 1) & (v75 + v44);
          v113 = v75 + 1;
          v45 = (__int64 *)(v11 + 16LL * v44);
          v46 = *v45;
          if ( v43 == *v45 )
            goto LABEL_20;
          v75 = v113;
        }
      }
      v45 = (__int64 *)(v11 + 16LL * (unsigned int)v9);
LABEL_20:
      v47 = v45[1];
      v118 = *v8;
      if ( !v17 )
        goto LABEL_25;
      v48 = v17 - 1;
      v49 = (v17 - 1) & (37 * v42);
      v50 = (int *)(v18 + 8LL * v49);
      v51 = *v50;
      if ( *v50 != v42 )
      {
        v111 = *v50;
        v105 = (v17 - 1) & (37 * v42);
        for ( k = 1; ; ++k )
        {
          if ( v111 == -1 )
            goto LABEL_25;
          v105 = v48 & (v105 + k);
          v111 = *(_DWORD *)(v18 + 8LL * v105);
          if ( v42 == v111 )
            break;
        }
        v85 = 1;
        v86 = 0;
        while ( v51 != -1 )
        {
          if ( v86 || v51 != -2 )
            v50 = v86;
          v49 = v48 & (v85 + v49);
          v97 = (int *)(v18 + 8LL * v49);
          v51 = *v97;
          if ( v42 == *v97 )
          {
            v98 = v97[1];
            v53 = v98 & 0x3F;
            v54 = 8LL * (v98 >> 6);
            goto LABEL_23;
          }
          ++v85;
          v86 = v50;
          v50 = (int *)(v18 + 8LL * v49);
        }
        v87 = *(_DWORD *)(v7 + 72);
        if ( !v86 )
          v86 = v50;
        ++*(_QWORD *)(v7 + 56);
        v88 = v87 + 1;
        if ( 4 * v88 >= 3 * v17 )
        {
          v17 *= 2;
        }
        else if ( v17 - *(_DWORD *)(v7 + 76) - v88 > v17 >> 3 )
        {
LABEL_87:
          *(_DWORD *)(v7 + 72) = v88;
          if ( *v86 != -1 )
            --*(_DWORD *)(v7 + 76);
          *v86 = v42;
          v53 = 0;
          v86[1] = 0;
          v54 = 0;
          goto LABEL_23;
        }
        sub_1BFDD60(v16, v17);
        sub_1BFD720(v16, &v118, v119);
        v86 = (int *)v119[0];
        v42 = v118;
        v88 = *(_DWORD *)(v7 + 72) + 1;
        goto LABEL_87;
      }
      v52 = v50[1];
      v53 = v52 & 0x3F;
      v54 = 8LL * (v52 >> 6);
LABEL_23:
      v55 = *(_QWORD *)(*(_QWORD *)(v47 + 24) + v54);
      if ( _bittest64(&v55, v53) )
      {
        v6 += 2;
        return v6;
      }
      v7 = v117;
      v8 = v116;
      v11 = *(_QWORD *)(v117 + 120);
      v9 = *(unsigned int *)(v117 + 136);
      v16 = v117 + 56;
      v18 = *(_QWORD *)(v117 + 64);
      v17 = *(_DWORD *)(v117 + 80);
LABEL_25:
      v56 = *v8;
      if ( (_DWORD)v9 )
      {
        v57 = v6[3];
        v58 = (v9 - 1) & (((unsigned int)v57 >> 9) ^ ((unsigned int)v57 >> 4));
        v59 = (__int64 *)(v11 + 16LL * v58);
        v60 = *v59;
        if ( v57 == *v59 )
          goto LABEL_27;
        v76 = 1;
        while ( v60 != -8 )
        {
          v94 = v76 + 1;
          v58 = (v9 - 1) & (v76 + v58);
          v59 = (__int64 *)(v11 + 16LL * v58);
          v60 = *v59;
          if ( v57 == *v59 )
            goto LABEL_27;
          v76 = v94;
        }
      }
      v59 = (__int64 *)(v11 + 16 * v9);
LABEL_27:
      v61 = v59[1];
      v118 = v56;
      if ( v17 )
      {
        v62 = v17 - 1;
        v63 = (v17 - 1) & (37 * v56);
        v64 = (int *)(v18 + 8LL * v63);
        v65 = *v64;
        if ( v56 != *v64 )
        {
          v70 = *v64;
          v71 = (v17 - 1) & (37 * v56);
          for ( m = 1; ; ++m )
          {
            if ( v70 == -1 )
              goto LABEL_34;
            v71 = v62 & (m + v71);
            v70 = *(_DWORD *)(v18 + 8LL * v71);
            if ( v56 == v70 )
              break;
          }
          v90 = 1;
          v91 = 0;
          while ( v65 != -1 )
          {
            if ( v91 || v65 != -2 )
              v64 = v91;
            v63 = v62 & (v90 + v63);
            v101 = (int *)(v18 + 8LL * v63);
            v65 = *v101;
            if ( v56 == *v101 )
            {
              v102 = v101[1];
              v67 = v102 & 0x3F;
              v68 = 8LL * (v102 >> 6);
              goto LABEL_30;
            }
            ++v90;
            v91 = v64;
            v64 = (int *)(v18 + 8LL * v63);
          }
          v92 = *(_DWORD *)(v7 + 72);
          if ( !v91 )
            v91 = v64;
          ++*(_QWORD *)(v7 + 56);
          v93 = v92 + 1;
          if ( 4 * (v92 + 1) >= 3 * v17 )
          {
            v17 *= 2;
          }
          else if ( v17 - *(_DWORD *)(v7 + 76) - v93 > v17 >> 3 )
          {
LABEL_103:
            *(_DWORD *)(v7 + 72) = v93;
            if ( *v91 != -1 )
              --*(_DWORD *)(v7 + 76);
            *v91 = v56;
            v67 = 0;
            v68 = 0;
            v91[1] = 0;
            goto LABEL_30;
          }
          sub_1BFDD60(v16, v17);
          sub_1BFD720(v16, &v118, v119);
          v91 = (int *)v119[0];
          v56 = v118;
          v93 = *(_DWORD *)(v7 + 72) + 1;
          goto LABEL_103;
        }
        v66 = v64[1];
        v67 = v66 & 0x3F;
        v68 = 8LL * (v66 >> 6);
LABEL_30:
        v69 = *(_QWORD *)(*(_QWORD *)(v61 + 24) + v68);
        if ( _bittest64(&v69, v67) )
        {
          v6 += 3;
          return v6;
        }
      }
LABEL_34:
      v6 += 4;
      if ( v6 == v115 )
      {
        v5 = (a2 - (__int64)v6) >> 3;
        goto LABEL_36;
      }
    }
  }
  v6 = a1;
LABEL_36:
  if ( v5 != 2 )
  {
    if ( v5 != 3 )
    {
      if ( v5 != 1 )
        return (__int64 *)a2;
      goto LABEL_120;
    }
    if ( (unsigned __int8)sub_21EBEB0((__int64)&v116, *v6) )
      return v6;
    ++v6;
  }
  if ( (unsigned __int8)sub_21EBEB0((__int64)&v116, *v6) )
    return v6;
  ++v6;
LABEL_120:
  if ( !(unsigned __int8)sub_21EBEB0((__int64)&v116, *v6) )
    return (__int64 *)a2;
  return v6;
}
