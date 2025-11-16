// Function: sub_1E1B900
// Address: 0x1e1b900
//
__int64 __fastcall sub_1E1B900(__int64 a1, int *a2, __int64 a3, __int64 a4)
{
  unsigned __int8 *v4; // r8
  __int64 result; // rax
  unsigned __int8 *v6; // r11
  char *v8; // r9
  unsigned int v9; // ecx
  __int64 v10; // r10
  int v11; // esi
  __int64 v12; // r12
  __int64 v13; // rdi
  unsigned int v14; // r13d
  __int16 v15; // dx
  _WORD *v16; // r14
  _WORD *v17; // r13
  unsigned __int16 v18; // dx
  _WORD *v19; // rdi
  int v20; // esi
  unsigned __int16 *v21; // r14
  unsigned int v22; // edi
  unsigned int i; // r12d
  bool v24; // cf
  int *v25; // r12
  int v26; // esi
  int v27; // r12d
  int v28; // esi
  __int64 v29; // r15
  __int64 v30; // r13
  __int64 v31; // r12
  unsigned int v32; // edi
  __int16 v33; // dx
  _WORD *v34; // r14
  __int16 *v35; // rdi
  unsigned __int16 v36; // dx
  unsigned int v37; // r14d
  _WORD *v38; // r12
  int v39; // esi
  unsigned __int16 *v40; // r13
  unsigned int m; // r12d
  bool v42; // cf
  __int16 *v43; // r14
  __int16 v44; // di
  int v45; // r12d
  int v46; // esi
  __int64 v47; // r13
  __int64 v48; // r12
  unsigned int v49; // edi
  __int16 v50; // dx
  _WORD *v51; // r14
  __int16 *v52; // rdi
  unsigned __int16 v53; // dx
  unsigned int v54; // r14d
  _WORD *v55; // r12
  int v56; // esi
  unsigned __int16 *v57; // r13
  unsigned int j; // r12d
  __int16 *v59; // r14
  __int16 v60; // di
  int v61; // r12d
  int v62; // esi
  __int64 v63; // r13
  __int64 v64; // r12
  unsigned int v65; // edi
  __int16 v66; // dx
  _WORD *v67; // r14
  __int16 *v68; // rdi
  unsigned __int16 v69; // dx
  unsigned int v70; // r14d
  _WORD *v71; // r12
  int v72; // esi
  unsigned __int16 *v73; // r13
  unsigned int k; // r12d
  __int16 *v75; // r14
  __int16 v76; // di
  int v77; // r12d
  __int64 v78; // rdx
  int v79; // edx
  __int64 v80; // r10
  __int64 v81; // rdi
  unsigned int v82; // r12d
  _WORD *v83; // r13
  unsigned __int16 v84; // dx
  _WORD *v85; // r12
  unsigned int v86; // esi
  _WORD *v87; // rdi
  int v88; // ecx
  unsigned __int16 *v89; // r10
  unsigned int n; // edi
  int v91; // edi
  int v92; // edi
  unsigned int v93; // esi
  unsigned __int16 *v94; // r13
  unsigned int v95; // edi
  unsigned __int16 v96; // dx
  int v97; // edx
  __int64 v98; // r10
  __int64 v99; // rdi
  unsigned int v100; // r12d
  _WORD *v101; // r13
  unsigned __int16 v102; // dx
  _WORD *v103; // r12
  _WORD *v104; // rdi
  unsigned int v105; // r10d
  int v106; // esi
  unsigned __int16 *v107; // r13
  unsigned int v108; // edi
  int v109; // edi
  int v110; // r13d
  __int64 v111; // rsi
  __int64 v112; // rdi
  unsigned int v113; // r10d
  __int16 v114; // dx
  _WORD *v115; // r10
  _WORD *v116; // r12
  _WORD *v117; // rdi
  unsigned int v118; // r10d
  __int64 v119; // [rsp+0h] [rbp-70h]
  __int64 v122; // [rsp+20h] [rbp-50h]
  int *v123; // [rsp+28h] [rbp-48h]
  char v125; // [rsp+3Fh] [rbp-31h]

  v4 = *(unsigned __int8 **)(a1 + 32);
  result = 5LL * *(unsigned int *)(a1 + 40);
  v6 = &v4[40 * *(unsigned int *)(a1 + 40)];
  if ( v6 == v4 )
    return result;
  v125 = 0;
  v123 = &a2[a3];
  v119 = (4 * a3) >> 2;
  v8 = (char *)a2 + ((4 * a3) & 0xFFFFFFFFFFFFFFF0LL);
  v122 = (4 * a3) >> 4;
  do
  {
    result = *v4;
    if ( (_BYTE)result == 12 )
    {
      v125 = 1;
    }
    else if ( !(_BYTE)result && (v4[3] & 0x10) != 0 )
    {
      v9 = *((_DWORD *)v4 + 2);
      if ( (int)v9 > 0 )
      {
        if ( v122 > 0 )
        {
          result = (__int64)a2;
          v10 = 24LL * v9;
          while ( 1 )
          {
            v11 = *(_DWORD *)result;
            if ( v9 == *(_DWORD *)result )
              goto LABEL_16;
            if ( v11 >= 0 )
            {
              v12 = *(_QWORD *)(a4 + 8);
              v13 = *(_QWORD *)(a4 + 56);
              v14 = *(_DWORD *)(v12 + 24LL * (unsigned int)v11 + 16);
              v15 = v11 * (v14 & 0xF);
              v16 = (_WORD *)(v13 + 2LL * (v14 >> 4));
              LODWORD(v12) = *(_DWORD *)(v12 + v10 + 16);
              v17 = v16 + 1;
              v18 = *v16 + v15;
              v20 = v9 * (v12 & 0xF);
              v19 = (_WORD *)(v13 + 2LL * ((unsigned int)v12 >> 4));
              LOWORD(v20) = *v19 + v9 * (v12 & 0xF);
              v21 = v19 + 1;
              v22 = v18;
              for ( i = (unsigned __int16)v20; ; i = (unsigned __int16)v20 )
              {
                v24 = v22 < i;
                if ( v22 == i )
                  break;
                while ( v24 )
                {
                  v18 += *v17;
                  if ( !*v17 )
                    goto LABEL_24;
                  v22 = v18;
                  ++v17;
                  v24 = v18 < i;
                  if ( v18 == i )
                    goto LABEL_16;
                }
                v27 = *v21;
                if ( !(_WORD)v27 )
                  goto LABEL_24;
                v20 += v27;
                ++v21;
              }
              goto LABEL_16;
            }
LABEL_24:
            v28 = *(_DWORD *)(result + 4);
            v29 = result + 4;
            if ( v9 == v28 )
              goto LABEL_31;
            if ( v28 >= 0 )
              break;
LABEL_35:
            v46 = *(_DWORD *)(result + 8);
            v29 = result + 8;
            if ( v9 == v46 )
              goto LABEL_31;
            if ( v46 >= 0 )
            {
              v47 = *(_QWORD *)(a4 + 8);
              v48 = *(_QWORD *)(a4 + 56);
              v49 = *(_DWORD *)(v47 + 24LL * (unsigned int)v46 + 16);
              v50 = v46 * (v49 & 0xF);
              v51 = (_WORD *)(v48 + 2LL * (v49 >> 4));
              LODWORD(v47) = *(_DWORD *)(v47 + v10 + 16);
              v52 = v51 + 1;
              v53 = *v51 + v50;
              v56 = v9 * (v47 & 0xF);
              v54 = v53;
              v55 = (_WORD *)(v48 + 2LL * ((unsigned int)v47 >> 4));
              LOWORD(v56) = *v55 + v9 * (v47 & 0xF);
              v57 = v55 + 1;
              for ( j = (unsigned __int16)v56; v54 != j; j = (unsigned __int16)v56 )
              {
                if ( v54 < j )
                {
                  do
                  {
                    v59 = v52 + 1;
                    v60 = *v52;
                    v53 += v60;
                    if ( !v60 )
                      goto LABEL_45;
                    v52 = v59;
                    v54 = v53;
                    if ( v53 == j )
                      goto LABEL_31;
                  }
                  while ( v53 < j );
                }
                v61 = *v57;
                if ( !(_WORD)v61 )
                  goto LABEL_45;
                v56 += v61;
                ++v57;
              }
              goto LABEL_31;
            }
LABEL_45:
            v62 = *(_DWORD *)(result + 12);
            v29 = result + 12;
            if ( v9 == v62 )
              goto LABEL_31;
            if ( v62 >= 0 )
            {
              v63 = *(_QWORD *)(a4 + 8);
              v64 = *(_QWORD *)(a4 + 56);
              v65 = *(_DWORD *)(v63 + 24LL * (unsigned int)v62 + 16);
              v66 = v62 * (v65 & 0xF);
              v67 = (_WORD *)(v64 + 2LL * (v65 >> 4));
              LODWORD(v63) = *(_DWORD *)(v63 + v10 + 16);
              v68 = v67 + 1;
              v69 = *v67 + v66;
              v72 = v9 * (v63 & 0xF);
              v70 = v69;
              v71 = (_WORD *)(v64 + 2LL * ((unsigned int)v63 >> 4));
              LOWORD(v72) = *v71 + v9 * (v63 & 0xF);
              v73 = v71 + 1;
              for ( k = (unsigned __int16)v72; v70 != k; k = (unsigned __int16)v72 )
              {
                if ( v70 < k )
                {
                  do
                  {
                    v75 = v68 + 1;
                    v76 = *v68;
                    v69 += v76;
                    if ( !v76 )
                      goto LABEL_55;
                    v68 = v75;
                    v70 = v69;
                    if ( v69 == k )
                      goto LABEL_31;
                  }
                  while ( v69 < k );
                }
                v77 = *v73;
                if ( !(_WORD)v77 )
                  goto LABEL_55;
                v72 += v77;
                ++v73;
              }
              goto LABEL_31;
            }
LABEL_55:
            result += 16;
            if ( (char *)result == v8 )
            {
              v78 = ((char *)v123 - v8) >> 2;
              goto LABEL_57;
            }
          }
          v30 = *(_QWORD *)(a4 + 8);
          v31 = *(_QWORD *)(a4 + 56);
          v32 = *(_DWORD *)(v30 + 24LL * (unsigned int)v28 + 16);
          v33 = v28 * (v32 & 0xF);
          v34 = (_WORD *)(v31 + 2LL * (v32 >> 4));
          LODWORD(v30) = *(_DWORD *)(v30 + v10 + 16);
          v35 = v34 + 1;
          v36 = *v34 + v33;
          v39 = v9 * (v30 & 0xF);
          v37 = v36;
          v38 = (_WORD *)(v31 + 2LL * ((unsigned int)v30 >> 4));
          LOWORD(v39) = *v38 + v9 * (v30 & 0xF);
          v40 = v38 + 1;
          for ( m = (unsigned __int16)v39; ; m = (unsigned __int16)v39 )
          {
            v42 = v37 < m;
            if ( v37 == m )
              break;
            while ( v42 )
            {
              v43 = v35 + 1;
              v44 = *v35;
              v36 += v44;
              if ( !v44 )
                goto LABEL_35;
              v35 = v43;
              v37 = v36;
              v42 = v36 < m;
              if ( v36 == m )
                goto LABEL_31;
            }
            v45 = *v40;
            if ( !(_WORD)v45 )
              goto LABEL_35;
            v39 += v45;
            ++v40;
          }
LABEL_31:
          result = v29;
          if ( v123 != (int *)v29 )
            goto LABEL_17;
          goto LABEL_32;
        }
        v78 = v119;
        result = (__int64)a2;
LABEL_57:
        if ( v78 == 2 )
          goto LABEL_77;
        if ( v78 != 3 )
        {
          if ( v78 == 1 )
            goto LABEL_60;
LABEL_32:
          v4[3] |= 0x40u;
          goto LABEL_17;
        }
        v110 = *(_DWORD *)result;
        if ( v9 != *(_DWORD *)result )
        {
          if ( v110 >= 0 )
          {
            v111 = *(_QWORD *)(a4 + 8);
            v112 = *(_QWORD *)(a4 + 56);
            v113 = *(_DWORD *)(v111 + 24LL * (unsigned int)v110 + 16);
            v114 = v110 * (v113 & 0xF);
            v115 = (_WORD *)(v112 + 2LL * (v113 >> 4));
            v96 = *v115 + v114;
            v116 = v115 + 1;
            LODWORD(v115) = *(_DWORD *)(v111 + 24LL * v9 + 16);
            v93 = v9 * ((unsigned __int8)v115 & 0xF);
            v117 = (_WORD *)(v112 + 2LL * ((unsigned int)v115 >> 4));
            v118 = v96;
            LOWORD(v93) = *v117 + v93;
            v94 = v117 + 1;
            v95 = (unsigned __int16)v93;
            while ( v118 != v95 )
            {
              if ( v118 >= v95 )
              {
                v92 = *v94;
                if ( !(_WORD)v92 )
                  goto LABEL_76;
                v93 += v92;
                ++v94;
                v95 = (unsigned __int16)v93;
              }
              else
              {
                v96 += *v116;
                if ( !*v116 )
                  goto LABEL_76;
                ++v116;
                v118 = v96;
              }
            }
            goto LABEL_16;
          }
LABEL_76:
          result += 4;
LABEL_77:
          v97 = *(_DWORD *)result;
          if ( v9 != *(_DWORD *)result )
          {
            if ( v97 < 0 )
            {
LABEL_83:
              result += 4;
LABEL_60:
              v79 = *(_DWORD *)result;
              if ( v9 != *(_DWORD *)result )
              {
                if ( v79 < 0 )
                  goto LABEL_32;
                v80 = *(_QWORD *)(a4 + 8);
                v81 = *(_QWORD *)(a4 + 56);
                v82 = *(_DWORD *)(v80 + 24LL * (unsigned int)v79 + 16);
                v83 = (_WORD *)(v81 + 2LL * (v82 >> 4));
                v84 = *v83 + (v82 & 0xF) * v79;
                v85 = v83 + 1;
                LODWORD(v80) = *(_DWORD *)(v80 + 24LL * v9 + 16);
                v88 = (v80 & 0xF) * v9;
                v86 = v84;
                v87 = (_WORD *)(v81 + 2LL * ((unsigned int)v80 >> 4));
                LOWORD(v88) = *v87 + v88;
                v89 = v87 + 1;
                for ( n = (unsigned __int16)v88; v86 != n; n = (unsigned __int16)v88 )
                {
                  if ( v86 < n )
                  {
                    do
                    {
                      v84 += *v85;
                      if ( !*v85 )
                        goto LABEL_32;
                      v86 = v84;
                      ++v85;
                      if ( v84 == n )
                        goto LABEL_16;
                    }
                    while ( v84 < n );
                  }
                  v91 = *v89;
                  if ( !(_WORD)v91 )
                    goto LABEL_32;
                  v88 += v91;
                  ++v89;
                }
              }
            }
            else
            {
              v98 = *(_QWORD *)(a4 + 8);
              v99 = *(_QWORD *)(a4 + 56);
              v100 = *(_DWORD *)(v98 + 24LL * (unsigned int)v97 + 16);
              v101 = (_WORD *)(v99 + 2LL * (v100 >> 4));
              v102 = *v101 + (v100 & 0xF) * v97;
              v103 = v101 + 1;
              LODWORD(v98) = *(_DWORD *)(v98 + 24LL * v9 + 16);
              v106 = v9 * (v98 & 0xF);
              v104 = (_WORD *)(v99 + 2LL * ((unsigned int)v98 >> 4));
              v105 = v102;
              LOWORD(v106) = *v104 + v106;
              v107 = v104 + 1;
              v108 = (unsigned __int16)v106;
              while ( v105 != v108 )
              {
                if ( v105 >= v108 )
                {
                  v109 = *v107;
                  if ( !(_WORD)v109 )
                    goto LABEL_83;
                  v106 += v109;
                  ++v107;
                  v108 = (unsigned __int16)v106;
                }
                else
                {
                  v102 += *v103;
                  if ( !*v103 )
                    goto LABEL_83;
                  ++v103;
                  v105 = v102;
                }
              }
            }
          }
        }
LABEL_16:
        if ( v123 != (int *)result )
          goto LABEL_17;
        goto LABEL_32;
      }
    }
LABEL_17:
    v4 += 40;
  }
  while ( v6 != v4 );
  v25 = a2;
  if ( v125 )
  {
    result = (__int64)a2;
    if ( v123 != a2 )
    {
      do
      {
        v26 = *v25++;
        result = sub_1E1B830(a1, v26, a4);
      }
      while ( v25 != v123 );
    }
  }
  return result;
}
