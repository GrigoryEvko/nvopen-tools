// Function: sub_1BE9A60
// Address: 0x1be9a60
//
__int64 *__fastcall sub_1BE9A60(__int64 *a1, __int64 a2, unsigned int a3)
{
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // r13
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 *v9; // rax
  __int64 *result; // rax
  __int64 *v11; // r13
  __int64 *v12; // rbx
  __int64 v13; // rax
  __int64 *v14; // r13
  __int64 v15; // r9
  unsigned int v16; // edx
  unsigned int v17; // r8d
  __int64 *v18; // rcx
  __int64 v19; // rdi
  unsigned int v20; // eax
  int v21; // eax
  __int64 v22; // rsi
  int v23; // eax
  __int64 v24; // r8
  int v25; // r9d
  unsigned int v26; // edi
  __int64 v27; // rcx
  __int64 v28; // rax
  __int64 v29; // rdi
  unsigned int v30; // edx
  __int64 *v31; // rcx
  __int64 v32; // r8
  __int64 v33; // rax
  __int64 v34; // rax
  unsigned int v35; // esi
  __int64 v36; // r12
  int v37; // esi
  int v38; // esi
  __int64 v39; // r8
  __int64 v40; // rdx
  int v41; // ecx
  __int64 *v42; // rax
  __int64 v43; // rdi
  __int64 v44; // r12
  __int64 v45; // r14
  __int64 *v46; // rax
  __int64 v47; // rbx
  __int64 v48; // r9
  unsigned int v49; // r10d
  unsigned int v50; // esi
  unsigned int v51; // r8d
  unsigned int v52; // edi
  __int64 *v53; // rdx
  __int64 v54; // rcx
  int v55; // esi
  int v56; // esi
  __int64 v57; // r8
  unsigned int v58; // edx
  int v59; // ecx
  __int64 *v60; // rax
  __int64 v61; // rdi
  int v62; // edi
  int v63; // esi
  int v64; // esi
  __int64 v65; // r8
  __int64 *v66; // r10
  int v67; // r11d
  __int64 v68; // rdx
  __int64 v69; // rdi
  int v70; // ecx
  int v71; // r10d
  int v72; // r11d
  int v73; // edi
  int v74; // esi
  int v75; // esi
  __int64 v76; // r8
  __int64 *v77; // r10
  int v78; // r11d
  __int64 v79; // rdx
  __int64 v80; // rdi
  __int64 *v81; // rdx
  __int64 *v82; // rax
  int v83; // edi
  int v84; // edx
  int v85; // eax
  int v86; // edx
  __int64 v87; // rsi
  __int64 *v88; // rdi
  int v89; // r9d
  unsigned int v90; // r11d
  __int64 v91; // rcx
  int v92; // eax
  int v93; // edx
  __int64 v94; // rsi
  unsigned int v95; // r11d
  __int64 v96; // rcx
  int v97; // r9d
  int v98; // r10d
  __int64 *v99; // r9
  __int64 *v100; // r8
  int v101; // eax
  __int64 *v102; // r11
  int v103; // r11d
  unsigned int v104; // [rsp+8h] [rbp-88h]
  __int64 *v105; // [rsp+8h] [rbp-88h]
  unsigned int v106; // [rsp+10h] [rbp-80h]
  int v108; // [rsp+18h] [rbp-78h]
  __int64 v109; // [rsp+20h] [rbp-70h]
  unsigned int v110; // [rsp+20h] [rbp-70h]
  __int64 v111; // [rsp+30h] [rbp-60h]
  __int64 *v113; // [rsp+38h] [rbp-58h]
  __int64 *v114; // [rsp+38h] [rbp-58h]
  __int64 *v115; // [rsp+40h] [rbp-50h]
  __int64 v116; // [rsp+40h] [rbp-50h]
  unsigned int v117; // [rsp+40h] [rbp-50h]
  int v118; // [rsp+40h] [rbp-50h]
  unsigned int v119; // [rsp+40h] [rbp-50h]
  unsigned int v120; // [rsp+40h] [rbp-50h]
  unsigned int v121; // [rsp+48h] [rbp-48h]
  __int64 *v122; // [rsp+48h] [rbp-48h]
  __int64 v123[7]; // [rsp+58h] [rbp-38h] BYREF

  v4 = *a1;
  v5 = (a1[1] - *a1) >> 3;
  v106 = v5;
  if ( (unsigned int)v5 > 1 )
  {
    v6 = 8;
    v7 = 8LL * (unsigned int)v5;
    while ( 1 )
    {
      v8 = *(_QWORD *)(v4 + v6);
      v6 += 8;
      v123[0] = v8;
      v9 = sub_1BE8E40((__int64)(a1 + 3), v123);
      v9[4] = *(_QWORD *)(*a1 + 8LL * *((unsigned int *)v9 + 3));
      if ( v7 == v6 )
        break;
      v4 = *a1;
    }
  }
  result = (__int64 *)(v106 - 1);
  if ( (unsigned int)result <= 1 )
    return result;
  v111 = 8LL * (_QWORD)result;
  v109 = (__int64)(a1 + 3);
  v121 = v106;
  do
  {
    v123[0] = *(_QWORD *)(*a1 + v111);
    v11 = sub_1BE8E40(v109, v123);
    v12 = (__int64 *)v11[5];
    *((_DWORD *)v11 + 4) = *((_DWORD *)v11 + 3);
    v13 = *((unsigned int *)v11 + 12);
    if ( v12 != &v12[v13] )
    {
      v115 = v11;
      v14 = &v12[v13];
      while ( 1 )
      {
        v21 = *((_DWORD *)a1 + 12);
        if ( !v21 )
          goto LABEL_13;
        v22 = *v12;
        v23 = v21 - 1;
        v24 = a1[4];
        v25 = 1;
        v26 = v23 & (((unsigned int)*v12 >> 4) ^ ((unsigned int)*v12 >> 9));
        v27 = *(_QWORD *)(v24 + 72LL * v26);
        if ( *v12 == v27 )
        {
LABEL_16:
          v28 = *(unsigned int *)(a2 + 48);
          if ( !(_DWORD)v28 )
            goto LABEL_21;
          v29 = *(_QWORD *)(a2 + 32);
          v30 = (v28 - 1) & (((unsigned int)*v12 >> 4) ^ ((unsigned int)*v12 >> 9));
          v31 = (__int64 *)(v29 + 16LL * v30);
          v32 = *v31;
          if ( v22 != *v31 )
          {
            v70 = 1;
            while ( v32 != -8 )
            {
              v71 = v70 + 1;
              v30 = (v28 - 1) & (v70 + v30);
              v31 = (__int64 *)(v29 + 16LL * v30);
              v32 = *v31;
              if ( v22 == *v31 )
                goto LABEL_18;
              v70 = v71;
            }
LABEL_21:
            v34 = sub_1BE90B0(a1, v22, v121);
            v35 = *((_DWORD *)a1 + 12);
            v36 = v34;
            if ( !v35 )
            {
              ++a1[3];
              goto LABEL_23;
            }
            v15 = a1[4];
            v16 = ((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4);
            v17 = (v35 - 1) & v16;
            v18 = (__int64 *)(v15 + 72LL * v17);
            v19 = *v18;
            if ( v34 == *v18 )
            {
              v20 = *((_DWORD *)v18 + 4);
LABEL_11:
              if ( *((_DWORD *)v115 + 4) > v20 )
                *((_DWORD *)v115 + 4) = v20;
              goto LABEL_13;
            }
            v72 = 1;
            v42 = 0;
            while ( v19 != -8 )
            {
              if ( v19 != -16 || v42 )
                v18 = v42;
              v101 = v72 + 1;
              v17 = (v35 - 1) & (v72 + v17);
              v102 = (__int64 *)(v15 + 72LL * v17);
              v19 = *v102;
              if ( v36 == *v102 )
              {
                v20 = *((_DWORD *)v102 + 4);
                goto LABEL_11;
              }
              v72 = v101;
              v42 = v18;
              v18 = (__int64 *)(v15 + 72LL * v17);
            }
            v73 = *((_DWORD *)a1 + 10);
            if ( !v42 )
              v42 = v18;
            ++a1[3];
            v41 = v73 + 1;
            if ( 4 * (v73 + 1) >= 3 * v35 )
            {
LABEL_23:
              sub_1BE8C00(v109, 2 * v35);
              v37 = *((_DWORD *)a1 + 12);
              if ( !v37 )
                goto LABEL_153;
              v38 = v37 - 1;
              v39 = a1[4];
              LODWORD(v40) = v38 & (((unsigned int)v36 >> 9) ^ ((unsigned int)v36 >> 4));
              v41 = *((_DWORD *)a1 + 10) + 1;
              v42 = (__int64 *)(v39 + 72LL * (unsigned int)v40);
              v43 = *v42;
              if ( v36 != *v42 )
              {
                v103 = 1;
                v77 = 0;
                while ( v43 != -8 )
                {
                  if ( !v77 && v43 == -16 )
                    v77 = v42;
                  v40 = v38 & (unsigned int)(v40 + v103);
                  v42 = (__int64 *)(v39 + 72 * v40);
                  v43 = *v42;
                  if ( v36 == *v42 )
                    goto LABEL_25;
                  ++v103;
                }
                goto LABEL_73;
              }
            }
            else if ( v35 - *((_DWORD *)a1 + 11) - v41 <= v35 >> 3 )
            {
              v104 = v16;
              sub_1BE8C00(v109, v35);
              v74 = *((_DWORD *)a1 + 12);
              if ( !v74 )
                goto LABEL_153;
              v75 = v74 - 1;
              v76 = a1[4];
              v77 = 0;
              v78 = 1;
              LODWORD(v79) = v75 & v104;
              v41 = *((_DWORD *)a1 + 10) + 1;
              v42 = (__int64 *)(v76 + 72LL * (v75 & v104));
              v80 = *v42;
              if ( v36 != *v42 )
              {
                while ( v80 != -8 )
                {
                  if ( !v77 && v80 == -16 )
                    v77 = v42;
                  v79 = v75 & (unsigned int)(v79 + v78);
                  v42 = (__int64 *)(v76 + 72 * v79);
                  v80 = *v42;
                  if ( v36 == *v42 )
                    goto LABEL_25;
                  ++v78;
                }
LABEL_73:
                if ( v77 )
                  v42 = v77;
              }
            }
LABEL_25:
            *((_DWORD *)a1 + 10) = v41;
            if ( *v42 != -8 )
              --*((_DWORD *)a1 + 11);
            *v42 = v36;
            v42[5] = (__int64)(v42 + 7);
            v42[6] = 0x200000000LL;
            *(_OWORD *)(v42 + 1) = 0;
            *(_OWORD *)(v42 + 3) = 0;
            *(_OWORD *)(v42 + 7) = 0;
            v20 = 0;
            goto LABEL_11;
          }
LABEL_18:
          if ( v31 == (__int64 *)(v29 + 16 * v28) )
            goto LABEL_21;
          v33 = v31[1];
          if ( !v33 || a3 <= *(_DWORD *)(v33 + 16) )
            goto LABEL_21;
LABEL_13:
          if ( v14 == ++v12 )
            break;
        }
        else
        {
          while ( v27 != -8 )
          {
            v26 = v23 & (v25 + v26);
            v27 = *(_QWORD *)(v24 + 72LL * v26);
            if ( v22 == v27 )
              goto LABEL_16;
            ++v25;
          }
          if ( v14 == ++v12 )
            break;
        }
      }
    }
    result = (__int64 *)--v121;
    v111 -= 8;
  }
  while ( v121 != 2 );
  if ( v106 > 2 )
  {
    v44 = v109;
    v45 = 16;
    do
    {
      v123[0] = *(_QWORD *)(*a1 + v45);
      v122 = sub_1BE8E40(v44, v123);
      v46 = sub_1BE8E40(v44, (__int64 *)(*a1 + 8LL * *((unsigned int *)v122 + 4)));
      v47 = v122[4];
      v48 = a1[4];
      v49 = *((_DWORD *)v46 + 2);
      v50 = *((_DWORD *)a1 + 12);
LABEL_36:
      if ( v50 )
      {
        while ( 1 )
        {
          v51 = v50 - 1;
          v52 = (v50 - 1) & (((unsigned int)v47 >> 9) ^ ((unsigned int)v47 >> 4));
          v53 = (__int64 *)(v48 + 72LL * v52);
          v54 = *v53;
          if ( *v53 != v47 )
            break;
          if ( v49 >= *((_DWORD *)v53 + 2) )
            goto LABEL_46;
          v47 = v53[4];
        }
        v116 = *v53;
        v60 = 0;
        v113 = (__int64 *)(v48 + 72LL * (v51 & (((unsigned int)v47 >> 9) ^ ((unsigned int)v47 >> 4))));
        v110 = (v50 - 1) & (((unsigned int)v47 >> 9) ^ ((unsigned int)v47 >> 4));
        v108 = 1;
        while ( v116 != -8 )
        {
          if ( v116 != -16 || v60 )
            v113 = v60;
          v110 = v51 & (v110 + v108);
          v105 = (__int64 *)(v48 + 72LL * v110);
          v116 = *v105;
          if ( *v105 == v47 )
          {
            v81 = (__int64 *)(v48 + 72LL * (v51 & (((unsigned int)v47 >> 9) ^ ((unsigned int)v47 >> 4))));
            if ( v49 >= *((_DWORD *)v105 + 2) )
              goto LABEL_46;
            v118 = 1;
            v82 = 0;
            while ( 2 )
            {
              if ( v54 == -8 )
              {
                v83 = *((_DWORD *)a1 + 10);
                if ( !v82 )
                  v82 = v81;
                ++a1[3];
                v84 = v83 + 1;
                if ( 4 * (v83 + 1) < 3 * v50 )
                {
                  if ( v50 - (v84 + *((_DWORD *)a1 + 11)) > v50 >> 3 )
                    goto LABEL_86;
                  v120 = v49;
                  sub_1BE8C00(v44, v50);
                  v92 = *((_DWORD *)a1 + 12);
                  if ( v92 )
                  {
                    v93 = v92 - 1;
                    v94 = a1[4];
                    v49 = v120;
                    v95 = (v92 - 1) & (((unsigned int)v47 >> 9) ^ ((unsigned int)v47 >> 4));
                    v82 = (__int64 *)(v94 + 72LL * v95);
                    v96 = *v82;
                    if ( *v82 != v47 )
                    {
                      v97 = 1;
                      v88 = 0;
                      while ( v96 != -8 )
                      {
                        if ( v88 || v96 != -16 )
                          v82 = v88;
                        v95 = v93 & (v97 + v95);
                        v100 = (__int64 *)(v94 + 72LL * v95);
                        v96 = *v100;
                        if ( *v100 == v47 )
                          goto LABEL_129;
                        ++v97;
                        v88 = v82;
                        v82 = (__int64 *)(v94 + 72LL * v95);
                      }
                      goto LABEL_96;
                    }
                    goto LABEL_91;
                  }
LABEL_153:
                  ++*((_DWORD *)a1 + 10);
                  BUG();
                }
                v119 = v49;
                sub_1BE8C00(v44, 2 * v50);
                v85 = *((_DWORD *)a1 + 12);
                if ( !v85 )
                  goto LABEL_153;
                v86 = v85 - 1;
                v87 = a1[4];
                v88 = 0;
                v49 = v119;
                v89 = 1;
                v90 = (v85 - 1) & (((unsigned int)v47 >> 9) ^ ((unsigned int)v47 >> 4));
                v82 = (__int64 *)(v87 + 72LL * v90);
                v91 = *v82;
                if ( *v82 == v47 )
                  goto LABEL_91;
                while ( v91 != -8 )
                {
                  if ( v88 || v91 != -16 )
                    v82 = v88;
                  v90 = v86 & (v89 + v90);
                  v100 = (__int64 *)(v87 + 72LL * v90);
                  v91 = *v100;
                  if ( *v100 == v47 )
                  {
LABEL_129:
                    v84 = *((_DWORD *)a1 + 10) + 1;
                    v82 = v100;
                    goto LABEL_86;
                  }
                  ++v89;
                  v88 = v82;
                  v82 = (__int64 *)(v87 + 72LL * v90);
                }
LABEL_96:
                if ( !v88 )
                {
LABEL_91:
                  v84 = *((_DWORD *)a1 + 10) + 1;
                }
                else
                {
                  v84 = *((_DWORD *)a1 + 10) + 1;
                  v82 = v88;
                }
LABEL_86:
                *((_DWORD *)a1 + 10) = v84;
                if ( *v82 != -8 )
                  --*((_DWORD *)a1 + 11);
                *v82 = v47;
                v47 = 0;
                v82[5] = (__int64)(v82 + 7);
                v82[6] = 0x200000000LL;
                *(_OWORD *)(v82 + 1) = 0;
                *(_OWORD *)(v82 + 3) = 0;
                *(_OWORD *)(v82 + 7) = 0;
                v48 = a1[4];
                v50 = *((_DWORD *)a1 + 12);
              }
              else
              {
                if ( v82 || v54 != -16 )
                  v81 = v82;
                v52 = v51 & (v118 + v52);
                v114 = (__int64 *)(v48 + 72LL * v52);
                v54 = *v114;
                if ( *v114 != v47 )
                {
                  ++v118;
                  v82 = v81;
                  v81 = (__int64 *)(v48 + 72LL * v52);
                  continue;
                }
                v47 = v114[4];
              }
              goto LABEL_36;
            }
          }
          ++v108;
          v60 = v113;
          v113 = (__int64 *)(v48 + 72LL * v110);
        }
        v62 = *((_DWORD *)a1 + 10);
        if ( !v60 )
          v60 = v113;
        ++a1[3];
        v59 = v62 + 1;
        if ( 4 * (v62 + 1) >= 3 * v50 )
          goto LABEL_41;
        if ( v50 - (v59 + *((_DWORD *)a1 + 11)) <= v50 >> 3 )
        {
          v117 = ((unsigned int)v47 >> 9) ^ ((unsigned int)v47 >> 4);
          sub_1BE8C00(v44, v50);
          v63 = *((_DWORD *)a1 + 12);
          if ( !v63 )
            goto LABEL_153;
          v64 = v63 - 1;
          v65 = a1[4];
          v66 = 0;
          v67 = 1;
          LODWORD(v68) = v64 & v117;
          v59 = *((_DWORD *)a1 + 10) + 1;
          v60 = (__int64 *)(v65 + 72LL * (v64 & v117));
          v69 = *v60;
          if ( *v60 != v47 )
          {
            while ( v69 != -8 )
            {
              if ( !v66 && v69 == -16 )
                v66 = v60;
              v68 = v64 & (unsigned int)(v68 + v67);
              v60 = (__int64 *)(v65 + 72 * v68);
              v69 = *v60;
              if ( *v60 == v47 )
                goto LABEL_43;
              ++v67;
            }
            if ( v66 )
              v60 = v66;
          }
        }
      }
      else
      {
        ++a1[3];
LABEL_41:
        sub_1BE8C00(v44, 2 * v50);
        v55 = *((_DWORD *)a1 + 12);
        if ( !v55 )
          goto LABEL_153;
        v56 = v55 - 1;
        v57 = a1[4];
        v58 = v56 & (((unsigned int)v47 >> 9) ^ ((unsigned int)v47 >> 4));
        v59 = *((_DWORD *)a1 + 10) + 1;
        v60 = (__int64 *)(v57 + 72LL * v58);
        v61 = *v60;
        if ( *v60 != v47 )
        {
          v98 = 1;
          v99 = 0;
          while ( v61 != -8 )
          {
            if ( v61 != -16 || v99 )
              v60 = v99;
            v58 = v56 & (v98 + v58);
            v61 = *(_QWORD *)(v57 + 72LL * v58);
            if ( v61 == v47 )
            {
              v60 = (__int64 *)(v57 + 72LL * v58);
              goto LABEL_43;
            }
            ++v98;
            v99 = v60;
            v60 = (__int64 *)(v57 + 72LL * v58);
          }
          if ( v99 )
            v60 = v99;
        }
      }
LABEL_43:
      *((_DWORD *)a1 + 10) = v59;
      if ( *v60 != -8 )
        --*((_DWORD *)a1 + 11);
      *v60 = v47;
      v60[5] = (__int64)(v60 + 7);
      v60[6] = 0x200000000LL;
      *(_OWORD *)(v60 + 1) = 0;
      *(_OWORD *)(v60 + 3) = 0;
      *(_OWORD *)(v60 + 7) = 0;
LABEL_46:
      result = v122;
      v45 += 8;
      v122[4] = v47;
    }
    while ( 8LL * v106 != v45 );
  }
  return result;
}
