// Function: sub_2BA4E90
// Address: 0x2ba4e90
//
__int64 __fastcall sub_2BA4E90(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r11
  __int64 v4; // r10
  __int64 v5; // r13
  __int64 v6; // r14
  __int64 v7; // r12
  _QWORD *v8; // r8
  __int64 v9; // rax
  __int64 v10; // rbx
  _QWORD *v11; // rax
  __int64 v12; // r8
  __int64 v13; // rax
  __int64 v14; // rdi
  __int64 v15; // r13
  __int64 v16; // r15
  __int64 v17; // r12
  __int64 v18; // rax
  int v19; // esi
  __int64 v20; // r8
  int v21; // esi
  unsigned int v22; // ecx
  __int64 *v23; // rdx
  __int64 v24; // r11
  __int64 v25; // rax
  __int64 v26; // rsi
  __int64 v27; // rax
  int v28; // ecx
  int v29; // edx
  __int64 *v30; // rbx
  __int64 *v31; // r14
  __int64 v32; // rax
  __int64 v33; // r12
  __int64 v34; // rax
  int v35; // ecx
  int v36; // edx
  int v37; // esi
  __int64 v38; // r8
  __int64 v39; // rdi
  unsigned int v40; // edx
  __int64 *v41; // rax
  __int64 v42; // rcx
  __int64 v43; // r9
  int v44; // edi
  int v45; // edx
  __int64 v46; // rax
  __int64 v47; // rbx
  __int64 result; // rax
  __int64 v49; // r14
  __int64 v50; // r12
  int v51; // ecx
  int v52; // edx
  unsigned int v53; // esi
  __int64 v54; // r8
  __int64 v55; // rdi
  unsigned int v56; // ecx
  __int64 v57; // rdx
  _QWORD *v58; // r9
  int v59; // eax
  int v60; // eax
  __int64 v61; // rax
  _QWORD *v62; // rbx
  __int64 v63; // r14
  __int64 v64; // r15
  _BYTE *v65; // rax
  int v66; // esi
  __int64 v67; // rdi
  int v68; // esi
  unsigned int v69; // ecx
  _QWORD *v70; // rdx
  _BYTE *v71; // r8
  __int64 v72; // rax
  __int64 v73; // rsi
  __int64 v74; // rax
  int v75; // ecx
  int v76; // edx
  int v77; // edx
  int v78; // r9d
  int v79; // edx
  int v80; // r9d
  int v81; // r15d
  int v82; // r15d
  unsigned int v83; // edx
  __int64 v84; // rdi
  int v85; // esi
  _QWORD *v86; // rcx
  int v87; // eax
  __int64 v88; // rdi
  _QWORD *v89; // rdx
  unsigned int v90; // r15d
  int v91; // ecx
  __int64 v92; // rsi
  int v93; // eax
  int v94; // r15d
  unsigned int v95; // ecx
  int v96; // edi
  __int64 *v97; // rsi
  int v98; // eax
  __int64 *v99; // rcx
  unsigned int v100; // r15d
  int v101; // esi
  __int64 v102; // rdi
  const void *v103; // [rsp+0h] [rbp-60h]
  __int64 v104; // [rsp+10h] [rbp-50h]
  unsigned int v105; // [rsp+10h] [rbp-50h]
  __int64 v106; // [rsp+10h] [rbp-50h]
  __int64 v107; // [rsp+10h] [rbp-50h]
  __int64 v108; // [rsp+10h] [rbp-50h]
  __int64 v109; // [rsp+10h] [rbp-50h]
  __int64 v110; // [rsp+10h] [rbp-50h]
  __int64 v111; // [rsp+10h] [rbp-50h]
  int v112; // [rsp+18h] [rbp-48h]
  int v113; // [rsp+18h] [rbp-48h]
  __int64 v114; // [rsp+18h] [rbp-48h]
  __int64 v115; // [rsp+18h] [rbp-48h]
  __int64 v116; // [rsp+18h] [rbp-48h]
  __int64 v117; // [rsp+18h] [rbp-48h]
  __int64 v118; // [rsp+18h] [rbp-48h]
  __int64 v119; // [rsp+18h] [rbp-48h]
  __int64 v120; // [rsp+20h] [rbp-40h] BYREF
  __int64 v121[7]; // [rsp+28h] [rbp-38h] BYREF

  v3 = a1;
  v4 = a2;
  v5 = a3;
  *(_BYTE *)(a2 + 152) = 1;
  v103 = (const void *)(a3 + 48);
  do
  {
    v6 = *(_QWORD *)(v4 + 8);
    v7 = *(_QWORD *)v4;
    if ( v6 )
    {
      v8 = *(_QWORD **)v6;
      v9 = *(unsigned int *)(v6 + 8);
      v120 = *(_QWORD *)v4;
      v10 = 0;
      v11 = sub_2B0C7B0(v8, (__int64)&v8[v9], &v120);
      v13 = (__int64)v11 - v12;
      if ( !*(_DWORD *)(v6 + 248) )
        goto LABEL_20;
      v14 = v5;
      v15 = *(unsigned int *)(v6 + 248);
      v16 = 8LL * (int)(v13 >> 3);
      v17 = v3;
      while ( 1 )
      {
        while ( 1 )
        {
          v18 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v6 + 240) + 80LL * (unsigned int)v10) + v16);
          if ( *(_BYTE *)v18 > 0x1Cu && *(_QWORD *)v17 == *(_QWORD *)(v18 + 40) )
          {
            v19 = *(_DWORD *)(v17 + 104);
            v20 = *(_QWORD *)(v17 + 88);
            if ( v19 )
            {
              v21 = v19 - 1;
              v22 = v21 & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
              v23 = (__int64 *)(v20 + 16LL * v22);
              v24 = *v23;
              if ( v18 != *v23 )
              {
                v77 = 1;
                while ( v24 != -4096 )
                {
                  v78 = v77 + 1;
                  v22 = v21 & (v77 + v22);
                  v23 = (__int64 *)(v20 + 16LL * v22);
                  v24 = *v23;
                  if ( v18 == *v23 )
                    goto LABEL_10;
                  v77 = v78;
                }
                goto LABEL_5;
              }
LABEL_10:
              v25 = v23[1];
              if ( v25 )
              {
                if ( *(_DWORD *)(v25 + 136) == *(_DWORD *)(v17 + 204) && *(_DWORD *)(v25 + 144) != -1 )
                  break;
              }
            }
          }
LABEL_5:
          if ( v15 == ++v10 )
            goto LABEL_19;
        }
        v26 = *(_QWORD *)(v25 + 16);
        --*(_DWORD *)(v25 + 148);
        if ( v26 )
        {
          v27 = v26;
          v28 = 0;
          while ( 1 )
          {
            v29 = *(_DWORD *)(v27 + 148);
            if ( v29 == -1 )
              goto LABEL_5;
            v27 = *(_QWORD *)(v27 + 24);
            v28 += v29;
            if ( !v27 )
            {
              if ( v28 )
                goto LABEL_5;
              break;
            }
          }
        }
        v121[0] = v26;
        ++v10;
        v104 = v4;
        sub_2BA3420(v14, v121);
        v4 = v104;
        if ( v15 == v10 )
        {
LABEL_19:
          v3 = v17;
          v5 = v14;
          goto LABEL_20;
        }
      }
    }
    v61 = 4LL * (*(_DWORD *)(v7 + 4) & 0x7FFFFFF);
    if ( (*(_BYTE *)(v7 + 7) & 0x40) != 0 )
    {
      v62 = *(_QWORD **)(v7 - 8);
      v7 = (__int64)&v62[v61];
    }
    else
    {
      v62 = (_QWORD *)(v7 - v61 * 8);
    }
    if ( v62 != (_QWORD *)v7 )
    {
      v63 = v3;
      v64 = v4;
      while ( 1 )
      {
        while ( 1 )
        {
          v65 = (_BYTE *)*v62;
          if ( *(_BYTE *)*v62 > 0x1Cu && *(_QWORD *)v63 == *((_QWORD *)v65 + 5) )
          {
            v66 = *(_DWORD *)(v63 + 104);
            v67 = *(_QWORD *)(v63 + 88);
            if ( v66 )
            {
              v68 = v66 - 1;
              v69 = v68 & (((unsigned int)v65 >> 9) ^ ((unsigned int)v65 >> 4));
              v70 = (_QWORD *)(v67 + 16LL * v69);
              v71 = (_BYTE *)*v70;
              if ( v65 != (_BYTE *)*v70 )
              {
                v79 = 1;
                while ( v71 != (_BYTE *)-4096LL )
                {
                  v80 = v79 + 1;
                  v69 = v68 & (v79 + v69);
                  v70 = (_QWORD *)(v67 + 16LL * v69);
                  v71 = (_BYTE *)*v70;
                  if ( v65 == (_BYTE *)*v70 )
                    goto LABEL_74;
                  v79 = v80;
                }
                goto LABEL_69;
              }
LABEL_74:
              v72 = v70[1];
              if ( v72 )
              {
                if ( *(_DWORD *)(v72 + 136) == *(_DWORD *)(v63 + 204) && *(_DWORD *)(v72 + 144) != -1 )
                  break;
              }
            }
          }
LABEL_69:
          v62 += 4;
          if ( (_QWORD *)v7 == v62 )
            goto LABEL_83;
        }
        v73 = *(_QWORD *)(v72 + 16);
        --*(_DWORD *)(v72 + 148);
        if ( v73 )
        {
          v74 = v73;
          v75 = 0;
          while ( 1 )
          {
            v76 = *(_DWORD *)(v74 + 148);
            if ( v76 == -1 )
              goto LABEL_69;
            v74 = *(_QWORD *)(v74 + 24);
            v75 += v76;
            if ( !v74 )
            {
              if ( v75 )
                goto LABEL_69;
              break;
            }
          }
        }
        v121[0] = v73;
        v62 += 4;
        sub_2BA3420(v5, v121);
        if ( (_QWORD *)v7 == v62 )
        {
LABEL_83:
          v3 = v63;
          v4 = v64;
          break;
        }
      }
    }
LABEL_20:
    v30 = *(__int64 **)(v4 + 40);
    v31 = &v30[*(unsigned int *)(v4 + 48)];
    if ( v31 != v30 )
    {
      while ( 1 )
      {
        v32 = *v30;
        if ( *(_DWORD *)(*v30 + 144) == -1 )
          goto LABEL_22;
        v33 = *(_QWORD *)(v32 + 16);
        --*(_DWORD *)(v32 + 148);
        if ( v33 )
        {
          v34 = v33;
          v35 = 0;
          do
          {
            v36 = *(_DWORD *)(v34 + 148);
            if ( v36 == -1 )
              goto LABEL_22;
            v34 = *(_QWORD *)(v34 + 24);
            v35 += v36;
          }
          while ( v34 );
          if ( v35 )
            goto LABEL_22;
        }
        v37 = *(_DWORD *)(v5 + 24);
        if ( !v37 )
        {
          ++*(_QWORD *)v5;
          goto LABEL_108;
        }
        v38 = (unsigned int)(v37 - 1);
        v39 = *(_QWORD *)(v5 + 8);
        v40 = v38 & (((unsigned int)v33 >> 9) ^ ((unsigned int)v33 >> 4));
        v41 = (__int64 *)(v39 + 8LL * v40);
        v42 = *v41;
        if ( v33 == *v41 )
        {
LABEL_22:
          if ( v31 == ++v30 )
            break;
        }
        else
        {
          v112 = 1;
          v43 = 0;
          v105 = *(_DWORD *)(v5 + 24);
          while ( v42 != -4096 )
          {
            if ( !v43 && v42 == -8192 )
              v43 = (__int64)v41;
            v40 = v38 & (v112 + v40);
            v41 = (__int64 *)(v39 + 8LL * v40);
            v42 = *v41;
            if ( v33 == *v41 )
              goto LABEL_22;
            ++v112;
          }
          v44 = *(_DWORD *)(v5 + 16);
          if ( v43 )
            v41 = (__int64 *)v43;
          ++*(_QWORD *)v5;
          v45 = v44 + 1;
          if ( 4 * (v44 + 1) < 3 * v105 )
          {
            if ( v105 - *(_DWORD *)(v5 + 20) - v45 <= v105 >> 3 )
            {
              v111 = v4;
              v119 = v3;
              sub_2BA3250(v5, v37);
              v98 = *(_DWORD *)(v5 + 24);
              if ( !v98 )
              {
LABEL_151:
                ++*(_DWORD *)(v5 + 16);
                BUG();
              }
              v43 = (unsigned int)(v98 - 1);
              v38 = *(_QWORD *)(v5 + 8);
              v99 = 0;
              v100 = v43 & (((unsigned int)v33 >> 9) ^ ((unsigned int)v33 >> 4));
              v3 = v119;
              v4 = v111;
              v45 = *(_DWORD *)(v5 + 16) + 1;
              v101 = 1;
              v41 = (__int64 *)(v38 + 8LL * v100);
              v102 = *v41;
              if ( v33 != *v41 )
              {
                while ( v102 != -4096 )
                {
                  if ( !v99 && v102 == -8192 )
                    v99 = v41;
                  v100 = v43 & (v101 + v100);
                  v41 = (__int64 *)(v38 + 8LL * v100);
                  v102 = *v41;
                  if ( v33 == *v41 )
                    goto LABEL_37;
                  ++v101;
                }
                if ( v99 )
                  v41 = v99;
              }
            }
            goto LABEL_37;
          }
LABEL_108:
          v110 = v4;
          v118 = v3;
          sub_2BA3250(v5, 2 * v37);
          v93 = *(_DWORD *)(v5 + 24);
          if ( !v93 )
            goto LABEL_151;
          v94 = v93 - 1;
          v43 = *(_QWORD *)(v5 + 8);
          v3 = v118;
          v4 = v110;
          v95 = (v93 - 1) & (((unsigned int)v33 >> 9) ^ ((unsigned int)v33 >> 4));
          v45 = *(_DWORD *)(v5 + 16) + 1;
          v41 = (__int64 *)(v43 + 8LL * v95);
          v38 = *v41;
          if ( v33 != *v41 )
          {
            v96 = 1;
            v97 = 0;
            while ( v38 != -4096 )
            {
              if ( v38 == -8192 && !v97 )
                v97 = v41;
              v95 = v94 & (v96 + v95);
              v41 = (__int64 *)(v43 + 8LL * v95);
              v38 = *v41;
              if ( v33 == *v41 )
                goto LABEL_37;
              ++v96;
            }
            if ( v97 )
              v41 = v97;
          }
LABEL_37:
          *(_DWORD *)(v5 + 16) = v45;
          if ( *v41 != -4096 )
            --*(_DWORD *)(v5 + 20);
          *v41 = v33;
          v46 = *(unsigned int *)(v5 + 40);
          if ( v46 + 1 > (unsigned __int64)*(unsigned int *)(v5 + 44) )
          {
            v109 = v4;
            v117 = v3;
            sub_C8D5F0(v5 + 32, v103, v46 + 1, 8u, v38, v43);
            v46 = *(unsigned int *)(v5 + 40);
            v4 = v109;
            v3 = v117;
          }
          ++v30;
          *(_QWORD *)(*(_QWORD *)(v5 + 32) + 8 * v46) = v33;
          ++*(_DWORD *)(v5 + 40);
          if ( v31 == v30 )
            break;
        }
      }
    }
    v47 = *(_QWORD *)(v4 + 88);
    result = *(unsigned int *)(v4 + 96);
    v49 = v47 + 8 * result;
    if ( v49 != v47 )
    {
      while ( 1 )
      {
        v50 = *(_QWORD *)(*(_QWORD *)v47 + 16LL);
        --*(_DWORD *)(*(_QWORD *)v47 + 148LL);
        if ( v50 )
        {
          result = v50;
          v51 = 0;
          do
          {
            v52 = *(_DWORD *)(result + 148);
            if ( v52 == -1 )
              goto LABEL_47;
            result = *(_QWORD *)(result + 24);
            v51 += v52;
          }
          while ( result );
          if ( v51 )
            goto LABEL_47;
        }
        v53 = *(_DWORD *)(v5 + 24);
        if ( !v53 )
        {
          ++*(_QWORD *)v5;
          goto LABEL_94;
        }
        v54 = v53 - 1;
        v55 = *(_QWORD *)(v5 + 8);
        v56 = v54 & (((unsigned int)v50 >> 9) ^ ((unsigned int)v50 >> 4));
        result = v55 + 8LL * v56;
        v57 = *(_QWORD *)result;
        if ( v50 == *(_QWORD *)result )
        {
LABEL_47:
          v47 += 8;
          if ( v49 == v47 )
            break;
        }
        else
        {
          v113 = 1;
          v58 = 0;
          while ( v57 != -4096 )
          {
            if ( v57 != -8192 || v58 )
              result = (__int64)v58;
            v56 = v54 & (v113 + v56);
            v57 = *(_QWORD *)(v55 + 8LL * v56);
            if ( v50 == v57 )
              goto LABEL_47;
            ++v113;
            v58 = (_QWORD *)result;
            result = v55 + 8LL * v56;
          }
          if ( !v58 )
            v58 = (_QWORD *)result;
          v59 = *(_DWORD *)(v5 + 16);
          ++*(_QWORD *)v5;
          v60 = v59 + 1;
          if ( 4 * v60 < 3 * v53 )
          {
            if ( v53 - *(_DWORD *)(v5 + 20) - v60 <= v53 >> 3 )
            {
              v108 = v4;
              v116 = v3;
              sub_2BA3250(v5, v53);
              v87 = *(_DWORD *)(v5 + 24);
              if ( !v87 )
                goto LABEL_151;
              v54 = (unsigned int)(v87 - 1);
              v88 = *(_QWORD *)(v5 + 8);
              v89 = 0;
              v3 = v116;
              v90 = v54 & (((unsigned int)v50 >> 9) ^ ((unsigned int)v50 >> 4));
              v4 = v108;
              v91 = 1;
              v58 = (_QWORD *)(v88 + 8LL * v90);
              v92 = *v58;
              v60 = *(_DWORD *)(v5 + 16) + 1;
              if ( v50 != *v58 )
              {
                while ( v92 != -4096 )
                {
                  if ( !v89 && v92 == -8192 )
                    v89 = v58;
                  v90 = v54 & (v91 + v90);
                  v58 = (_QWORD *)(v88 + 8LL * v90);
                  v92 = *v58;
                  if ( v50 == *v58 )
                    goto LABEL_59;
                  ++v91;
                }
                if ( v89 )
                  v58 = v89;
              }
            }
            goto LABEL_59;
          }
LABEL_94:
          v106 = v4;
          v114 = v3;
          sub_2BA3250(v5, 2 * v53);
          v81 = *(_DWORD *)(v5 + 24);
          if ( !v81 )
            goto LABEL_151;
          v82 = v81 - 1;
          v54 = *(_QWORD *)(v5 + 8);
          v3 = v114;
          v4 = v106;
          v83 = v82 & (((unsigned int)v50 >> 9) ^ ((unsigned int)v50 >> 4));
          v58 = (_QWORD *)(v54 + 8LL * v83);
          v84 = *v58;
          v60 = *(_DWORD *)(v5 + 16) + 1;
          if ( v50 != *v58 )
          {
            v85 = 1;
            v86 = 0;
            while ( v84 != -4096 )
            {
              if ( !v86 && v84 == -8192 )
                v86 = v58;
              v83 = v82 & (v85 + v83);
              v58 = (_QWORD *)(v54 + 8LL * v83);
              v84 = *v58;
              if ( v50 == *v58 )
                goto LABEL_59;
              ++v85;
            }
            if ( v86 )
              v58 = v86;
          }
LABEL_59:
          *(_DWORD *)(v5 + 16) = v60;
          if ( *v58 != -4096 )
            --*(_DWORD *)(v5 + 20);
          *v58 = v50;
          result = *(unsigned int *)(v5 + 40);
          if ( result + 1 > (unsigned __int64)*(unsigned int *)(v5 + 44) )
          {
            v107 = v4;
            v115 = v3;
            sub_C8D5F0(v5 + 32, v103, result + 1, 8u, v54, (__int64)v58);
            result = *(unsigned int *)(v5 + 40);
            v4 = v107;
            v3 = v115;
          }
          v47 += 8;
          *(_QWORD *)(*(_QWORD *)(v5 + 32) + 8 * result) = v50;
          ++*(_DWORD *)(v5 + 40);
          if ( v49 == v47 )
            break;
        }
      }
    }
    v4 = *(_QWORD *)(v4 + 24);
  }
  while ( v4 );
  return result;
}
