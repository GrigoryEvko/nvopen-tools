// Function: sub_3570360
// Address: 0x3570360
//
__int64 __fastcall sub_3570360(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 *v4; // r12
  int v5; // edx
  __int64 v6; // rcx
  int v7; // ebx
  __int64 v8; // rax
  __int64 v9; // rcx
  int v10; // edx
  __int64 v11; // r13
  __int64 v12; // rdi
  __int64 v13; // rax
  __int64 v14; // r14
  __int64 v15; // rbx
  __int64 v16; // rsi
  __int64 *v17; // rdx
  __int64 v18; // rdx
  int v19; // ebx
  __int64 v20; // rdx
  unsigned int v21; // esi
  __int64 v22; // r9
  _QWORD *v23; // r15
  int v24; // r11d
  unsigned int v25; // ecx
  _QWORD *v26; // rax
  __int64 v27; // r8
  _DWORD *v28; // rax
  __int64 *v29; // rbx
  __int64 v30; // r14
  unsigned __int64 v31; // rax
  __int64 v32; // r8
  __int64 v33; // r9
  __int64 v34; // r15
  int v35; // eax
  unsigned int v36; // r14d
  __int64 *v37; // rcx
  __int64 v38; // rdx
  _QWORD *v39; // rax
  __int64 v40; // rbx
  __int64 v41; // r13
  __int64 v42; // rbx
  __int64 v43; // rax
  __int64 v44; // r11
  __int64 **v45; // rbx
  __int64 v46; // r9
  unsigned int v47; // edi
  __int64 *v48; // rax
  __int64 v49; // r8
  unsigned int v50; // esi
  __int64 v51; // r8
  int v52; // r10d
  _QWORD *v53; // rdx
  unsigned int v54; // ecx
  _QWORD *v55; // rax
  __int64 v56; // rdi
  __int64 v57; // rax
  __int64 v58; // rax
  __int64 v59; // r13
  __int64 v60; // r12
  unsigned int v61; // esi
  int v62; // r14d
  int v63; // eax
  int v64; // r9d
  __int64 v65; // r10
  unsigned int v66; // ecx
  int v67; // eax
  __int64 *v68; // rdx
  __int64 v69; // r8
  int v70; // eax
  int v71; // r13d
  __int64 v72; // r9
  unsigned int v73; // ecx
  int v74; // eax
  __int64 v75; // r8
  int v76; // edi
  _QWORD *v77; // rsi
  __int64 v78; // rax
  int v79; // eax
  int v80; // eax
  __int64 v81; // rax
  int v82; // eax
  int v83; // r9d
  __int64 v84; // r11
  unsigned int v85; // r10d
  __int64 v86; // rdi
  int v87; // esi
  _QWORD *v88; // rcx
  int v89; // eax
  int v90; // r9d
  __int64 v91; // r11
  int v92; // esi
  unsigned int v93; // r10d
  __int64 v94; // rdi
  int v95; // eax
  int v96; // eax
  int v97; // r9d
  __int64 v98; // r10
  __int64 *v99; // rsi
  int v100; // edi
  unsigned int v101; // ecx
  __int64 v102; // r8
  int v103; // eax
  int v104; // r10d
  int v105; // r10d
  __int64 v106; // r8
  _QWORD *v107; // rcx
  unsigned int v108; // r13d
  int v109; // esi
  __int64 v110; // rdi
  int v111; // edi
  __int64 v112; // [rsp+8h] [rbp-128h]
  __int64 v113; // [rsp+10h] [rbp-120h]
  __int64 v114; // [rsp+10h] [rbp-120h]
  int v115; // [rsp+10h] [rbp-120h]
  unsigned int v116; // [rsp+10h] [rbp-120h]
  __int64 v117; // [rsp+10h] [rbp-120h]
  __int64 *v118; // [rsp+18h] [rbp-118h]
  __int64 v119; // [rsp+20h] [rbp-110h]
  __int64 **v121; // [rsp+30h] [rbp-100h]
  __int64 **v123; // [rsp+40h] [rbp-F0h]
  __int64 v124; // [rsp+40h] [rbp-F0h]
  __int64 **v125; // [rsp+48h] [rbp-E8h]
  __int64 *v126; // [rsp+48h] [rbp-E8h]
  __int64 v127; // [rsp+48h] [rbp-E8h]
  __int64 v128; // [rsp+48h] [rbp-E8h]
  __m128i v129; // [rsp+50h] [rbp-E0h] BYREF
  __int64 v130; // [rsp+60h] [rbp-D0h] BYREF
  __int64 v131; // [rsp+68h] [rbp-C8h]
  __int64 v132; // [rsp+70h] [rbp-C0h]

  result = *(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8);
  v123 = (__int64 **)result;
  v119 = a1 + 24;
  if ( result == *(_QWORD *)a2 )
    return result;
  v125 = *(__int64 ***)a2;
  do
  {
    v4 = *v125;
    if ( (__int64 *)v4[2] != v4 )
      goto LABEL_3;
    v5 = *((_DWORD *)v4 + 10);
    if ( v5 )
    {
      v6 = v4[6];
      v7 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)v6 + 16LL) + 8LL);
      if ( v7 )
      {
        if ( v5 == 1 )
        {
LABEL_73:
          *(_DWORD *)sub_356E5F0(*(_QWORD *)(a1 + 8), *v125) = v7;
          v78 = v4[6];
          *((_DWORD *)v4 + 2) = v7;
          v4[2] = *(_QWORD *)(*(_QWORD *)v78 + 16LL);
          goto LABEL_3;
        }
        v8 = v6 + 8;
        v9 = v6 + 8LL * (unsigned int)(v5 - 2) + 16;
        while ( 1 )
        {
          v10 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)v8 + 16LL) + 8LL);
          if ( !v10 || v7 != v10 )
            break;
          v8 += 8;
          if ( v9 == v8 )
            goto LABEL_73;
        }
      }
    }
    v11 = *v4;
    v12 = *v4;
    v129.m128i_i64[0] = (__int64)&v130;
    v129.m128i_i64[1] = 0x1400000000LL;
    v13 = sub_2E311E0(v12);
    v14 = *(_QWORD *)(v11 + 56);
    v15 = v13;
    if ( v14 != v13 )
    {
      while ( !(unsigned __int8)sub_356FB10(a1, v14, (__int64)&v129) )
      {
        if ( !v14 )
          BUG();
        if ( (*(_BYTE *)v14 & 4) != 0 )
        {
          v14 = *(_QWORD *)(v14 + 8);
          if ( v15 == v14 )
            goto LABEL_19;
        }
        else
        {
          while ( (*(_BYTE *)(v14 + 44) & 8) != 0 )
            v14 = *(_QWORD *)(v14 + 8);
          v14 = *(_QWORD *)(v14 + 8);
          if ( v15 == v14 )
            goto LABEL_19;
        }
      }
      v44 = *(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8);
      if ( *(_QWORD *)a2 == v44 )
        goto LABEL_19;
      v118 = v4;
      v45 = *(__int64 ***)a2;
      while ( 2 )
      {
        v58 = (*v45)[7];
        if ( v58 )
        {
          v59 = *(_QWORD *)(a1 + 8);
          v60 = *(_QWORD *)(v58 + 24);
          v61 = *(_DWORD *)(v59 + 24);
          v62 = *(_DWORD *)(*(_QWORD *)(v58 + 32) + 8LL);
          if ( !v61 )
          {
            ++*(_QWORD *)v59;
            goto LABEL_59;
          }
          v46 = *(_QWORD *)(v59 + 8);
          v47 = (v61 - 1) & (((unsigned int)v60 >> 9) ^ ((unsigned int)v60 >> 4));
          v48 = (__int64 *)(v46 + 16LL * v47);
          v49 = *v48;
          if ( v60 == *v48 )
          {
LABEL_51:
            *((_DWORD *)v48 + 2) = v62;
            v50 = *(_DWORD *)(a1 + 48);
            if ( v50 )
              goto LABEL_52;
LABEL_64:
            ++*(_QWORD *)(a1 + 24);
LABEL_65:
            v114 = v44;
            sub_356EA90(v119, 2 * v50);
            v70 = *(_DWORD *)(a1 + 48);
            if ( !v70 )
              goto LABEL_169;
            v71 = v70 - 1;
            v72 = *(_QWORD *)(a1 + 32);
            v44 = v114;
            v73 = (v70 - 1) & (((unsigned int)v60 >> 9) ^ ((unsigned int)v60 >> 4));
            v74 = *(_DWORD *)(a1 + 40) + 1;
            v53 = (_QWORD *)(v72 + 16LL * v73);
            v75 = *v53;
            if ( v60 != *v53 )
            {
              v76 = 1;
              v77 = 0;
              while ( v75 != -4096 )
              {
                if ( v75 == -8192 && !v77 )
                  v77 = v53;
                v73 = v71 & (v76 + v73);
                v53 = (_QWORD *)(v72 + 16LL * v73);
                v75 = *v53;
                if ( v60 == *v53 )
                  goto LABEL_125;
                ++v76;
              }
              if ( v77 )
                v53 = v77;
            }
            goto LABEL_125;
          }
          v115 = 1;
          v68 = 0;
          while ( v49 != -4096 )
          {
            if ( v49 == -8192 && !v68 )
              v68 = v48;
            v47 = (v61 - 1) & (v115 + v47);
            v48 = (__int64 *)(v46 + 16LL * v47);
            v49 = *v48;
            if ( v60 == *v48 )
              goto LABEL_51;
            ++v115;
          }
          if ( !v68 )
            v68 = v48;
          v95 = *(_DWORD *)(v59 + 16);
          ++*(_QWORD *)v59;
          v67 = v95 + 1;
          if ( 4 * v67 >= 3 * v61 )
          {
LABEL_59:
            v113 = v44;
            sub_34F9190(v59, 2 * v61);
            v63 = *(_DWORD *)(v59 + 24);
            if ( !v63 )
              goto LABEL_168;
            v64 = v63 - 1;
            v65 = *(_QWORD *)(v59 + 8);
            v44 = v113;
            v66 = (v63 - 1) & (((unsigned int)v60 >> 9) ^ ((unsigned int)v60 >> 4));
            v67 = *(_DWORD *)(v59 + 16) + 1;
            v68 = (__int64 *)(v65 + 16LL * v66);
            v69 = *v68;
            if ( v60 != *v68 )
            {
              v111 = 1;
              v99 = 0;
              while ( v69 != -4096 )
              {
                if ( v69 == -8192 && !v99 )
                  v99 = v68;
                v66 = v64 & (v111 + v66);
                v68 = (__int64 *)(v65 + 16LL * v66);
                v69 = *v68;
                if ( v60 == *v68 )
                  goto LABEL_61;
                ++v111;
              }
              goto LABEL_112;
            }
          }
          else if ( v61 - *(_DWORD *)(v59 + 20) - v67 <= v61 >> 3 )
          {
            v112 = v44;
            v116 = ((unsigned int)v60 >> 9) ^ ((unsigned int)v60 >> 4);
            sub_34F9190(v59, v61);
            v96 = *(_DWORD *)(v59 + 24);
            if ( !v96 )
            {
LABEL_168:
              ++*(_DWORD *)(v59 + 16);
              BUG();
            }
            v97 = v96 - 1;
            v98 = *(_QWORD *)(v59 + 8);
            v99 = 0;
            v44 = v112;
            v100 = 1;
            v101 = v97 & v116;
            v67 = *(_DWORD *)(v59 + 16) + 1;
            v68 = (__int64 *)(v98 + 16LL * (v97 & v116));
            v102 = *v68;
            if ( v60 != *v68 )
            {
              while ( v102 != -4096 )
              {
                if ( !v99 && v102 == -8192 )
                  v99 = v68;
                v101 = v97 & (v100 + v101);
                v68 = (__int64 *)(v98 + 16LL * v101);
                v102 = *v68;
                if ( v60 == *v68 )
                  goto LABEL_61;
                ++v100;
              }
LABEL_112:
              if ( v99 )
                v68 = v99;
            }
          }
LABEL_61:
          *(_DWORD *)(v59 + 16) = v67;
          if ( *v68 != -4096 )
            --*(_DWORD *)(v59 + 20);
          *v68 = v60;
          *((_DWORD *)v68 + 2) = 0;
          *((_DWORD *)v68 + 2) = v62;
          v50 = *(_DWORD *)(a1 + 48);
          if ( !v50 )
            goto LABEL_64;
LABEL_52:
          v51 = *(_QWORD *)(a1 + 32);
          v52 = 1;
          v53 = 0;
          v54 = (v50 - 1) & (((unsigned int)v60 >> 9) ^ ((unsigned int)v60 >> 4));
          v55 = (_QWORD *)(v51 + 16LL * v54);
          v56 = *v55;
          if ( v60 == *v55 )
          {
LABEL_53:
            v57 = v55[1];
          }
          else
          {
            while ( v56 != -4096 )
            {
              if ( !v53 && v56 == -8192 )
                v53 = v55;
              v54 = (v50 - 1) & (v52 + v54);
              v55 = (_QWORD *)(v51 + 16LL * v54);
              v56 = *v55;
              if ( v60 == *v55 )
                goto LABEL_53;
              ++v52;
            }
            if ( !v53 )
              v53 = v55;
            v103 = *(_DWORD *)(a1 + 40);
            ++*(_QWORD *)(a1 + 24);
            v74 = v103 + 1;
            if ( 4 * v74 >= 3 * v50 )
              goto LABEL_65;
            if ( v50 - *(_DWORD *)(a1 + 44) - v74 <= v50 >> 3 )
            {
              v117 = v44;
              sub_356EA90(v119, v50);
              v104 = *(_DWORD *)(a1 + 48);
              if ( !v104 )
              {
LABEL_169:
                ++*(_DWORD *)(a1 + 40);
                BUG();
              }
              v105 = v104 - 1;
              v106 = *(_QWORD *)(a1 + 32);
              v107 = 0;
              v108 = v105 & (((unsigned int)v60 >> 9) ^ ((unsigned int)v60 >> 4));
              v44 = v117;
              v109 = 1;
              v74 = *(_DWORD *)(a1 + 40) + 1;
              v53 = (_QWORD *)(v106 + 16LL * v108);
              v110 = *v53;
              if ( v60 != *v53 )
              {
                while ( v110 != -4096 )
                {
                  if ( !v107 && v110 == -8192 )
                    v107 = v53;
                  v108 = v105 & (v109 + v108);
                  v53 = (_QWORD *)(v106 + 16LL * v108);
                  v110 = *v53;
                  if ( v60 == *v53 )
                    goto LABEL_125;
                  ++v109;
                }
                if ( v107 )
                  v53 = v107;
              }
            }
LABEL_125:
            *(_DWORD *)(a1 + 40) = v74;
            if ( *v53 != -4096 )
              --*(_DWORD *)(a1 + 44);
            *v53 = v60;
            v57 = 0;
            v53[1] = 0;
          }
          *(_DWORD *)(v57 + 8) = v62;
        }
        if ( (__int64 **)v44 == ++v45 )
        {
          v4 = v118;
          break;
        }
        continue;
      }
    }
LABEL_19:
    if ( (__int64 *)v129.m128i_i64[0] != &v130 )
      _libc_free(v129.m128i_u64[0]);
    if ( !*((_DWORD *)v4 + 2) )
    {
      v16 = *v4;
      v17 = (__int64 *)(*(_QWORD *)(*v4 + 48) & 0xFFFFFFFFFFFFFFF8LL);
      if ( v17 != (__int64 *)(*v4 + 48) )
        v17 = *(__int64 **)(v16 + 56);
      sub_356E080(
        0,
        v16,
        v17,
        *(_QWORD *)(*(_QWORD *)a1 + 8LL),
        *(_QWORD *)(*(_QWORD *)a1 + 16LL),
        *(_QWORD *)(*(_QWORD *)a1 + 40LL),
        *(_QWORD *)(*(_QWORD *)a1 + 32LL));
      v19 = *(_DWORD *)(*(_QWORD *)(v18 + 32) + 8LL);
      *((_DWORD *)v4 + 2) = v19;
      *(_DWORD *)sub_356E5F0(*(_QWORD *)(a1 + 8), v4) = v19;
    }
LABEL_3:
    ++v125;
  }
  while ( v123 != v125 );
  result = *(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8);
  v121 = *(__int64 ***)a2;
  v124 = result;
  if ( *(_QWORD *)a2 != result )
  {
    while ( 1 )
    {
      v29 = *(__int64 **)(v124 - 8);
      v30 = v29[2];
      if ( (__int64 *)v30 == v29 )
      {
        v31 = sub_2EBEE10(*(_QWORD *)(*(_QWORD *)a1 + 40LL), *((_DWORD *)v29 + 2));
        v34 = v31;
        if ( v31 )
        {
          v35 = *(unsigned __int16 *)(v31 + 68);
          if ( (!v35 || v35 == 68) && (*(_DWORD *)(v34 + 40) & 0xFFFFFFu) <= 1 )
          {
            if ( *((_DWORD *)v29 + 10) )
            {
              v36 = 0;
              v37 = v29;
              do
              {
                v38 = v36;
                v126 = v37;
                ++v36;
                v39 = *(_QWORD **)(v37[6] + 8 * v38);
                v40 = *v39;
                v41 = *(_QWORD *)(*v39 + 32LL);
                LODWORD(v39) = *(_DWORD *)(v39[2] + 8LL);
                v129.m128i_i64[0] = 0;
                v130 = 0;
                v129.m128i_i32[2] = (int)v39;
                v131 = 0;
                v132 = 0;
                sub_2E8EAD0(v34, v41, &v129);
                v129.m128i_i8[0] = 4;
                v130 = 0;
                v129.m128i_i32[0] &= 0xFFF000FF;
                v131 = v40;
                sub_2E8EAD0(v34, v41, &v129);
                v37 = v126;
              }
              while ( *((_DWORD *)v126 + 10) != v36 );
            }
            v42 = *(_QWORD *)(a1 + 16);
            if ( v42 )
            {
              v43 = *(unsigned int *)(v42 + 8);
              if ( v43 + 1 > (unsigned __int64)*(unsigned int *)(v42 + 12) )
              {
                sub_C8D5F0(*(_QWORD *)(a1 + 16), (const void *)(v42 + 16), v43 + 1, 8u, v32, v33);
                v43 = *(unsigned int *)(v42 + 8);
              }
              *(_QWORD *)(*(_QWORD *)v42 + 8 * v43) = v34;
              ++*(_DWORD *)(v42 + 8);
            }
          }
        }
        goto LABEL_35;
      }
      v20 = *(_QWORD *)(a1 + 8);
      v21 = *(_DWORD *)(v20 + 24);
      if ( !v21 )
        break;
      v22 = *(_QWORD *)(v20 + 8);
      v23 = 0;
      v24 = 1;
      v25 = (v21 - 1) & (((unsigned int)*v29 >> 9) ^ ((unsigned int)*v29 >> 4));
      v26 = (_QWORD *)(v22 + 16LL * v25);
      v27 = *v26;
      if ( *v29 != *v26 )
      {
        while ( v27 != -4096 )
        {
          if ( v27 == -8192 && !v23 )
            v23 = v26;
          v25 = (v21 - 1) & (v24 + v25);
          v26 = (_QWORD *)(v22 + 16LL * v25);
          v27 = *v26;
          if ( *v29 == *v26 )
            goto LABEL_33;
          ++v24;
        }
        if ( !v23 )
          v23 = v26;
        v79 = *(_DWORD *)(v20 + 16);
        ++*(_QWORD *)v20;
        v80 = v79 + 1;
        if ( 4 * v80 < 3 * v21 )
        {
          if ( v21 - *(_DWORD *)(v20 + 20) - v80 <= v21 >> 3 )
          {
            v128 = v20;
            sub_34F9190(v20, v21);
            v20 = v128;
            v89 = *(_DWORD *)(v128 + 24);
            if ( !v89 )
            {
LABEL_166:
              ++*(_DWORD *)(v20 + 16);
              BUG();
            }
            v90 = v89 - 1;
            v91 = *(_QWORD *)(v128 + 8);
            v92 = 1;
            v88 = 0;
            v93 = (v89 - 1) & (((unsigned int)*v29 >> 9) ^ ((unsigned int)*v29 >> 4));
            v23 = (_QWORD *)(v91 + 16LL * v93);
            v94 = *v23;
            v80 = *(_DWORD *)(v128 + 16) + 1;
            if ( *v23 != *v29 )
            {
              while ( v94 != -4096 )
              {
                if ( !v88 && v94 == -8192 )
                  v88 = v23;
                v93 = v90 & (v92 + v93);
                v23 = (_QWORD *)(v91 + 16LL * v93);
                v94 = *v23;
                if ( *v29 == *v23 )
                  goto LABEL_84;
                ++v92;
              }
              goto LABEL_92;
            }
          }
          goto LABEL_84;
        }
LABEL_88:
        v127 = v20;
        sub_34F9190(v20, 2 * v21);
        v20 = v127;
        v82 = *(_DWORD *)(v127 + 24);
        if ( !v82 )
          goto LABEL_166;
        v83 = v82 - 1;
        v84 = *(_QWORD *)(v127 + 8);
        v85 = (v82 - 1) & (((unsigned int)*v29 >> 9) ^ ((unsigned int)*v29 >> 4));
        v23 = (_QWORD *)(v84 + 16LL * v85);
        v86 = *v23;
        v80 = *(_DWORD *)(v127 + 16) + 1;
        if ( *v23 != *v29 )
        {
          v87 = 1;
          v88 = 0;
          while ( v86 != -4096 )
          {
            if ( v86 == -8192 && !v88 )
              v88 = v23;
            v85 = v83 & (v87 + v85);
            v23 = (_QWORD *)(v84 + 16LL * v85);
            v86 = *v23;
            if ( *v29 == *v23 )
              goto LABEL_84;
            ++v87;
          }
LABEL_92:
          if ( v88 )
            v23 = v88;
        }
LABEL_84:
        *(_DWORD *)(v20 + 16) = v80;
        if ( *v23 != -4096 )
          --*(_DWORD *)(v20 + 20);
        v81 = *v29;
        *((_DWORD *)v23 + 2) = 0;
        *v23 = v81;
        v28 = v23 + 1;
        goto LABEL_34;
      }
LABEL_33:
      v28 = v26 + 1;
LABEL_34:
      *v28 = *(_DWORD *)(v30 + 8);
LABEL_35:
      v124 -= 8;
      result = v124;
      if ( v121 == (__int64 **)v124 )
        return result;
    }
    ++*(_QWORD *)v20;
    goto LABEL_88;
  }
  return result;
}
