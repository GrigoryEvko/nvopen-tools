// Function: sub_1387DB0
// Address: 0x1387db0
//
__int64 __fastcall sub_1387DB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r15
  __int64 v7; // r13
  __int64 v9; // r12
  __int64 v10; // r8
  __int64 v11; // rsi
  __int64 v12; // r14
  __int64 v13; // rbx
  _QWORD *v14; // rax
  __int64 v15; // rdx
  _QWORD *v16; // rcx
  __int64 v17; // rdi
  __int64 v18; // rax
  __int64 v19; // r14
  __int64 v20; // rcx
  __int64 v21; // rax
  char *v22; // r15
  __int64 v23; // r14
  char *v24; // rbx
  __m128i *v25; // r12
  __m128i *v26; // rsi
  unsigned __int64 v27; // rax
  unsigned int *v28; // r12
  unsigned int *v29; // rdi
  __m128i *v30; // rdi
  size_t v31; // rbx
  __m128i *v32; // rax
  __int64 v33; // rdx
  __m128i *v34; // rcx
  __m128i *v35; // rax
  _QWORD *v36; // rbx
  __int64 v37; // r12
  unsigned __int64 v38; // rdi
  unsigned __int64 v39; // rdi
  int v41; // eax
  __int64 v42; // rax
  __m128i *v43; // rax
  __int64 v44; // rax
  __int64 v45; // rsi
  __int64 *v46; // rax
  __int64 v47; // r14
  unsigned int *v48; // r15
  __int64 v49; // rdi
  unsigned int *v50; // r14
  __int64 v51; // r15
  unsigned int *v52; // rbx
  unsigned int *v53; // r15
  unsigned int v54; // ecx
  __m128i *v55; // rax
  unsigned int v56; // esi
  unsigned int v57; // r10d
  unsigned int v58; // edi
  unsigned int v59; // edx
  __int64 v60; // rax
  __int64 v61; // r13
  __int64 v62; // rsi
  __int64 v63; // r14
  __int64 v64; // r10
  __int64 v65; // rax
  unsigned int v66; // eax
  __int64 v67; // rcx
  __int64 v68; // rax
  unsigned int v69; // edx
  __int64 v70; // rax
  unsigned int v71; // eax
  __int64 v72; // rcx
  __int64 v73; // rax
  unsigned int v74; // edx
  __int64 v75; // rax
  __int64 v76; // rax
  __m128i *v77; // rax
  __int64 v78; // rdx
  int v79; // r11d
  __int64 v80; // rdx
  int v81; // eax
  __int64 v82; // rax
  int v83; // r11d
  __int64 v84; // rdx
  int v85; // eax
  __int64 v86; // rax
  unsigned int v87; // edx
  __int64 v88; // rsi
  __int64 v89; // rcx
  unsigned int v90; // edx
  __int64 v91; // rsi
  __int64 v92; // rcx
  unsigned int v93; // edx
  __int64 v94; // rsi
  unsigned int v95; // edx
  __int64 v96; // rsi
  __int64 v97; // [rsp+8h] [rbp-C8h]
  __int64 v98; // [rsp+8h] [rbp-C8h]
  __int64 v99; // [rsp+8h] [rbp-C8h]
  unsigned int v100; // [rsp+10h] [rbp-C0h]
  unsigned int v101; // [rsp+10h] [rbp-C0h]
  __int64 v102; // [rsp+10h] [rbp-C0h]
  __int64 v103; // [rsp+10h] [rbp-C0h]
  __int64 v104; // [rsp+10h] [rbp-C0h]
  __int64 v105; // [rsp+18h] [rbp-B8h]
  __int64 v106; // [rsp+20h] [rbp-B0h]
  __int64 v107; // [rsp+28h] [rbp-A8h]
  __int32 v108; // [rsp+28h] [rbp-A8h]
  __int64 v109; // [rsp+28h] [rbp-A8h]
  unsigned int *v110; // [rsp+30h] [rbp-A0h]
  __int64 v111; // [rsp+38h] [rbp-98h]
  __int64 v112; // [rsp+38h] [rbp-98h]
  __int64 v113; // [rsp+38h] [rbp-98h]
  __int64 v114; // [rsp+48h] [rbp-88h] BYREF
  char v115; // [rsp+50h] [rbp-80h]
  __int64 v116; // [rsp+54h] [rbp-7Ch] BYREF
  char v117; // [rsp+5Ch] [rbp-74h]
  __m128i v118; // [rsp+60h] [rbp-70h] BYREF
  __int64 v119; // [rsp+70h] [rbp-60h]
  __m128i v120; // [rsp+80h] [rbp-50h] BYREF
  __int64 v121; // [rsp+90h] [rbp-40h]
  unsigned int v122; // [rsp+98h] [rbp-38h]

  v6 = a3;
  v7 = a1;
  if ( (*(_BYTE *)(a2 + 18) & 1) != 0 )
  {
    sub_15E08E0(a2);
    v9 = *(_QWORD *)(a2 + 88);
    v10 = v9 + 40LL * *(_QWORD *)(a2 + 96);
    if ( (*(_BYTE *)(a2 + 18) & 1) != 0 )
    {
      v113 = v9 + 40LL * *(_QWORD *)(a2 + 96);
      sub_15E08E0(a2);
      v9 = *(_QWORD *)(a2 + 88);
      v10 = v113;
    }
  }
  else
  {
    v9 = *(_QWORD *)(a2 + 88);
    v10 = v9 + 40LL * *(_QWORD *)(a2 + 96);
  }
  if ( v9 == v10 )
    goto LABEL_15;
  v11 = a1 + 16;
  v12 = a4;
  v13 = v10;
  do
  {
    v14 = *(_QWORD **)v6;
    v15 = 8LL * *(unsigned int *)(v6 + 8);
    v16 = (_QWORD *)(*(_QWORD *)v6 + v15);
    v17 = v15 >> 3;
    a3 = v15 >> 5;
    if ( a3 )
    {
      a3 = (__int64)&v14[4 * a3];
      while ( *v14 != v9 )
      {
        if ( v14[1] == v9 )
        {
          ++v14;
          break;
        }
        if ( v14[2] == v9 )
        {
          v14 += 2;
          break;
        }
        if ( v14[3] == v9 )
        {
          v14 += 3;
          break;
        }
        v14 += 4;
        if ( (_QWORD *)a3 == v14 )
        {
          v17 = v16 - v14;
          goto LABEL_38;
        }
      }
    }
    else
    {
LABEL_38:
      if ( v17 == 2 )
        goto LABEL_48;
      if ( v17 == 3 )
      {
        if ( *v14 != v9 )
        {
          ++v14;
LABEL_48:
          if ( *v14 != v9 && *++v14 != v9 )
            goto LABEL_13;
        }
      }
      else if ( v17 != 1 || *v14 != v9 )
      {
        goto LABEL_13;
      }
    }
    if ( v16 != v14 )
    {
      v41 = *(_DWORD *)(v9 + 32);
      v121 = 0;
      v120 = (__m128i)(unsigned int)(v41 + 1);
      v42 = *(unsigned int *)(v7 + 8);
      if ( (unsigned int)v42 >= *(_DWORD *)(v7 + 12) )
      {
        sub_16CD150(v7, v11, 0, 24);
        v42 = *(unsigned int *)(v7 + 8);
      }
      v43 = (__m128i *)(*(_QWORD *)v7 + 24 * v42);
      *v43 = _mm_loadu_si128(&v120);
      LODWORD(a3) = v121;
      v43[1].m128i_i64[0] = v121;
      ++*(_DWORD *)(v7 + 8);
    }
LABEL_13:
    v9 += 40;
  }
  while ( v13 != v9 );
  a4 = v12;
LABEL_15:
  v18 = *(unsigned int *)(a4 + 24);
  v19 = *(_QWORD *)(a4 + 8);
  v120 = 0u;
  v20 = *(unsigned int *)(a4 + 16);
  v121 = 0;
  v122 = 0;
  v111 = v19 + 48 * v18;
  if ( !(_DWORD)v20 || v19 == v111 )
    goto LABEL_16;
  while ( 2 )
  {
    if ( *(_QWORD *)v19 == -8 )
    {
      if ( *(_DWORD *)(v19 + 8) != -1 )
        break;
      goto LABEL_136;
    }
    if ( *(_QWORD *)v19 == -16 && *(_DWORD *)(v19 + 8) == -2 )
    {
LABEL_136:
      v19 += 48;
      if ( v111 == v19 )
        goto LABEL_16;
      continue;
    }
    break;
  }
  if ( v111 == v19 )
    goto LABEL_16;
  v106 = v7;
  sub_1381950((__int64)&v114, *(_QWORD *)v19, *(_DWORD *)(v19 + 8), v6);
  while ( 2 )
  {
    if ( !v115 )
      goto LABEL_62;
    if ( !*(_DWORD *)(v19 + 32) )
      goto LABEL_62;
    v60 = *(_QWORD *)(v19 + 24);
    v61 = v60 + 24LL * *(unsigned int *)(v19 + 40);
    if ( v60 == v61 )
      goto LABEL_62;
    while ( 2 )
    {
      v62 = *(_QWORD *)v60;
      if ( *(_QWORD *)v60 == -8 )
      {
        if ( *(_DWORD *)(v60 + 8) != -1 )
          break;
        goto LABEL_157;
      }
      if ( v62 == -16 && *(_DWORD *)(v60 + 8) == -2 )
      {
LABEL_157:
        v60 += 24;
        if ( v61 == v60 )
          goto LABEL_62;
        continue;
      }
      break;
    }
    if ( v61 == v60 )
      goto LABEL_62;
    v105 = v19;
    v63 = v60;
LABEL_111:
    sub_1381950((__int64)&v116, v62, *(_DWORD *)(v63 + 8), v6);
    if ( !v117 )
    {
      v64 = *(_QWORD *)v63;
      v108 = *(_DWORD *)(v63 + 8);
      v65 = *(_QWORD *)(v63 + 16);
      if ( (v65 & 5) == 0 )
        goto LABEL_113;
      if ( v122 )
      {
        v10 = v120.m128i_i64[1];
        v101 = ((unsigned int)v64 >> 9) ^ ((unsigned int)v64 >> 4);
        v71 = (v122 - 1) & v101;
        a6 = v120.m128i_i64[1] + 136LL * v71;
        v72 = *(_QWORD *)a6;
        if ( v64 == *(_QWORD *)a6 )
        {
LABEL_127:
          v73 = *(unsigned int *)(a6 + 16);
          v74 = *(_DWORD *)(a6 + 20);
          v20 = v114;
          v118.m128i_i64[0] = v114;
          v118.m128i_i32[2] = v108;
          if ( (unsigned int)v73 >= v74 )
          {
            v97 = v64;
            v102 = a6;
            sub_16CD150(a6 + 8, a6 + 24, 0, 12);
            a6 = v102;
            v64 = v97;
            v73 = *(unsigned int *)(v102 + 16);
          }
LABEL_129:
          v75 = *(_QWORD *)(a6 + 8) + 12 * v73;
          *(_QWORD *)v75 = v118.m128i_i64[0];
          *(_DWORD *)(v75 + 8) = v118.m128i_i32[2];
          ++*(_DWORD *)(a6 + 16);
          v65 = *(_QWORD *)(v63 + 16);
LABEL_113:
          if ( (v65 & 0x28) == 0 )
            goto LABEL_106;
          if ( v122 )
          {
            v10 = v120.m128i_i64[1];
            v100 = ((unsigned int)v64 >> 9) ^ ((unsigned int)v64 >> 4);
            v66 = (v122 - 1) & v100;
            a6 = v120.m128i_i64[1] + 136LL * v66;
            v67 = *(_QWORD *)a6;
            if ( v64 == *(_QWORD *)a6 )
            {
LABEL_116:
              v68 = *(unsigned int *)(a6 + 80);
              v69 = *(_DWORD *)(a6 + 84);
              v20 = v114;
              v118.m128i_i64[0] = v114;
              v118.m128i_i32[2] = v108;
              if ( v69 <= (unsigned int)v68 )
              {
                v109 = a6;
                sub_16CD150(a6 + 72, a6 + 88, 0, 12);
                a6 = v109;
                v68 = *(unsigned int *)(v109 + 80);
              }
              goto LABEL_118;
            }
            v83 = 1;
            v84 = 0;
            while ( v67 != -8 )
            {
              if ( !v84 && v67 == -16 )
                v84 = a6;
              v66 = (v122 - 1) & (v83 + v66);
              a6 = v120.m128i_i64[1] + 136LL * v66;
              v67 = *(_QWORD *)a6;
              if ( v64 == *(_QWORD *)a6 )
                goto LABEL_116;
              ++v83;
            }
            if ( v84 )
              a6 = v84;
            ++v120.m128i_i64[0];
            v85 = v121 + 1;
            if ( 4 * ((int)v121 + 1) < 3 * v122 )
            {
              if ( v122 - HIDWORD(v121) - v85 <= v122 >> 3 )
              {
                v98 = v64;
                sub_13826E0((__int64)&v120, v122);
                if ( !v122 )
                {
LABEL_215:
                  LODWORD(v121) = v121 + 1;
                  BUG();
                }
                v10 = 1;
                v64 = v98;
                v93 = (v122 - 1) & v100;
                v92 = 0;
                a6 = v120.m128i_i64[1] + 136LL * v93;
                v94 = *(_QWORD *)a6;
                v85 = v121 + 1;
                if ( v98 != *(_QWORD *)a6 )
                {
                  while ( v94 != -8 )
                  {
                    if ( !v92 && v94 == -16 )
                      v92 = a6;
                    v93 = (v122 - 1) & (v10 + v93);
                    a6 = v120.m128i_i64[1] + 136LL * v93;
                    v94 = *(_QWORD *)a6;
                    if ( v98 == *(_QWORD *)a6 )
                      goto LABEL_153;
                    v10 = (unsigned int)(v10 + 1);
                  }
                  goto LABEL_186;
                }
              }
              goto LABEL_153;
            }
          }
          else
          {
            ++v120.m128i_i64[0];
          }
          v104 = v64;
          sub_13826E0((__int64)&v120, 2 * v122);
          if ( !v122 )
            goto LABEL_215;
          v64 = v104;
          v90 = (v122 - 1) & (((unsigned int)v104 >> 9) ^ ((unsigned int)v104 >> 4));
          a6 = v120.m128i_i64[1] + 136LL * v90;
          v91 = *(_QWORD *)a6;
          v85 = v121 + 1;
          if ( *(_QWORD *)a6 != v104 )
          {
            v10 = 1;
            v92 = 0;
            while ( v91 != -8 )
            {
              if ( !v92 && v91 == -16 )
                v92 = a6;
              v90 = (v122 - 1) & (v10 + v90);
              a6 = v120.m128i_i64[1] + 136LL * v90;
              v91 = *(_QWORD *)a6;
              if ( v104 == *(_QWORD *)a6 )
                goto LABEL_153;
              v10 = (unsigned int)(v10 + 1);
            }
LABEL_186:
            if ( v92 )
              a6 = v92;
          }
LABEL_153:
          LODWORD(v121) = v85;
          if ( *(_QWORD *)a6 != -8 )
            --HIDWORD(v121);
          *(_QWORD *)a6 = v64;
          memset((void *)(a6 + 8), 0, 0x80u);
          v20 = 0;
          *(_QWORD *)(a6 + 8) = a6 + 24;
          *(_QWORD *)(a6 + 72) = a6 + 88;
          v86 = v114;
          *(_QWORD *)(a6 + 16) = 0x400000000LL;
          v118.m128i_i64[0] = v86;
          *(_QWORD *)(a6 + 80) = 0x400000000LL;
          v118.m128i_i32[2] = v108;
          v68 = 0;
LABEL_118:
          v63 += 24;
          v70 = *(_QWORD *)(a6 + 72) + 12 * v68;
          *(_QWORD *)v70 = v118.m128i_i64[0];
          *(_DWORD *)(v70 + 8) = v118.m128i_i32[2];
          ++*(_DWORD *)(a6 + 80);
          if ( v63 == v61 )
            goto LABEL_119;
          while ( 1 )
          {
LABEL_107:
            if ( *(_QWORD *)v63 == -8 )
            {
              if ( *(_DWORD *)(v63 + 8) != -1 )
                goto LABEL_109;
            }
            else if ( *(_QWORD *)v63 != -16 || *(_DWORD *)(v63 + 8) != -2 )
            {
LABEL_109:
              if ( v61 == v63 )
                goto LABEL_119;
              v62 = *(_QWORD *)v63;
              goto LABEL_111;
            }
            v63 += 24;
            if ( v61 == v63 )
              goto LABEL_119;
          }
        }
        v79 = 1;
        v80 = 0;
        while ( v72 != -8 )
        {
          if ( v72 == -16 && !v80 )
            v80 = a6;
          v71 = (v122 - 1) & (v79 + v71);
          a6 = v120.m128i_i64[1] + 136LL * v71;
          v72 = *(_QWORD *)a6;
          if ( *(_QWORD *)a6 == v64 )
            goto LABEL_127;
          ++v79;
        }
        if ( v80 )
          a6 = v80;
        ++v120.m128i_i64[0];
        v81 = v121 + 1;
        if ( 4 * ((int)v121 + 1) < 3 * v122 )
        {
          if ( v122 - HIDWORD(v121) - v81 <= v122 >> 3 )
          {
            v99 = v64;
            sub_13826E0((__int64)&v120, v122);
            if ( !v122 )
            {
LABEL_216:
              LODWORD(v121) = v121 + 1;
              BUG();
            }
            v10 = 1;
            v64 = v99;
            v95 = (v122 - 1) & v101;
            v89 = 0;
            a6 = v120.m128i_i64[1] + 136LL * v95;
            v96 = *(_QWORD *)a6;
            v81 = v121 + 1;
            if ( v99 != *(_QWORD *)a6 )
            {
              while ( v96 != -8 )
              {
                if ( !v89 && v96 == -16 )
                  v89 = a6;
                v95 = (v122 - 1) & (v10 + v95);
                a6 = v120.m128i_i64[1] + 136LL * v95;
                v96 = *(_QWORD *)a6;
                if ( v99 == *(_QWORD *)a6 )
                  goto LABEL_144;
                v10 = (unsigned int)(v10 + 1);
              }
              goto LABEL_192;
            }
          }
          goto LABEL_144;
        }
      }
      else
      {
        ++v120.m128i_i64[0];
      }
      v103 = v64;
      sub_13826E0((__int64)&v120, 2 * v122);
      if ( !v122 )
        goto LABEL_216;
      v64 = v103;
      v87 = (v122 - 1) & (((unsigned int)v103 >> 9) ^ ((unsigned int)v103 >> 4));
      a6 = v120.m128i_i64[1] + 136LL * v87;
      v88 = *(_QWORD *)a6;
      v81 = v121 + 1;
      if ( v103 != *(_QWORD *)a6 )
      {
        v10 = 1;
        v89 = 0;
        while ( v88 != -8 )
        {
          if ( !v89 && v88 == -16 )
            v89 = a6;
          v87 = (v122 - 1) & (v10 + v87);
          a6 = v120.m128i_i64[1] + 136LL * v87;
          v88 = *(_QWORD *)a6;
          if ( *(_QWORD *)a6 == v103 )
            goto LABEL_144;
          v10 = (unsigned int)(v10 + 1);
        }
LABEL_192:
        if ( v89 )
          a6 = v89;
      }
LABEL_144:
      LODWORD(v121) = v81;
      if ( *(_QWORD *)a6 != -8 )
        --HIDWORD(v121);
      *(_QWORD *)a6 = v64;
      memset((void *)(a6 + 8), 0, 0x80u);
      v20 = 0;
      *(_QWORD *)(a6 + 8) = a6 + 24;
      *(_QWORD *)(a6 + 72) = a6 + 88;
      v82 = v114;
      *(_QWORD *)(a6 + 16) = 0x400000000LL;
      v118.m128i_i64[0] = v82;
      *(_QWORD *)(a6 + 80) = 0x400000000LL;
      v118.m128i_i32[2] = v108;
      v73 = 0;
      goto LABEL_129;
    }
    if ( v116 != v114 && (*(_BYTE *)(v63 + 16) & 5) != 0 )
    {
      v118.m128i_i64[0] = v114;
      v118.m128i_i64[1] = v116;
      v119 = 0x7FFFFFFFFFFFFFFFLL;
      v76 = *(unsigned int *)(v106 + 8);
      if ( (unsigned int)v76 >= *(_DWORD *)(v106 + 12) )
      {
        sub_16CD150(v106, v106 + 16, 0, 24);
        v76 = *(unsigned int *)(v106 + 8);
      }
      v77 = (__m128i *)(*(_QWORD *)v106 + 24 * v76);
      v78 = v119;
      *v77 = _mm_loadu_si128(&v118);
      v77[1].m128i_i64[0] = v78;
      ++*(_DWORD *)(v106 + 8);
    }
LABEL_106:
    v63 += 24;
    if ( v63 != v61 )
      goto LABEL_107;
LABEL_119:
    v19 = v105;
LABEL_62:
    v44 = v19 + 48;
    if ( v19 + 48 == v111 )
      break;
    while ( 2 )
    {
      v45 = *(_QWORD *)v44;
      v19 = v44;
      if ( *(_QWORD *)v44 == -8 )
      {
        if ( *(_DWORD *)(v44 + 8) != -1 )
          goto LABEL_59;
LABEL_65:
        v44 += 48;
        if ( v111 == v44 )
          goto LABEL_66;
        continue;
      }
      break;
    }
    if ( v45 == -16 && *(_DWORD *)(v44 + 8) == -2 )
      goto LABEL_65;
LABEL_59:
    if ( v111 != v44 )
    {
      sub_1381950((__int64)&v114, v45, *(_DWORD *)(v44 + 8), v6);
      continue;
    }
    break;
  }
LABEL_66:
  LODWORD(a3) = v121;
  v7 = v106;
  if ( (_DWORD)v121
    && (v46 = (__int64 *)v120.m128i_i64[1],
        LODWORD(a3) = 17 * v122,
        v20 = v120.m128i_i64[1] + 136LL * v122,
        v107 = v20,
        v120.m128i_i64[1] != v20) )
  {
    while ( 1 )
    {
      a3 = *v46;
      v10 = (__int64)v46;
      if ( *v46 != -16 && a3 != -8 )
        break;
      v46 += 17;
      if ( (__int64 *)v20 == v46 )
        goto LABEL_16;
    }
    v21 = *(unsigned int *)(v106 + 8);
    if ( v10 != v20 )
    {
      v47 = v10;
      do
      {
        v48 = *(unsigned int **)(v47 + 8);
        LODWORD(a3) = 3 * *(_DWORD *)(v47 + 16);
        v110 = &v48[3 * *(unsigned int *)(v47 + 16)];
        if ( v48 != v110 )
        {
          v49 = v47;
          v50 = *(unsigned int **)(v47 + 8);
          v51 = v49;
          do
          {
            v52 = *(unsigned int **)(v51 + 72);
            LODWORD(a3) = 3 * *(_DWORD *)(v51 + 80);
            v10 = (__int64)&v52[3 * *(unsigned int *)(v51 + 80)];
            if ( v52 != (unsigned int *)v10 )
            {
              v112 = v51;
              v53 = &v52[3 * *(unsigned int *)(v51 + 80)];
              do
              {
                v56 = v52[2];
                v57 = v50[2];
                if ( v56 != v57 )
                {
                  a6 = *v50;
                  v54 = v50[1];
                  v58 = *v52;
                  v59 = v52[1];
                  if ( v56 > v57 )
                    v54 = v56 + v54 - v57;
                  else
                    v59 = v57 + v59 - v56;
                  v118.m128i_i64[0] = __PAIR64__(v54, a6);
                  v118.m128i_i64[1] = __PAIR64__(v59, v58);
                  v119 = 0x7FFFFFFFFFFFFFFFLL;
                  if ( (unsigned int)v21 >= *(_DWORD *)(v106 + 12) )
                  {
                    sub_16CD150(v106, v106 + 16, 0, 24);
                    v21 = *(unsigned int *)(v106 + 8);
                  }
                  v55 = (__m128i *)(*(_QWORD *)v106 + 24 * v21);
                  a3 = v119;
                  *v55 = _mm_loadu_si128(&v118);
                  v55[1].m128i_i64[0] = a3;
                  v21 = (unsigned int)(*(_DWORD *)(v106 + 8) + 1);
                  *(_DWORD *)(v106 + 8) = v21;
                }
                v52 += 3;
              }
              while ( v53 != v52 );
              v51 = v112;
            }
            v50 += 3;
          }
          while ( v110 != v50 );
          v47 = v51;
        }
        v20 = v107;
        v47 += 136;
        if ( v47 == v107 )
          break;
        while ( 1 )
        {
          a3 = *(_QWORD *)v47;
          if ( *(_QWORD *)v47 != -8 && a3 != -16 )
            break;
          v47 += 136;
          if ( v107 == v47 )
            goto LABEL_17;
        }
      }
      while ( v107 != v47 );
    }
  }
  else
  {
LABEL_16:
    v21 = *(unsigned int *)(v7 + 8);
  }
LABEL_17:
  v22 = *(char **)v7;
  v23 = 24 * v21;
  v24 = (char *)(*(_QWORD *)v7 + 24 * v21);
  v25 = (__m128i *)v24;
  if ( *(char **)v7 != v24 )
  {
    v26 = (__m128i *)(*(_QWORD *)v7 + 24 * v21);
    _BitScanReverse64(&v27, 0xAAAAAAAAAAAAAAABLL * (v23 >> 3));
    sub_13876C0(*(_QWORD *)v7, v26, 2LL * (int)(63 - (v27 ^ 0x3F)), v20, v10, a6);
    if ( (unsigned __int64)v23 <= 0x180 )
    {
      sub_13820B0(v22, v24);
    }
    else
    {
      v28 = (unsigned int *)(v22 + 384);
      sub_13820B0(v22, v22 + 384);
      if ( v24 != v22 + 384 )
      {
        do
        {
          v29 = v28;
          v28 += 6;
          sub_1382000(v29);
        }
        while ( v24 != (char *)v28 );
        v24 = *(char **)v7;
        v25 = (__m128i *)(*(_QWORD *)v7 + 24LL * *(unsigned int *)(v7 + 8));
        goto LABEL_22;
      }
    }
    v24 = *(char **)v7;
    v25 = (__m128i *)(*(_QWORD *)v7 + 24LL * *(unsigned int *)(v7 + 8));
  }
LABEL_22:
  v30 = (__m128i *)v24;
  v31 = 0;
  v32 = sub_1383FE0(v30, v25, a3);
  v33 = *(_QWORD *)v7;
  v34 = v32;
  if ( v25 != (__m128i *)(*(_QWORD *)v7 + 24LL * *(unsigned int *)(v7 + 8)) )
  {
    v31 = *(_QWORD *)v7 + 24LL * *(unsigned int *)(v7 + 8) - (_QWORD)v25;
    v35 = (__m128i *)memmove(v32, v25, v31);
    v33 = *(_QWORD *)v7;
    v34 = v35;
  }
  *(_DWORD *)(v7 + 8) = -1431655765 * ((__int64)((__int64)v34->m128i_i64 + v31 - v33) >> 3);
  if ( v122 )
  {
    v36 = (_QWORD *)v120.m128i_i64[1];
    v37 = v120.m128i_i64[1] + 136LL * v122;
    do
    {
      if ( *v36 != -8 && *v36 != -16 )
      {
        v38 = v36[9];
        if ( (_QWORD *)v38 != v36 + 11 )
          _libc_free(v38);
        v39 = v36[1];
        if ( (_QWORD *)v39 != v36 + 3 )
          _libc_free(v39);
      }
      v36 += 17;
    }
    while ( (_QWORD *)v37 != v36 );
  }
  return j___libc_free_0(v120.m128i_i64[1]);
}
