// Function: sub_2ADE2D0
// Address: 0x2ade2d0
//
__int64 *__fastcall sub_2ADE2D0(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v3; // r13
  __int64 v4; // rdi
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // r12
  __int64 v8; // rax
  __int64 *v9; // r13
  _QWORD *v10; // rbx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 *v16; // r13
  __int64 *v17; // r14
  __int64 v18; // rax
  __int64 v19; // rcx
  __int64 v20; // r13
  __int64 v21; // r14
  __int64 v22; // rbx
  char v23; // al
  __int64 v24; // rcx
  __int64 *v25; // rbx
  __int64 *v26; // r14
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // r9
  __int64 v32; // r12
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rdx
  __int64 v36; // r12
  __int64 *i; // rax
  unsigned int v38; // ebx
  __int64 v39; // rax
  _BYTE **v40; // rdx
  _BYTE *v41; // rsi
  __int64 v42; // r13
  __int64 v43; // rdx
  __int64 j; // r13
  __int64 v45; // r15
  __int64 v46; // rdi
  __int64 v47; // rsi
  _QWORD *v48; // rax
  __int64 *v49; // rsi
  int v50; // eax
  bool v51; // al
  __int64 v52; // rsi
  _DWORD *v53; // rbx
  __int64 v54; // r12
  __int64 v55; // r13
  __int64 v56; // rax
  __int64 v57; // rdi
  __int64 v58; // rdx
  __int64 v59; // rax
  __int64 v60; // r10
  __int64 v61; // r15
  unsigned __int8 *v62; // rbx
  __int64 v63; // rdi
  __int64 v64; // rsi
  _QWORD *v65; // rax
  __int64 *v66; // rsi
  __int64 v67; // r15
  unsigned __int8 *v68; // rbx
  __int64 v69; // rdi
  __int64 v70; // rsi
  _QWORD *v71; // rax
  __int64 *v72; // rsi
  __int64 v73; // rdx
  __int64 v74; // rcx
  __int64 v75; // r8
  __int64 v76; // r9
  unsigned int v77; // eax
  __int64 v78; // rsi
  int v79; // edi
  unsigned int v80; // eax
  unsigned __int8 *v81; // rsi
  int v82; // edi
  int v83; // eax
  unsigned int v84; // eax
  unsigned __int8 *v85; // rsi
  int v86; // edi
  int v87; // eax
  bool v88; // al
  __int64 v89; // rax
  __int64 *v90; // r14
  __int64 v91; // r12
  __int64 *v92; // r13
  __int64 v93; // rdx
  __int64 *result; // rax
  __int64 v95; // rbx
  __int64 v96; // rax
  __int64 v97; // rdx
  __int64 v98; // r12
  __int64 v99; // r13
  __m128i v100; // rax
  __m128i v101; // xmm0
  __m128i v102; // xmm1
  __int64 v103; // [rsp+0h] [rbp-260h]
  __int64 v104; // [rsp+18h] [rbp-248h]
  unsigned __int8 v105; // [rsp+18h] [rbp-248h]
  unsigned __int8 v106; // [rsp+18h] [rbp-248h]
  __int64 v107; // [rsp+30h] [rbp-230h]
  _DWORD *v108; // [rsp+30h] [rbp-230h]
  __int64 v109; // [rsp+40h] [rbp-220h]
  __int64 v110; // [rsp+40h] [rbp-220h]
  __int64 v111; // [rsp+40h] [rbp-220h]
  _DWORD *v112; // [rsp+40h] [rbp-220h]
  __int64 v113; // [rsp+48h] [rbp-218h] BYREF
  __int64 v114; // [rsp+50h] [rbp-210h] BYREF
  __int64 v115; // [rsp+58h] [rbp-208h] BYREF
  __int64 *v116; // [rsp+60h] [rbp-200h] BYREF
  __int64 v117; // [rsp+68h] [rbp-1F8h]
  __int64 *v118; // [rsp+70h] [rbp-1F0h] BYREF
  __int64 v119; // [rsp+78h] [rbp-1E8h]
  _QWORD *v120; // [rsp+80h] [rbp-1E0h]
  __int64 v121; // [rsp+88h] [rbp-1D8h]
  _QWORD v122[6]; // [rsp+90h] [rbp-1D0h] BYREF
  __int64 v123[6]; // [rsp+C0h] [rbp-1A0h] BYREF
  __m128i v124; // [rsp+F0h] [rbp-170h] BYREF
  __m128i v125; // [rsp+100h] [rbp-160h] BYREF
  char v126; // [rsp+110h] [rbp-150h] BYREF
  __m128i v127; // [rsp+150h] [rbp-110h] BYREF
  __m128i v128; // [rsp+160h] [rbp-100h]
  __int64 *v129; // [rsp+170h] [rbp-F0h]
  __int64 v130; // [rsp+178h] [rbp-E8h]
  _BYTE v131[64]; // [rsp+180h] [rbp-E0h] BYREF
  __int64 v132; // [rsp+1C0h] [rbp-A0h] BYREF
  __int64 v133; // [rsp+1C8h] [rbp-98h]
  __int64 v134; // [rsp+1D0h] [rbp-90h]
  __int64 v135; // [rsp+1D8h] [rbp-88h]
  __int64 *v136; // [rsp+1E0h] [rbp-80h]
  __int64 v137; // [rsp+1E8h] [rbp-78h]
  _BYTE v138[112]; // [rsp+1F0h] [rbp-70h] BYREF

  v2 = a1;
  v3 = a1 + 160;
  v113 = a2;
  if ( !BYTE4(a2) )
  {
    v4 = *(_QWORD *)(a1 + 416);
    v127 = 0u;
    v129 = (__int64 *)v131;
    v130 = 0x800000000LL;
    v137 = 0x800000000LL;
    v136 = (__int64 *)v138;
    v128 = 0u;
    v132 = 0;
    v133 = 0;
    v134 = 0;
    v135 = 0;
    v124.m128i_i64[0] = 0;
    v124.m128i_i64[1] = (__int64)&v126;
    v125.m128i_i64[0] = 8;
    v125.m128i_i32[2] = 0;
    v125.m128i_i8[12] = 1;
    v117 = v2;
    v104 = sub_D47930(v4);
    v122[0] = &v114;
    v116 = &v113;
    v114 = v2;
    v122[1] = &v127;
    v122[2] = &v116;
    v122[3] = &v132;
    v122[4] = &v124;
    v5 = sub_2ACAB60(v3, (__int64)&v113);
    if ( *(_BYTE *)(v5 + 28) )
      v6 = *(unsigned int *)(v5 + 20);
    else
      v6 = *(unsigned int *)(v5 + 16);
    v7 = *(_QWORD *)(v5 + 8) + 8 * v6;
    v8 = sub_2ACAB60(v3, (__int64)&v113);
    v9 = *(__int64 **)(v8 + 8);
    v10 = (_QWORD *)v8;
    v11 = sub_254BB00(v8);
    v118 = v9;
    v119 = v11;
    sub_254BBF0((__int64)&v118);
    v120 = v10;
    v16 = v118;
    v121 = *v10;
    if ( v118 != (__int64 *)v7 )
    {
      v109 = v2;
      v17 = (__int64 *)v119;
      do
      {
        v123[0] = *v16;
        sub_2ADDD60((__int64)&v127, v123, v123[0], v13, v14, v15);
        do
          ++v16;
        while ( v17 != v16 && (unsigned __int64)*v16 >= 0xFFFFFFFFFFFFFFFELL );
      }
      while ( v16 != (__int64 *)v7 );
      v2 = v109;
    }
    v18 = *(_QWORD *)(v2 + 416);
    v19 = *(_QWORD *)(v18 + 40);
    v20 = *(_QWORD *)(v18 + 32);
    v110 = v19;
    if ( v19 == v20 )
    {
LABEL_23:
      v25 = &v136[(unsigned int)v137];
      if ( v25 != v136 )
      {
        v111 = v2;
        v26 = v136;
        do
        {
          while ( 1 )
          {
            v123[0] = *v26;
            if ( !(unsigned __int8)sub_B19060((__int64)&v124, v123[0], v12, v19) )
              break;
            if ( v25 == ++v26 )
              goto LABEL_28;
          }
          ++v26;
          sub_2ADDD60((__int64)&v127, v123, v12, v19, v27, v28);
        }
        while ( v25 != v26 );
LABEL_28:
        v2 = v111;
      }
      v32 = sub_2ABFD00(v2 + 224, (__int64)&v113);
      if ( v32 && v32 != *(_QWORD *)(v2 + 232) + 72LL * *(unsigned int *)(v2 + 248) )
      {
        v33 = *(_QWORD *)(v32 + 16);
        v34 = *(_BYTE *)(v32 + 36) ? v33 + 8LL * *(unsigned int *)(v32 + 28) : v33 + 8LL * *(unsigned int *)(v32 + 24);
        v123[0] = *(_QWORD *)(v32 + 16);
        v123[1] = v34;
        sub_254BBF0((__int64)v123);
        v123[2] = v32 + 8;
        v123[3] = *(_QWORD *)(v32 + 8);
        v35 = *(_BYTE *)(v32 + 36) ? *(unsigned int *)(v32 + 28) : *(unsigned int *)(v32 + 24);
        v36 = *(_QWORD *)(v32 + 16) + 8 * v35;
        for ( i = (__int64 *)v123[0]; v123[0] != v36; i = (__int64 *)v123[0] )
        {
          v115 = *i;
          sub_2ADDD60((__int64)&v127, &v115, v35, v29, v30, v31);
          v123[0] += 8;
          sub_254BBF0((__int64)v123);
        }
      }
      if ( (_DWORD)v130 )
      {
        v38 = 0;
        v39 = 0;
        do
        {
          while ( 1 )
          {
            ++v38;
            v42 = v129[v39];
            if ( (*(_BYTE *)(v42 + 7) & 0x40) == 0 )
              break;
            v40 = *(_BYTE ***)(v42 - 8);
            v41 = *v40;
            if ( **v40 == 63 )
              goto LABEL_43;
LABEL_40:
            v39 = v38;
            if ( v38 == (_DWORD)v130 )
              goto LABEL_59;
          }
          v40 = (_BYTE **)(v42 - 32LL * (*(_DWORD *)(v42 + 4) & 0x7FFFFFF));
          v41 = *v40;
          if ( **v40 != 63 )
            goto LABEL_40;
LABEL_43:
          if ( (unsigned __int8)sub_D48480(*(_QWORD *)(v114 + 416), (__int64)v41, (__int64)v40, v29) )
            goto LABEL_40;
          v115 = *(_QWORD *)sub_986520(v42);
          for ( j = *(_QWORD *)(v115 + 16); j; j = *(_QWORD *)(j + 8) )
          {
            v45 = *(_QWORD *)(j + 24);
            v46 = *(_QWORD *)(v2 + 416);
            v123[0] = v45;
            v47 = *(_QWORD *)(v45 + 40);
            if ( *(_BYTE *)(v46 + 84) )
            {
              v48 = *(_QWORD **)(v46 + 64);
              v43 = (__int64)&v48[*(unsigned int *)(v46 + 76)];
              if ( v48 == (_QWORD *)v43 )
                continue;
              while ( v47 != *v48 )
              {
                if ( (_QWORD *)v43 == ++v48 )
                  goto LABEL_57;
              }
              if ( !v128.m128i_i32[0] )
                goto LABEL_51;
            }
            else
            {
              if ( !sub_C8CA60(v46 + 56, v47) )
                continue;
              if ( !v128.m128i_i32[0] )
              {
LABEL_51:
                v49 = &v129[(unsigned int)v130];
                if ( v49 == sub_2AA81A0(v129, (__int64)v49, v123) )
                  goto LABEL_52;
                continue;
              }
            }
            v29 = v127.m128i_i64[1];
            if ( !v128.m128i_i32[2] )
              goto LABEL_52;
            v43 = (unsigned int)(v128.m128i_i32[2] - 1);
            v77 = v43 & (((unsigned int)v45 >> 9) ^ ((unsigned int)v45 >> 4));
            v78 = *(_QWORD *)(v127.m128i_i64[1] + 8LL * v77);
            if ( v45 != v78 )
            {
              v79 = 1;
              while ( v78 != -4096 )
              {
                v30 = (unsigned int)(v79 + 1);
                v77 = v43 & (v79 + v77);
                v78 = *(_QWORD *)(v127.m128i_i64[1] + 8LL * v77);
                if ( v45 == v78 )
                  goto LABEL_57;
                ++v79;
              }
LABEL_52:
              if ( *(_BYTE *)v45 != 61 )
              {
                if ( *(_BYTE *)v45 != 62 )
                  goto LABEL_40;
                v50 = sub_2AAA2B0(v117, v45, *(_DWORD *)v116, *((_BYTE *)v116 + 4));
                v29 = *(_QWORD *)(v45 - 64);
                if ( v115 == v29 )
                  v51 = v50 == 5;
                else
LABEL_55:
                  v51 = v50 != 4;
                if ( !v51 )
                  goto LABEL_40;
                continue;
              }
              v50 = sub_2AAA2B0(v117, v45, *(_DWORD *)v116, *((_BYTE *)v116 + 4));
              goto LABEL_55;
            }
LABEL_57:
            ;
          }
          sub_2ADDD60((__int64)&v127, &v115, v43, v29, v30, v31);
          v39 = v38;
        }
        while ( v38 != (_DWORD)v130 );
      }
LABEL_59:
      v52 = *(_QWORD *)(v2 + 440);
      v53 = *(_DWORD **)(v52 + 160);
      v108 = &v53[22 * *(unsigned int *)(v52 + 168)];
      if ( v108 == v53 )
        goto LABEL_134;
      v112 = *(_DWORD **)(v52 + 160);
      v54 = v104;
      while ( 1 )
      {
        v55 = *(_QWORD *)v112;
        v56 = 0x1FFFFFFFE0LL;
        v57 = *(_QWORD *)(*(_QWORD *)v112 - 8LL);
        v58 = *(_DWORD *)(*(_QWORD *)v112 + 4LL) & 0x7FFFFFF;
        if ( (*(_DWORD *)(*(_QWORD *)v112 + 4LL) & 0x7FFFFFF) != 0 )
        {
          v59 = 0;
          v29 = v57 + 32LL * *(unsigned int *)(v55 + 72);
          do
          {
            if ( v54 == *(_QWORD *)(v29 + 8 * v59) )
            {
              v56 = 32 * v59;
              goto LABEL_66;
            }
            ++v59;
          }
          while ( (_DWORD)v58 != (_DWORD)v59 );
          v56 = 0x1FFFFFFFE0LL;
        }
LABEL_66:
        v60 = *(_QWORD *)(v57 + v56);
        v115 = v60;
        if ( v55 != *(_QWORD *)(v52 + 72) )
          break;
        if ( !*(_BYTE *)(v2 + 108) )
          break;
        v29 = *(unsigned int *)(v2 + 100);
        if ( !(_DWORD)v29 )
          break;
LABEL_92:
        v112 += 22;
        if ( v108 == v112 )
        {
LABEL_134:
          v89 = sub_2ACAB60(v2 + 192, (__int64)&v113);
          v90 = v129;
          v91 = v89;
          v92 = &v129[(unsigned int)v130];
          if ( v129 != v92 )
          {
            do
            {
              v93 = *v90++;
              sub_BED950((__int64)v123, v91, v93);
            }
            while ( v92 != v90 );
          }
          if ( !v125.m128i_i8[12] )
            _libc_free(v124.m128i_u64[1]);
          if ( v136 != (__int64 *)v138 )
            _libc_free((unsigned __int64)v136);
          sub_C7D6A0(v133, 8LL * (unsigned int)v135, 8);
          if ( v129 != (__int64 *)v131 )
            _libc_free((unsigned __int64)v129);
          return (__int64 *)sub_C7D6A0(v127.m128i_i64[1], 8LL * v128.m128i_u32[2], 8);
        }
        v52 = *(_QWORD *)(v2 + 440);
      }
      v61 = *(_QWORD *)(v55 + 16);
      if ( v61 )
      {
        while ( 1 )
        {
          v62 = *(unsigned __int8 **)(v61 + 24);
          v123[0] = (__int64)v62;
          if ( v62 != (unsigned __int8 *)v60 )
            break;
LABEL_77:
          v61 = *(_QWORD *)(v61 + 8);
          if ( !v61 )
            goto LABEL_78;
        }
        v63 = *(_QWORD *)(v2 + 416);
        v64 = *((_QWORD *)v62 + 5);
        if ( *(_BYTE *)(v63 + 84) )
        {
          v65 = *(_QWORD **)(v63 + 64);
          v58 = (__int64)&v65[*(unsigned int *)(v63 + 76)];
          if ( v65 == (_QWORD *)v58 )
            goto LABEL_77;
          while ( v64 != *v65 )
          {
            if ( (_QWORD *)v58 == ++v65 )
              goto LABEL_77;
          }
          if ( !v128.m128i_i32[0] )
          {
LABEL_75:
            v66 = &v129[(unsigned int)v130];
            if ( v66 != sub_2AA81A0(v129, (__int64)v66, v123) )
            {
LABEL_76:
              v60 = v115;
              goto LABEL_77;
            }
LABEL_110:
            if ( v112[8] != 2 )
              goto LABEL_92;
            v30 = *v62;
            v105 = *v62;
            if ( (unsigned __int8)(v30 - 61) > 1u || v55 != sub_228AED0(v62) )
              goto LABEL_92;
            v83 = sub_2AAA2B0(v117, (__int64)v62, *(_DWORD *)v116, *((_BYTE *)v116 + 4));
            v30 = v105;
            if ( v105 == 62 && (v58 = *((_QWORD *)v62 - 8), v55 == v58) && v58 )
            {
              if ( v83 != 5 )
                goto LABEL_92;
            }
            else if ( v83 == 4 )
            {
              goto LABEL_92;
            }
            goto LABEL_76;
          }
        }
        else
        {
          if ( !sub_C8CA60(v63 + 56, v64) )
            goto LABEL_76;
          if ( !v128.m128i_i32[0] )
            goto LABEL_75;
        }
        v29 = v127.m128i_i64[1];
        if ( v128.m128i_i32[2] )
        {
          v58 = (unsigned int)(v128.m128i_i32[2] - 1);
          v80 = v58 & (((unsigned int)v62 >> 9) ^ ((unsigned int)v62 >> 4));
          v81 = *(unsigned __int8 **)(v127.m128i_i64[1] + 8LL * v80);
          if ( v62 == v81 )
            goto LABEL_76;
          v82 = 1;
          while ( v81 != (unsigned __int8 *)-4096LL )
          {
            v30 = (unsigned int)(v82 + 1);
            v80 = v58 & (v82 + v80);
            v81 = *(unsigned __int8 **)(v127.m128i_i64[1] + 8LL * v80);
            if ( v62 == v81 )
              goto LABEL_76;
            ++v82;
          }
        }
        goto LABEL_110;
      }
LABEL_78:
      if ( *(_BYTE *)v60 == 84 )
      {
        if ( (unsigned __int8)sub_31A6BC0(*(_QWORD *)(v2 + 440), v60) )
          goto LABEL_92;
        v60 = v115;
      }
      v67 = *(_QWORD *)(v60 + 16);
      if ( !v67 )
      {
LABEL_91:
        v123[0] = v55;
        sub_2ADDD60((__int64)&v127, v123, v58, v29, v30, v31);
        sub_2ADDD60((__int64)&v127, &v115, v73, v74, v75, v76);
        goto LABEL_92;
      }
      while ( 1 )
      {
        v68 = *(unsigned __int8 **)(v67 + 24);
        v123[0] = (__int64)v68;
        if ( (unsigned __int8 *)v55 == v68 )
          goto LABEL_90;
        v69 = *(_QWORD *)(v2 + 416);
        v70 = *((_QWORD *)v68 + 5);
        if ( *(_BYTE *)(v69 + 84) )
        {
          v71 = *(_QWORD **)(v69 + 64);
          v58 = (__int64)&v71[*(unsigned int *)(v69 + 76)];
          if ( v71 == (_QWORD *)v58 )
            goto LABEL_90;
          while ( v70 != *v71 )
          {
            if ( (_QWORD *)v58 == ++v71 )
              goto LABEL_90;
          }
          if ( !v128.m128i_i32[0] )
            goto LABEL_89;
        }
        else
        {
          if ( !sub_C8CA60(v69 + 56, v70) )
            goto LABEL_90;
          if ( !v128.m128i_i32[0] )
          {
LABEL_89:
            v72 = &v129[(unsigned int)v130];
            if ( v72 != sub_2AA81A0(v129, (__int64)v72, v123) )
              goto LABEL_90;
            goto LABEL_124;
          }
        }
        v29 = v127.m128i_i64[1];
        if ( v128.m128i_i32[2] )
        {
          v58 = (unsigned int)(v128.m128i_i32[2] - 1);
          v84 = v58 & (((unsigned int)v68 >> 9) ^ ((unsigned int)v68 >> 4));
          v85 = *(unsigned __int8 **)(v127.m128i_i64[1] + 8LL * v84);
          if ( v85 == v68 )
            goto LABEL_90;
          v86 = 1;
          while ( v85 != (unsigned __int8 *)-4096LL )
          {
            v30 = (unsigned int)(v86 + 1);
            v84 = v58 & (v86 + v84);
            v85 = *(unsigned __int8 **)(v127.m128i_i64[1] + 8LL * v84);
            if ( v68 == v85 )
              goto LABEL_90;
            ++v86;
          }
        }
LABEL_124:
        if ( v112[8] != 2 )
          goto LABEL_92;
        v30 = *v68;
        v106 = *v68;
        if ( (unsigned __int8)(v30 - 61) > 1u )
          goto LABEL_92;
        v103 = v115;
        if ( v103 != sub_228AED0(v68) )
          goto LABEL_92;
        v87 = sub_2AAA2B0(v117, (__int64)v68, *(_DWORD *)v116, *((_BYTE *)v116 + 4));
        v30 = v106;
        if ( v106 == 62 && (v29 = v103, v103 == *((_QWORD *)v68 - 8)) )
          v88 = v87 == 5;
        else
          v88 = v87 != 4;
        if ( !v88 )
          goto LABEL_92;
LABEL_90:
        v67 = *(_QWORD *)(v67 + 8);
        if ( !v67 )
          goto LABEL_91;
      }
    }
    v107 = v2;
    while ( 1 )
    {
      v21 = *(_QWORD *)(*(_QWORD *)v20 + 56LL);
      v22 = *(_QWORD *)v20 + 48LL;
      if ( v21 != v22 )
        break;
LABEL_21:
      v20 += 8;
      if ( v110 == v20 )
      {
        v2 = v107;
        goto LABEL_23;
      }
    }
    while ( 1 )
    {
      while ( 1 )
      {
        if ( !v21 )
          BUG();
        v23 = *(_BYTE *)(v21 - 24);
        if ( v23 != 61 )
          break;
        sub_2ADE150((__int64)v122, (_BYTE *)(v21 - 24), *(_QWORD *)(v21 - 56), v19);
LABEL_16:
        v21 = *(_QWORD *)(v21 + 8);
        if ( v22 == v21 )
          goto LABEL_21;
      }
      if ( v23 != 62 )
        goto LABEL_16;
      sub_2ADE150((__int64)v122, (_BYTE *)(v21 - 24), *(_QWORD *)(v21 - 56), v19);
      sub_2ADE150((__int64)v122, (_BYTE *)(v21 - 24), *(_QWORD *)(v21 - 88), v24);
      v21 = *(_QWORD *)(v21 + 8);
      if ( v22 == v21 )
        goto LABEL_21;
    }
  }
  v95 = sub_2ACAB60(a1 + 192, (__int64)&v113);
  v96 = sub_2ACAB60(v3, (__int64)&v113);
  if ( *(_BYTE *)(v96 + 28) )
    v97 = *(unsigned int *)(v96 + 20);
  else
    v97 = *(unsigned int *)(v96 + 16);
  v98 = *(_QWORD *)(v96 + 8) + 8 * v97;
  v99 = sub_2ACAB60(v3, (__int64)&v113);
  v100.m128i_i64[0] = *(_QWORD *)(v99 + 8);
  if ( *(_BYTE *)(v99 + 28) )
    v100.m128i_i64[1] = v100.m128i_i64[0] + 8LL * *(unsigned int *)(v99 + 20);
  else
    v100.m128i_i64[1] = v100.m128i_i64[0] + 8LL * *(unsigned int *)(v99 + 16);
  v124 = v100;
  sub_254BBF0((__int64)&v124);
  v125.m128i_i64[0] = v99;
  v101 = _mm_loadu_si128(&v124);
  v125.m128i_i64[1] = *(_QWORD *)v99;
  result = (__int64 *)v124.m128i_i64[0];
  v102 = _mm_loadu_si128(&v125);
  v127 = v101;
  v128 = v102;
  if ( v124.m128i_i64[0] != v98 )
  {
    do
    {
      sub_BED950((__int64)&v132, v95, *result);
      v127.m128i_i64[0] += 8;
      sub_254BBF0((__int64)&v127);
      result = (__int64 *)v127.m128i_i64[0];
    }
    while ( v127.m128i_i64[0] != v98 );
  }
  return result;
}
