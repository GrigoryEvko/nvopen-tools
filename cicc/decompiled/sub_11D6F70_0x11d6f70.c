// Function: sub_11D6F70
// Address: 0x11d6f70
//
__int64 __fastcall sub_11D6F70(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v3; // r13
  __int64 v4; // r12
  __int64 v5; // r15
  __int64 v6; // rbx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // r10
  __int64 v10; // rax
  unsigned __int64 v11; // rdx
  __int64 *v12; // rax
  unsigned int v13; // eax
  __int64 v14; // r12
  __int64 v15; // r15
  __int64 v17; // r14
  __int64 v18; // rdx
  char v19; // bl
  __int64 v20; // r15
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 v23; // rax
  unsigned __int64 v24; // rdx
  __int64 *v25; // rax
  __int64 v26; // rax
  __int64 v27; // r14
  __int64 v28; // rax
  __int64 v29; // rbx
  __int64 *v30; // r15
  __int64 v31; // rsi
  __int64 *v32; // r13
  int v33; // eax
  unsigned int v34; // esi
  __int64 v35; // rax
  __int64 v36; // rsi
  __int64 v37; // rsi
  __int64 v38; // r14
  __int64 v39; // r12
  int v40; // eax
  __int64 v41; // rdx
  __int64 v42; // rcx
  __int64 v43; // r8
  __int64 v44; // r9
  __int64 v45; // r8
  __int64 v46; // r9
  __int64 v47; // r12
  __int64 v48; // rsi
  __int64 v49; // rsi
  __m128i *v50; // r12
  __int64 v51; // r12
  __int64 v52; // rax
  __int64 *v53; // rbx
  __int64 v54; // rax
  __int64 *v55; // r14
  unsigned __int64 v56; // r15
  unsigned __int64 v57; // r12
  __int64 v58; // rcx
  __int64 *v59; // rdx
  __int64 *v60; // rax
  int v61; // r8d
  __int64 *v62; // rdi
  unsigned int v63; // edx
  __int64 *v64; // rax
  __int64 v65; // r11
  int v66; // ecx
  unsigned __int32 v67; // eax
  __int64 *v68; // r9
  unsigned __int32 v69; // edx
  unsigned int v70; // edi
  __int64 v71; // rdx
  __int64 v72; // r15
  __int64 v73; // rax
  __int64 v74; // rdi
  __int64 v75; // r14
  __int64 v76; // r12
  __int64 *v77; // r9
  unsigned int v78; // edx
  __int64 *v79; // rax
  __int64 v80; // r10
  __int64 v81; // r13
  unsigned __int32 v82; // eax
  __int64 *v83; // r8
  unsigned __int32 v84; // edx
  unsigned int v85; // edi
  __int64 v86; // rax
  int v87; // r11d
  int v88; // ecx
  __int64 v89; // rdx
  __int64 v90; // rdi
  __int64 *v91; // rax
  int v92; // r9d
  int v93; // ecx
  __int64 v94; // rdx
  __int64 v95; // rdi
  int v96; // r9d
  int v97; // r12d
  __int64 v98; // rsi
  unsigned __int8 *v99; // rsi
  __int64 v100; // rax
  int v101; // ecx
  __int64 *v102; // r10
  unsigned int v103; // edx
  __int64 v104; // rdi
  __int64 *v105; // rax
  int v106; // ecx
  __int64 *v107; // r10
  unsigned int v108; // edx
  __int64 v109; // rdi
  unsigned int v110; // r11d
  unsigned int v111; // r11d
  __int64 v112; // [rsp+8h] [rbp-188h]
  __int64 v113; // [rsp+10h] [rbp-180h]
  __int64 v114; // [rsp+10h] [rbp-180h]
  __int64 v115; // [rsp+18h] [rbp-178h]
  int v116; // [rsp+28h] [rbp-168h]
  _QWORD *v117; // [rsp+28h] [rbp-168h]
  __int64 v118; // [rsp+28h] [rbp-168h]
  __int64 v119; // [rsp+38h] [rbp-158h] BYREF
  __int64 *v120; // [rsp+40h] [rbp-150h] BYREF
  __int64 v121; // [rsp+48h] [rbp-148h]
  _BYTE v122[128]; // [rsp+50h] [rbp-140h] BYREF
  __m128i v123; // [rsp+D0h] [rbp-C0h] BYREF
  __int64 *v124; // [rsp+E0h] [rbp-B0h] BYREF
  __int64 v125; // [rsp+E8h] [rbp-A8h]
  __int64 v126; // [rsp+F0h] [rbp-A0h]
  __int64 v127; // [rsp+F8h] [rbp-98h]
  __int64 v128; // [rsp+100h] [rbp-90h]
  __int64 v129; // [rsp+108h] [rbp-88h]
  __int16 v130; // [rsp+110h] [rbp-80h]
  char v131; // [rsp+160h] [rbp-30h] BYREF

  v2 = *(_QWORD *)(a2 + 56);
  v120 = (__int64 *)v122;
  v115 = a2;
  v121 = 0x800000000LL;
  if ( !v2 )
    BUG();
  v3 = a1;
  if ( *(_BYTE *)(v2 - 24) == 84 )
  {
    v4 = 0;
    v116 = *(_DWORD *)(v2 - 20) & 0x7FFFFFF;
    if ( v116 )
    {
      v5 = 0;
      do
      {
        while ( 1 )
        {
          v6 = *(_QWORD *)(*(_QWORD *)(v2 - 32) + 32LL * *(unsigned int *)(v2 + 48) + 8 * v4);
          a2 = v6;
          v9 = sub_11D6F60((__int64 *)a1, v6);
          v10 = (unsigned int)v121;
          v11 = (unsigned int)v121 + 1LL;
          if ( v11 > HIDWORD(v121) )
          {
            a2 = (__int64)v122;
            v112 = v9;
            sub_C8D5F0((__int64)&v120, v122, v11, 0x10u, v7, v8);
            v10 = (unsigned int)v121;
            v9 = v112;
          }
          v12 = &v120[2 * v10];
          *v12 = v6;
          v12[1] = v9;
          v13 = v121 + 1;
          LODWORD(v121) = v121 + 1;
          if ( v4 )
            break;
          v5 = v9;
          v4 = 1;
          if ( v116 == 1 )
            goto LABEL_12;
        }
        if ( v9 != v5 )
          v5 = 0;
        ++v4;
      }
      while ( v116 != (_DWORD)v4 );
LABEL_12:
      v14 = v5;
      v15 = v13;
      goto LABEL_13;
    }
LABEL_31:
    v14 = sub_ACADE0(*(__int64 ***)(a1 + 8));
    goto LABEL_15;
  }
  v17 = *(_QWORD *)(a2 + 16);
  if ( !v17 )
    goto LABEL_31;
  while ( 1 )
  {
    v18 = *(_QWORD *)(v17 + 24);
    if ( (unsigned __int8)(*(_BYTE *)v18 - 30) <= 0xAu )
      break;
    v17 = *(_QWORD *)(v17 + 8);
    if ( !v17 )
    {
      v14 = sub_ACADE0(*(__int64 ***)(a1 + 8));
      goto LABEL_15;
    }
  }
  v14 = 0;
  v19 = 1;
LABEL_21:
  v20 = *(_QWORD *)(v18 + 40);
  a2 = v20;
  v21 = sub_11D6F60((__int64 *)a1, v20);
  v23 = (unsigned int)v121;
  v24 = (unsigned int)v121 + 1LL;
  if ( v24 > HIDWORD(v121) )
  {
    a2 = (__int64)v122;
    v113 = v21;
    sub_C8D5F0((__int64)&v120, v122, v24, 0x10u, v21, v22);
    v23 = (unsigned int)v121;
    v21 = v113;
  }
  v25 = &v120[2 * v23];
  *v25 = v20;
  v25[1] = v21;
  v15 = (unsigned int)(v121 + 1);
  LODWORD(v121) = v121 + 1;
  if ( v19 )
  {
    v14 = v21;
  }
  else if ( v21 != v14 )
  {
    v14 = 0;
  }
  v17 = *(_QWORD *)(v17 + 8);
  if ( v17 )
  {
    v19 = 0;
    do
    {
      v18 = *(_QWORD *)(v17 + 24);
      if ( (unsigned __int8)(*(_BYTE *)v18 - 30) <= 0xAu )
        goto LABEL_21;
      v17 = *(_QWORD *)(v17 + 8);
    }
    while ( v17 );
    if ( (_DWORD)v15 )
      goto LABEL_14;
    goto LABEL_31;
  }
LABEL_13:
  if ( !(_DWORD)v15 )
    goto LABEL_31;
LABEL_14:
  if ( v14 )
    goto LABEL_15;
  v26 = *(_QWORD *)(v115 + 56);
  if ( !v26 )
    BUG();
  if ( *(_BYTE *)(v26 - 24) != 84 )
    goto LABEL_37;
  v53 = v120;
  v123.m128i_i64[0] = 0;
  v54 = 16 * v15;
  v55 = &v120[2 * v15];
  v56 = (unsigned __int64)(16 * v15) >> 5;
  v57 = ((((((((((v54 >> 4) | v56) >> 2) | (v54 >> 4) | v56) >> 4) | (((v54 >> 4) | v56) >> 2) | (v54 >> 4) | v56) >> 8)
         | (((((v54 >> 4) | v56) >> 2) | (v54 >> 4) | v56) >> 4)
         | (((v54 >> 4) | v56) >> 2)
         | (v54 >> 4)
         | v56) >> 16)
       | (((((((v54 >> 4) | v56) >> 2) | (v54 >> 4) | v56) >> 4) | (((v54 >> 4) | v56) >> 2) | (v54 >> 4) | v56) >> 8)
       | (((((v54 >> 4) | v56) >> 2) | (v54 >> 4) | v56) >> 4)
       | (((v54 >> 4) | v56) >> 2)
       | (v54 >> 4)
       | v56)
      + 1;
  if ( (unsigned int)v57 > 8 )
  {
    v123.m128i_i8[8] &= ~1u;
    v100 = sub_C7D670(16LL * (unsigned int)v57, 8);
    LODWORD(v125) = v57;
    v124 = (__int64 *)v100;
  }
  else
  {
    v123.m128i_i8[8] |= 1u;
  }
  v58 = v123.m128i_i8[8] & 1;
  v123.m128i_i64[1] = v58;
  if ( (_BYTE)v58 )
  {
    v59 = (__int64 *)&v131;
    v60 = (__int64 *)&v124;
  }
  else
  {
    v60 = v124;
    v59 = &v124[2 * (unsigned int)v125];
    if ( v124 == v59 )
      goto LABEL_83;
  }
  do
  {
    if ( v60 )
      *v60 = -4096;
    v60 += 2;
  }
  while ( v59 != v60 );
  for ( LOBYTE(v58) = v123.m128i_i8[8]; ; LOBYTE(v58) = v123.m128i_i8[8] )
  {
LABEL_83:
    v66 = v58 & 1;
    if ( v66 )
    {
      v61 = 7;
      v62 = (__int64 *)&v124;
    }
    else
    {
      a2 = (unsigned int)v125;
      v62 = v124;
      v61 = v125 - 1;
      if ( !(_DWORD)v125 )
      {
        v67 = v123.m128i_u32[2];
        ++v123.m128i_i64[0];
        v68 = 0;
        v69 = ((unsigned __int32)v123.m128i_i32[2] >> 1) + 1;
LABEL_86:
        v70 = 3 * a2;
        goto LABEL_87;
      }
    }
    a2 = *v53;
    v63 = v61 & (((unsigned int)*v53 >> 9) ^ ((unsigned int)*v53 >> 4));
    v64 = &v62[2 * v63];
    v65 = *v64;
    if ( *v53 == *v64 )
      goto LABEL_81;
    v68 = 0;
    v97 = 1;
    while ( v65 != -4096 )
    {
      if ( !v68 && v65 == -8192 )
        v68 = v64;
      v63 = v61 & (v97 + v63);
      v64 = &v62[2 * v63];
      v65 = *v64;
      if ( a2 == *v64 )
        goto LABEL_81;
      ++v97;
    }
    if ( !v68 )
      v68 = v64;
    v67 = v123.m128i_u32[2];
    ++v123.m128i_i64[0];
    v69 = ((unsigned __int32)v123.m128i_i32[2] >> 1) + 1;
    if ( !(_BYTE)v66 )
    {
      a2 = (unsigned int)v125;
      goto LABEL_86;
    }
    v70 = 24;
    a2 = 8;
LABEL_87:
    if ( v70 <= 4 * v69 )
    {
      a2 = (unsigned int)(2 * a2);
      sub_FADF20((__int64)&v123, a2);
      if ( (v123.m128i_i8[8] & 1) != 0 )
      {
        v101 = 7;
        v102 = (__int64 *)&v124;
      }
      else
      {
        v102 = v124;
        if ( !(_DWORD)v125 )
          goto LABEL_214;
        v101 = v125 - 1;
      }
      v67 = v123.m128i_u32[2];
      v103 = v101 & (((unsigned int)*v53 >> 9) ^ ((unsigned int)*v53 >> 4));
      v68 = &v102[2 * v103];
      v104 = *v68;
      if ( *v53 == *v68 )
        goto LABEL_89;
      v105 = 0;
      a2 = 1;
      while ( v104 != -4096 )
      {
        if ( v104 == -8192 && !v105 )
          v105 = v68;
        v111 = a2 + 1;
        a2 = v101 & (v103 + (unsigned int)a2);
        v103 = a2;
        v68 = &v102[2 * (unsigned int)a2];
        v104 = *v68;
        if ( *v53 == *v68 )
          goto LABEL_180;
        a2 = v111;
      }
    }
    else
    {
      if ( (_DWORD)a2 - v123.m128i_i32[3] - v69 > (unsigned int)a2 >> 3 )
        goto LABEL_89;
      sub_FADF20((__int64)&v123, a2);
      if ( (v123.m128i_i8[8] & 1) != 0 )
      {
        v106 = 7;
        v107 = (__int64 *)&v124;
      }
      else
      {
        v107 = v124;
        if ( !(_DWORD)v125 )
        {
LABEL_214:
          v123.m128i_i32[2] = (2 * ((unsigned __int32)v123.m128i_i32[2] >> 1) + 2) | v123.m128i_i8[8] & 1;
          BUG();
        }
        v106 = v125 - 1;
      }
      v67 = v123.m128i_u32[2];
      v108 = v106 & (((unsigned int)*v53 >> 9) ^ ((unsigned int)*v53 >> 4));
      v68 = &v107[2 * v108];
      v109 = *v68;
      if ( *v53 == *v68 )
        goto LABEL_89;
      v105 = 0;
      a2 = 1;
      while ( v109 != -4096 )
      {
        if ( v109 == -8192 && !v105 )
          v105 = v68;
        v110 = a2 + 1;
        a2 = v106 & (v108 + (unsigned int)a2);
        v108 = a2;
        v68 = &v107[2 * (unsigned int)a2];
        v109 = *v68;
        if ( *v53 == *v68 )
          goto LABEL_180;
        a2 = v110;
      }
    }
    if ( v105 )
      v68 = v105;
LABEL_180:
    v67 = v123.m128i_u32[2];
LABEL_89:
    v123.m128i_i32[2] = (2 * (v67 >> 1) + 2) | v67 & 1;
    if ( *v68 != -4096 )
      --v123.m128i_i32[3];
    *v68 = *v53;
    v68[1] = v53[1];
LABEL_81:
    v53 += 2;
    if ( v55 == v53 )
      break;
  }
  v29 = sub_AA5930(v115);
  if ( v29 == v71 )
    goto LABEL_116;
  v118 = v3;
  v72 = v71;
  while ( 2 )
  {
    v73 = *(_DWORD *)(v29 + 4) & 0x7FFFFFF;
    if ( (_DWORD)v73 == (unsigned __int32)v123.m128i_i32[2] >> 1 )
    {
      if ( !(_DWORD)v73 )
      {
LABEL_125:
        if ( (v123.m128i_i8[8] & 1) == 0 )
        {
          a2 = 16LL * (unsigned int)v125;
          sub_C7D6A0((__int64)v124, a2, 8);
        }
        goto LABEL_127;
      }
      v74 = *(_QWORD *)(v29 - 8);
      v75 = 8 * v73;
      v76 = 0;
      while ( 2 )
      {
        v81 = *(_QWORD *)(v74 + 32LL * *(unsigned int *)(v29 + 72) + v76);
        if ( (v123.m128i_i8[8] & 1) != 0 )
        {
          a2 = 7;
          v77 = (__int64 *)&v124;
          goto LABEL_98;
        }
        a2 = (unsigned int)v125;
        v77 = v124;
        if ( !(_DWORD)v125 )
        {
          v82 = v123.m128i_u32[2];
          ++v123.m128i_i64[0];
          v83 = 0;
          v84 = ((unsigned __int32)v123.m128i_i32[2] >> 1) + 1;
          goto LABEL_105;
        }
        a2 = (unsigned int)(v125 - 1);
LABEL_98:
        v78 = a2 & (((unsigned int)v81 >> 9) ^ ((unsigned int)v81 >> 4));
        v79 = &v77[2 * v78];
        v80 = *v79;
        if ( *v79 == v81 )
        {
LABEL_99:
          if ( v79[1] != *(_QWORD *)(v74 + 4 * v76) )
            goto LABEL_111;
LABEL_100:
          v76 += 8;
          if ( v75 == v76 )
            goto LABEL_125;
          continue;
        }
        break;
      }
      v83 = 0;
      v87 = 1;
      while ( v80 != -4096 )
      {
        if ( !v83 && v80 == -8192 )
          v83 = v79;
        v78 = a2 & (v87 + v78);
        v79 = &v77[2 * v78];
        v80 = *v79;
        if ( v81 == *v79 )
          goto LABEL_99;
        ++v87;
      }
      v85 = 24;
      a2 = 8;
      if ( !v83 )
        v83 = v79;
      v82 = v123.m128i_u32[2];
      ++v123.m128i_i64[0];
      v84 = ((unsigned __int32)v123.m128i_i32[2] >> 1) + 1;
      if ( (v123.m128i_i8[8] & 1) == 0 )
      {
        a2 = (unsigned int)v125;
LABEL_105:
        v85 = 3 * a2;
      }
      if ( 4 * v84 >= v85 )
      {
        sub_FADF20((__int64)&v123, 2 * a2);
        if ( (v123.m128i_i8[8] & 1) != 0 )
        {
          v88 = 7;
          a2 = (__int64)&v124;
        }
        else
        {
          a2 = (__int64)v124;
          if ( !(_DWORD)v125 )
            goto LABEL_213;
          v88 = v125 - 1;
        }
        v82 = v123.m128i_u32[2];
        LODWORD(v89) = v88 & (((unsigned int)v81 >> 9) ^ ((unsigned int)v81 >> 4));
        v83 = (__int64 *)(a2 + 16LL * (unsigned int)v89);
        v90 = *v83;
        if ( *v83 != v81 )
        {
          v91 = 0;
          v92 = 1;
          while ( v90 != -4096 )
          {
            if ( !v91 && v90 == -8192 )
              v91 = v83;
            v89 = v88 & (unsigned int)(v89 + v92);
            v83 = (__int64 *)(a2 + 16 * v89);
            v90 = *v83;
            if ( v81 == *v83 )
              goto LABEL_145;
            ++v92;
          }
LABEL_143:
          if ( v91 )
            v83 = v91;
LABEL_145:
          v82 = v123.m128i_u32[2];
        }
      }
      else if ( (_DWORD)a2 - v123.m128i_i32[3] - v84 <= (unsigned int)a2 >> 3 )
      {
        sub_FADF20((__int64)&v123, a2);
        if ( (v123.m128i_i8[8] & 1) != 0 )
        {
          v93 = 7;
          a2 = (__int64)&v124;
        }
        else
        {
          a2 = (__int64)v124;
          if ( !(_DWORD)v125 )
          {
LABEL_213:
            v123.m128i_i32[2] = (2 * ((unsigned __int32)v123.m128i_i32[2] >> 1) + 2) | v123.m128i_i8[8] & 1;
            BUG();
          }
          v93 = v125 - 1;
        }
        v82 = v123.m128i_u32[2];
        LODWORD(v94) = v93 & (((unsigned int)v81 >> 9) ^ ((unsigned int)v81 >> 4));
        v83 = (__int64 *)(a2 + 16LL * (unsigned int)v94);
        v95 = *v83;
        if ( *v83 != v81 )
        {
          v91 = 0;
          v96 = 1;
          while ( v95 != -4096 )
          {
            if ( !v91 && v95 == -8192 )
              v91 = v83;
            v94 = v93 & (unsigned int)(v94 + v96);
            v83 = (__int64 *)(a2 + 16 * v94);
            v95 = *v83;
            if ( v81 == *v83 )
              goto LABEL_145;
            ++v96;
          }
          goto LABEL_143;
        }
      }
      v123.m128i_i32[2] = (2 * (v82 >> 1) + 2) | v82 & 1;
      if ( *v83 != -4096 )
        --v123.m128i_i32[3];
      *v83 = v81;
      v83[1] = 0;
      v74 = *(_QWORD *)(v29 - 8);
      if ( *(_QWORD *)(v74 + 4 * v76) )
        goto LABEL_111;
      goto LABEL_100;
    }
LABEL_111:
    v86 = *(_QWORD *)(v29 + 32);
    if ( !v86 )
      BUG();
    v29 = 0;
    if ( *(_BYTE *)(v86 - 24) == 84 )
      v29 = v86 - 24;
    if ( v72 != v29 )
      continue;
    break;
  }
  v3 = v118;
LABEL_116:
  if ( (v123.m128i_i8[8] & 1) == 0 )
    sub_C7D6A0((__int64)v124, 16LL * (unsigned int)v125, 8);
  LODWORD(v15) = v121;
LABEL_37:
  v27 = *(_QWORD *)(v3 + 8);
  LOWORD(v126) = 260;
  v123.m128i_i64[0] = v3 + 16;
  v28 = sub_BD2DA0(80);
  v29 = v28;
  if ( v28 )
  {
    v117 = (_QWORD *)v28;
    sub_B44260(v28, v27, 55, 0x8000000u, 0, 0);
    *(_DWORD *)(v29 + 72) = v15;
    sub_BD6B50((unsigned __int8 *)v29, (const char **)&v123);
    sub_BD2A10(v29, *(_DWORD *)(v29 + 72), 1);
    sub_B44220(v117, *(_QWORD *)(v115 + 56), 1);
  }
  else
  {
    v117 = 0;
    sub_B44220(0, *(_QWORD *)(v115 + 56), 1);
  }
  v30 = v120;
  v31 = 2LL * (unsigned int)v121;
  if ( v120 != &v120[v31] )
  {
    v114 = v3;
    v32 = &v120[v31];
    do
    {
      v38 = *v30;
      v39 = v30[1];
      v40 = *(_DWORD *)(v29 + 4) & 0x7FFFFFF;
      if ( v40 == *(_DWORD *)(v29 + 72) )
      {
        sub_B48D90(v29);
        v40 = *(_DWORD *)(v29 + 4) & 0x7FFFFFF;
      }
      v33 = (v40 + 1) & 0x7FFFFFF;
      v34 = v33 | *(_DWORD *)(v29 + 4) & 0xF8000000;
      v35 = *(_QWORD *)(v29 - 8) + 32LL * (unsigned int)(v33 - 1);
      *(_DWORD *)(v29 + 4) = v34;
      if ( *(_QWORD *)v35 )
      {
        v36 = *(_QWORD *)(v35 + 8);
        **(_QWORD **)(v35 + 16) = v36;
        if ( v36 )
          *(_QWORD *)(v36 + 16) = *(_QWORD *)(v35 + 16);
      }
      *(_QWORD *)v35 = v39;
      if ( v39 )
      {
        v37 = *(_QWORD *)(v39 + 16);
        *(_QWORD *)(v35 + 8) = v37;
        if ( v37 )
          *(_QWORD *)(v37 + 16) = v35 + 8;
        *(_QWORD *)(v35 + 16) = v39 + 16;
        *(_QWORD *)(v39 + 16) = v35;
      }
      v30 += 2;
      *(_QWORD *)(*(_QWORD *)(v29 - 8)
                + 32LL * *(unsigned int *)(v29 + 72)
                + 8LL * ((*(_DWORD *)(v29 + 4) & 0x7FFFFFFu) - 1)) = v38;
    }
    while ( v32 != v30 );
    v3 = v114;
  }
  a2 = (__int64)&v123;
  v123 = (__m128i)(unsigned __int64)sub_AA4E30(v115);
  v124 = 0;
  v125 = 0;
  v126 = 0;
  v127 = 0;
  v128 = 0;
  v129 = 0;
  v130 = 257;
  v14 = sub_1020E10(v29, &v123, v41, v42, v43, v44);
  if ( v14 )
  {
    sub_B43D60(v117);
    goto LABEL_15;
  }
  v119 = 0;
  v47 = sub_AA4FF0(v115);
  if ( !v47 )
  {
LABEL_57:
    v48 = v119;
    if ( &v119 == (__int64 *)(v47 + 48) )
      goto LABEL_61;
    v49 = *(_QWORD *)(v47 + 48);
    v119 = v49;
    if ( v49 )
    {
      sub_B96E90((__int64)&v119, v49, 1);
      goto LABEL_60;
    }
    v123.m128i_i64[0] = 0;
    v50 = (__m128i *)(v29 + 48);
LABEL_159:
    if ( v50 == &v123 )
      goto LABEL_65;
    v98 = *(_QWORD *)(v29 + 48);
    if ( !v98 )
    {
      *(_QWORD *)(v29 + 48) = v123.m128i_i64[0];
      goto LABEL_65;
    }
LABEL_161:
    sub_B91220((__int64)v50, v98);
    goto LABEL_162;
  }
  if ( v47 != v115 + 48 )
  {
    v47 -= 24;
    goto LABEL_57;
  }
LABEL_60:
  v48 = v119;
LABEL_61:
  v123.m128i_i64[0] = v48;
  v50 = (__m128i *)(v29 + 48);
  if ( !v48 )
    goto LABEL_159;
  sub_B96E90((__int64)&v123, v48, 1);
  if ( v50 == &v123 )
  {
    if ( v123.m128i_i64[0] )
      sub_B91220((__int64)&v123, v123.m128i_i64[0]);
    goto LABEL_65;
  }
  v98 = *(_QWORD *)(v29 + 48);
  if ( v98 )
    goto LABEL_161;
LABEL_162:
  v99 = (unsigned __int8 *)v123.m128i_i64[0];
  *(_QWORD *)(v29 + 48) = v123.m128i_i64[0];
  if ( v99 )
    sub_B976B0((__int64)&v123, v99, (__int64)v50);
LABEL_65:
  v51 = *(_QWORD *)(v3 + 48);
  if ( v51 )
  {
    v52 = *(unsigned int *)(v51 + 8);
    if ( v52 + 1 > (unsigned __int64)*(unsigned int *)(v51 + 12) )
    {
      sub_C8D5F0(*(_QWORD *)(v3 + 48), (const void *)(v51 + 16), v52 + 1, 8u, v45, v46);
      v52 = *(unsigned int *)(v51 + 8);
    }
    *(_QWORD *)(*(_QWORD *)v51 + 8 * v52) = v29;
    ++*(_DWORD *)(v51 + 8);
  }
  a2 = v119;
  if ( v119 )
    sub_B91220((__int64)&v119, v119);
LABEL_127:
  v14 = v29;
LABEL_15:
  if ( v120 != (__int64 *)v122 )
    _libc_free(v120, a2);
  return v14;
}
