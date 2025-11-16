// Function: sub_18AD9B0
// Address: 0x18ad9b0
//
__int64 __fastcall sub_18AD9B0(__int64 a1, __int64 a2, char a3)
{
  __int64 v3; // r14
  unsigned int v4; // esi
  __int64 v5; // rax
  __int64 v6; // r9
  unsigned int v7; // ecx
  __int64 *v8; // rdx
  __int64 v9; // r8
  __int64 v10; // rax
  __int64 *v11; // rax
  unsigned int v12; // esi
  __int64 v13; // rbx
  __int64 v14; // r8
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rdi
  __int64 v18; // r15
  __int64 *v19; // r12
  __int64 v20; // r15
  bool v21; // zf
  _QWORD *v22; // rax
  _QWORD *v23; // rdx
  unsigned int v24; // esi
  __int64 v25; // r10
  int v26; // r11d
  unsigned int v27; // edi
  unsigned __int64 v28; // rdx
  unsigned __int64 v29; // rdx
  int v30; // eax
  __int64 *v31; // rdx
  unsigned int k; // eax
  __int64 *v33; // rdi
  __int64 v34; // rcx
  __int64 *v36; // rax
  __int64 *v37; // r15
  __int64 v38; // rbx
  __int64 v39; // r12
  _QWORD *v40; // rax
  _QWORD *v41; // rdx
  unsigned int v42; // esi
  __int64 v43; // r10
  int v44; // r11d
  unsigned int v45; // edi
  unsigned __int64 v46; // rdx
  unsigned __int64 v47; // rdx
  int v48; // eax
  __int64 *v49; // rdx
  unsigned int j; // eax
  __int64 *v51; // rdi
  __int64 v52; // rcx
  __int64 *v53; // rbx
  const __m128i *v54; // r12
  unsigned __int64 v55; // rbx
  __int64 v56; // rdi
  __int64 v57; // rsi
  __int64 v58; // rdx
  __int64 v59; // rcx
  __int64 v60; // r8
  __int64 v61; // r9
  unsigned __int64 v62; // rbx
  __int64 *v63; // rbx
  __int64 *v64; // rax
  __int64 *v65; // rax
  __int64 *v66; // rdx
  __int64 *v67; // rax
  __int64 v68; // r15
  __int64 v69; // r12
  __int64 *v70; // rbx
  __int64 *v71; // r14
  __int64 v72; // rax
  __int64 v73; // rdx
  __int64 v74; // rcx
  __int64 v75; // r8
  __int64 v76; // r9
  bool v77; // bl
  __int64 *v78; // rax
  __int64 v79; // rsi
  __int64 *v80; // rax
  __int64 *v81; // rdx
  __int64 *v82; // rax
  __int64 v83; // r15
  __int64 v84; // r12
  __int64 *v85; // rbx
  __int64 *v86; // r14
  __int64 v87; // rax
  __int64 v88; // rdx
  __int64 v89; // rcx
  __int64 v90; // r8
  __int64 v91; // r9
  __int64 v92; // rax
  int v93; // edi
  __int64 v94; // rax
  __int64 v95; // rdi
  unsigned __int64 v96; // rbx
  int v97; // r10d
  __int64 *v98; // rcx
  int v99; // eax
  int v100; // edx
  __int64 v101; // r8
  int v102; // edi
  __int64 v103; // rax
  unsigned int v104; // eax
  int v105; // ebx
  __int64 *v106; // rdi
  int v107; // ecx
  int v108; // ecx
  unsigned int v109; // eax
  int v110; // eax
  __int64 *v111; // r12
  int v112; // eax
  __int64 v113; // [rsp+0h] [rbp-F0h]
  __int64 v114; // [rsp+8h] [rbp-E8h]
  __int64 v115; // [rsp+10h] [rbp-E0h]
  __int64 v116; // [rsp+18h] [rbp-D8h]
  __int64 v117; // [rsp+18h] [rbp-D8h]
  unsigned __int8 v118; // [rsp+26h] [rbp-CAh]
  __int64 v120; // [rsp+28h] [rbp-C8h]
  __int64 v121; // [rsp+38h] [rbp-B8h]
  unsigned __int64 v122; // [rsp+40h] [rbp-B0h]
  int v123; // [rsp+48h] [rbp-A8h]
  int i; // [rsp+4Ch] [rbp-A4h]
  __int64 *v125; // [rsp+50h] [rbp-A0h]
  __int64 *v126; // [rsp+50h] [rbp-A0h]
  unsigned int v127; // [rsp+58h] [rbp-98h]
  __int64 v128; // [rsp+58h] [rbp-98h]
  __int64 v129; // [rsp+58h] [rbp-98h]
  unsigned __int64 v130; // [rsp+58h] [rbp-98h]
  __int64 v131; // [rsp+68h] [rbp-88h] BYREF
  __int64 v132; // [rsp+70h] [rbp-80h] BYREF
  __int64 *v133; // [rsp+78h] [rbp-78h] BYREF
  __int64 v134; // [rsp+80h] [rbp-70h] BYREF
  __int64 v135; // [rsp+88h] [rbp-68h] BYREF
  __int64 v136; // [rsp+90h] [rbp-60h] BYREF
  __int64 v137; // [rsp+98h] [rbp-58h]
  __int64 v138; // [rsp+A0h] [rbp-50h] BYREF
  __int64 v139; // [rsp+A8h] [rbp-48h]
  __m128i v140[4]; // [rsp+B0h] [rbp-40h] BYREF

  v114 = a2 + 72;
  v121 = *(_QWORD *)(a2 + 80);
  if ( a2 + 72 == v121 )
    return 0;
  v118 = 0;
  v113 = a1 + 936;
  v120 = a1 + 64;
  v3 = a1;
  do
  {
    v4 = *(_DWORD *)(v3 + 960);
    v5 = v121 - 24;
    if ( !v121 )
      v5 = 0;
    v131 = v5;
    if ( !v4 )
    {
      ++*(_QWORD *)(v3 + 936);
      goto LABEL_147;
    }
    v6 = *(_QWORD *)(v3 + 944);
    v7 = (v4 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
    v8 = (__int64 *)(v6 + 16LL * v7);
    v9 = *v8;
    if ( v5 != *v8 )
    {
      v105 = 1;
      v106 = 0;
      while ( v9 != -8 )
      {
        if ( !v106 && v9 == -16 )
          v106 = v8;
        v7 = (v4 - 1) & (v105 + v7);
        v8 = (__int64 *)(v6 + 16LL * v7);
        v9 = *v8;
        if ( v5 == *v8 )
          goto LABEL_7;
        ++v105;
      }
      v107 = *(_DWORD *)(v3 + 952);
      if ( !v106 )
        v106 = v8;
      ++*(_QWORD *)(v3 + 936);
      v108 = v107 + 1;
      if ( 4 * v108 < 3 * v4 )
      {
        if ( v4 - *(_DWORD *)(v3 + 956) - v108 > v4 >> 3 )
        {
LABEL_143:
          *(_DWORD *)(v3 + 952) = v108;
          if ( *v106 != -8 )
            --*(_DWORD *)(v3 + 956);
          *v106 = v5;
          v10 = 0;
          v106[1] = 0;
          goto LABEL_8;
        }
LABEL_148:
        sub_18AC470(v113, v4);
        sub_18A8880(v113, &v131, v140);
        v106 = (__int64 *)v140[0].m128i_i64[0];
        v5 = v131;
        v108 = *(_DWORD *)(v3 + 952) + 1;
        goto LABEL_143;
      }
LABEL_147:
      v4 *= 2;
      goto LABEL_148;
    }
LABEL_7:
    v10 = v8[1];
LABEL_8:
    v132 = v10;
    for ( i = 0; ; i = 1 )
    {
      v134 = 0;
      v135 = 0;
      v136 = 0;
      v137 = 0;
      v138 = 0;
      v139 = 0;
      if ( !i )
      {
        v123 = *((_DWORD *)sub_18AD860(v3 + 1088, &v131) + 4);
        v36 = sub_18AD860(v3 + 1088, &v131);
        v37 = (__int64 *)v36[1];
        v126 = &v37[*((unsigned int *)v36 + 4)];
        if ( v37 == v126 )
        {
          if ( v123 == 1 )
          {
            v127 = 0;
            v122 = 0;
            goto LABEL_69;
          }
          goto LABEL_117;
        }
        v127 = 0;
        v122 = 0;
        v117 = v3 + 32;
LABEL_48:
        v38 = *v37;
        v39 = v131;
        v21 = *(_QWORD *)(v3 + 928) == 0;
        v140[0].m128i_i64[0] = *v37;
        v140[0].m128i_i64[1] = v131;
        if ( v21 )
        {
          v40 = *(_QWORD **)(v3 + 360);
          v41 = &v40[2 * *(unsigned int *)(v3 + 368)];
          if ( v40 == v41 )
            goto LABEL_64;
          while ( v38 != *v40 || v131 != v40[1] )
          {
            v40 += 2;
            if ( v41 == v40 )
              goto LABEL_64;
          }
        }
        else
        {
          v40 = sub_18A9EA0(v3 + 888, (unsigned __int64 *)v140);
          v41 = (_QWORD *)(v3 + 896);
        }
        if ( v41 == v40 )
        {
LABEL_64:
          ++v127;
          v134 = v38;
          v135 = v39;
          goto LABEL_65;
        }
        v42 = *(_DWORD *)(v3 + 56);
        if ( !v42 )
        {
          ++*(_QWORD *)(v3 + 32);
LABEL_119:
          v42 *= 2;
          goto LABEL_120;
        }
        v43 = *(_QWORD *)(v3 + 40);
        v44 = 1;
        v45 = (unsigned int)v39 >> 9;
        v46 = (((v45 ^ ((unsigned int)v39 >> 4)
               | ((unsigned __int64)(((unsigned int)v38 >> 9) ^ ((unsigned int)v38 >> 4)) << 32))
              - 1
              - ((unsigned __int64)(v45 ^ ((unsigned int)v39 >> 4)) << 32)) >> 22)
            ^ ((v45 ^ ((unsigned int)v39 >> 4)
              | ((unsigned __int64)(((unsigned int)v38 >> 9) ^ ((unsigned int)v38 >> 4)) << 32))
             - 1
             - ((unsigned __int64)(v45 ^ ((unsigned int)v39 >> 4)) << 32));
        v47 = ((9 * (((v46 - 1 - (v46 << 13)) >> 8) ^ (v46 - 1 - (v46 << 13)))) >> 15)
            ^ (9 * (((v46 - 1 - (v46 << 13)) >> 8) ^ (v46 - 1 - (v46 << 13))));
        v48 = ((v47 - 1 - (v47 << 27)) >> 31) ^ (v47 - 1 - ((_DWORD)v47 << 27));
        v49 = 0;
        for ( j = (v42 - 1) & v48; ; j = (v42 - 1) & v109 )
        {
          v51 = (__int64 *)(v43 + 24LL * j);
          v52 = *v51;
          if ( v38 == *v51 && v39 == v51[1] )
          {
            v122 += v51[2];
            goto LABEL_65;
          }
          if ( v52 == -8 )
          {
            if ( v51[1] == -8 )
            {
              v112 = *(_DWORD *)(v3 + 48);
              if ( !v49 )
                v49 = v51;
              ++*(_QWORD *)(v3 + 32);
              v102 = v112 + 1;
              if ( 4 * (v112 + 1) >= 3 * v42 )
                goto LABEL_119;
              v101 = v38;
              if ( v42 - *(_DWORD *)(v3 + 52) - v102 <= v42 >> 3 )
              {
LABEL_120:
                sub_18AD1A0(v117, v42);
                sub_18AA1F0(v117, v140[0].m128i_i64, &v133);
                v49 = v133;
                v101 = v140[0].m128i_i64[0];
                v102 = *(_DWORD *)(v3 + 48) + 1;
              }
              *(_DWORD *)(v3 + 48) = v102;
              if ( *v49 != -8 || v49[1] != -8 )
                --*(_DWORD *)(v3 + 52);
              *v49 = v101;
              v103 = v140[0].m128i_i64[1];
              v49[2] = 0;
              v49[1] = v103;
LABEL_65:
              if ( v38 == v39 )
              {
                v136 = v38;
                v137 = v38;
              }
              if ( v126 == ++v37 )
              {
                if ( v123 != 1 )
                  goto LABEL_35;
LABEL_69:
                v123 = 1;
                v138 = *(_QWORD *)sub_18AD860(v3 + 1088, &v131)[1];
                v139 = v131;
                if ( v127 > 1 )
                  goto LABEL_36;
                goto LABEL_70;
              }
              goto LABEL_48;
            }
          }
          else if ( v52 == -16 && v51[1] == -16 && !v49 )
          {
            v49 = (__int64 *)(v43 + 24LL * j);
          }
          v109 = v44 + j;
          ++v44;
        }
      }
      v116 = v3 + 1120;
      v11 = sub_18AD860(v3 + 1120, &v131);
      v12 = *(_DWORD *)(v3 + 1144);
      v123 = *((_DWORD *)v11 + 4);
      if ( !v12 )
      {
        ++*(_QWORD *)(v3 + 1120);
        goto LABEL_133;
      }
      v13 = v131;
      v14 = *(_QWORD *)(v3 + 1128);
      LODWORD(v15) = (v12 - 1) & (((unsigned int)v131 >> 9) ^ ((unsigned int)v131 >> 4));
      v16 = v14 + 88LL * (unsigned int)v15;
      v17 = *(_QWORD *)v16;
      if ( v131 != *(_QWORD *)v16 )
      {
        v97 = i;
        v98 = 0;
        while ( v17 != -8 )
        {
          if ( v17 == -16 && !v98 )
            v98 = (__int64 *)v16;
          v15 = (v12 - 1) & ((_DWORD)v15 + v97);
          v16 = v14 + 88 * v15;
          v17 = *(_QWORD *)v16;
          if ( v131 == *(_QWORD *)v16 )
            goto LABEL_12;
          ++v97;
        }
        v99 = *(_DWORD *)(v3 + 1136);
        if ( !v98 )
          v98 = (__int64 *)v16;
        ++*(_QWORD *)(v3 + 1120);
        v100 = v99 + 1;
        if ( 4 * (v99 + 1) >= 3 * v12 )
        {
LABEL_133:
          v12 *= 2;
        }
        else if ( v12 - *(_DWORD *)(v3 + 1140) - v100 > v12 >> 3 )
        {
LABEL_113:
          *(_DWORD *)(v3 + 1136) = v100;
          if ( *v98 != -8 )
            --*(_DWORD *)(v3 + 1140);
          *v98 = v13;
          v98[1] = (__int64)(v98 + 3);
          v98[2] = 0x800000000LL;
LABEL_116:
          if ( v123 == 1 )
          {
            v127 = 0;
            v122 = 0;
LABEL_95:
            v92 = *(_QWORD *)sub_18AD860(v116, &v131)[1];
            v138 = v131;
            v139 = v92;
            v123 = i;
            goto LABEL_35;
          }
LABEL_117:
          sub_18AAE60(v3, &v132);
          sub_1377F70(v120, v132);
          v122 = 0;
          goto LABEL_39;
        }
        sub_18AD550(v116, v12);
        sub_18AA320(v116, &v131, v140);
        v98 = (__int64 *)v140[0].m128i_i64[0];
        v13 = v131;
        v100 = *(_DWORD *)(v3 + 1136) + 1;
        goto LABEL_113;
      }
LABEL_12:
      v18 = *(_QWORD *)(v16 + 8);
      if ( v18 + 8LL * *(unsigned int *)(v16 + 16) == v18 )
        goto LABEL_116;
      v125 = (__int64 *)(v18 + 8LL * *(unsigned int *)(v16 + 16));
      v19 = *(__int64 **)(v16 + 8);
      v127 = 0;
      v122 = 0;
      v115 = v3 + 32;
      while ( 2 )
      {
        v20 = *v19;
        v21 = *(_QWORD *)(v3 + 928) == 0;
        v140[0].m128i_i64[0] = v13;
        v140[0].m128i_i64[1] = v20;
        if ( v21 )
        {
          v22 = *(_QWORD **)(v3 + 360);
          v23 = &v22[2 * *(unsigned int *)(v3 + 368)];
          if ( v22 == v23 )
            goto LABEL_30;
          while ( v13 != *v22 || v20 != v22[1] )
          {
            v22 += 2;
            if ( v23 == v22 )
              goto LABEL_30;
          }
        }
        else
        {
          v22 = sub_18A9EA0(v3 + 888, (unsigned __int64 *)v140);
          v23 = (_QWORD *)(v3 + 896);
        }
        if ( v23 == v22 )
        {
LABEL_30:
          ++v127;
          v134 = v13;
          v135 = v20;
          goto LABEL_31;
        }
        v24 = *(_DWORD *)(v3 + 56);
        if ( !v24 )
        {
          ++*(_QWORD *)(v3 + 32);
LABEL_97:
          v24 *= 2;
          goto LABEL_98;
        }
        v25 = *(_QWORD *)(v3 + 40);
        v26 = i;
        v27 = (unsigned int)v20 >> 9;
        v28 = (((v27 ^ ((unsigned int)v20 >> 4)
               | ((unsigned __int64)(((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4)) << 32))
              - 1
              - ((unsigned __int64)(v27 ^ ((unsigned int)v20 >> 4)) << 32)) >> 22)
            ^ ((v27 ^ ((unsigned int)v20 >> 4)
              | ((unsigned __int64)(((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4)) << 32))
             - 1
             - ((unsigned __int64)(v27 ^ ((unsigned int)v20 >> 4)) << 32));
        v29 = ((9 * (((v28 - 1 - (v28 << 13)) >> 8) ^ (v28 - 1 - (v28 << 13)))) >> 15)
            ^ (9 * (((v28 - 1 - (v28 << 13)) >> 8) ^ (v28 - 1 - (v28 << 13))));
        v30 = ((v29 - 1 - (v29 << 27)) >> 31) ^ (v29 - 1 - ((_DWORD)v29 << 27));
        v31 = 0;
        for ( k = (v24 - 1) & v30; ; k = (v24 - 1) & v104 )
        {
          v33 = (__int64 *)(v25 + 24LL * k);
          v34 = *v33;
          if ( v13 == *v33 && v20 == v33[1] )
          {
            v122 += v33[2];
            goto LABEL_31;
          }
          if ( v34 == -8 )
            break;
          if ( v34 == -16 && v33[1] == -16 && !v31 )
            v31 = (__int64 *)(v25 + 24LL * k);
LABEL_127:
          v104 = v26 + k;
          ++v26;
        }
        if ( v33[1] != -8 )
          goto LABEL_127;
        v110 = *(_DWORD *)(v3 + 48);
        if ( !v31 )
          v31 = v33;
        ++*(_QWORD *)(v3 + 32);
        v93 = v110 + 1;
        if ( 4 * (v110 + 1) >= 3 * v24 )
          goto LABEL_97;
        if ( v24 - *(_DWORD *)(v3 + 52) - v93 <= v24 >> 3 )
        {
LABEL_98:
          sub_18AD1A0(v115, v24);
          sub_18AA1F0(v115, v140[0].m128i_i64, &v133);
          v31 = v133;
          v13 = v140[0].m128i_i64[0];
          v93 = *(_DWORD *)(v3 + 48) + 1;
        }
        *(_DWORD *)(v3 + 48) = v93;
        if ( *v31 != -8 || v31[1] != -8 )
          --*(_DWORD *)(v3 + 52);
        *v31 = v13;
        v94 = v140[0].m128i_i64[1];
        v31[2] = 0;
        v31[1] = v94;
LABEL_31:
        if ( v125 != ++v19 )
        {
          v13 = v131;
          continue;
        }
        break;
      }
      if ( v123 == 1 )
        goto LABEL_95;
LABEL_35:
      if ( v127 > 1 )
      {
LABEL_36:
        if ( sub_1377F70(v120, v132) && !sub_18AAE60(v3, &v132)[1] )
        {
          if ( i )
          {
            v80 = sub_18AD860(v3 + 1120, &v131);
            v81 = (__int64 *)v80[1];
            v82 = &v81[*((unsigned int *)v80 + 4)];
            if ( v82 != v81 )
            {
              v129 = v3;
              v83 = v3 + 360;
              v84 = v3 + 32;
              v85 = v82;
              v86 = v81;
              do
              {
                v87 = *v86++;
                v140[0].m128i_i64[0] = v131;
                v140[0].m128i_i64[1] = v87;
                sub_18AD450(v84, v140[0].m128i_i64)[2] = 0;
                sub_18A8AC0(v83, v140, v88, v89, v90, v91);
              }
              while ( v85 != v86 );
              v3 = v129;
            }
            goto LABEL_39;
          }
          v65 = sub_18AD860(v3 + 1088, &v131);
          v66 = (__int64 *)v65[1];
          v67 = &v66[*((unsigned int *)v65 + 4)];
          if ( v67 != v66 )
          {
            v128 = v3;
            v68 = v3 + 360;
            v69 = v3 + 32;
            v70 = v67;
            v71 = v66;
            do
            {
              v72 = *v71++;
              v140[0].m128i_i64[0] = v72;
              v140[0].m128i_i64[1] = v131;
              sub_18AD450(v69, v140[0].m128i_i64)[2] = 0;
              sub_18A8AC0(v68, v140, v73, v74, v75, v76);
            }
            while ( v70 != v71 );
            v3 = v128;
          }
          if ( a3 )
          {
LABEL_84:
            v77 = !sub_1377F70(v120, v132) && v122 != 0;
            if ( v77 )
            {
              v78 = sub_18AAE60(v3, &v132);
              v79 = v132;
              v78[1] = v122;
              sub_1412190(v120, v79);
              v118 = v77;
            }
            goto LABEL_40;
          }
          continue;
        }
        if ( !v136 || !sub_1377F70(v120, v132) )
          goto LABEL_39;
        v54 = (const __m128i *)&v136;
        v95 = v3 + 32;
        v96 = sub_18AAE60(v3, &v131)[1];
        if ( v96 < v122 )
          sub_18AD450(v95, &v136)[2] = 0;
        else
          sub_18AD450(v95, &v136)[2] = v96 - v122;
LABEL_106:
        sub_18A8AC0(v3 + 360, v54, v58, v59, v60, v61);
        v118 = 1;
        goto LABEL_39;
      }
LABEL_70:
      v53 = sub_18AAE60(v3, &v132);
      if ( v127 )
      {
        if ( !sub_1377F70(v120, v132) )
          goto LABEL_39;
        v54 = (const __m128i *)&v134;
        v55 = v53[1];
        v56 = v3 + 32;
        if ( v55 < v122 )
        {
          sub_18AD450(v56, &v134)[2] = 0;
          if ( !i )
            goto LABEL_152;
LABEL_74:
          v57 = sub_18AC630(v113, &v135)[1];
          v140[0].m128i_i64[0] = v57;
        }
        else
        {
          sub_18AD450(v56, &v134)[2] = v55 - v122;
          if ( i )
            goto LABEL_74;
LABEL_152:
          v57 = sub_18AC630(v113, &v134)[1];
          v140[0].m128i_i64[0] = v57;
        }
        if ( sub_1377F70(v120, v57) )
        {
          v62 = sub_18AD450(v3 + 32, &v134)[2];
          if ( v62 > sub_18AAE60(v3, v140[0].m128i_i64)[1] )
          {
            v63 = sub_18AAE60(v3, v140[0].m128i_i64);
            v64 = sub_18AD450(v3 + 32, &v134);
            v58 = v63[1];
            v64[2] = v58;
          }
        }
        goto LABEL_106;
      }
      if ( sub_1377F70(v120, v132) )
      {
        if ( v123 == 1 )
        {
          v130 = sub_18AD450(v3 + 32, &v138)[2];
          if ( v130 < sub_18AAE60(v3, &v132)[1] )
          {
            v111 = sub_18AAE60(v3, &v132);
            v118 = 1;
            sub_18AD450(v3 + 32, &v138)[2] = v111[1];
          }
        }
      }
      else if ( v53[1] < v122 )
      {
        v53[1] = v122;
        v118 = 1;
      }
LABEL_39:
      if ( a3 )
        goto LABEL_84;
LABEL_40:
      if ( i == 1 )
        break;
    }
    v121 = *(_QWORD *)(v121 + 8);
  }
  while ( v114 != v121 );
  return v118;
}
