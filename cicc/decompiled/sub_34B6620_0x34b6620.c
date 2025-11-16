// Function: sub_34B6620
// Address: 0x34b6620
//
__int64 __fastcall sub_34B6620(__int64 *a1, unsigned int a2, int a3, _QWORD *a4, _QWORD *a5)
{
  unsigned __int64 v5; // r12
  unsigned int *v6; // rbx
  unsigned int v7; // r14d
  __int64 v8; // r8
  _QWORD *v9; // rax
  int *v10; // r15
  __int64 v11; // rcx
  __int64 v12; // rdx
  __int64 v13; // r8
  __int16 *v14; // rdi
  __int64 v15; // rax
  unsigned __int64 v16; // rdx
  unsigned __int64 v17; // rax
  __int32 v18; // r10d
  __int16 *v19; // rsi
  __int64 v20; // rdx
  __int64 v21; // r15
  __int64 v22; // r12
  unsigned int *v23; // r15
  unsigned int v24; // esi
  __int16 *v25; // rax
  int v26; // edi
  __int64 v27; // rax
  __int32 v28; // esi
  __int64 v29; // rdi
  __int64 v30; // rbx
  __int32 v31; // eax
  unsigned int v32; // r15d
  unsigned __int64 v34; // rdi
  _WORD *v35; // rbx
  __int64 v36; // r12
  _QWORD *v37; // rax
  __int64 v38; // rsi
  __int64 v39; // rcx
  __int64 v40; // rdx
  int v41; // eax
  unsigned int *v42; // r15
  __int64 *v43; // rbx
  __int32 v44; // r12d
  _QWORD *v45; // r14
  unsigned __int16 v46; // r13
  int *v47; // rcx
  unsigned __int64 v48; // rax
  unsigned int *v49; // r14
  int v50; // edx
  __int64 v51; // r15
  int v52; // eax
  __int64 v53; // r8
  unsigned int v54; // r13d
  _QWORD *v55; // rax
  __int64 v56; // rax
  __int64 v57; // rdx
  char *v58; // rax
  __int64 v59; // rdx
  __int64 v60; // r8
  __int64 v61; // r11
  __int64 v62; // rax
  __int64 v63; // rdx
  __int64 v64; // r15
  __int64 v65; // r12
  __int64 v66; // r14
  unsigned int v67; // eax
  __int64 v68; // rax
  __int64 v69; // rdx
  __int64 v70; // r15
  __int64 v71; // r12
  __int64 v72; // rax
  __int32 v73; // r14d
  unsigned __int64 v74; // rsi
  _QWORD *v75; // r12
  _QWORD *v76; // rbx
  _QWORD *v77; // rax
  __int64 v78; // rsi
  _QWORD *v79; // rax
  _QWORD *v80; // rdx
  char v81; // di
  unsigned __int64 v82; // rdi
  bool v83; // al
  _QWORD *v84; // rdx
  __int64 v85; // r14
  int *v86; // rdi
  int *v87; // rax
  _QWORD *v88; // [rsp+8h] [rbp-178h]
  int *v89; // [rsp+18h] [rbp-168h]
  __int64 v90; // [rsp+20h] [rbp-160h]
  unsigned int *v91; // [rsp+20h] [rbp-160h]
  __int32 v92; // [rsp+20h] [rbp-160h]
  __int64 *v93; // [rsp+28h] [rbp-158h]
  int *v96; // [rsp+40h] [rbp-140h]
  __int64 v97; // [rsp+48h] [rbp-138h]
  __int64 v98; // [rsp+50h] [rbp-130h]
  __int32 v99; // [rsp+50h] [rbp-130h]
  unsigned int *v100; // [rsp+58h] [rbp-128h]
  unsigned int v101; // [rsp+58h] [rbp-128h]
  __int32 v102; // [rsp+60h] [rbp-120h]
  int v103; // [rsp+60h] [rbp-120h]
  __int32 v104; // [rsp+60h] [rbp-120h]
  __int32 v105; // [rsp+60h] [rbp-120h]
  __int64 v106; // [rsp+60h] [rbp-120h]
  unsigned __int8 v108; // [rsp+68h] [rbp-118h]
  _WORD *src; // [rsp+70h] [rbp-110h]
  __int32 v110; // [rsp+78h] [rbp-108h]
  __int64 v112; // [rsp+90h] [rbp-F0h]
  unsigned int *v113; // [rsp+98h] [rbp-E8h]
  unsigned int *v114; // [rsp+98h] [rbp-E8h]
  __int32 v115; // [rsp+98h] [rbp-E8h]
  __int32 v116; // [rsp+98h] [rbp-E8h]
  unsigned int v117; // [rsp+ACh] [rbp-D4h] BYREF
  unsigned int *v118; // [rsp+B0h] [rbp-D0h] BYREF
  unsigned __int64 v119; // [rsp+B8h] [rbp-C8h]
  __int64 v120; // [rsp+C0h] [rbp-C0h]
  __int64 v121; // [rsp+D0h] [rbp-B0h] BYREF
  int v122; // [rsp+D8h] [rbp-A8h] BYREF
  _QWORD *v123; // [rsp+E0h] [rbp-A0h]
  int *v124; // [rsp+E8h] [rbp-98h]
  int *v125; // [rsp+F0h] [rbp-90h]
  __int64 v126; // [rsp+F8h] [rbp-88h]
  __m128i v127; // [rsp+100h] [rbp-80h] BYREF
  __int16 v128; // [rsp+110h] [rbp-70h] BYREF
  int v129; // [rsp+118h] [rbp-68h]
  __int64 v130; // [rsp+120h] [rbp-60h]
  __int16 v131; // [rsp+128h] [rbp-58h]
  int v132; // [rsp+140h] [rbp-40h]

  v100 = (unsigned int *)a1[15];
  v112 = (__int64)(v100 + 14);
  v118 = 0;
  v119 = 0;
  v120 = 0;
  sub_34B5CB0(v100, a3, (unsigned __int64 *)&v118, (__int64)(v100 + 14));
  v5 = v119;
  v6 = v118;
  if ( v118 == (unsigned int *)v119 )
  {
    v32 = 0;
    goto LABEL_36;
  }
  v122 = 0;
  v124 = &v122;
  v125 = &v122;
  v123 = 0;
  v126 = 0;
  v113 = (unsigned int *)v119;
  do
  {
    while ( 1 )
    {
      v7 = *v6;
      v117 = *v6;
      if ( sub_34B40C0(v112, &v117) )
      {
        v9 = v123;
        v10 = &v122;
        if ( !v123 )
          goto LABEL_12;
        do
        {
          while ( 1 )
          {
            v11 = v9[2];
            v12 = v9[3];
            if ( *((_DWORD *)v9 + 8) >= v7 )
              break;
            v9 = (_QWORD *)v9[3];
            if ( !v12 )
              goto LABEL_10;
          }
          v10 = (int *)v9;
          v9 = (_QWORD *)v9[2];
        }
        while ( v11 );
LABEL_10:
        if ( v10 == &v122 || v7 < v10[8] )
        {
LABEL_12:
          v127.m128i_i64[0] = (__int64)&v117;
          v10 = (int *)sub_34B6540(&v121, (__int64)v10, (unsigned int **)&v127);
        }
        sub_34B51C0((__int64)&v127, a1, v117, v11, v8, (__int64)(v10 + 10));
        v14 = (__int16 *)v127.m128i_i64[0];
        if ( v10 + 10 != (int *)&v127 )
        {
          v15 = v127.m128i_i64[0];
          if ( (__int16 *)v127.m128i_i64[0] != &v128 )
          {
            v34 = *((_QWORD *)v10 + 5);
            if ( (int *)v34 != v10 + 14 )
            {
              _libc_free(v34);
              v15 = v127.m128i_i64[0];
            }
            *((_QWORD *)v10 + 5) = v15;
            *((_QWORD *)v10 + 6) = v127.m128i_i64[1];
            v10[26] = v132;
            goto LABEL_3;
          }
          v16 = v127.m128i_u32[2];
          v17 = (unsigned int)v10[12];
          v18 = v127.m128i_i32[2];
          if ( v127.m128i_u32[2] <= v17 )
          {
            v14 = &v128;
            if ( v127.m128i_i32[2] )
            {
              v105 = v127.m128i_i32[2];
              memmove(*((void **)v10 + 5), &v128, 8LL * v127.m128i_u32[2]);
              v14 = (__int16 *)v127.m128i_i64[0];
              v18 = v105;
            }
          }
          else
          {
            if ( v127.m128i_u32[2] > (unsigned __int64)(unsigned int)v10[13] )
            {
              v10[12] = 0;
              v104 = v16;
              sub_C8D5F0((__int64)(v10 + 10), v10 + 14, v16, 8u, v13, (__int64)(v10 + 10));
              v14 = (__int16 *)v127.m128i_i64[0];
              v16 = v127.m128i_u32[2];
              v17 = 0;
              v18 = v104;
              v19 = (__int16 *)v127.m128i_i64[0];
            }
            else
            {
              v14 = &v128;
              v19 = &v128;
              if ( v10[12] )
              {
                v99 = v127.m128i_i32[2];
                v106 = 8 * v17;
                memmove(*((void **)v10 + 5), &v128, 8 * v17);
                v14 = (__int16 *)v127.m128i_i64[0];
                v16 = v127.m128i_u32[2];
                v18 = v99;
                v19 = (__int16 *)(v127.m128i_i64[0] + v106);
                v17 = v106;
              }
            }
            v20 = 4 * v16;
            if ( v19 != &v14[v20] )
            {
              v102 = v18;
              memcpy((void *)(v17 + *((_QWORD *)v10 + 5)), v19, v20 * 2 - v17);
              v14 = (__int16 *)v127.m128i_i64[0];
              v18 = v102;
            }
          }
          v10[12] = v18;
        }
        v10[26] = v132;
        if ( v14 != &v128 )
          break;
      }
LABEL_3:
      if ( v113 == ++v6 )
        goto LABEL_24;
    }
    _libc_free((unsigned __int64)v14);
    ++v6;
  }
  while ( v113 != v6 );
LABEL_24:
  v21 = a1[4];
  if ( v118 != (unsigned int *)v119 )
  {
    v114 = (unsigned int *)v119;
    v22 = a1[4];
    v23 = v118;
    do
    {
      v24 = *v23;
      if ( *v23 != a2 )
      {
        v117 = a2;
        v25 = (__int16 *)(*(_QWORD *)(v22 + 56) + 2LL * *(unsigned int *)(*(_QWORD *)(v22 + 8) + 24LL * v24 + 8));
        v26 = *v25;
        v27 = (__int64)(v25 + 1);
        v129 = 0;
        v130 = 0;
        v28 = v26 + v24;
        if ( !(_WORD)v26 )
          v27 = 0;
        v127.m128i_i32[0] = v28;
        v128 = v28;
        v131 = 0;
        v127.m128i_i64[1] = v27;
        if ( !sub_2E46590(v127.m128i_i32, (int *)&v117) )
          goto LABEL_34;
      }
      ++v23;
    }
    while ( v114 != v23 );
    v21 = v22;
  }
  v93 = sub_2FF6500(v21, a2, 1);
  v29 = a1[5];
  v30 = *(_QWORD *)v29 + 24LL * *(unsigned __int16 *)(*v93 + 24);
  if ( *(_DWORD *)(v29 + 8) == *(_DWORD *)v30 )
  {
    v31 = *(_DWORD *)(v30 + 4);
    v115 = v31;
    if ( !v31 )
      goto LABEL_34;
  }
  else
  {
    sub_2F60630(v29, (unsigned __int16 ***)v93);
    v31 = *(_DWORD *)(v30 + 4);
    v115 = v31;
    if ( !v31 )
      goto LABEL_34;
  }
  v35 = *(_WORD **)(v30 + 16);
  v127.m128i_i32[2] = v31;
  src = v35;
  v127.m128i_i64[0] = (__int64)v93;
  v36 = (__int64)(a4 + 1);
  sub_34B5FA0((__int64)a4, &v127);
  v37 = (_QWORD *)a4[2];
  v88 = a4 + 1;
  if ( !v37 )
  {
    v36 = (__int64)(a4 + 1);
LABEL_111:
    v78 = v36;
    v36 = sub_22077B0(0x30u);
    *(_DWORD *)(v36 + 40) = 0;
    *(_QWORD *)(v36 + 32) = v93;
    v79 = sub_34B6050(a4, v78, (unsigned __int64 *)(v36 + 32));
    if ( v80 )
    {
      v81 = v79 || v88 == v80 || (unsigned __int64)v93 < v80[4];
      v38 = v36;
      sub_220F040(v81, v36, v80, v88);
      ++a4[5];
    }
    else
    {
      v82 = v36;
      v38 = 48;
      v36 = (__int64)v79;
      j_j___libc_free_0(v82);
    }
    goto LABEL_49;
  }
  v38 = (__int64)v93;
  do
  {
    while ( 1 )
    {
      v39 = v37[2];
      v40 = v37[3];
      if ( v37[4] >= (unsigned __int64)v93 )
        break;
      v37 = (_QWORD *)v37[3];
      if ( !v40 )
        goto LABEL_47;
    }
    v36 = (__int64)v37;
    v37 = (_QWORD *)v37[2];
  }
  while ( v39 );
LABEL_47:
  if ( (_QWORD *)v36 == v88 || *(_QWORD *)(v36 + 32) > (unsigned __int64)v93 )
    goto LABEL_111;
LABEL_49:
  v41 = 0;
  v42 = v100;
  v43 = a1;
  if ( *(_DWORD *)(v36 + 40) != v115 )
    v41 = *(_DWORD *)(v36 + 40);
  v44 = *(_DWORD *)(v36 + 40);
  v103 = v41;
  while ( 2 )
  {
    if ( !v44 )
      v44 = v115;
    v45 = (_QWORD *)v43[2];
    v46 = src[--v44];
    v40 = *(_QWORD *)(*(_QWORD *)((*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(*v45 + 16LL)
                                                                                               + 200LL))(
                                    *(_QWORD *)(*v45 + 16LL),
                                    v38,
                                    v40,
                                    v39)
                                + 248)
                    + 16LL);
    v108 = *(_BYTE *)(v40 + v46);
    if ( !v108 )
      goto LABEL_52;
    v40 = v45[48];
    v101 = v46;
    v39 = v46;
    v98 = 1LL << v46;
    v38 = *(_QWORD *)(v40 + 8LL * (v46 >> 6)) & (1LL << v46);
    v97 = 8LL * (v46 >> 6);
    if ( v38 || v46 == a2 )
      goto LABEL_52;
    sub_34B5950((__int64)a5, a5[2]);
    v47 = (int *)v119;
    a5[2] = 0;
    a5[3] = a5 + 1;
    a5[4] = a5 + 1;
    v48 = (unsigned __int64)v118;
    a5[5] = 0;
    v89 = v47;
    if ( v47 == (int *)v48 )
    {
LABEL_95:
      v32 = v108;
      v73 = v44;
      v74 = a4[2];
      if ( v74 )
      {
        v75 = a4 + 1;
        v76 = (_QWORD *)a4[2];
        while ( 1 )
        {
          if ( v76[4] < (unsigned __int64)v93 )
          {
            v76 = (_QWORD *)v76[3];
          }
          else
          {
            v77 = (_QWORD *)v76[2];
            if ( v76[4] <= (unsigned __int64)v93 )
            {
              v84 = (_QWORD *)v76[3];
              while ( v84 )
              {
                if ( v84[4] <= (unsigned __int64)v93 )
                {
                  v84 = (_QWORD *)v84[3];
                }
                else
                {
                  v75 = v84;
                  v84 = (_QWORD *)v84[2];
                }
              }
              while ( v77 )
              {
                if ( v77[4] < (unsigned __int64)v93 )
                {
                  v77 = (_QWORD *)v77[3];
                }
                else
                {
                  v76 = v77;
                  v77 = (_QWORD *)v77[2];
                }
              }
              if ( (_QWORD *)a4[3] != v76 || v88 != v75 )
              {
                if ( v76 != v75 )
                {
                  v116 = v73;
                  v85 = (__int64)v76;
                  do
                  {
                    v86 = (int *)v85;
                    v85 = sub_220EF30(v85);
                    v87 = sub_220F330(v86, v88);
                    j_j___libc_free_0((unsigned __int64)v87);
                    --a4[5];
                  }
                  while ( (_QWORD *)v85 != v75 );
                  v73 = v116;
                  v32 = v108;
                }
                goto LABEL_125;
              }
LABEL_124:
              sub_34B5590((__int64)a4, v74);
              a4[2] = 0;
              a4[5] = 0;
              a4[3] = v88;
              a4[4] = v88;
              goto LABEL_125;
            }
            v75 = v76;
            v76 = (_QWORD *)v76[2];
          }
          if ( !v76 )
          {
            v83 = v88 == v75;
            goto LABEL_122;
          }
        }
      }
      v75 = a4 + 1;
      v83 = v108;
LABEL_122:
      if ( (_QWORD *)a4[3] == v75 && v83 )
        goto LABEL_124;
LABEL_125:
      v127.m128i_i32[2] = v73;
      v127.m128i_i64[0] = (__int64)v93;
      sub_34B5FA0((__int64)a4, &v127);
      goto LABEL_35;
    }
    v96 = (int *)v48;
    v49 = v42;
    while ( 2 )
    {
      v50 = *v96;
      v117 = v50;
      if ( v50 != a2 )
      {
        v51 = 1;
        v52 = sub_E91E30((_QWORD *)v43[4], a2, v50);
        v53 = 0;
        v54 = v52;
        if ( v52 )
        {
          v54 = sub_E91CF0((_QWORD *)v43[4], v101, v52);
          v51 = 1LL << v54;
          v53 = 8LL * (v54 >> 6);
        }
        v55 = v123;
        if ( v123 )
          goto LABEL_64;
LABEL_109:
        v38 = (__int64)&v122;
        goto LABEL_70;
      }
      v55 = v123;
      v53 = v97;
      v51 = v98;
      v54 = v101;
      if ( !v123 )
        goto LABEL_109;
LABEL_64:
      v38 = (__int64)&v122;
      do
      {
        while ( 1 )
        {
          v39 = v55[2];
          v40 = v55[3];
          if ( *((_DWORD *)v55 + 8) >= v117 )
            break;
          v55 = (_QWORD *)v55[3];
          if ( !v40 )
            goto LABEL_68;
        }
        v38 = (__int64)v55;
        v55 = (_QWORD *)v55[2];
      }
      while ( v39 );
LABEL_68:
      if ( (int *)v38 == &v122 || v117 < *(_DWORD *)(v38 + 32) )
      {
LABEL_70:
        v90 = v53;
        v127.m128i_i64[0] = (__int64)&v117;
        v56 = sub_34B6540(&v121, v38, (unsigned int **)&v127);
        v53 = v90;
        v38 = v56;
      }
      if ( (*(_QWORD *)(*(_QWORD *)(v38 + 40) + v53) & v51) == 0 )
        goto LABEL_107;
      v57 = v43[15];
      v39 = *(_QWORD *)(v57 + 104);
      if ( *(_DWORD *)(v39 + 4LL * v54) != -1 )
      {
        v40 = *(_QWORD *)(v57 + 128);
        if ( *(_DWORD *)(v40 + 4LL * v54) == -1 )
          goto LABEL_107;
      }
      v40 = *((_QWORD *)v49 + 16);
      v38 = v117;
      v39 = *((_QWORD *)v49 + 13);
      if ( *(_DWORD *)(v39 + 4LL * v117) > *(_DWORD *)(v40 + 4LL * v54) )
        goto LABEL_107;
      v58 = sub_E922F0((_QWORD *)v43[4], v54);
      v60 = (__int64)&v58[2 * v59 - 2];
      if ( v58 != (char *)v60 )
      {
        v61 = v43[15];
        v39 = 4LL * v117;
        while ( 1 )
        {
          v40 = *(unsigned __int16 *)v58;
          if ( *(_DWORD *)(*(_QWORD *)(v61 + 104) + 4 * v40) != -1 )
          {
            v38 = *(_QWORD *)(v61 + 128);
            if ( *(_DWORD *)(v38 + 4 * v40) == -1 )
              break;
          }
          v38 = *((_QWORD *)v49 + 16);
          v40 = *(unsigned int *)(v38 + 4 * v40);
          if ( *(_DWORD *)(*((_QWORD *)v49 + 13) + 4LL * v117) > (unsigned int)v40 )
            break;
          v58 += 2;
          if ( (char *)v60 == v58 )
            goto LABEL_81;
        }
LABEL_107:
        v42 = v49;
        goto LABEL_52;
      }
LABEL_81:
      v62 = sub_34B4480(v112, &v117);
      if ( v62 == v63 )
      {
LABEL_87:
        v68 = sub_34B4480(v112, &v117);
        v70 = v69;
        if ( v68 != v69 )
        {
          v92 = v44;
          v71 = v68;
          while ( 1 )
          {
            v72 = *(_QWORD *)(v71 + 40);
            if ( (*(_BYTE *)(v72 + 3) & 0x10) != 0 && (*(_BYTE *)(v72 + 4) & 4) != 0 )
            {
              v38 = v54;
              if ( (unsigned int)sub_2E89C70(*(_QWORD *)(v72 + 16), v54, v43[4], 0) != -1 )
                break;
            }
            v71 = sub_220EEE0(v71);
            if ( v71 == v70 )
            {
              v44 = v92;
              goto LABEL_94;
            }
          }
          v44 = v92;
          v42 = v49;
          goto LABEL_52;
        }
LABEL_94:
        v127.m128i_i64[0] = __PAIR64__(v54, v117);
        sub_34B5490(a5, (unsigned int *)&v127);
        if ( v89 == ++v96 )
          goto LABEL_95;
        continue;
      }
      break;
    }
    v91 = v49;
    v64 = v63;
    v110 = v44;
    v65 = v62;
    while ( 1 )
    {
      v38 = v54;
      v66 = *(_QWORD *)(*(_QWORD *)(v65 + 40) + 16LL);
      v67 = sub_2E8E710(v66, v54, v43[4], 0, 1);
      if ( v67 != -1 )
      {
        v40 = *(_QWORD *)(v66 + 32);
        if ( (*(_BYTE *)(v40 + 40LL * v67 + 4) & 4) != 0 )
          break;
      }
      v65 = sub_220EEE0(v65);
      if ( v65 == v64 )
      {
        v49 = v91;
        v44 = v110;
        goto LABEL_87;
      }
    }
    v42 = v91;
    v44 = v110;
LABEL_52:
    if ( v44 != v103 )
      continue;
    break;
  }
LABEL_34:
  v32 = 0;
LABEL_35:
  sub_34B2F40(v123);
  v5 = (unsigned __int64)v118;
LABEL_36:
  if ( v5 )
    j_j___libc_free_0(v5);
  return v32;
}
