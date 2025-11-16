// Function: sub_F44160
// Address: 0xf44160
//
__int64 __fastcall sub_F44160(__int64 a1, unsigned int a2, __int64 a3, void **a4)
{
  __int64 v5; // r12
  __int64 v6; // rax
  __int64 v7; // r9
  unsigned __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rdi
  int v12; // esi
  _QWORD *v13; // rax
  __int64 v14; // rsi
  unsigned int v15; // ecx
  _QWORD *v16; // rdx
  _QWORD *v17; // r8
  __int64 v18; // r15
  __int64 v19; // r13
  __int64 v20; // rdx
  __int64 v21; // r8
  int v22; // ecx
  int v23; // ecx
  unsigned int v24; // edx
  __int64 *v25; // rax
  __int64 v26; // r10
  __int64 v27; // rax
  unsigned __int64 v28; // rdx
  __int64 *v29; // rdi
  __int64 v30; // rax
  __int64 *v31; // rdx
  __int64 *v32; // r8
  __int64 v33; // rcx
  __int64 v34; // rax
  unsigned __int64 v35; // rcx
  __int64 v36; // rsi
  unsigned __int64 v37; // rcx
  __int64 v38; // rsi
  unsigned __int64 v39; // rcx
  __int64 v40; // rsi
  unsigned __int64 v41; // rcx
  unsigned __int64 v42; // rax
  unsigned __int64 v43; // rax
  int v44; // r13d
  __int64 v45; // r15
  __int64 v46; // rax
  __int64 v47; // r13
  unsigned __int16 v48; // r14
  __m128i *v49; // rax
  __m128i *v50; // r15
  __int64 v51; // rsi
  __m128i *v52; // r8
  __int64 v53; // r14
  __int64 *v54; // r15
  __int64 v55; // rcx
  __int64 v56; // rax
  unsigned int v57; // edx
  __int64 v58; // r10
  __int64 v59; // r9
  __int64 *v60; // rax
  __int64 v61; // rax
  __int64 v63; // rdx
  __int128 v64; // rax
  __int64 v65; // r15
  __int64 v66; // rax
  unsigned int v67; // r14d
  int v68; // r15d
  unsigned __int64 v69; // rax
  __int64 v70; // r14
  int v71; // r15d
  unsigned int v72; // ebx
  unsigned int v73; // r13d
  unsigned int v74; // r15d
  __int64 v75; // rdx
  int v76; // edx
  int v77; // edx
  unsigned int v78; // ecx
  _QWORD *v79; // rax
  _QWORD *v80; // r8
  __int64 v81; // rbx
  unsigned int v82; // ecx
  __int64 *v83; // rax
  __int64 v84; // rdi
  __int64 *v85; // rdi
  _QWORD *v86; // rax
  _QWORD *v87; // rax
  _QWORD *v88; // rdx
  __int64 v89; // rsi
  unsigned __int8 *v90; // rsi
  int v91; // eax
  char v92; // al
  _QWORD *v93; // rsi
  int v94; // edx
  __int64 v95; // rdx
  __int64 *v96; // rax
  unsigned __int64 *v97; // rsi
  int v98; // eax
  int v99; // edi
  __m128i v100; // xmm3
  int v101; // eax
  int v102; // r9d
  int v103; // eax
  int v104; // r8d
  unsigned __int64 v105; // rax
  __int64 v106; // [rsp+10h] [rbp-450h]
  __int64 v107; // [rsp+18h] [rbp-448h]
  int v108; // [rsp+18h] [rbp-448h]
  __int64 v109; // [rsp+20h] [rbp-440h]
  __int64 *v110; // [rsp+28h] [rbp-438h]
  __int64 v112; // [rsp+38h] [rbp-428h]
  __int128 v113; // [rsp+38h] [rbp-428h]
  __m128i *v114; // [rsp+38h] [rbp-428h]
  __int64 v116; // [rsp+48h] [rbp-418h]
  __int64 v117; // [rsp+50h] [rbp-410h]
  _QWORD *v119; // [rsp+68h] [rbp-3F8h] BYREF
  __int128 v120; // [rsp+70h] [rbp-3F0h] BYREF
  __int128 v121; // [rsp+80h] [rbp-3E0h]
  __int64 v122; // [rsp+90h] [rbp-3D0h]
  __m128i v123; // [rsp+A0h] [rbp-3C0h]
  __int16 v124; // [rsp+C0h] [rbp-3A0h]
  __m128i v125; // [rsp+D0h] [rbp-390h] BYREF
  __m128i v126; // [rsp+E0h] [rbp-380h] BYREF
  __int64 v127; // [rsp+F0h] [rbp-370h]
  __int64 **v128; // [rsp+100h] [rbp-360h] BYREF
  __int64 v129; // [rsp+108h] [rbp-358h]
  _BYTE v130[32]; // [rsp+110h] [rbp-350h] BYREF
  unsigned __int64 *v131; // [rsp+130h] [rbp-330h] BYREF
  __int64 v132; // [rsp+138h] [rbp-328h]
  unsigned __int64 v133[2]; // [rsp+140h] [rbp-320h] BYREF
  __int64 v134; // [rsp+150h] [rbp-310h]
  unsigned __int64 v135; // [rsp+158h] [rbp-308h]
  _QWORD *v136; // [rsp+160h] [rbp-300h]
  unsigned __int64 v137; // [rsp+168h] [rbp-2F8h]
  __m128i v138; // [rsp+170h] [rbp-2F0h] BYREF
  __m128i v139; // [rsp+180h] [rbp-2E0h] BYREF
  __int64 v140; // [rsp+190h] [rbp-2D0h]

  v119 = *(_QWORD **)(a1 + 40);
  v5 = sub_B46EC0(a1, a2);
  v6 = sub_AA4FF0(v5);
  if ( !v6 )
    BUG();
  v8 = (unsigned int)*(unsigned __int8 *)(v6 - 24) - 39;
  if ( (unsigned int)v8 <= 0x38 )
  {
    v9 = 0x100060000000001LL;
    if ( _bittest64(&v9, v8) )
      return 0;
  }
  if ( *(_BYTE *)(a3 + 35) )
  {
    v61 = sub_AA50C0(v5, 1);
    if ( !v61 )
      BUG();
    if ( *(_BYTE *)(v61 - 24) == 36 )
      return 0;
  }
  v10 = *(_QWORD *)(a3 + 16);
  v128 = (__int64 **)v130;
  v129 = 0x400000000LL;
  v117 = v10;
  if ( v10 )
  {
    v11 = *(_QWORD *)(v10 + 8);
    v12 = *(_DWORD *)(v10 + 24);
    v13 = v119;
    if ( v12 )
    {
      v14 = (unsigned int)(v12 - 1);
      v15 = v14 & (((unsigned int)v119 >> 9) ^ ((unsigned int)v119 >> 4));
      v16 = (_QWORD *)(v11 + 16LL * v15);
      v17 = (_QWORD *)*v16;
      if ( v119 == (_QWORD *)*v16 )
      {
LABEL_8:
        v18 = v16[1];
        if ( !v18 )
          goto LABEL_51;
        v19 = *(_QWORD *)(v5 + 16);
        if ( !v19 )
          goto LABEL_51;
        while ( 1 )
        {
          v20 = *(_QWORD *)(v19 + 24);
          if ( (unsigned __int8)(*(_BYTE *)v20 - 30) <= 0xAu )
            break;
          v19 = *(_QWORD *)(v19 + 8);
          if ( !v19 )
            goto LABEL_51;
        }
LABEL_14:
        v21 = *(_QWORD *)(v20 + 40);
        if ( (_QWORD *)v21 == v13 )
          goto LABEL_21;
        v22 = *(_DWORD *)(v117 + 24);
        v14 = *(_QWORD *)(v117 + 8);
        if ( v22 )
        {
          v23 = v22 - 1;
          v24 = v23 & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
          v25 = (__int64 *)(v14 + 16LL * v24);
          v26 = *v25;
          if ( v21 == *v25 )
          {
LABEL_17:
            if ( v18 == v25[1] )
            {
              v27 = (unsigned int)v129;
              v28 = (unsigned int)v129 + 1LL;
              if ( v28 > HIDWORD(v129) )
              {
                v14 = (__int64)v130;
                v116 = v21;
                sub_C8D5F0((__int64)&v128, v130, v28, 8u, v21, v7);
                v27 = (unsigned int)v129;
                v21 = v116;
              }
              v128[v27] = (__int64 *)v21;
              LODWORD(v129) = v129 + 1;
LABEL_21:
              while ( 1 )
              {
                v19 = *(_QWORD *)(v19 + 8);
                if ( !v19 )
                  break;
                v20 = *(_QWORD *)(v19 + 24);
                if ( (unsigned __int8)(*(_BYTE *)v20 - 30) <= 0xAu )
                {
                  v13 = v119;
                  goto LABEL_14;
                }
              }
              v29 = (__int64 *)v128;
              v30 = 8LL * (unsigned int)v129;
              v31 = (__int64 *)v128;
              v32 = (__int64 *)&v128[(unsigned __int64)v30 / 8];
              v33 = v30 >> 3;
              v34 = v30 >> 5;
              if ( v34 )
              {
                while ( 1 )
                {
                  v35 = *(_QWORD *)(*v31 + 48) & 0xFFFFFFFFFFFFFFF8LL;
                  if ( v35 == *v31 + 48 )
                    goto LABEL_208;
                  if ( !v35 )
                    BUG();
                  v14 = (unsigned int)*(unsigned __int8 *)(v35 - 24) - 30;
                  if ( (unsigned int)v14 > 0xA )
LABEL_208:
                    BUG();
                  if ( *(_BYTE *)(v35 - 24) == 33 )
                    goto LABEL_133;
                  v36 = v31[1];
                  v37 = *(_QWORD *)(v36 + 48) & 0xFFFFFFFFFFFFFFF8LL;
                  if ( v37 == v36 + 48 )
                    goto LABEL_205;
                  if ( !v37 )
                    BUG();
                  v14 = (unsigned int)*(unsigned __int8 *)(v37 - 24) - 30;
                  if ( (unsigned int)v14 > 0xA )
LABEL_205:
                    BUG();
                  if ( *(_BYTE *)(v37 - 24) == 33 )
                  {
                    ++v31;
                    goto LABEL_133;
                  }
                  v38 = v31[2];
                  v39 = *(_QWORD *)(v38 + 48) & 0xFFFFFFFFFFFFFFF8LL;
                  if ( v39 == v38 + 48 )
                    goto LABEL_202;
                  if ( !v39 )
                    BUG();
                  v14 = (unsigned int)*(unsigned __int8 *)(v39 - 24) - 30;
                  if ( (unsigned int)v14 > 0xA )
LABEL_202:
                    BUG();
                  if ( *(_BYTE *)(v39 - 24) == 33 )
                  {
                    v31 += 2;
                    goto LABEL_133;
                  }
                  v40 = v31[3];
                  v41 = *(_QWORD *)(v40 + 48) & 0xFFFFFFFFFFFFFFF8LL;
                  if ( v41 == v40 + 48 )
                    goto LABEL_209;
                  if ( !v41 )
                    BUG();
                  v14 = (unsigned int)*(unsigned __int8 *)(v41 - 24) - 30;
                  if ( (unsigned int)v14 > 0xA )
LABEL_209:
                    BUG();
                  if ( *(_BYTE *)(v41 - 24) == 33 )
                  {
                    v31 += 3;
                    goto LABEL_133;
                  }
                  v31 += 4;
                  if ( !--v34 )
                  {
                    v33 = v32 - v31;
                    break;
                  }
                }
              }
              if ( v33 != 2 )
              {
                if ( v33 != 3 )
                {
                  if ( v33 != 1 )
                    goto LABEL_51;
                  goto LABEL_47;
                }
                v105 = *(_QWORD *)(*v31 + 48) & 0xFFFFFFFFFFFFFFF8LL;
                if ( v105 == *v31 + 48 )
                  goto LABEL_217;
                if ( !v105 )
                  BUG();
                if ( (unsigned int)*(unsigned __int8 *)(v105 - 24) - 30 > 0xA )
LABEL_217:
                  BUG();
                if ( *(_BYTE *)(v105 - 24) == 33 )
                  goto LABEL_133;
                ++v31;
              }
              v42 = *(_QWORD *)(*v31 + 48) & 0xFFFFFFFFFFFFFFF8LL;
              if ( v42 == *v31 + 48 )
                goto LABEL_213;
              if ( !v42 )
                BUG();
              if ( (unsigned int)*(unsigned __int8 *)(v42 - 24) - 30 > 0xA )
LABEL_213:
                BUG();
              if ( *(_BYTE *)(v42 - 24) == 33 )
                goto LABEL_133;
              ++v31;
LABEL_47:
              v43 = *(_QWORD *)(*v31 + 48) & 0xFFFFFFFFFFFFFFF8LL;
              if ( v43 == *v31 + 48 )
                goto LABEL_211;
              if ( !v43 )
                BUG();
              if ( (unsigned int)*(unsigned __int8 *)(v43 - 24) - 30 > 0xA )
LABEL_211:
                BUG();
              if ( *(_BYTE *)(v43 - 24) != 33 )
                goto LABEL_51;
LABEL_133:
              if ( v31 == v32 )
                goto LABEL_51;
              if ( *(_BYTE *)(a3 + 36) )
              {
                v47 = 0;
                goto LABEL_122;
              }
            }
          }
          else
          {
            v91 = 1;
            while ( v26 != -4096 )
            {
              v99 = v91 + 1;
              v24 = v23 & (v91 + v24);
              v25 = (__int64 *)(v14 + 16LL * v24);
              v26 = *v25;
              if ( v21 == *v25 )
                goto LABEL_17;
              v91 = v99;
            }
          }
        }
        LODWORD(v129) = 0;
        goto LABEL_51;
      }
      v94 = 1;
      while ( v17 != (_QWORD *)-4096LL )
      {
        v7 = (unsigned int)(v94 + 1);
        v15 = v14 & (v94 + v15);
        v16 = (_QWORD *)(v11 + 16LL * v15);
        v17 = (_QWORD *)*v16;
        if ( v119 == (_QWORD *)*v16 )
          goto LABEL_8;
        v94 = v7;
      }
    }
  }
LABEL_51:
  sub_CA0F50(v138.m128i_i64, a4);
  v44 = sub_2241AC0(&v138, byte_3F871B3);
  if ( (__m128i *)v138.m128i_i64[0] != &v139 )
    j_j___libc_free_0(v138.m128i_i64[0], v139.m128i_i64[0] + 1);
  if ( v44 )
  {
    v45 = sub_BD5C60(a1);
    v46 = sub_22077B0(80);
    v47 = v46;
    if ( v46 )
      sub_AA4D50(v46, v45, (__int64)a4, 0, 0);
  }
  else
  {
    v131 = (unsigned __int64 *)"_crit_edge";
    LOWORD(v134) = 259;
    v124 = 261;
    v123.m128i_i64[0] = (__int64)sub_BD5D20(v5);
    v123.m128i_i64[1] = v63;
    *(_QWORD *)&v64 = sub_BD5D20((__int64)v119);
    v120 = v64;
    *(_QWORD *)&v121 = ".";
    LOWORD(v122) = 773;
    v92 = v134;
    v125.m128i_i64[0] = (__int64)&v120;
    v126 = v123;
    LOWORD(v127) = 1282;
    if ( (_BYTE)v134 )
    {
      if ( (_BYTE)v134 == 1 )
      {
        v100 = _mm_load_si128(&v126);
        v138 = _mm_load_si128(&v125);
        v140 = v127;
        v139 = v100;
      }
      else
      {
        if ( BYTE1(v134) == 1 )
        {
          v93 = v131;
          v107 = v132;
        }
        else
        {
          v93 = &v131;
          v92 = 2;
        }
        v138.m128i_i64[0] = (__int64)&v125;
        v139.m128i_i64[0] = (__int64)v93;
        v138.m128i_i64[1] = v109;
        LOBYTE(v140) = 2;
        v139.m128i_i64[1] = v107;
        BYTE1(v140) = v92;
      }
    }
    else
    {
      LOWORD(v140) = 256;
    }
    v65 = sub_BD5C60(a1);
    v66 = sub_22077B0(80);
    v47 = v66;
    if ( v66 )
      sub_AA4D50(v66, v65, (__int64)&v138, 0, 0);
  }
  sub_B43C20((__int64)&v138, v47);
  v48 = v138.m128i_u16[4];
  v112 = v138.m128i_i64[0];
  v49 = (__m128i *)sub_BD2C40(72, 1u);
  v50 = v49;
  if ( v49 )
    sub_B4C8F0((__int64)v49, v5, 1u, v112, v48);
  v51 = *(_QWORD *)(a1 + 48);
  v52 = v50 + 3;
  v138.m128i_i64[0] = v51;
  if ( v51 )
  {
    sub_B96E90((__int64)&v138, v51, 1);
    v52 = v50 + 3;
    if ( &v50[3] == &v138 )
    {
      if ( v138.m128i_i64[0] )
        sub_B91220((__int64)&v138, v138.m128i_i64[0]);
      goto LABEL_62;
    }
    v89 = v50[3].m128i_i64[0];
    if ( !v89 )
    {
LABEL_130:
      v90 = (unsigned __int8 *)v138.m128i_i64[0];
      v50[3].m128i_i64[0] = v138.m128i_i64[0];
      if ( v90 )
        sub_B976B0((__int64)&v138, v90, (__int64)v52);
      goto LABEL_62;
    }
LABEL_129:
    v114 = v52;
    sub_B91220((__int64)v52, v89);
    v52 = v114;
    goto LABEL_130;
  }
  if ( v52 != &v138 )
  {
    v89 = v50[3].m128i_i64[0];
    if ( v89 )
      goto LABEL_129;
  }
LABEL_62:
  v53 = v119[9];
  v54 = (__int64 *)v119[4];
  sub_B2B790(v53 + 72, v47);
  v55 = *v54;
  v56 = *(_QWORD *)(v47 + 24);
  *(_QWORD *)(v47 + 32) = v54;
  v55 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)(v47 + 24) = v55 | v56 & 7;
  *(_QWORD *)(v55 + 8) = v47 + 24;
  *v54 = *v54 & 7 | (v47 + 24);
  sub_AA4C30(v47, *(_BYTE *)(v53 + 128));
  sub_B46F90((unsigned __int8 *)a1, a2, v47);
  v14 = *(_QWORD *)(v5 + 56);
  v57 = 0;
  while ( 1 )
  {
    if ( !v14 )
      BUG();
    if ( *(_BYTE *)(v14 - 24) != 84 )
      break;
    v58 = *(_QWORD *)(v14 - 32);
    v59 = 32LL * *(unsigned int *)(v14 + 48);
    v60 = (__int64 *)(v58 + v59 + 8LL * v57);
    if ( v119 == (_QWORD *)*v60 )
      goto LABEL_70;
    if ( (*(_DWORD *)(v14 - 20) & 0x7FFFFFF) != 0 )
    {
      v60 = (__int64 *)(v58 + v59);
      v57 = 0;
      while ( v119 != (_QWORD *)*v60 )
      {
        ++v57;
        ++v60;
        if ( (*(_DWORD *)(v14 - 20) & 0x7FFFFFF) == v57 )
          goto LABEL_81;
      }
LABEL_70:
      *v60 = v47;
      v14 = *(_QWORD *)(v14 + 8);
    }
    else
    {
LABEL_81:
      v57 = -1;
      *(_QWORD *)(v58 + v59 + 0x7FFFFFFF8LL) = v47;
      v14 = *(_QWORD *)(v14 + 8);
    }
  }
  if ( *(_BYTE *)(a3 + 32) )
  {
    v67 = a2 + 1;
    v68 = sub_B46E30(a1);
    if ( a2 + 1 != v68 )
    {
      do
      {
        while ( 1 )
        {
          v14 = v67;
          if ( v5 == sub_B46EC0(a1, v67) )
            break;
          if ( v68 == ++v67 )
            goto LABEL_88;
        }
        sub_AA5980(v5, (__int64)v119, *(_BYTE *)(a3 + 33));
        v14 = v67;
        sub_B46F90((unsigned __int8 *)a1, v67++, v47);
      }
      while ( v68 != v67 );
    }
  }
LABEL_88:
  *((_QWORD *)&v113 + 1) = *(_QWORD *)a3;
  *(_QWORD *)&v113 = *(_QWORD *)(a3 + 8);
  v110 = *(__int64 **)(a3 + 24);
  if ( v110 )
  {
    v14 = v5;
    v138.m128i_i64[0] = (__int64)v119;
    sub_D6E1B0(v110, v5, v47, (__int64 **)&v138, 1, *(_BYTE *)(a3 + 32));
  }
  if ( v113 != 0 )
  {
    v134 = v47;
    v131 = v133;
    v133[0] = (unsigned __int64)v119;
    v133[1] = v47 & 0xFFFFFFFFFFFFFFFBLL;
    v135 = v5 & 0xFFFFFFFFFFFFFFFBLL;
    v132 = 0x300000002LL;
    v69 = v119[6] & 0xFFFFFFFFFFFFFFF8LL;
    if ( (_QWORD *)v69 != v119 + 6 )
    {
      if ( !v69 )
        BUG();
      v70 = v69 - 24;
      if ( (unsigned int)*(unsigned __int8 *)(v69 - 24) - 30 <= 0xA )
      {
        v108 = sub_B46E30(v70);
        v71 = v108 >> 2;
        if ( v108 >> 2 <= 0 )
        {
          v98 = v108;
          v72 = 0;
        }
        else
        {
          v106 = v47;
          v72 = 0;
          do
          {
            if ( v5 == sub_B46EC0(v70, v72) )
            {
              v47 = v106;
              goto LABEL_103;
            }
            v73 = v72 + 1;
            if ( v5 == sub_B46EC0(v70, v72 + 1)
              || (v73 = v72 + 2, v5 == sub_B46EC0(v70, v72 + 2))
              || (v73 = v72 + 3, v5 == sub_B46EC0(v70, v72 + 3)) )
            {
              v74 = v73;
              v47 = v106;
              v72 = v74;
              goto LABEL_103;
            }
            v72 += 4;
            --v71;
          }
          while ( v71 );
          v47 = v106;
          v98 = v108 - v72;
        }
        switch ( v98 )
        {
          case 2:
LABEL_167:
            if ( v5 != sub_B46EC0(v70, v72) )
            {
              ++v72;
              goto LABEL_169;
            }
LABEL_103:
            v75 = 2;
            if ( v108 != v72 )
            {
LABEL_104:
              if ( *((_QWORD *)&v113 + 1) )
              {
                sub_B26290((__int64)&v138, v133, v75, 1u);
                v14 = (__int64)&v138;
                sub_B24D40(*((__int64 *)&v113 + 1), (__int64)&v138, 0);
                sub_B1A8B0(v14, v14);
                if ( !(_QWORD)v113 )
                {
LABEL_106:
                  if ( v131 != v133 )
                    _libc_free(v131, &v138);
                  goto LABEL_108;
                }
                v97 = v131;
                v75 = (unsigned int)v132;
              }
              else
              {
                v97 = v133;
              }
              sub_B26B80((__int64)&v138, v97, v75, 1u);
              v14 = (__int64)&v138;
              sub_B2A420(v113, (__int64)&v138, 0);
              sub_B1AA80(v14, v14);
              goto LABEL_106;
            }
            break;
          case 3:
            if ( v5 != sub_B46EC0(v70, v72) )
            {
              ++v72;
              goto LABEL_167;
            }
            goto LABEL_103;
          case 1:
LABEL_169:
            if ( v5 == sub_B46EC0(v70, v72) )
              goto LABEL_103;
            break;
        }
      }
    }
    v75 = 3;
    LODWORD(v132) = 3;
    v136 = v119;
    v137 = v5 & 0xFFFFFFFFFFFFFFFBLL | 4;
    goto LABEL_104;
  }
LABEL_108:
  if ( !v117 || (v76 = *(_DWORD *)(v117 + 24), v14 = *(_QWORD *)(v117 + 8), !v76) )
  {
LABEL_121:
    v29 = (__int64 *)v128;
    goto LABEL_122;
  }
  v77 = v76 - 1;
  v78 = v77 & (((unsigned int)v119 >> 9) ^ ((unsigned int)v119 >> 4));
  v79 = (_QWORD *)(v14 + 16LL * v78);
  v80 = (_QWORD *)*v79;
  if ( v119 != (_QWORD *)*v79 )
  {
    v101 = 1;
    while ( v80 != (_QWORD *)-4096LL )
    {
      v102 = v101 + 1;
      v78 = v77 & (v101 + v78);
      v79 = (_QWORD *)(v14 + 16LL * v78);
      v80 = (_QWORD *)*v79;
      if ( v119 == (_QWORD *)*v79 )
        goto LABEL_111;
      v101 = v102;
    }
    goto LABEL_121;
  }
LABEL_111:
  v81 = v79[1];
  if ( !v81 )
    goto LABEL_121;
  v82 = v77 & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
  v83 = (__int64 *)(v14 + 16LL * v82);
  v84 = *v83;
  if ( v5 == *v83 )
  {
LABEL_113:
    v85 = (__int64 *)v83[1];
    if ( v85 )
    {
      v86 = (_QWORD *)v83[1];
      if ( (__int64 *)v81 != v85 )
      {
        while ( 1 )
        {
          v86 = (_QWORD *)*v86;
          if ( (_QWORD *)v81 == v86 )
            break;
          if ( !v86 )
          {
            if ( v85 != (__int64 *)v81 )
            {
              v96 = (__int64 *)v81;
              while ( 1 )
              {
                v96 = (__int64 *)*v96;
                if ( v85 == v96 )
                  break;
                if ( !v96 )
                {
                  v85 = (__int64 *)*v85;
                  if ( v85 )
                    break;
                  goto LABEL_116;
                }
              }
            }
            v14 = v47;
            sub_D4F330(v85, v47, v117);
            goto LABEL_116;
          }
        }
      }
      v14 = v47;
      sub_D4F330((__int64 *)v81, v47, v117);
    }
  }
  else
  {
    v103 = 1;
    while ( v84 != -4096 )
    {
      v104 = v103 + 1;
      v82 = v77 & (v103 + v82);
      v83 = (__int64 *)(v14 + 16LL * v82);
      v84 = *v83;
      if ( v5 == *v83 )
        goto LABEL_113;
      v103 = v104;
    }
  }
LABEL_116:
  if ( !*(_BYTE *)(v81 + 84) )
  {
    v14 = v5;
    if ( !sub_C8CA60(v81 + 56, v5) )
      goto LABEL_151;
    goto LABEL_121;
  }
  v87 = *(_QWORD **)(v81 + 64);
  v88 = &v87[*(unsigned int *)(v81 + 76)];
  if ( v87 != v88 )
  {
    while ( v5 != *v87 )
    {
      if ( v88 == ++v87 )
        goto LABEL_151;
    }
    goto LABEL_121;
  }
LABEL_151:
  if ( *(_BYTE *)(a3 + 34) )
  {
    v14 = 1;
    sub_F34B50((__int64 *)&v119, 1, v47, v5);
  }
  v29 = (__int64 *)v128;
  if ( (_DWORD)v129 )
  {
    v14 = (__int64)v128;
    v95 = sub_F40FB0(v5, v128, (unsigned int)v129, "split", *((__int64 *)&v113 + 1), v117, v110, *(_BYTE *)(a3 + 34));
    if ( *(_BYTE *)(a3 + 34) )
    {
      v14 = (unsigned int)v129;
      sub_F34B50((__int64 *)v128, (unsigned int)v129, v95, v5);
    }
    goto LABEL_121;
  }
LABEL_122:
  if ( v29 != (__int64 *)v130 )
    _libc_free(v29, v14);
  return v47;
}
