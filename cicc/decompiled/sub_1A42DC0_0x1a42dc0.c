// Function: sub_1A42DC0
// Address: 0x1a42dc0
//
__int64 __fastcall sub_1A42DC0(
        __int64 a1,
        _QWORD *a2,
        __m128 a3,
        __m128i a4,
        __m128i a5,
        __m128i a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  unsigned __int64 v10; // r15
  __int64 *v11; // rax
  __int64 v12; // r13
  __int64 v13; // r14
  __int64 v14; // rbx
  _QWORD *v15; // rax
  __int64 v16; // r8
  int v17; // r9d
  unsigned __int8 *v18; // rsi
  __int64 v19; // rax
  int v20; // r8d
  int v21; // r9d
  double v22; // xmm4_8
  __m128i v23; // xmm5
  _BYTE *v24; // rdx
  _QWORD *v25; // rax
  _QWORD *i; // rdx
  __int64 v27; // rbx
  __int64 v28; // r12
  _QWORD *v29; // rcx
  char v30; // al
  char v31; // dl
  __m128i *v32; // rcx
  char v33; // al
  const char **v34; // rsi
  char v35; // dl
  __m128i *v36; // rsi
  __m128i *v37; // rcx
  __int64 v38; // rax
  __int64 v39; // r13
  _BYTE *v40; // rax
  __int64 v41; // rdx
  __int64 result; // rax
  unsigned int v43; // r13d
  int v44; // ebx
  __int64 j; // r15
  __int64 *v46; // r15
  __m128i v47; // rax
  __m128i *v48; // r8
  int v49; // r9d
  unsigned __int64 v50; // rcx
  unsigned int v51; // r15d
  __int64 v52; // rdx
  unsigned int v53; // esi
  _QWORD *v54; // r14
  unsigned int v55; // eax
  _QWORD *v56; // rax
  _QWORD *v57; // r14
  unsigned __int64 *v58; // r12
  __int64 v59; // rax
  unsigned __int64 v60; // rcx
  __int64 v61; // rsi
  unsigned __int8 *v62; // rsi
  int v63; // eax
  __int64 v64; // rdx
  char v65; // al
  __m128i *v66; // rdx
  __int64 **v67; // rdx
  __int64 *v68; // rbx
  __int64 v69; // rsi
  __int64 v70; // rax
  __int64 v71; // rsi
  __int64 v72; // rdx
  unsigned __int8 *v73; // rsi
  __int64 *v74; // r13
  __int64 v75; // rax
  __int64 v76; // rcx
  __int64 v77; // rsi
  unsigned __int8 *v78; // rsi
  __int64 v79; // r12
  __m128i *v80; // rdx
  char v81; // al
  __int64 **v82; // r13
  __int64 v83; // r14
  __int64 *v84; // rbx
  __int64 v85; // rdx
  __int64 v86; // rsi
  __int64 v87; // rax
  __int64 v88; // rsi
  __int64 v89; // rdx
  unsigned __int8 *v90; // rsi
  __int64 **v91; // [rsp+30h] [rbp-340h]
  __int64 v92; // [rsp+38h] [rbp-338h]
  __int64 v93; // [rsp+50h] [rbp-320h]
  __int64 *v94; // [rsp+50h] [rbp-320h]
  unsigned int v95; // [rsp+58h] [rbp-318h]
  unsigned int v96; // [rsp+5Ch] [rbp-314h]
  __int64 v97; // [rsp+60h] [rbp-310h]
  unsigned int v98; // [rsp+70h] [rbp-300h]
  int v99; // [rsp+80h] [rbp-2F0h]
  __int64 v100; // [rsp+80h] [rbp-2F0h]
  __int64 **v101; // [rsp+88h] [rbp-2E8h]
  unsigned int v102; // [rsp+90h] [rbp-2E0h]
  __int64 v103; // [rsp+90h] [rbp-2E0h]
  __int64 *v104; // [rsp+90h] [rbp-2E0h]
  unsigned __int8 *v105; // [rsp+A8h] [rbp-2C8h] BYREF
  _QWORD v106[2]; // [rsp+B0h] [rbp-2C0h] BYREF
  __m128i v107; // [rsp+C0h] [rbp-2B0h] BYREF
  __int64 v108; // [rsp+D0h] [rbp-2A0h]
  _QWORD *v109; // [rsp+E0h] [rbp-290h] BYREF
  __int16 v110; // [rsp+F0h] [rbp-280h]
  __m128i v111; // [rsp+100h] [rbp-270h] BYREF
  __int64 v112; // [rsp+110h] [rbp-260h]
  const char *v113; // [rsp+120h] [rbp-250h] BYREF
  __int64 v114; // [rsp+128h] [rbp-248h]
  char v115; // [rsp+130h] [rbp-240h]
  char v116; // [rsp+131h] [rbp-23Fh]
  __m128i v117; // [rsp+140h] [rbp-230h] BYREF
  __int64 v118; // [rsp+150h] [rbp-220h]
  __m128i v119; // [rsp+160h] [rbp-210h] BYREF
  __int64 v120; // [rsp+170h] [rbp-200h]
  __m128i v121; // [rsp+180h] [rbp-1F0h] BYREF
  __int64 v122; // [rsp+190h] [rbp-1E0h]
  unsigned __int8 *v123; // [rsp+1A0h] [rbp-1D0h] BYREF
  __int64 v124; // [rsp+1A8h] [rbp-1C8h]
  __int64 *v125; // [rsp+1B0h] [rbp-1C0h]
  _QWORD *v126; // [rsp+1B8h] [rbp-1B8h]
  __int64 v127; // [rsp+1C0h] [rbp-1B0h]
  int v128; // [rsp+1C8h] [rbp-1A8h]
  __int64 v129; // [rsp+1D0h] [rbp-1A0h]
  __int64 v130; // [rsp+1D8h] [rbp-198h]
  _BYTE *v131; // [rsp+1F0h] [rbp-180h] BYREF
  __int64 v132; // [rsp+1F8h] [rbp-178h]
  _BYTE v133[64]; // [rsp+200h] [rbp-170h] BYREF
  __int64 v134[5]; // [rsp+240h] [rbp-130h] BYREF
  char *v135; // [rsp+268h] [rbp-108h]
  char v136; // [rsp+278h] [rbp-F8h] BYREF
  unsigned __int8 *v137[2]; // [rsp+2C0h] [rbp-B0h] BYREF
  __int16 v138; // [rsp+2D0h] [rbp-A0h]
  char *v139; // [rsp+2E8h] [rbp-88h]
  char v140; // [rsp+2F8h] [rbp-78h] BYREF

  v10 = (unsigned __int64)a2;
  if ( *(_DWORD *)(a1 + 496) )
  {
    result = sub_1A3F5B0(a1, (__int64)a2);
    if ( !(_BYTE)result )
      return result;
  }
  v93 = *a2;
  v11 = (__int64 *)*(a2 - 3);
  if ( *(_BYTE *)(*a2 + 8LL) != 16 )
    return 0;
  v12 = *v11;
  if ( *(_BYTE *)(*v11 + 8) != 16 )
    return 0;
  v13 = *(_QWORD *)(v12 + 32);
  v14 = *(_QWORD *)(v93 + 32);
  v96 = v13;
  v95 = v14;
  v15 = (_QWORD *)sub_16498A0((__int64)a2);
  v18 = (unsigned __int8 *)a2[6];
  v123 = 0;
  v126 = v15;
  v19 = *(_QWORD *)(v10 + 40);
  v127 = 0;
  v124 = v19;
  v128 = 0;
  v129 = 0;
  v130 = 0;
  v125 = (__int64 *)(v10 + 24);
  v137[0] = v18;
  if ( v18 )
  {
    sub_1623A60((__int64)v137, (__int64)v18, 2);
    v123 = v137[0];
    if ( v137[0] )
      sub_1623210((__int64)v137, v137[0], (__int64)&v123);
  }
  sub_1A41500((__int64)v134, (_QWORD *)a1, v10, *(_QWORD *)(v10 - 24), v16, v17);
  v131 = v133;
  v132 = 0x800000000LL;
  v92 = (unsigned int)v14;
  if ( (_DWORD)v14 )
  {
    v24 = v133;
    v25 = v133;
    if ( (unsigned int)v14 > 8uLL )
    {
      sub_16CD150((__int64)&v131, v133, (unsigned int)v14, 8, v20, v21);
      v24 = v131;
      v25 = &v131[8 * (unsigned int)v132];
    }
    for ( i = &v24[8 * (unsigned int)v14]; i != v25; ++v25 )
    {
      if ( v25 )
        *v25 = 0;
    }
    LODWORD(v132) = v14;
  }
  if ( (_DWORD)v14 == (_DWORD)v13 )
  {
    v79 = 0;
    if ( (_DWORD)v14 )
    {
      do
      {
        v119.m128i_i32[0] = v79;
        LOWORD(v120) = 265;
        v113 = sub_1649960(v10);
        v114 = v85;
        v117.m128i_i64[0] = (__int64)&v113;
        LOWORD(v118) = 773;
        v117.m128i_i64[1] = (__int64)".i";
        v81 = v120;
        if ( (_BYTE)v120 )
        {
          if ( (_BYTE)v120 == 1 )
          {
            v121 = _mm_loadu_si128(&v117);
            v122 = v118;
          }
          else
          {
            v80 = (__m128i *)v119.m128i_i64[0];
            if ( BYTE1(v120) != 1 )
            {
              v80 = &v119;
              v81 = 2;
            }
            v121.m128i_i64[1] = (__int64)v80;
            LOBYTE(v122) = 2;
            v121.m128i_i64[0] = (__int64)&v117;
            BYTE1(v122) = v81;
          }
        }
        else
        {
          LOWORD(v122) = 256;
        }
        v82 = *(__int64 ***)(v93 + 24);
        v83 = (__int64)sub_1A3F820(v134, v79);
        v84 = (__int64 *)&v131[8 * v79];
        if ( v82 != *(__int64 ***)v83 )
        {
          if ( *(_BYTE *)(v83 + 16) > 0x10u )
          {
            v138 = 257;
            v83 = sub_15FDBD0(47, v83, (__int64)v82, (__int64)v137, 0);
            if ( v124 )
            {
              v104 = v125;
              sub_157E9D0(v124 + 40, v83);
              v86 = *v104;
              v87 = *(_QWORD *)(v83 + 24) & 7LL;
              *(_QWORD *)(v83 + 32) = v104;
              v86 &= 0xFFFFFFFFFFFFFFF8LL;
              *(_QWORD *)(v83 + 24) = v86 | v87;
              *(_QWORD *)(v86 + 8) = v83 + 24;
              *v104 = *v104 & 7 | (v83 + 24);
            }
            sub_164B780(v83, v121.m128i_i64);
            if ( v123 )
            {
              v111.m128i_i64[0] = (__int64)v123;
              sub_1623A60((__int64)&v111, (__int64)v123, 2);
              v88 = *(_QWORD *)(v83 + 48);
              v89 = v83 + 48;
              if ( v88 )
              {
                sub_161E7C0(v83 + 48, v88);
                v89 = v83 + 48;
              }
              v90 = (unsigned __int8 *)v111.m128i_i64[0];
              *(_QWORD *)(v83 + 48) = v111.m128i_i64[0];
              if ( v90 )
                sub_1623210((__int64)&v111, v90, v89);
            }
          }
          else
          {
            v83 = sub_15A46C0(47, (__int64 ***)v83, v82, 0);
          }
        }
        *v84 = v83;
        ++v79;
      }
      while ( v79 != v92 );
    }
  }
  else
  {
    if ( (unsigned int)v14 <= (unsigned int)v13 )
    {
      v98 = (unsigned int)v13 / (unsigned int)v14;
      v91 = (__int64 **)sub_16463B0(*(__int64 **)(v12 + 24), (unsigned int)v13 / (unsigned int)v14);
      v97 = 0;
      v99 = 0;
      while ( 1 )
      {
        v27 = 0;
        v28 = sub_1599EF0(v91);
        do
        {
          while ( 1 )
          {
            LOWORD(v120) = 265;
            v119.m128i_i32[0] = v27;
            v113 = ".upto";
            v116 = 1;
            v115 = 3;
            v110 = 265;
            LODWORD(v109) = v97;
            v106[0] = sub_1649960(v10);
            LOWORD(v108) = 773;
            v106[1] = v41;
            v107.m128i_i64[0] = (__int64)v106;
            v107.m128i_i64[1] = (__int64)".i";
            v30 = v110;
            if ( !(_BYTE)v110 )
            {
              LOWORD(v112) = 256;
LABEL_40:
              LOWORD(v118) = 256;
LABEL_41:
              LOWORD(v122) = 256;
              goto LABEL_34;
            }
            if ( (_BYTE)v110 != 1 )
            {
              v29 = v109;
              if ( HIBYTE(v110) != 1 )
              {
                v29 = &v109;
                v30 = 2;
              }
              v111.m128i_i64[1] = (__int64)v29;
              v111.m128i_i64[0] = (__int64)&v107;
              v31 = v115;
              LOBYTE(v112) = 2;
              BYTE1(v112) = v30;
              if ( !v115 )
                goto LABEL_40;
              if ( v115 == 1 )
              {
LABEL_94:
                a4 = _mm_loadu_si128(&v111);
                v33 = v112;
                v117 = a4;
                v118 = v112;
                if ( !(_BYTE)v112 )
                  goto LABEL_41;
                v35 = v120;
                if ( !(_BYTE)v120 )
                  goto LABEL_41;
                if ( (_BYTE)v112 == 1 )
                {
                  a5 = _mm_loadu_si128(&v119);
                  v121 = a5;
                  v122 = v120;
                  goto LABEL_34;
                }
                goto LABEL_28;
              }
LABEL_24:
              v32 = &v111;
              v33 = 2;
              goto LABEL_25;
            }
            a3 = (__m128)_mm_loadu_si128(&v107);
            v31 = v115;
            v112 = v108;
            v111 = (__m128i)a3;
            if ( !v115 )
              goto LABEL_40;
            if ( v115 == 1 )
              goto LABEL_94;
            if ( BYTE1(v112) != 1 )
              goto LABEL_24;
            v32 = (__m128i *)v111.m128i_i64[0];
            v33 = 5;
LABEL_25:
            v34 = (const char **)v113;
            if ( v116 != 1 )
            {
              v34 = &v113;
              v31 = 2;
            }
            BYTE1(v118) = v31;
            v35 = v120;
            v117.m128i_i64[0] = (__int64)v32;
            v117.m128i_i64[1] = (__int64)v34;
            LOBYTE(v118) = v33;
            if ( !(_BYTE)v120 )
              goto LABEL_41;
LABEL_28:
            if ( v35 == 1 )
            {
              a6 = _mm_loadu_si128(&v117);
              v121 = a6;
              v122 = v118;
            }
            else
            {
              v36 = (__m128i *)v117.m128i_i64[0];
              if ( BYTE1(v118) != 1 )
              {
                v36 = &v117;
                v33 = 2;
              }
              v37 = (__m128i *)v119.m128i_i64[0];
              if ( BYTE1(v120) != 1 )
              {
                v37 = &v119;
                v35 = 2;
              }
              v121.m128i_i64[0] = (__int64)v36;
              v121.m128i_i64[1] = (__int64)v37;
              LOBYTE(v122) = v33;
              BYTE1(v122) = v35;
            }
LABEL_34:
            v38 = sub_1643350(v126);
            v39 = sub_159C470(v38, v27, 0);
            v40 = sub_1A3F820(v134, v99 + (int)v27);
            if ( *(_BYTE *)(v28 + 16) > 0x10u || v40[16] > 0x10u || *(_BYTE *)(v39 + 16) > 0x10u )
              break;
            ++v27;
            v28 = sub_15A3890((__int64 *)v28, (__int64)v40, v39, 0);
            if ( v98 <= (unsigned int)v27 )
              goto LABEL_78;
          }
          v103 = (__int64)v40;
          v138 = 257;
          v56 = sub_1648A60(56, 3u);
          v57 = v56;
          if ( v56 )
            sub_15FA480((__int64)v56, (__int64 *)v28, v103, v39, (__int64)v137, 0);
          if ( v124 )
          {
            v58 = (unsigned __int64 *)v125;
            sub_157E9D0(v124 + 40, (__int64)v57);
            v59 = v57[3];
            v60 = *v58;
            v57[4] = v58;
            v60 &= 0xFFFFFFFFFFFFFFF8LL;
            v57[3] = v60 | v59 & 7;
            *(_QWORD *)(v60 + 8) = v57 + 3;
            *v58 = *v58 & 7 | (unsigned __int64)(v57 + 3);
          }
          sub_164B780((__int64)v57, v121.m128i_i64);
          if ( v123 )
          {
            v105 = v123;
            sub_1623A60((__int64)&v105, (__int64)v123, 2);
            v61 = v57[6];
            if ( v61 )
              sub_161E7C0((__int64)(v57 + 6), v61);
            v62 = v105;
            v57[6] = v105;
            if ( v62 )
              sub_1623210((__int64)&v105, v62, (__int64)(v57 + 6));
          }
          v28 = (__int64)v57;
          ++v27;
        }
        while ( v98 > (unsigned int)v27 );
LABEL_78:
        v63 = 0;
        if ( v95 <= v96 )
          v63 = v96 / v95 - 1;
        v99 += v63 + 1;
        LOWORD(v120) = 265;
        v119.m128i_i32[0] = v97;
        v113 = sub_1649960(v10);
        v114 = v64;
        v117.m128i_i64[0] = (__int64)&v113;
        LOWORD(v118) = 773;
        v117.m128i_i64[1] = (__int64)".i";
        v65 = v120;
        if ( (_BYTE)v120 )
        {
          if ( (_BYTE)v120 == 1 )
          {
            v23 = _mm_loadu_si128(&v117);
            v121 = v23;
            v122 = v118;
          }
          else
          {
            v66 = (__m128i *)v119.m128i_i64[0];
            if ( BYTE1(v120) != 1 )
            {
              v66 = &v119;
              v65 = 2;
            }
            v121.m128i_i64[1] = (__int64)v66;
            LOBYTE(v122) = 2;
            v121.m128i_i64[0] = (__int64)&v117;
            BYTE1(v122) = v65;
          }
        }
        else
        {
          LOWORD(v122) = 256;
        }
        v67 = *(__int64 ***)(v93 + 24);
        v68 = (__int64 *)&v131[8 * v97];
        if ( v67 != *(__int64 ***)v28 )
        {
          if ( *(_BYTE *)(v28 + 16) > 0x10u )
          {
            v138 = 257;
            v28 = sub_15FDBD0(47, v28, (__int64)v67, (__int64)v137, 0);
            if ( v124 )
            {
              v74 = v125;
              sub_157E9D0(v124 + 40, v28);
              v75 = *(_QWORD *)(v28 + 24);
              v76 = *v74;
              *(_QWORD *)(v28 + 32) = v74;
              v76 &= 0xFFFFFFFFFFFFFFF8LL;
              *(_QWORD *)(v28 + 24) = v76 | v75 & 7;
              *(_QWORD *)(v76 + 8) = v28 + 24;
              *v74 = *v74 & 7 | (v28 + 24);
            }
            sub_164B780(v28, v121.m128i_i64);
            if ( v123 )
            {
              v111.m128i_i64[0] = (__int64)v123;
              sub_1623A60((__int64)&v111, (__int64)v123, 2);
              v77 = *(_QWORD *)(v28 + 48);
              if ( v77 )
                sub_161E7C0(v28 + 48, v77);
              v78 = (unsigned __int8 *)v111.m128i_i64[0];
              *(_QWORD *)(v28 + 48) = v111.m128i_i64[0];
              if ( v78 )
                sub_1623210((__int64)&v111, v78, v28 + 48);
            }
          }
          else
          {
            v28 = sub_15A46C0(47, (__int64 ***)v28, v67, 0);
          }
        }
        ++v97;
        *v68 = v28;
        if ( v92 == v97 )
          goto LABEL_62;
      }
    }
    v43 = (unsigned int)v14 / (unsigned int)v13;
    v101 = (__int64 **)sub_16463B0(*(__int64 **)(v93 + 24), (unsigned int)v14 / (unsigned int)v13);
    v100 = v10;
    v44 = 0;
    v102 = 0;
    do
    {
      for ( j = (__int64)sub_1A3F820(v134, v102); *(_BYTE *)(j + 16) == 71; j = *v46 )
      {
        if ( (*(_BYTE *)(j + 23) & 0x40) != 0 )
          v46 = *(__int64 **)(j - 8);
        else
          v46 = (__int64 *)(j - 24LL * (*(_DWORD *)(j + 20) & 0xFFFFFFF));
      }
      v47.m128i_i64[0] = (__int64)sub_1649960(j);
      v119 = v47;
      LOWORD(v122) = 773;
      v121.m128i_i64[0] = (__int64)&v119;
      v121.m128i_i64[1] = (__int64)".cast";
      if ( v101 != *(__int64 ***)j )
      {
        if ( *(_BYTE *)(j + 16) > 0x10u )
        {
          v138 = 257;
          j = sub_15FDBD0(47, j, (__int64)v101, (__int64)v137, 0);
          if ( v124 )
          {
            v94 = v125;
            sub_157E9D0(v124 + 40, j);
            v69 = *v94;
            v70 = *(_QWORD *)(j + 24) & 7LL;
            *(_QWORD *)(j + 32) = v94;
            v69 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)(j + 24) = v69 | v70;
            *(_QWORD *)(v69 + 8) = j + 24;
            *v94 = *v94 & 7 | (j + 24);
          }
          sub_164B780(j, v121.m128i_i64);
          if ( v123 )
          {
            v117.m128i_i64[0] = (__int64)v123;
            sub_1623A60((__int64)&v117, (__int64)v123, 2);
            v71 = *(_QWORD *)(j + 48);
            v48 = &v117;
            v72 = j + 48;
            if ( v71 )
            {
              sub_161E7C0(j + 48, v71);
              v48 = &v117;
              v72 = j + 48;
            }
            v73 = (unsigned __int8 *)v117.m128i_i64[0];
            *(_QWORD *)(j + 48) = v117.m128i_i64[0];
            if ( v73 )
              sub_1623210((__int64)&v117, v73, v72);
          }
        }
        else
        {
          j = sub_15A46C0(47, (__int64 ***)j, v101, 0);
        }
      }
      v50 = j;
      v51 = 0;
      sub_1A41500((__int64)v137, (_QWORD *)a1, v100, v50, (__int64)v48, v49);
      do
      {
        v52 = v51 + v44;
        v53 = v51++;
        v54 = &v131[8 * v52];
        *v54 = sub_1A3F820((__int64 *)v137, v53);
      }
      while ( v43 > v51 );
      v55 = 0;
      if ( v95 >= v96 )
        v55 = v43 - 1;
      v44 += v55 + 1;
      if ( v139 != &v140 )
        _libc_free((unsigned __int64)v139);
      ++v102;
    }
    while ( v96 != v102 );
    v10 = v100;
  }
LABEL_62:
  sub_1A41120(
    a1,
    v10,
    &v131,
    a3,
    *(double *)a4.m128i_i64,
    *(double *)a5.m128i_i64,
    *(double *)a6.m128i_i64,
    v22,
    *(double *)v23.m128i_i64,
    a9,
    a10);
  if ( v131 != v133 )
    _libc_free((unsigned __int64)v131);
  if ( v135 != &v136 )
    _libc_free((unsigned __int64)v135);
  result = 1;
  if ( v123 )
  {
    sub_161E7C0((__int64)&v123, (__int64)v123);
    return 1;
  }
  return result;
}
