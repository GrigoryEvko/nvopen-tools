// Function: sub_18BE580
// Address: 0x18be580
//
void __fastcall sub_18BE580(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  _QWORD *v10; // rax
  _QWORD *v11; // rbx
  __int64 v12; // r8
  __int64 v13; // rax
  __int64 v14; // r12
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rdi
  __int64 v18; // rax
  __int64 v19; // rsi
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 *v22; // rdi
  __int64 v23; // r15
  __int64 **v24; // rdx
  __int64 v25; // rbx
  _QWORD *v26; // rax
  __int64 v27; // r12
  __int64 *v28; // rbx
  __int64 v29; // rax
  __int64 v30; // rcx
  double v31; // xmm4_8
  double v32; // xmm5_8
  __int64 v33; // rsi
  unsigned __int8 *v34; // rsi
  _QWORD **v35; // r15
  _QWORD **v36; // rbx
  _QWORD *v37; // r14
  __int64 v38; // rbx
  __int64 v39; // rdi
  __int64 v40; // rax
  __int64 v41; // rsi
  __int64 v42; // rax
  double v43; // xmm4_8
  double v44; // xmm5_8
  _QWORD **v45; // r15
  unsigned __int64 v46; // rbx
  _QWORD **v47; // r14
  _QWORD *v48; // r12
  __int64 v49; // r14
  __int64 v50; // rax
  __int64 v51; // rsi
  __int64 v52; // r15
  _QWORD *v53; // r14
  double v54; // xmm4_8
  double v55; // xmm5_8
  _QWORD *v56; // r12
  _QWORD *v57; // rax
  _QWORD *v58; // r14
  __int64 v59; // rcx
  __int64 v60; // rdx
  __int64 v61; // rax
  _QWORD *v62; // rsi
  _QWORD *v63; // rax
  _QWORD *v64; // rdx
  _BOOL8 v65; // rdi
  __int64 v66; // r12
  bool v67; // zf
  __int64 *v68; // rax
  __int64 *v69; // r12
  __m128i *v70; // rsi
  __m128i v71; // rcx
  _QWORD *v72; // rax
  _QWORD *v73; // rax
  _QWORD *v74; // rax
  __int64 v75; // r12
  unsigned __int64 *v76; // r15
  __int64 v77; // rax
  unsigned __int64 v78; // rcx
  __int64 v79; // rsi
  unsigned __int8 *v80; // rsi
  _QWORD *v81; // rax
  __int64 *v82; // r12
  __int64 v83; // rax
  __int64 v84; // rcx
  __int64 v85; // rsi
  unsigned __int8 *v86; // rsi
  __int64 *v87; // rbx
  __int64 v88; // rax
  __int64 v89; // rcx
  __int64 v90; // rsi
  unsigned __int8 *v91; // rsi
  _QWORD *v92; // rdi
  __int64 v93; // [rsp+8h] [rbp-238h]
  __int64 v94; // [rsp+38h] [rbp-208h]
  __int64 v95; // [rsp+40h] [rbp-200h]
  __int64 v97; // [rsp+58h] [rbp-1E8h]
  __int64 v98; // [rsp+60h] [rbp-1E0h]
  __int64 v99; // [rsp+60h] [rbp-1E0h]
  __int64 *i; // [rsp+60h] [rbp-1E0h]
  __int64 v101; // [rsp+60h] [rbp-1E0h]
  _BYTE *v102; // [rsp+68h] [rbp-1D8h]
  char v103; // [rsp+73h] [rbp-1CDh] BYREF
  int v104; // [rsp+74h] [rbp-1CCh] BYREF
  __int64 v105; // [rsp+78h] [rbp-1C8h] BYREF
  _BYTE *v106; // [rsp+80h] [rbp-1C0h] BYREF
  __int64 v107; // [rsp+88h] [rbp-1B8h]
  _BYTE v108[16]; // [rsp+90h] [rbp-1B0h] BYREF
  _BYTE *v109; // [rsp+A0h] [rbp-1A0h] BYREF
  __int64 v110; // [rsp+A8h] [rbp-198h]
  _BYTE v111[16]; // [rsp+B0h] [rbp-190h] BYREF
  __int64 v112[2]; // [rsp+C0h] [rbp-180h] BYREF
  __int16 v113; // [rsp+D0h] [rbp-170h]
  __m128i v114; // [rsp+E0h] [rbp-160h] BYREF
  __int16 v115; // [rsp+F0h] [rbp-150h]
  __int64 *v116; // [rsp+100h] [rbp-140h] BYREF
  __int64 v117; // [rsp+108h] [rbp-138h]
  _BYTE v118[16]; // [rsp+110h] [rbp-130h] BYREF
  __int64 v119; // [rsp+120h] [rbp-120h] BYREF
  __int64 v120; // [rsp+128h] [rbp-118h]
  __int64 *v121; // [rsp+130h] [rbp-110h]
  __int64 v122; // [rsp+138h] [rbp-108h]
  __int64 v123; // [rsp+140h] [rbp-100h]
  int v124; // [rsp+148h] [rbp-F8h]
  __int64 v125; // [rsp+150h] [rbp-F0h]
  __int64 v126; // [rsp+158h] [rbp-E8h]
  __int64 v127[2]; // [rsp+170h] [rbp-D0h] BYREF
  __int64 v128; // [rsp+180h] [rbp-C0h]
  __int64 v129; // [rsp+188h] [rbp-B8h]
  __int64 v130; // [rsp+190h] [rbp-B0h]
  int v131; // [rsp+198h] [rbp-A8h]
  __int64 v132; // [rsp+1A0h] [rbp-A0h]
  __int64 v133; // [rsp+1A8h] [rbp-98h]
  __m128i v134; // [rsp+1C0h] [rbp-80h] BYREF
  __int64 *v135; // [rsp+1D0h] [rbp-70h]
  __int64 v136; // [rsp+1D8h] [rbp-68h]
  __int64 v137; // [rsp+1E0h] [rbp-60h]
  int v138; // [rsp+1E8h] [rbp-58h]
  __int64 v139; // [rsp+1F0h] [rbp-50h]
  __int64 v140; // [rsp+1F8h] [rbp-48h]

  v93 = sub_15E26F0(*(__int64 **)a1, 208, 0, 0);
  v94 = *(_QWORD *)(a2 + 8);
  if ( v94 )
  {
    while ( 1 )
    {
      v10 = sub_1648700(v94);
      v95 = (__int64)v10;
      v11 = v10;
      v94 = *(_QWORD *)(v94 + 8);
      if ( *((_BYTE *)v10 + 16) == 78 )
        break;
LABEL_80:
      if ( !v94 )
        return;
    }
    v12 = (__int64)v10;
    v13 = *((_DWORD *)v10 + 5) & 0xFFFFFFF;
    v102 = (_BYTE *)v11[-3 * v13];
    v14 = v11[3 * (1 - v13)];
    v15 = v11[3 * (2 - v13)];
    v106 = v108;
    v98 = v15;
    v16 = *(_QWORD *)(v15 + 24);
    v109 = v111;
    v97 = v16;
    v116 = (__int64 *)v118;
    v117 = 0x100000000LL;
    v107 = 0x100000000LL;
    v110 = 0x100000000LL;
    v103 = 0;
    sub_14A88D0((__int64)&v116, (__int64)&v106, (__int64)&v109, &v103, v12);
    if ( (_DWORD)v107 == 1 )
    {
      if ( !v103 )
      {
        v11 = *(_QWORD **)v106;
        v17 = *(_QWORD *)v106;
LABEL_5:
        v18 = sub_16498A0(v17);
        v125 = 0;
        v126 = 0;
        v19 = v11[6];
        v122 = v18;
        v124 = 0;
        v20 = v11[5];
        v119 = 0;
        v120 = v20;
        v121 = v11 + 3;
        v123 = 0;
        v134.m128i_i64[0] = v19;
        if ( v19 )
        {
          sub_1623A60((__int64)&v134, v19, 2);
          if ( v119 )
            sub_161E7C0((__int64)&v119, v119);
          v119 = v134.m128i_i64[0];
          if ( v134.m128i_i64[0] )
            sub_1623210((__int64)&v134, (unsigned __int8 *)v134.m128i_i64[0], (__int64)&v119);
        }
        LOWORD(v135) = 257;
        v21 = sub_12815B0(&v119, *(_QWORD *)(a1 + 40), v102, v14, (__int64)&v134);
        v22 = *(__int64 **)(a1 + 48);
        LOWORD(v128) = 257;
        v23 = v21;
        v24 = (__int64 **)sub_1646BA0(v22, 0);
        if ( v24 != *(__int64 ***)v23 )
        {
          if ( *(_BYTE *)(v23 + 16) > 0x10u )
          {
            LOWORD(v135) = 257;
            v23 = sub_15FDBD0(47, v23, (__int64)v24, (__int64)&v134, 0);
            if ( v120 )
            {
              v87 = v121;
              sub_157E9D0(v120 + 40, v23);
              v88 = *(_QWORD *)(v23 + 24);
              v89 = *v87;
              *(_QWORD *)(v23 + 32) = v87;
              v89 &= 0xFFFFFFFFFFFFFFF8LL;
              *(_QWORD *)(v23 + 24) = v89 | v88 & 7;
              *(_QWORD *)(v89 + 8) = v23 + 24;
              *v87 = *v87 & 7 | (v23 + 24);
            }
            sub_164B780(v23, v127);
            if ( v119 )
            {
              v114.m128i_i64[0] = v119;
              sub_1623A60((__int64)&v114, v119, 2);
              v90 = *(_QWORD *)(v23 + 48);
              if ( v90 )
                sub_161E7C0(v23 + 48, v90);
              v91 = (unsigned __int8 *)v114.m128i_i64[0];
              *(_QWORD *)(v23 + 48) = v114.m128i_i64[0];
              if ( v91 )
                sub_1623210((__int64)&v114, v91, v23 + 48);
            }
          }
          else
          {
            v23 = sub_15A46C0(47, (__int64 ***)v23, v24, 0);
          }
        }
        LOWORD(v135) = 257;
        v25 = *(_QWORD *)(a1 + 48);
        v26 = sub_1648A60(64, 1u);
        v27 = (__int64)v26;
        if ( v26 )
          sub_15F9210((__int64)v26, v25, v23, 0, 0, 0);
        if ( v120 )
        {
          v28 = v121;
          sub_157E9D0(v120 + 40, v27);
          v29 = *(_QWORD *)(v27 + 24);
          v30 = *v28;
          *(_QWORD *)(v27 + 32) = v28;
          v30 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v27 + 24) = v30 | v29 & 7;
          *(_QWORD *)(v30 + 8) = v27 + 24;
          *v28 = *v28 & 7 | (v27 + 24);
        }
        sub_164B780(v27, v134.m128i_i64);
        if ( v119 )
        {
          v127[0] = v119;
          sub_1623A60((__int64)v127, v119, 2);
          v33 = *(_QWORD *)(v27 + 48);
          if ( v33 )
            sub_161E7C0(v27 + 48, v33);
          v34 = (unsigned __int8 *)v127[0];
          *(_QWORD *)(v27 + 48) = v127[0];
          if ( v34 )
            sub_1623210((__int64)v127, v34, v27 + 48);
        }
        v35 = (_QWORD **)v106;
        v36 = (_QWORD **)&v106[8 * (unsigned int)v107];
        if ( v106 != (_BYTE *)v36 )
        {
          do
          {
            v37 = *v35++;
            sub_164D160((__int64)v37, v27, a3, a4, a5, a6, v31, v32, a9, a10);
            sub_15F20C0(v37);
          }
          while ( v36 != v35 );
        }
        if ( (_DWORD)v110 != 1 || v103 )
        {
          v38 = v95;
          v39 = v95;
        }
        else
        {
          v38 = *(_QWORD *)v109;
          v39 = *(_QWORD *)v109;
        }
        v40 = sub_16498A0(v39);
        v127[0] = 0;
        v129 = v40;
        v130 = 0;
        v131 = 0;
        v132 = 0;
        v133 = 0;
        v127[1] = *(_QWORD *)(v38 + 40);
        v128 = v38 + 24;
        v41 = *(_QWORD *)(v38 + 48);
        v134.m128i_i64[0] = v41;
        if ( v41 )
        {
          sub_1623A60((__int64)&v134, v41, 2);
          if ( v127[0] )
            sub_161E7C0((__int64)v127, v127[0]);
          v127[0] = v134.m128i_i64[0];
          if ( v134.m128i_i64[0] )
            sub_1623210((__int64)&v134, (unsigned __int8 *)v134.m128i_i64[0], (__int64)v127);
        }
        LOWORD(v135) = 257;
        v114.m128i_i64[0] = (__int64)v102;
        v114.m128i_i64[1] = v98;
        v42 = sub_1285290(v127, *(_QWORD *)(v93 + 24), v93, (int)&v114, 2, (__int64)&v134, 0);
        v45 = (_QWORD **)v109;
        v46 = v42;
        if ( v109 != &v109[8 * (unsigned int)v110] )
        {
          v99 = v27;
          v47 = (_QWORD **)&v109[8 * (unsigned int)v110];
          do
          {
            v48 = *v45++;
            sub_164D160((__int64)v48, v46, a3, a4, a5, a6, v43, v44, a9, a10);
            sub_15F20C0(v48);
          }
          while ( v47 != v45 );
          v27 = v99;
        }
        if ( *(_QWORD *)(v95 + 8) )
        {
          v49 = sub_1599EF0(*(__int64 ***)v95);
          v50 = sub_16498A0(v95);
          v134.m128i_i64[0] = 0;
          v136 = v50;
          v137 = 0;
          v138 = 0;
          v139 = 0;
          v140 = 0;
          v134.m128i_i64[1] = *(_QWORD *)(v95 + 40);
          v135 = (__int64 *)(v95 + 24);
          v51 = *(_QWORD *)(v95 + 48);
          v114.m128i_i64[0] = v51;
          if ( v51 )
          {
            sub_1623A60((__int64)&v114, v51, 2);
            if ( v134.m128i_i64[0] )
              sub_161E7C0((__int64)&v134, v134.m128i_i64[0]);
            v134.m128i_i64[0] = v114.m128i_i64[0];
            if ( v114.m128i_i64[0] )
              sub_1623210((__int64)&v114, (unsigned __int8 *)v114.m128i_i64[0], (__int64)&v134);
          }
          v104 = 0;
          v113 = 257;
          if ( *(_BYTE *)(v49 + 16) > 0x10u || *(_BYTE *)(v27 + 16) > 0x10u )
          {
            v115 = 257;
            v81 = sub_1648A60(88, 2u);
            v52 = (__int64)v81;
            if ( v81 )
            {
              v101 = (__int64)v81;
              sub_15F1EA0((__int64)v81, *(_QWORD *)v49, 63, (__int64)(v81 - 6), 2, 0);
              *(_QWORD *)(v52 + 56) = v52 + 72;
              *(_QWORD *)(v52 + 64) = 0x400000000LL;
              sub_15FAD90(v52, v49, v27, &v104, 1, (__int64)&v114);
            }
            else
            {
              v101 = 0;
            }
            if ( v134.m128i_i64[1] )
            {
              v82 = v135;
              sub_157E9D0(v134.m128i_i64[1] + 40, v52);
              v83 = *(_QWORD *)(v52 + 24);
              v84 = *v82;
              *(_QWORD *)(v52 + 32) = v82;
              v84 &= 0xFFFFFFFFFFFFFFF8LL;
              *(_QWORD *)(v52 + 24) = v84 | v83 & 7;
              *(_QWORD *)(v84 + 8) = v52 + 24;
              *v82 = *v82 & 7 | (v52 + 24);
            }
            sub_164B780(v101, v112);
            if ( v134.m128i_i64[0] )
            {
              v105 = v134.m128i_i64[0];
              sub_1623A60((__int64)&v105, v134.m128i_i64[0], 2);
              v85 = *(_QWORD *)(v52 + 48);
              if ( v85 )
                sub_161E7C0(v52 + 48, v85);
              v86 = (unsigned __int8 *)v105;
              *(_QWORD *)(v52 + 48) = v105;
              if ( v86 )
                sub_1623210((__int64)&v105, v86, v52 + 48);
            }
          }
          else
          {
            v52 = sub_15A3A20((__int64 *)v49, (__int64 *)v27, &v104, 1, 0);
          }
          v104 = 1;
          v113 = 257;
          if ( *(_BYTE *)(v52 + 16) > 0x10u || *(_BYTE *)(v46 + 16) > 0x10u )
          {
            v115 = 257;
            v74 = sub_1648A60(88, 2u);
            v53 = v74;
            if ( v74 )
            {
              v75 = (__int64)v74;
              sub_15F1EA0((__int64)v74, *(_QWORD *)v52, 63, (__int64)(v74 - 6), 2, 0);
              v53[7] = v53 + 9;
              v53[8] = 0x400000000LL;
              sub_15FAD90((__int64)v53, v52, v46, &v104, 1, (__int64)&v114);
            }
            else
            {
              v75 = 0;
            }
            if ( v134.m128i_i64[1] )
            {
              v76 = (unsigned __int64 *)v135;
              sub_157E9D0(v134.m128i_i64[1] + 40, (__int64)v53);
              v77 = v53[3];
              v78 = *v76;
              v53[4] = v76;
              v78 &= 0xFFFFFFFFFFFFFFF8LL;
              v53[3] = v78 | v77 & 7;
              *(_QWORD *)(v78 + 8) = v53 + 3;
              *v76 = *v76 & 7 | (unsigned __int64)(v53 + 3);
            }
            sub_164B780(v75, v112);
            if ( v134.m128i_i64[0] )
            {
              v105 = v134.m128i_i64[0];
              sub_1623A60((__int64)&v105, v134.m128i_i64[0], 2);
              v79 = v53[6];
              if ( v79 )
                sub_161E7C0((__int64)(v53 + 6), v79);
              v80 = (unsigned __int8 *)v105;
              v53[6] = v105;
              if ( v80 )
                sub_1623210((__int64)&v105, v80, (__int64)(v53 + 6));
            }
          }
          else
          {
            v53 = (_QWORD *)sub_15A3A20((__int64 *)v52, (__int64 *)v46, &v104, 1, 0);
          }
          sub_164D160(v95, (__int64)v53, a3, a4, a5, a6, v54, v55, a9, a10);
          if ( v134.m128i_i64[0] )
            sub_161E7C0((__int64)&v134, v134.m128i_i64[0]);
        }
        v56 = (_QWORD *)(a1 + 168);
        v57 = *(_QWORD **)(a1 + 176);
        v58 = (_QWORD *)(a1 + 168);
        if ( !v57 )
          goto LABEL_56;
        do
        {
          while ( 1 )
          {
            v59 = v57[2];
            v60 = v57[3];
            if ( v57[4] >= v46 )
              break;
            v57 = (_QWORD *)v57[3];
            if ( !v60 )
              goto LABEL_54;
          }
          v58 = v57;
          v57 = (_QWORD *)v57[2];
        }
        while ( v59 );
LABEL_54:
        if ( v58 == v56 || v58[4] > v46 )
        {
LABEL_56:
          v61 = sub_22077B0(48);
          v62 = v58;
          *(_QWORD *)(v61 + 32) = v46;
          v58 = (_QWORD *)v61;
          *(_DWORD *)(v61 + 40) = 0;
          v63 = sub_18BE480((_QWORD *)(a1 + 160), v62, (unsigned __int64 *)(v61 + 32));
          if ( v64 )
          {
            v65 = v56 == v64 || v63 || v46 < v64[4];
            sub_220F040(v65, v58, v64, v56);
            ++*(_QWORD *)(a1 + 200);
          }
          else
          {
            v92 = v58;
            v58 = v63;
            j_j___libc_free_0(v92, 48);
          }
        }
        v66 = (unsigned int)v117;
        v67 = v103 == 0;
        *((_DWORD *)v58 + 10) = v117;
        if ( !v67 )
          *((_DWORD *)v58 + 10) = v66 + 1;
        v68 = &v116[2 * v66];
        v69 = v116;
        for ( i = v68; i != v69; v73[1] = (char *)v70 + 24 )
        {
          while ( 1 )
          {
            v71.m128i_i64[1] = v69[1];
            v114.m128i_i64[1] = *v69;
            v114.m128i_i64[0] = v97;
            v72 = (_QWORD *)sub_18B7900(a1 + 104, &v114);
            v73 = sub_18BB8E0(v72, v71.m128i_i64[1]);
            v71.m128i_i64[0] = (__int64)v102;
            *((_BYTE *)v73 + 24) = 0;
            v134 = v71;
            v135 = v58 + 5;
            v70 = (__m128i *)v73[1];
            if ( v70 != (__m128i *)v73[2] )
              break;
            v69 += 2;
            sub_18B49F0((const __m128i **)v73, v70, &v134);
            if ( i == v69 )
              goto LABEL_70;
          }
          if ( v70 )
          {
            a3 = (__m128)_mm_loadu_si128(&v134);
            *v70 = (__m128i)a3;
            v70[1].m128i_i64[0] = (__int64)v135;
            v70 = (__m128i *)v73[1];
          }
          v69 += 2;
        }
LABEL_70:
        sub_15F20C0((_QWORD *)v95);
        if ( v127[0] )
          sub_161E7C0((__int64)v127, v127[0]);
        if ( v119 )
          sub_161E7C0((__int64)&v119, v119);
        if ( v109 != v111 )
          _libc_free((unsigned __int64)v109);
        if ( v106 != v108 )
          _libc_free((unsigned __int64)v106);
        if ( v116 != (__int64 *)v118 )
          _libc_free((unsigned __int64)v116);
        goto LABEL_80;
      }
      v11 = (_QWORD *)v95;
    }
    v17 = (__int64)v11;
    goto LABEL_5;
  }
}
