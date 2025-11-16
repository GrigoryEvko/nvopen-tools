// Function: sub_121E800
// Address: 0x121e800
//
__int64 __fastcall sub_121E800(__int64 **a1, __int64 a2, unsigned int *a3, __int64 *a4, __int64 *a5, int a6)
{
  int v6; // eax
  unsigned int v7; // r15d
  __int64 v11; // rcx
  __int64 v13; // rsi
  const char *v14; // rax
  __int64 v16; // rdx
  __int64 v17; // rsi
  int v18; // r9d
  const char *v19; // rax
  unsigned __int64 v20; // rsi
  __int64 v21; // rsi
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rax
  __m128i *v26; // rax
  __int64 v27; // rcx
  __m128i *v28; // rax
  __int64 v29; // rcx
  __m128i *v30; // r9
  unsigned __int64 v31; // rax
  unsigned __int64 v32; // rdi
  unsigned __int64 v33; // rcx
  __m128i *v34; // rax
  __m128i *v35; // rcx
  __m128i *v36; // rdx
  __int64 v37; // rcx
  __m128i *v38; // rax
  unsigned __int64 v39; // rsi
  int v40; // r9d
  __int64 *v41; // rax
  __int64 v42; // rax
  __m128i *v43; // rax
  __int64 v44; // rcx
  __m128i *v45; // rax
  unsigned __int64 v46; // rsi
  int v47; // r9d
  __int64 v48; // rax
  __int64 *v49; // rax
  __int64 v50; // r8
  __int64 v51; // rsi
  __m128i *v52; // rax
  __int64 v53; // rcx
  __m128i *v54; // rax
  __int64 v55; // rcx
  __m128i *v56; // r9
  unsigned __int64 v57; // rax
  unsigned __int64 v58; // rdi
  unsigned __int64 v59; // rcx
  __m128i *v60; // rax
  __int64 v61; // rcx
  __m128i *v62; // rdx
  __int64 v63; // rcx
  __m128i *v64; // rax
  unsigned __int64 v65; // rsi
  unsigned __int64 v66; // rax
  unsigned __int64 v67; // rsi
  __int64 v68; // rax
  __int64 v69; // rdx
  unsigned int v70; // eax
  char v71; // r15
  char **v72; // rsi
  unsigned int v73; // edx
  __int64 v74; // rax
  __int64 v75; // rdi
  __int64 *v76; // rax
  int v77; // eax
  __int64 *v78; // rsi
  __int64 i; // rax
  _BYTE *v80; // rdi
  char v81; // al
  __int64 v82; // rsi
  _DWORD *v83; // rax
  bool v84; // [rsp+Ch] [rbp-234h]
  __int64 *v85; // [rsp+18h] [rbp-228h]
  __int64 *v86; // [rsp+20h] [rbp-220h]
  __int64 v87; // [rsp+20h] [rbp-220h]
  __int64 *v88; // [rsp+20h] [rbp-220h]
  char v89; // [rsp+20h] [rbp-220h]
  __int64 *v90; // [rsp+28h] [rbp-218h]
  unsigned int v91; // [rsp+28h] [rbp-218h]
  __int64 v92[2]; // [rsp+30h] [rbp-210h] BYREF
  __int64 v93; // [rsp+40h] [rbp-200h] BYREF
  __m128i *v94; // [rsp+50h] [rbp-1F0h] BYREF
  __int64 v95; // [rsp+58h] [rbp-1E8h]
  __m128i v96; // [rsp+60h] [rbp-1E0h] BYREF
  __int64 v97[2]; // [rsp+70h] [rbp-1D0h] BYREF
  __int64 v98; // [rsp+80h] [rbp-1C0h] BYREF
  __m128i *v99; // [rsp+90h] [rbp-1B0h] BYREF
  __int64 v100; // [rsp+98h] [rbp-1A8h]
  __m128i v101; // [rsp+A0h] [rbp-1A0h] BYREF
  __m128i *v102; // [rsp+B0h] [rbp-190h] BYREF
  __int64 v103; // [rsp+B8h] [rbp-188h]
  __m128i v104; // [rsp+C0h] [rbp-180h] BYREF
  _QWORD *v105; // [rsp+D0h] [rbp-170h] BYREF
  __int64 v106; // [rsp+D8h] [rbp-168h]
  _QWORD v107[2]; // [rsp+E0h] [rbp-160h] BYREF
  __m128i *v108; // [rsp+F0h] [rbp-150h] BYREF
  __int64 v109; // [rsp+F8h] [rbp-148h]
  __m128i v110; // [rsp+100h] [rbp-140h] BYREF
  __int64 v111[2]; // [rsp+110h] [rbp-130h] BYREF
  __int64 v112; // [rsp+120h] [rbp-120h] BYREF
  __m128i *v113; // [rsp+130h] [rbp-110h] BYREF
  __int64 v114; // [rsp+138h] [rbp-108h]
  __m128i v115; // [rsp+140h] [rbp-100h] BYREF
  __m128i *v116; // [rsp+150h] [rbp-F0h] BYREF
  __int64 v117; // [rsp+158h] [rbp-E8h]
  __m128i v118; // [rsp+160h] [rbp-E0h] BYREF
  __int64 v119; // [rsp+170h] [rbp-D0h] BYREF
  __int64 v120; // [rsp+178h] [rbp-C8h]
  _QWORD v121[2]; // [rsp+180h] [rbp-C0h] BYREF
  unsigned __int64 v122; // [rsp+190h] [rbp-B0h] BYREF
  __int64 v123; // [rsp+198h] [rbp-A8h]
  __m128i v124; // [rsp+1A0h] [rbp-A0h] BYREF
  __m128i *v125; // [rsp+1B0h] [rbp-90h] BYREF
  __int64 v126; // [rsp+1B8h] [rbp-88h]
  __m128i v127; // [rsp+1C0h] [rbp-80h] BYREF
  __int16 v128; // [rsp+1D0h] [rbp-70h]
  __int64 *v129; // [rsp+1E0h] [rbp-60h] BYREF
  unsigned int v130; // [rsp+1E8h] [rbp-58h]
  const char *v131; // [rsp+1F0h] [rbp-50h]
  __int16 v132; // [rsp+200h] [rbp-40h]

  v11 = *(unsigned __int8 *)(a2 + 8);
  if ( (_BYTE)v11 == 13 )
  {
    HIBYTE(v132) = 1;
    v14 = "functions are not values, refer to them as pointers";
    goto LABEL_4;
  }
  v13 = *a3;
  switch ( (int)v13 )
  {
    case 0:
      if ( !a5 )
        goto LABEL_94;
      v48 = sub_121DDC0((__int64)a5, a3[4], a2, *((_QWORD *)a3 + 1));
      *a4 = v48;
      LOBYTE(v7) = v48 == 0;
      return v7;
    case 1:
      v21 = a3[4];
      v22 = sub_121C740((__int64)a1, v21, a2, *((_QWORD *)a3 + 1));
      *a4 = v22;
      if ( !v22 )
        goto LABEL_25;
      goto LABEL_23;
    case 2:
      if ( a5 )
      {
        v24 = sub_121D710(a5, (__int64)(a3 + 8), a2, *((_QWORD *)a3 + 1));
        *a4 = v24;
        LOBYTE(v7) = v24 == 0;
        return v7;
      }
LABEL_94:
      HIBYTE(v132) = 1;
      v14 = "invalid use of function-local name";
      goto LABEL_4;
    case 3:
      v21 = (__int64)(a3 + 8);
      v22 = sub_121C160(a1, (__int64)(a3 + 8), a2, *((_QWORD *)a3 + 1));
      *a4 = v22;
      if ( v22 )
      {
LABEL_23:
        if ( *((_BYTE *)a3 + 152) )
        {
          v22 = sub_ACCB80(v22, v21, v23);
          *a4 = v22;
        }
      }
LABEL_25:
      LOBYTE(v7) = v22 == 0;
      return v7;
    case 4:
      if ( (_BYTE)v11 != 12 )
      {
        HIBYTE(v132) = 1;
        v14 = "integer constant must have integer type";
        goto LABEL_4;
      }
      v68 = sub_BCAE30(a2);
      v126 = v69;
      v125 = (__m128i *)v68;
      v70 = sub_CA1930(&v125);
      v71 = *((_BYTE *)a3 + 108);
      v72 = (char **)(a3 + 24);
      if ( v71 )
        sub_C44AB0((__int64)&v129, (__int64)v72, v70);
      else
        sub_C44B10((__int64)&v129, v72, v70);
      v73 = v130;
      v74 = (__int64)v129;
      if ( a3[26] > 0x40 )
      {
        v75 = *((_QWORD *)a3 + 12);
        if ( v75 )
        {
          v87 = (__int64)v129;
          v91 = v130;
          j_j___libc_free_0_0(v75);
          v74 = v87;
          v73 = v91;
        }
      }
      *((_BYTE *)a3 + 108) = v71;
      v7 = 0;
      *((_QWORD *)a3 + 12) = v74;
      a3[26] = v73;
      *a4 = sub_ACCFD0(*a1, (__int64)(a3 + 24));
      return v7;
    case 5:
      if ( (unsigned __int8)v11 > 3u && (_BYTE)v11 != 5 && (v11 = (unsigned int)v11 & 0xFFFFFFFD, (_BYTE)v11 != 4)
        || (v90 = (__int64 *)(a3 + 28),
            v7 = sub_AC36F0(a2, (_QWORD *)a3 + 14, (unsigned int)v13, v11, (__int64)a5),
            !(_BYTE)v7) )
      {
        HIBYTE(v132) = 1;
        v14 = "floating point constant invalid for type";
LABEL_4:
        v129 = (__int64 *)v14;
        LOBYTE(v132) = 3;
LABEL_5:
        sub_11FD800((__int64)(a1 + 22), *((_QWORD *)a3 + 1), (__int64)&v129, 1);
        return 1;
      }
      v86 = (__int64 *)*((_QWORD *)a3 + 14);
      v41 = (__int64 *)sub_C33320();
      if ( v41 != v86 )
        goto LABEL_69;
      v88 = v41;
      v80 = a3 + 28;
      v85 = (__int64 *)sub_C33340();
      if ( v88 == v85 )
        v80 = (_BYTE *)*((_QWORD *)a3 + 15);
      v89 = sub_C35FD0(v80);
      v81 = *(_BYTE *)(a2 + 8);
      if ( v81 )
      {
        if ( v81 == 1 )
        {
          v83 = sub_C33300();
        }
        else
        {
          if ( v81 != 2 )
            goto LABEL_175;
          v83 = sub_C33310();
        }
      }
      else
      {
        v83 = sub_C332F0();
      }
      sub_C41640(v90, v83, 1, (bool *)&v122);
LABEL_175:
      if ( !v89 )
        goto LABEL_69;
      if ( v85 == *((__int64 **)a3 + 14) )
        sub_C3E660((__int64)&v125, (__int64)v90);
      else
        sub_C3A850((__int64)&v125, v90);
      v82 = *((_QWORD *)a3 + 14);
      if ( v85 == (__int64 *)v82 )
      {
        v84 = (*(_BYTE *)(*((_QWORD *)a3 + 15) + 20LL) & 8) != 0;
        sub_C3C500(&v129, (__int64)v85);
      }
      else
      {
        v84 = (a3[33] & 8) != 0;
        sub_C373C0(&v129, v82);
      }
      if ( v85 == v129 )
        sub_C3D480((__int64)&v129, 1u, v84, (unsigned __int64 *)&v125);
      else
        sub_C36070((__int64)&v129, 1, v84, (unsigned __int64 *)&v125);
      if ( v85 == *((__int64 **)a3 + 14) )
      {
        if ( v85 == v129 )
        {
          sub_969EE0((__int64)v90);
          sub_C3C840(v90, &v129);
          goto LABEL_185;
        }
        sub_969EE0((__int64)v90);
LABEL_192:
        if ( v85 == v129 )
          sub_C3C840(v90, &v129);
        else
          sub_C338E0((__int64)v90, (__int64)&v129);
        goto LABEL_185;
      }
      if ( v85 == v129 )
      {
        sub_C338F0((__int64)v90);
        goto LABEL_192;
      }
      sub_C33870((__int64)v90, (__int64)&v129);
LABEL_185:
      sub_91D830(&v129);
      sub_969240((__int64 *)&v125);
LABEL_69:
      v42 = sub_AC8EA0(*a1, v90);
      *a4 = v42;
      if ( a2 == *(_QWORD *)(v42 + 8) )
        return 0;
      sub_1207630(v92, a2);
      v43 = (__m128i *)sub_2241130(v92, 0, 0, "floating point constant does not have type '", 44);
      v94 = &v96;
      if ( (__m128i *)v43->m128i_i64[0] == &v43[1] )
      {
        v96 = _mm_loadu_si128(v43 + 1);
      }
      else
      {
        v94 = (__m128i *)v43->m128i_i64[0];
        v96.m128i_i64[0] = v43[1].m128i_i64[0];
      }
      v44 = v43->m128i_i64[1];
      v43[1].m128i_i8[0] = 0;
      v95 = v44;
      v43->m128i_i64[0] = (__int64)v43[1].m128i_i64;
      v43->m128i_i64[1] = 0;
      if ( v95 == 0x3FFFFFFFFFFFFFFFLL )
LABEL_202:
        sub_4262D8((__int64)"basic_string::append");
      v45 = (__m128i *)sub_2241490(&v94, "'", 1, v44);
      v125 = &v127;
      if ( (__m128i *)v45->m128i_i64[0] == &v45[1] )
      {
        v127 = _mm_loadu_si128(v45 + 1);
      }
      else
      {
        v125 = (__m128i *)v45->m128i_i64[0];
        v127.m128i_i64[0] = v45[1].m128i_i64[0];
      }
      v126 = v45->m128i_i64[1];
      v45->m128i_i64[0] = (__int64)v45[1].m128i_i64;
      v45->m128i_i64[1] = 0;
      v45[1].m128i_i8[0] = 0;
      v46 = *((_QWORD *)a3 + 1);
      v132 = 260;
      v129 = (__int64 *)&v125;
      sub_11FD800((__int64)(a1 + 22), v46, (__int64)&v129, 1);
      if ( v125 != &v127 )
        j_j___libc_free_0(v125, v127.m128i_i64[0] + 1);
      if ( v94 != &v96 )
        j_j___libc_free_0(v94, v96.m128i_i64[0] + 1);
      if ( (__int64 *)v92[0] != &v93 )
        j_j___libc_free_0(v92[0], v93 + 1);
      return v7;
    case 6:
      if ( (_BYTE)v11 != 14 )
      {
        HIBYTE(v132) = 1;
        v14 = "null must be a pointer type";
        goto LABEL_4;
      }
      v7 = 0;
      *a4 = sub_AC9EC0((__int64 **)a2);
      return v7;
    case 7:
      LOBYTE(a6) = (_BYTE)v11 != 7;
      LOBYTE(v6) = (_BYTE)v11 != 13;
      v18 = v6 & a6 ^ 1;
      LOBYTE(v18) = ((_BYTE)v11 == 8) | v18;
      v7 = v18;
      if ( (_BYTE)v18 )
      {
        HIBYTE(v132) = 1;
        v19 = "invalid type for undef constant";
        goto LABEL_16;
      }
      *a4 = sub_ACA8A0((__int64 **)a2);
      return v7;
    case 8:
      LOBYTE(a6) = (_BYTE)v11 != 13;
      LOBYTE(v6) = (_BYTE)v11 != 7;
      v40 = v6 & a6 ^ 1;
      LOBYTE(v40) = ((_BYTE)v11 == 8) | v40;
      v7 = v40;
      if ( (_BYTE)v40 )
      {
        HIBYTE(v132) = 1;
        v19 = "invalid type for null constant";
LABEL_16:
        v20 = *((_QWORD *)a3 + 1);
        v129 = (__int64 *)v19;
        LOBYTE(v132) = 3;
        sub_11FD800((__int64)(a1 + 22), v20, (__int64)&v129, 1);
        return v7;
      }
      if ( (_BYTE)v11 != 20 || (v13 = 1, sub_BCEE90(a2, 1)) )
      {
        *a4 = sub_AD6530(a2, v13);
        return v7;
      }
      HIBYTE(v132) = 1;
      v14 = "invalid type for null constant";
      goto LABEL_4;
    case 9:
      if ( (_BYTE)v11 != 11 )
      {
        HIBYTE(v132) = 1;
        v14 = "invalid type for none constant";
        goto LABEL_4;
      }
      v7 = 0;
      *a4 = sub_AD6530(a2, v13);
      return v7;
    case 10:
      LOBYTE(a6) = (_BYTE)v11 != 7;
      LOBYTE(v6) = (_BYTE)v11 != 13;
      v47 = v6 & a6 ^ 1;
      LOBYTE(v47) = ((_BYTE)v11 == 8) | v47;
      v7 = v47;
      if ( !(_BYTE)v47 )
        goto LABEL_96;
      HIBYTE(v132) = 1;
      v19 = "invalid type for poison constant";
      goto LABEL_16;
    case 11:
      if ( (_BYTE)v11 != 16 || *(_QWORD *)(a2 + 32) )
      {
        HIBYTE(v132) = 1;
        v14 = "invalid empty array initializer";
        goto LABEL_4;
      }
LABEL_96:
      v7 = 0;
      *a4 = sub_ACADE0((__int64 **)a2);
      return v7;
    case 12:
      v25 = *((_QWORD *)a3 + 17);
      if ( a2 == *(_QWORD *)(v25 + 8) )
        goto LABEL_92;
      sub_1207630((__int64 *)&v105, a2);
      sub_1207630(v97, *(_QWORD *)(*((_QWORD *)a3 + 17) + 8LL));
      v26 = (__m128i *)sub_2241130(v97, 0, 0, "constant expression type mismatch: got type '", 45);
      v99 = &v101;
      if ( (__m128i *)v26->m128i_i64[0] == &v26[1] )
      {
        v101 = _mm_loadu_si128(v26 + 1);
      }
      else
      {
        v99 = (__m128i *)v26->m128i_i64[0];
        v101.m128i_i64[0] = v26[1].m128i_i64[0];
      }
      v27 = v26->m128i_i64[1];
      v26[1].m128i_i8[0] = 0;
      v100 = v27;
      v26->m128i_i64[0] = (__int64)v26[1].m128i_i64;
      v26->m128i_i64[1] = 0;
      if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v100) <= 0xF )
        goto LABEL_202;
      v28 = (__m128i *)sub_2241490(&v99, "' but expected '", 16, v27);
      v102 = &v104;
      if ( (__m128i *)v28->m128i_i64[0] == &v28[1] )
      {
        v104 = _mm_loadu_si128(v28 + 1);
      }
      else
      {
        v102 = (__m128i *)v28->m128i_i64[0];
        v104.m128i_i64[0] = v28[1].m128i_i64[0];
      }
      v29 = v28->m128i_i64[1];
      v28[1].m128i_i8[0] = 0;
      v103 = v29;
      v28->m128i_i64[0] = (__int64)v28[1].m128i_i64;
      v30 = v102;
      v28->m128i_i64[1] = 0;
      v31 = 15;
      v32 = 15;
      if ( v30 != &v104 )
        v32 = v104.m128i_i64[0];
      v33 = v103 + v106;
      if ( v103 + v106 <= v32 )
        goto LABEL_45;
      if ( v105 != v107 )
        v31 = v107[0];
      if ( v33 <= v31 )
      {
        v34 = (__m128i *)sub_2241130(&v105, 0, 0, v30, v103);
        v108 = &v110;
        v35 = (__m128i *)v34->m128i_i64[0];
        v36 = v34 + 1;
        if ( (__m128i *)v34->m128i_i64[0] != &v34[1] )
          goto LABEL_46;
      }
      else
      {
LABEL_45:
        v34 = (__m128i *)sub_2241490(&v102, v105, v106, v33);
        v108 = &v110;
        v35 = (__m128i *)v34->m128i_i64[0];
        v36 = v34 + 1;
        if ( (__m128i *)v34->m128i_i64[0] != &v34[1] )
        {
LABEL_46:
          v108 = v35;
          v110.m128i_i64[0] = v34[1].m128i_i64[0];
          goto LABEL_47;
        }
      }
      v110 = _mm_loadu_si128(v34 + 1);
LABEL_47:
      v37 = v34->m128i_i64[1];
      v109 = v37;
      v34->m128i_i64[0] = (__int64)v36;
      v34->m128i_i64[1] = 0;
      v34[1].m128i_i8[0] = 0;
      if ( v109 == 0x3FFFFFFFFFFFFFFFLL )
        goto LABEL_202;
      v38 = (__m128i *)sub_2241490(&v108, "'", 1, v37);
      v125 = &v127;
      if ( (__m128i *)v38->m128i_i64[0] == &v38[1] )
      {
        v127 = _mm_loadu_si128(v38 + 1);
      }
      else
      {
        v125 = (__m128i *)v38->m128i_i64[0];
        v127.m128i_i64[0] = v38[1].m128i_i64[0];
      }
      v126 = v38->m128i_i64[1];
      v38->m128i_i64[0] = (__int64)v38[1].m128i_i64;
      v38->m128i_i64[1] = 0;
      v38[1].m128i_i8[0] = 0;
      v132 = 260;
      v39 = *((_QWORD *)a3 + 1);
      v129 = (__int64 *)&v125;
      sub_11FD800((__int64)(a1 + 22), v39, (__int64)&v129, 1);
      if ( v125 != &v127 )
        j_j___libc_free_0(v125, v127.m128i_i64[0] + 1);
      if ( v108 != &v110 )
        j_j___libc_free_0(v108, v110.m128i_i64[0] + 1);
      if ( v102 != &v104 )
        j_j___libc_free_0(v102, v104.m128i_i64[0] + 1);
      if ( v99 != &v101 )
        j_j___libc_free_0(v99, v101.m128i_i64[0] + 1);
      if ( (__int64 *)v97[0] != &v98 )
        j_j___libc_free_0(v97[0], v98 + 1);
      if ( v105 != v107 )
        j_j___libc_free_0(v105, v107[0] + 1LL);
      return 1;
    case 13:
      if ( (_BYTE)v11 == 18 )
      {
        v76 = *(__int64 **)(a2 + 16);
        v50 = *((_QWORD *)a3 + 17);
        v51 = *v76;
        if ( *(_QWORD *)(v50 + 8) != *v76 )
          goto LABEL_98;
      }
      else
      {
        if ( (unsigned __int8)v11 != 17 )
        {
          HIBYTE(v132) = 1;
          v14 = "vector constant must have vector type";
          goto LABEL_4;
        }
        v49 = *(__int64 **)(a2 + 16);
        v50 = *((_QWORD *)a3 + 17);
        v51 = *v49;
        if ( *v49 != *(_QWORD *)(v50 + 8) )
        {
LABEL_98:
          sub_1207630(&v119, v51);
          sub_1207630(v111, *(_QWORD *)(*((_QWORD *)a3 + 17) + 8LL));
          v52 = (__m128i *)sub_2241130(v111, 0, 0, "constant expression type mismatch: got type '", 45);
          v113 = &v115;
          if ( (__m128i *)v52->m128i_i64[0] == &v52[1] )
          {
            v115 = _mm_loadu_si128(v52 + 1);
          }
          else
          {
            v113 = (__m128i *)v52->m128i_i64[0];
            v115.m128i_i64[0] = v52[1].m128i_i64[0];
          }
          v53 = v52->m128i_i64[1];
          v52[1].m128i_i8[0] = 0;
          v114 = v53;
          v52->m128i_i64[0] = (__int64)v52[1].m128i_i64;
          v52->m128i_i64[1] = 0;
          if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v114) <= 0xF )
            goto LABEL_202;
          v54 = (__m128i *)sub_2241490(&v113, "' but expected '", 16, v53);
          v116 = &v118;
          if ( (__m128i *)v54->m128i_i64[0] == &v54[1] )
          {
            v118 = _mm_loadu_si128(v54 + 1);
          }
          else
          {
            v116 = (__m128i *)v54->m128i_i64[0];
            v118.m128i_i64[0] = v54[1].m128i_i64[0];
          }
          v55 = v54->m128i_i64[1];
          v54[1].m128i_i8[0] = 0;
          v117 = v55;
          v54->m128i_i64[0] = (__int64)v54[1].m128i_i64;
          v56 = v116;
          v54->m128i_i64[1] = 0;
          v57 = 15;
          v58 = 15;
          if ( v56 != &v118 )
            v58 = v118.m128i_i64[0];
          v59 = v117 + v120;
          if ( v117 + v120 <= v58 )
            goto LABEL_109;
          if ( (_QWORD *)v119 != v121 )
            v57 = v121[0];
          if ( v59 <= v57 )
          {
            v60 = (__m128i *)sub_2241130(&v119, 0, 0, v56, v117);
            v122 = (unsigned __int64)&v124;
            v61 = v60->m128i_i64[0];
            v62 = v60 + 1;
            if ( (__m128i *)v60->m128i_i64[0] != &v60[1] )
              goto LABEL_110;
          }
          else
          {
LABEL_109:
            v60 = (__m128i *)sub_2241490(&v116, v119, v120, v59);
            v122 = (unsigned __int64)&v124;
            v61 = v60->m128i_i64[0];
            v62 = v60 + 1;
            if ( (__m128i *)v60->m128i_i64[0] != &v60[1] )
            {
LABEL_110:
              v122 = v61;
              v124.m128i_i64[0] = v60[1].m128i_i64[0];
LABEL_111:
              v63 = v60->m128i_i64[1];
              v123 = v63;
              v60->m128i_i64[0] = (__int64)v62;
              v60->m128i_i64[1] = 0;
              v60[1].m128i_i8[0] = 0;
              if ( v123 == 0x3FFFFFFFFFFFFFFFLL )
                goto LABEL_202;
              v64 = (__m128i *)sub_2241490(&v122, "'", 1, v63);
              v125 = &v127;
              if ( (__m128i *)v64->m128i_i64[0] == &v64[1] )
              {
                v127 = _mm_loadu_si128(v64 + 1);
              }
              else
              {
                v125 = (__m128i *)v64->m128i_i64[0];
                v127.m128i_i64[0] = v64[1].m128i_i64[0];
              }
              v126 = v64->m128i_i64[1];
              v64->m128i_i64[0] = (__int64)v64[1].m128i_i64;
              v64->m128i_i64[1] = 0;
              v64[1].m128i_i8[0] = 0;
              v65 = *((_QWORD *)a3 + 1);
              v132 = 260;
              v129 = (__int64 *)&v125;
              sub_11FD800((__int64)(a1 + 22), v65, (__int64)&v129, 1);
              if ( v125 != &v127 )
                j_j___libc_free_0(v125, v127.m128i_i64[0] + 1);
              if ( (__m128i *)v122 != &v124 )
                j_j___libc_free_0(v122, v124.m128i_i64[0] + 1);
              if ( v116 != &v118 )
                j_j___libc_free_0(v116, v118.m128i_i64[0] + 1);
              if ( v113 != &v115 )
                j_j___libc_free_0(v113, v115.m128i_i64[0] + 1);
              if ( (__int64 *)v111[0] != &v112 )
                j_j___libc_free_0(v111[0], v112 + 1);
              if ( (_QWORD *)v119 != v121 )
                j_j___libc_free_0(v119, v121[0] + 1LL);
              return 1;
            }
          }
          v124 = _mm_loadu_si128(v60 + 1);
          goto LABEL_111;
        }
      }
      v77 = *(_DWORD *)(a2 + 32);
      BYTE4(v129) = (_BYTE)v11 == 18;
      v7 = 0;
      LODWORD(v129) = v77;
      *a4 = sub_AD5E10((__int64)v129, (unsigned __int8 *)v50);
      return v7;
    case 14:
      v17 = *((_QWORD *)a3 + 3);
      if ( !v17 )
      {
        HIBYTE(v132) = 1;
        v14 = "invalid type for inline asm constraint string";
        goto LABEL_4;
      }
      sub_B43120(&v119, v17, *((char **)a3 + 8), *((_QWORD *)a3 + 9), (__int64)a5);
      if ( (v119 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      {
        v66 = v119 & 0xFFFFFFFFFFFFFFFELL | 1;
        v119 = 0;
        v122 = v66;
        sub_C64870((__int64)&v125, (__int64 *)&v122);
        v67 = *((_QWORD *)a3 + 1);
        v132 = 260;
        v129 = (__int64 *)&v125;
        sub_11FD800((__int64)(a1 + 22), v67, (__int64)&v129, 1);
        if ( v125 != &v127 )
        {
          v67 = v127.m128i_i64[0] + 1;
          j_j___libc_free_0(v125, v127.m128i_i64[0] + 1);
        }
        if ( (v122 & 1) != 0 || (v122 & 0xFFFFFFFFFFFFFFFELL) != 0 )
          sub_C63C30(&v122, v67);
        v7 = (v119 & 1) == 0;
        if ( (v119 & 1) != 0 || (v119 & 0xFFFFFFFFFFFFFFFELL) != 0 )
          sub_C63C30(&v119, v67);
      }
      else
      {
        v7 = 0;
        *a4 = sub_B41A60(
                *((_QWORD ***)a3 + 3),
                *((_QWORD *)a3 + 4),
                *((_QWORD *)a3 + 5),
                *((_QWORD *)a3 + 8),
                *((_QWORD *)a3 + 9),
                a3[4] & 1,
                (a3[4] & 2) != 0,
                (a3[4] >> 2) & 1,
                (a3[4] & 8) != 0);
      }
      return v7;
    case 15:
    case 16:
      if ( (_BYTE)v11 != 15 )
      {
        HIBYTE(v132) = 1;
        v14 = "constant expression type mismatch";
        goto LABEL_4;
      }
      v16 = a3[4];
      if ( (_DWORD)v16 != *(_DWORD *)(a2 + 12) )
      {
        HIBYTE(v132) = 1;
        v14 = "initializer with struct type has wrong # elements";
        goto LABEL_4;
      }
      if ( ((_DWORD)v13 == 16) != ((*(_DWORD *)(a2 + 8) & 0x200) != 0) )
      {
        HIBYTE(v132) = 1;
        v14 = "packed'ness of initializer and type don't match";
        goto LABEL_4;
      }
      v78 = (__int64 *)*((_QWORD *)a3 + 18);
      if ( (_DWORD)v16 )
      {
        for ( i = 0; i != v16; ++i )
        {
          if ( *(_QWORD *)(*(_QWORD *)(a2 + 16) + 8 * i) != *(_QWORD *)(v78[i] + 8) )
          {
            v127.m128i_i32[0] = i;
            v128 = 2307;
            v129 = (__int64 *)&v125;
            v125 = (__m128i *)"element ";
            v131 = " of struct initializer doesn't match struct element type";
            v132 = 770;
            goto LABEL_5;
          }
        }
      }
      else
      {
        v16 = 0;
      }
      v25 = sub_AD24A0((__int64 **)a2, v78, v16);
LABEL_92:
      *a4 = v25;
      return 0;
    default:
      BUG();
  }
}
