// Function: sub_31FFCF0
// Address: 0x31ffcf0
//
__int64 __fastcall sub_31FFCF0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // rdi
  void (*v7)(); // rax
  __int64 v8; // rax
  __int64 v9; // rdi
  void (*v10)(); // rax
  unsigned __int64 *v11; // rbx
  unsigned __int64 v12; // r15
  unsigned int v13; // esi
  int v14; // r10d
  __int64 v15; // rdx
  __int64 v16; // rdi
  unsigned int i; // eax
  _QWORD *v18; // r8
  __int64 v19; // rcx
  unsigned int *v20; // r8
  void (*v21)(void); // rax
  unsigned __int64 v22; // r13
  unsigned __int64 v23; // rsi
  unsigned __int8 v24; // al
  unsigned __int64 *v25; // rdx
  unsigned int v26; // eax
  _QWORD *v27; // r8
  __int64 v28; // rax
  unsigned __int64 v29; // rax
  unsigned __int8 v30; // al
  unsigned __int8 v31; // dl
  const char *v32; // rdi
  __int64 v33; // rax
  __int64 v34; // rdx
  unsigned __int8 v35; // al
  __int64 v36; // rdi
  __int64 v37; // rax
  __int64 v38; // rcx
  __int64 v39; // r9
  char v40; // al
  unsigned __int64 v41; // rdx
  __int64 *v42; // rdi
  __int64 v43; // rax
  void (*v44)(); // rdx
  void (*v45)(); // rax
  const char *v46; // rcx
  __int64 *v47; // rdi
  void (*v48)(); // rax
  __int64 v49; // rax
  void (*v50)(); // rdx
  void (*v51)(); // rax
  char v53; // dl
  char v54; // al
  _QWORD *v55; // rdi
  char v56; // dl
  __m128i *v57; // rdi
  char v58; // dl
  __m128i v59; // xmm7
  __m128i *v60; // rdi
  __m128i v61; // xmm2
  __m128i v62; // xmm3
  __m128i v63; // xmm6
  __m128i v64; // xmm7
  __m128i v65; // xmm4
  __m128i v66; // xmm5
  __m128i v67; // xmm1
  __m128i v68; // xmm5
  __int64 v69; // r8
  int v70; // edx
  __int64 v71; // rax
  unsigned int v72; // eax
  int v73; // eax
  __int64 v74; // [rsp+0h] [rbp-270h]
  __int64 v75; // [rsp+8h] [rbp-268h]
  __int64 v76; // [rsp+28h] [rbp-248h]
  __int64 v77; // [rsp+30h] [rbp-240h]
  unsigned __int64 *v78; // [rsp+38h] [rbp-238h]
  __int64 v79; // [rsp+40h] [rbp-230h]
  __int64 v80; // [rsp+48h] [rbp-228h]
  __int64 v81; // [rsp+50h] [rbp-220h]
  __int64 v82; // [rsp+58h] [rbp-218h]
  unsigned int v83; // [rsp+64h] [rbp-20Ch]
  __int64 v84; // [rsp+68h] [rbp-208h]
  _QWORD *v85; // [rsp+70h] [rbp-200h]
  _QWORD *v86; // [rsp+70h] [rbp-200h]
  void (*v87)(); // [rsp+78h] [rbp-1F8h]
  unsigned int v88; // [rsp+8Ch] [rbp-1E4h]
  __m128i v89; // [rsp+90h] [rbp-1E0h] BYREF
  unsigned __int128 v90; // [rsp+A0h] [rbp-1D0h] BYREF
  __int64 v91; // [rsp+B0h] [rbp-1C0h]
  _QWORD v92[4]; // [rsp+C0h] [rbp-1B0h] BYREF
  char v93; // [rsp+E0h] [rbp-190h]
  char v94; // [rsp+E1h] [rbp-18Fh]
  __m128i v95; // [rsp+F0h] [rbp-180h] BYREF
  __m128i v96; // [rsp+100h] [rbp-170h] BYREF
  __int64 v97; // [rsp+110h] [rbp-160h]
  const char *v98; // [rsp+120h] [rbp-150h] BYREF
  __int64 v99; // [rsp+128h] [rbp-148h]
  __int16 v100; // [rsp+140h] [rbp-130h]
  __m128i v101; // [rsp+150h] [rbp-120h] BYREF
  __m128i v102; // [rsp+160h] [rbp-110h] BYREF
  __int64 v103; // [rsp+170h] [rbp-100h]
  __m128i v104; // [rsp+180h] [rbp-F0h] BYREF
  __m128i v105; // [rsp+190h] [rbp-E0h] BYREF
  __int64 v106; // [rsp+1A0h] [rbp-D0h]
  __m128i v107; // [rsp+1B0h] [rbp-C0h] BYREF
  __m128i v108; // [rsp+1C0h] [rbp-B0h] BYREF
  __int64 v109; // [rsp+1D0h] [rbp-A0h]
  __m128i v110; // [rsp+1E0h] [rbp-90h] BYREF
  __m128i v111; // [rsp+1F0h] [rbp-80h] BYREF
  __int64 v112; // [rsp+200h] [rbp-70h]
  __m128i v113; // [rsp+210h] [rbp-60h] BYREF
  __m128i v114; // [rsp+220h] [rbp-50h]
  __int64 v115; // [rsp+230h] [rbp-40h]

  v6 = *(_QWORD *)(a1 + 528);
  v7 = *(void (**)())(*(_QWORD *)v6 + 120LL);
  v113.m128i_i64[0] = (__int64)"Inlinee lines subsection";
  LOWORD(v115) = 259;
  if ( v7 != nullsub_98 )
    ((void (__fastcall *)(__int64, __m128i *, __int64))v7)(v6, &v113, 1);
  v8 = sub_31F8650(a1, 246, a3, a4, a5);
  v9 = *(_QWORD *)(a1 + 528);
  v75 = v8;
  v10 = *(void (**)())(*(_QWORD *)v9 + 120LL);
  v113.m128i_i64[0] = (__int64)"Inlinee lines signature";
  LOWORD(v115) = 259;
  if ( v10 != nullsub_98 )
  {
    ((void (__fastcall *)(__int64, __m128i *, __int64))v10)(v9, &v113, 1);
    v9 = *(_QWORD *)(a1 + 528);
  }
  (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v9 + 536LL))(v9, 0, 4);
  v11 = *(unsigned __int64 **)(a1 + 1168);
  v78 = &v11[*(unsigned int *)(a1 + 1176)];
  if ( v11 == v78 )
    return sub_31F8740(a1, v75);
  v74 = a1 + 1216;
  do
  {
    v12 = *v11;
    v13 = *(_DWORD *)(a1 + 1240);
    v113 = (__m128i)*v11;
    if ( !v13 )
    {
      ++*(_QWORD *)(a1 + 1216);
      v110.m128i_i64[0] = 0;
LABEL_93:
      v13 *= 2;
      goto LABEL_94;
    }
    v14 = 1;
    v15 = *(_QWORD *)(a1 + 1224);
    v16 = 0;
    for ( i = (v13 - 1) & (969526130 * (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4))); ; i = (v13 - 1) & v72 )
    {
      v18 = (_QWORD *)(v15 + 24LL * i);
      v19 = *v18;
      if ( v12 == *v18 && !v18[1] )
      {
        v20 = (unsigned int *)(v18 + 2);
        goto LABEL_18;
      }
      if ( v19 == -4096 )
        break;
      if ( v19 == -8192 && v18[1] == -8192 && !v16 )
        v16 = v15 + 24LL * i;
LABEL_101:
      v72 = v14 + i;
      ++v14;
    }
    if ( v18[1] != -4096 )
      goto LABEL_101;
    v73 = *(_DWORD *)(a1 + 1232);
    if ( !v16 )
      v16 = (__int64)v18;
    ++*(_QWORD *)(a1 + 1216);
    v70 = v73 + 1;
    v110.m128i_i64[0] = v16;
    if ( 4 * (v73 + 1) >= 3 * v13 )
      goto LABEL_93;
    v69 = v12;
    if ( v13 - *(_DWORD *)(a1 + 1236) - v70 <= v13 >> 3 )
    {
LABEL_94:
      sub_31FE9B0(v74, v13);
      sub_31FB320(v74, v113.m128i_i64, (__int64 **)&v110);
      v69 = v113.m128i_i64[0];
      v16 = v110.m128i_i64[0];
      v70 = *(_DWORD *)(a1 + 1232) + 1;
    }
    *(_DWORD *)(a1 + 1232) = v70;
    if ( *(_QWORD *)v16 != -4096 || *(_QWORD *)(v16 + 8) != -4096 )
      --*(_DWORD *)(a1 + 1236);
    *(_QWORD *)v16 = v69;
    v71 = v113.m128i_i64[1];
    v20 = (unsigned int *)(v16 + 16);
    *(_DWORD *)(v16 + 16) = 0;
    *(_QWORD *)(v16 + 8) = v71;
LABEL_18:
    v88 = *v20;
    v21 = *(void (**)(void))(**(_QWORD **)(a1 + 528) + 160LL);
    if ( v21 != nullsub_99 )
      v21();
    v22 = v12 - 16;
    v23 = v12;
    if ( *(_BYTE *)v12 != 16 )
    {
      v24 = *(_BYTE *)(v12 - 16);
      if ( (v24 & 2) != 0 )
        v25 = *(unsigned __int64 **)(v12 - 32);
      else
        v25 = (unsigned __int64 *)(v22 - 8LL * ((v24 >> 2) & 0xF));
      v23 = *v25;
    }
    v26 = sub_31FF830(a1, v23);
    v27 = *(_QWORD **)(a1 + 528);
    v83 = v26;
    v28 = *v27;
    v104.m128i_i64[0] = 58;
    v87 = *(void (**)())(v28 + 120);
    LODWORD(v28) = *(_DWORD *)(v12 + 16);
    LOWORD(v112) = 265;
    v110.m128i_i32[0] = v28;
    LOWORD(v106) = 264;
    v29 = v12;
    if ( *(_BYTE *)v12 == 16 )
      goto LABEL_27;
    v30 = *(_BYTE *)(v12 - 16);
    if ( (v30 & 2) == 0 )
    {
      v29 = *(_QWORD *)(v22 - 8LL * ((v30 >> 2) & 0xF));
      if ( !v29 )
        goto LABEL_70;
LABEL_27:
      v31 = *(_BYTE *)(v29 - 16);
      if ( (v31 & 2) != 0 )
      {
        v32 = **(const char ***)(v29 - 32);
        if ( v32 )
        {
LABEL_29:
          v85 = v27;
          v33 = sub_B91420((__int64)v32);
          v27 = v85;
          v32 = (const char *)v33;
          goto LABEL_30;
        }
      }
      else
      {
        v32 = *(const char **)(v29 - 16 - 8LL * ((v31 >> 2) & 0xF));
        if ( v32 )
          goto LABEL_29;
      }
      v34 = 0;
      goto LABEL_30;
    }
    v29 = **(_QWORD **)(v12 - 32);
    if ( v29 )
      goto LABEL_27;
LABEL_70:
    v34 = 0;
    v32 = byte_3F871B3;
LABEL_30:
    v98 = v32;
    v100 = 261;
    v99 = v34;
    v94 = 1;
    v92[0] = " starts at ";
    v93 = 3;
    v35 = *(_BYTE *)(v12 - 16);
    if ( (v35 & 2) == 0 )
    {
      v36 = *(_QWORD *)(v12 - 8LL * ((v35 >> 2) & 0xF));
      if ( !v36 )
        goto LABEL_51;
LABEL_32:
      v86 = v27;
      v37 = sub_B91420(v36);
      v38 = (__int64)"Inlined function ";
      v27 = v86;
      v39 = v37;
      v40 = v93;
      LOWORD(v91) = 1283;
      v89.m128i_i64[0] = (__int64)"Inlined function ";
      v90 = __PAIR128__(v41, v39);
      if ( !v93 )
      {
        v23 = 256;
        LOWORD(v97) = 256;
LABEL_34:
        v38 = 256;
        LOWORD(v103) = 256;
LABEL_35:
        LOWORD(v109) = 256;
        goto LABEL_36;
      }
      if ( v93 != 1 )
        goto LABEL_52;
      v67 = _mm_loadu_si128((const __m128i *)&v90);
      v53 = v100;
      v95 = _mm_loadu_si128(&v89);
      v97 = v91;
      v96 = v67;
      if ( !(_BYTE)v100 )
        goto LABEL_34;
      if ( (_BYTE)v100 == 1 )
        goto LABEL_79;
      if ( BYTE1(v97) == 1 )
      {
        v77 = v95.m128i_i64[1];
        v23 = v95.m128i_i64[0];
        v54 = 3;
        if ( HIBYTE(v100) == 1 )
          goto LABEL_90;
      }
      else
      {
LABEL_56:
        v23 = (unsigned __int64)&v95;
        v54 = 2;
        if ( HIBYTE(v100) == 1 )
        {
LABEL_90:
          v55 = v98;
          v76 = v99;
LABEL_58:
          BYTE1(v103) = v53;
          v56 = v106;
          v101.m128i_i64[0] = v23;
          v101.m128i_i64[1] = v77;
          v38 = v76;
          v102.m128i_i64[0] = (__int64)v55;
          v102.m128i_i64[1] = v76;
          LOBYTE(v103) = v54;
          if ( (_BYTE)v106 )
            goto LABEL_59;
          goto LABEL_35;
        }
      }
      v55 = &v98;
      v53 = 2;
      goto LABEL_58;
    }
    v36 = *(_QWORD *)(*(_QWORD *)(v12 - 32) + 16LL);
    if ( v36 )
      goto LABEL_32;
LABEL_51:
    v90 = 0u;
    LOWORD(v91) = 1283;
    v89.m128i_i64[0] = (__int64)"Inlined function ";
    v40 = 3;
LABEL_52:
    if ( v94 == 1 )
    {
      v23 = v92[0];
      v84 = v92[1];
    }
    else
    {
      v23 = (unsigned __int64)v92;
      v40 = 2;
    }
    v38 = v84;
    v96.m128i_i64[0] = v23;
    v95.m128i_i64[0] = (__int64)&v89;
    v53 = v100;
    v96.m128i_i64[1] = v84;
    LOBYTE(v97) = 2;
    BYTE1(v97) = v40;
    if ( !(_BYTE)v100 )
      goto LABEL_34;
    if ( (_BYTE)v100 != 1 )
      goto LABEL_56;
LABEL_79:
    v54 = v97;
    v61 = _mm_loadu_si128(&v95);
    v62 = _mm_loadu_si128(&v96);
    v103 = v97;
    v101 = v61;
    v102 = v62;
    if ( !(_BYTE)v97 )
      goto LABEL_35;
    v56 = v106;
    if ( !(_BYTE)v106 )
      goto LABEL_35;
    if ( (_BYTE)v97 == 1 )
    {
      v63 = _mm_loadu_si128(&v104);
      v64 = _mm_loadu_si128(&v105);
      v109 = v106;
      v54 = v106;
      v107 = v63;
      v108 = v64;
      goto LABEL_65;
    }
LABEL_59:
    if ( v56 == 1 )
    {
      v54 = v103;
      v65 = _mm_loadu_si128(&v101);
      v66 = _mm_loadu_si128(&v102);
      v109 = v103;
      v107 = v65;
      v108 = v66;
      if ( (_BYTE)v103 )
        goto LABEL_65;
LABEL_36:
      LOWORD(v115) = 256;
LABEL_37:
      if ( v87 != nullsub_98 )
        goto LABEL_68;
      goto LABEL_38;
    }
    if ( BYTE1(v103) == 1 )
    {
      v80 = v101.m128i_i64[1];
      v57 = (__m128i *)v101.m128i_i64[0];
    }
    else
    {
      v57 = &v101;
      v54 = 2;
    }
    if ( BYTE1(v106) == 1 )
    {
      v79 = v104.m128i_i64[1];
      v23 = v104.m128i_i64[0];
    }
    else
    {
      v23 = (unsigned __int64)&v104;
      v56 = 2;
    }
    v107.m128i_i64[0] = (__int64)v57;
    v108.m128i_i64[0] = v23;
    v107.m128i_i64[1] = v80;
    v38 = v79;
    LOBYTE(v109) = v54;
    v108.m128i_i64[1] = v79;
    BYTE1(v109) = v56;
LABEL_65:
    v58 = v112;
    if ( !(_BYTE)v112 )
      goto LABEL_36;
    if ( v54 != 1 )
    {
      if ( (_BYTE)v112 == 1 )
      {
        v68 = _mm_loadu_si128(&v108);
        v113 = _mm_loadu_si128(&v107);
        v115 = v109;
        v114 = v68;
        goto LABEL_37;
      }
      if ( BYTE1(v109) == 1 )
      {
        v82 = v107.m128i_i64[1];
        v60 = (__m128i *)v107.m128i_i64[0];
        if ( BYTE1(v112) == 1 )
          goto LABEL_105;
LABEL_77:
        v23 = (unsigned __int64)&v110;
        v58 = 2;
      }
      else
      {
        v60 = &v107;
        v54 = 2;
        if ( BYTE1(v112) != 1 )
          goto LABEL_77;
LABEL_105:
        v81 = v110.m128i_i64[1];
        v23 = v110.m128i_i64[0];
      }
      v113.m128i_i64[0] = (__int64)v60;
      v114.m128i_i64[0] = v23;
      v113.m128i_i64[1] = v82;
      v38 = v81;
      LOBYTE(v115) = v54;
      v114.m128i_i64[1] = v81;
      BYTE1(v115) = v58;
      goto LABEL_37;
    }
    v59 = _mm_loadu_si128(&v111);
    v113 = _mm_loadu_si128(&v110);
    v115 = v112;
    v114 = v59;
    if ( v87 != nullsub_98 )
    {
LABEL_68:
      v23 = (unsigned __int64)&v113;
      ((void (__fastcall *)(_QWORD *, __m128i *, __int64))v87)(v27, &v113, 1);
    }
LABEL_38:
    v42 = *(__int64 **)(a1 + 528);
    v43 = *v42;
    v44 = *(void (**)())(*v42 + 160);
    if ( v44 != nullsub_99 )
    {
      ((void (__fastcall *)(__int64 *, unsigned __int64, void (*)(), __int64, _QWORD *))v44)(v42, v23, v44, v38, v27);
      v42 = *(__int64 **)(a1 + 528);
      v43 = *v42;
    }
    v45 = *(void (**)())(v43 + 120);
    v46 = "Type index of inlined function";
    v113.m128i_i64[0] = (__int64)"Type index of inlined function";
    LOWORD(v115) = 259;
    if ( v45 != nullsub_98 )
    {
      ((void (__fastcall *)(__int64 *, __m128i *, __int64, const char *, _QWORD *))v45)(
        v42,
        &v113,
        1,
        "Type index of inlined function",
        v27);
      v42 = *(__int64 **)(a1 + 528);
    }
    (*(void (__fastcall **)(__int64 *, _QWORD, __int64, const char *, _QWORD *))(*v42 + 536))(v42, v88, 4, v46, v27);
    v47 = *(__int64 **)(a1 + 528);
    v48 = *(void (**)())(*v47 + 120);
    v113.m128i_i64[0] = (__int64)"Offset into filechecksum table";
    LOWORD(v115) = 259;
    if ( v48 != nullsub_98 )
    {
      ((void (__fastcall *)(__int64 *, __m128i *, __int64))v48)(v47, &v113, 1);
      v47 = *(__int64 **)(a1 + 528);
    }
    v49 = *v47;
    v50 = *(void (**)())(*v47 + 816);
    if ( v50 != nullsub_111 )
    {
      ((void (__fastcall *)(__int64 *, _QWORD))v50)(v47, v83);
      v47 = *(__int64 **)(a1 + 528);
      v49 = *v47;
    }
    v51 = *(void (**)())(v49 + 120);
    v113.m128i_i64[0] = (__int64)"Starting line number";
    LOWORD(v115) = 259;
    if ( v51 != nullsub_98 )
    {
      ((void (__fastcall *)(__int64 *, __m128i *, __int64))v51)(v47, &v113, 1);
      v47 = *(__int64 **)(a1 + 528);
    }
    ++v11;
    (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(*v47 + 536))(v47, *(unsigned int *)(v12 + 16), 4);
  }
  while ( v78 != v11 );
  return sub_31F8740(a1, v75);
}
