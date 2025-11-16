// Function: sub_26A57A0
// Address: 0x26a57a0
//
__int64 __fastcall sub_26A57A0(__int64 a1, _DWORD *a2, __int64 a3, __int64 a4)
{
  __int64 v8; // r15
  __int64 v9; // r13
  __int64 v10; // r14
  __int64 v11; // r8
  __int64 v12; // r9
  unsigned int v13; // edx
  __int64 *v14; // rax
  __int64 v15; // rdi
  __int64 v16; // rax
  unsigned __int64 v17; // rdx
  __int64 v18; // rsi
  __int64 v19; // rdx
  __int64 v20; // r13
  __int64 v21; // r14
  __int64 *v22; // r8
  unsigned int v23; // edx
  unsigned int v24; // eax
  __int64 *v25; // rsi
  __int64 v26; // rdi
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rcx
  bool v30; // al
  __m128i v31; // xmm1
  bool v32; // zf
  int v33; // eax
  __m128i v34; // xmm0
  __m128i v35; // xmm0
  __m128i v36; // xmm2
  __m128i v37; // xmm3
  __int64 (__fastcall *v38)(__int64 (__fastcall **)(__int64, __int64), __int64, __int64); // rax
  __int64 *v39; // r14
  __int64 *v40; // rbx
  unsigned __int64 i; // rax
  __int64 v42; // rdi
  unsigned int v43; // ecx
  __int64 v44; // rsi
  __int64 *v45; // rbx
  __int64 *v46; // r13
  __int64 v47; // rsi
  __int64 v48; // rdi
  int v49; // eax
  __int64 v50; // rsi
  unsigned int v51; // r11d
  int v52; // r9d
  __int64 *v53; // rsi
  int v54; // esi
  int v55; // r9d
  __int64 v56; // r13
  __int64 v57; // r14
  unsigned int v58; // eax
  __int64 *v59; // rsi
  __int64 v60; // rdi
  int v61; // esi
  int v62; // r9d
  int v63; // r10d
  __int64 *v64; // r9
  __int64 v65; // [rsp+58h] [rbp-9F48h]
  char v66; // [rsp+78h] [rbp-9F28h]
  __int64 v68; // [rsp+88h] [rbp-9F18h]
  __int64 v69; // [rsp+98h] [rbp-9F08h] BYREF
  _QWORD v70[2]; // [rsp+A0h] [rbp-9F00h] BYREF
  char v71; // [rsp+B0h] [rbp-9EF0h]
  __int64 v72; // [rsp+C0h] [rbp-9EE0h] BYREF
  __int64 v73; // [rsp+C8h] [rbp-9ED8h]
  __int64 v74; // [rsp+D0h] [rbp-9ED0h]
  unsigned int v75; // [rsp+D8h] [rbp-9EC8h]
  char v76[8]; // [rsp+E0h] [rbp-9EC0h] BYREF
  __int64 v77; // [rsp+E8h] [rbp-9EB8h]
  unsigned int v78; // [rsp+F8h] [rbp-9EA8h]
  __int64 *v79; // [rsp+100h] [rbp-9EA0h]
  __int64 v80; // [rsp+110h] [rbp-9E90h] BYREF
  __int64 v81; // [rsp+118h] [rbp-9E88h]
  __int64 v82; // [rsp+120h] [rbp-9E80h]
  __int64 v83; // [rsp+128h] [rbp-9E78h]
  __int64 *v84; // [rsp+130h] [rbp-9E70h]
  __int64 v85; // [rsp+138h] [rbp-9E68h]
  __int64 v86[2]; // [rsp+140h] [rbp-9E60h] BYREF
  __int64 *v87; // [rsp+150h] [rbp-9E50h]
  __int64 v88; // [rsp+158h] [rbp-9E48h]
  _BYTE v89[32]; // [rsp+160h] [rbp-9E40h] BYREF
  __int64 *v90; // [rsp+180h] [rbp-9E20h]
  __int64 v91; // [rsp+188h] [rbp-9E18h]
  _QWORD v92[2]; // [rsp+190h] [rbp-9E10h] BYREF
  _BYTE *v93; // [rsp+1A0h] [rbp-9E00h] BYREF
  __int64 v94; // [rsp+1A8h] [rbp-9DF8h]
  _BYTE v95[128]; // [rsp+1B0h] [rbp-9DF0h] BYREF
  __int64 v96; // [rsp+230h] [rbp-9D70h]
  __m128i v97; // [rsp+238h] [rbp-9D68h] BYREF
  __int64 (__fastcall *v98)(__int64 *, __m128i *, int); // [rsp+248h] [rbp-9D58h]
  __int64 (__fastcall *v99)(__int64 (__fastcall **)(__int64, __int64), __int64, __int64); // [rsp+250h] [rbp-9D50h]
  _BYTE v100[16]; // [rsp+258h] [rbp-9D48h] BYREF
  __int64 (__fastcall *v101)(__int64 *, __int64); // [rsp+268h] [rbp-9D38h]
  __int64 *v102; // [rsp+270h] [rbp-9D30h]
  __int64 *v103; // [rsp+278h] [rbp-9D28h]
  __m128i *v104; // [rsp+280h] [rbp-9D20h]
  __int64 v105; // [rsp+288h] [rbp-9D18h]
  __m128i v106; // [rsp+290h] [rbp-9D10h] BYREF
  const char *v107; // [rsp+2A0h] [rbp-9D00h]
  __m128i v108; // [rsp+2A8h] [rbp-9CF8h] BYREF
  __int64 (__fastcall *v109)(_QWORD *, __int64, int); // [rsp+2B8h] [rbp-9CE8h]
  __int64 (__fastcall *v110)(__int64, __int64); // [rsp+2C0h] [rbp-9CE0h]
  __m128i v111; // [rsp+2D0h] [rbp-9CD0h] BYREF
  __int64 v112; // [rsp+2E0h] [rbp-9CC0h]
  __int64 (__fastcall *v113)(__int64 *, __m128i *, int); // [rsp+2E8h] [rbp-9CB8h]
  __int64 (__fastcall *v114)(__int64 (__fastcall **)(__int64, __int64), __int64, __int64); // [rsp+2F0h] [rbp-9CB0h]
  _QWORD v115[2]; // [rsp+2F8h] [rbp-9CA8h] BYREF
  __int64 (__fastcall *v116)(__int64 *, __int64); // [rsp+308h] [rbp-9C98h]
  __int64 *v117; // [rsp+310h] [rbp-9C90h]
  __int64 *v118; // [rsp+318h] [rbp-9C88h]
  __m128i *v119; // [rsp+320h] [rbp-9C80h]
  __int64 v120; // [rsp+328h] [rbp-9C78h]
  __m128i v121; // [rsp+330h] [rbp-9C70h]
  const char *v122; // [rsp+340h] [rbp-9C60h]
  _BYTE v123[16]; // [rsp+348h] [rbp-9C58h] BYREF
  __int64 (__fastcall *v124)(_QWORD *, __int64, int); // [rsp+358h] [rbp-9C48h]
  __int64 (__fastcall *v125)(__int64, __int64); // [rsp+360h] [rbp-9C40h]
  __int64 v126; // [rsp+370h] [rbp-9C30h] BYREF
  _BYTE *v127; // [rsp+378h] [rbp-9C28h]
  __int64 v128; // [rsp+380h] [rbp-9C20h]
  int v129; // [rsp+388h] [rbp-9C18h]
  char v130; // [rsp+38Ch] [rbp-9C14h]
  _BYTE v131[128]; // [rsp+390h] [rbp-9C10h] BYREF
  _BYTE *v132; // [rsp+410h] [rbp-9B90h]
  __int64 v133; // [rsp+418h] [rbp-9B88h]
  _BYTE v134[128]; // [rsp+420h] [rbp-9B80h] BYREF
  _BYTE *v135; // [rsp+4A0h] [rbp-9B00h]
  __int64 v136; // [rsp+4A8h] [rbp-9AF8h]
  _BYTE v137[128]; // [rsp+4B0h] [rbp-9AF0h] BYREF
  __int64 v138; // [rsp+530h] [rbp-9A70h]
  __int64 v139; // [rsp+538h] [rbp-9A68h]
  __int64 v140; // [rsp+540h] [rbp-9A60h]
  __int64 v141; // [rsp+548h] [rbp-9A58h]
  __int64 v142; // [rsp+550h] [rbp-9A50h]
  __m128i v143; // [rsp+560h] [rbp-9A40h] BYREF
  __int64 (__fastcall *v144)(_QWORD *, __int64, int); // [rsp+570h] [rbp-9A30h]
  void *v145; // [rsp+578h] [rbp-9A28h]
  char v146[344]; // [rsp+5B0h] [rbp-99F0h] BYREF
  __int64 v147; // [rsp+708h] [rbp-9898h]
  __int64 v148[10]; // [rsp+16C0h] [rbp-88E0h] BYREF
  char v149[344]; // [rsp+1710h] [rbp-8890h] BYREF
  __int64 v150; // [rsp+1868h] [rbp-8738h]

  if ( !sub_2674810(a3) || (_BYTE)qword_4FF54C8 )
  {
    *(_BYTE *)(a1 + 76) = 1;
    *(_QWORD *)(a1 + 8) = a1 + 32;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
    return a1;
  }
  v8 = a3 + 24;
  v65 = *(_QWORD *)(sub_BC0510(a4, &unk_4F82418, a3) + 8);
  sub_269A220((__int64)v76, a3);
  v72 = 0;
  v73 = 0;
  v74 = 0;
  v75 = 0;
  v66 = sub_2674830(a3);
  if ( !v66 )
    goto LABEL_6;
  v20 = *(_QWORD *)(a3 + 32);
  v126 = 0;
  v127 = v131;
  v128 = 16;
  v129 = 0;
  v130 = 1;
  while ( v20 != v8 )
  {
    v21 = v20 - 56;
    if ( !v20 )
      v21 = 0;
    if ( !sub_B2FC80(v21) )
    {
      v22 = (__int64 *)(v77 + 8LL * v78);
      if ( !v78 )
        goto LABEL_37;
      v23 = v78 - 1;
      v24 = (v78 - 1) & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
      v25 = (__int64 *)(v77 + 8LL * v24);
      v26 = *v25;
      if ( v21 != *v25 )
      {
        v50 = *v25;
        v51 = (v78 - 1) & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
        v52 = 1;
        while ( v50 != -4096 )
        {
          v63 = v52 + 1;
          v51 = v23 & (v52 + v51);
          v64 = (__int64 *)(v77 + 8LL * v51);
          v50 = *v64;
          if ( v21 == *v64 )
          {
            if ( v22 != v64 )
              goto LABEL_30;
            goto LABEL_89;
          }
          v52 = v63;
        }
        v24 = v23 & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
        v53 = (__int64 *)(v77 + 8LL * v24);
        v26 = *v53;
        if ( v21 != *v53 )
        {
LABEL_89:
          v54 = 1;
          while ( v26 != -4096 )
          {
            v55 = v54 + 1;
            v24 = v23 & (v54 + v24);
            v53 = (__int64 *)(v77 + 8LL * v24);
            v26 = *v53;
            if ( v21 == *v53 )
              goto LABEL_92;
            v54 = v55;
          }
          goto LABEL_37;
        }
LABEL_92:
        if ( v22 != v53 )
          goto LABEL_41;
LABEL_37:
        v27 = *(_QWORD *)(v21 + 16);
        if ( v27 )
        {
          while ( **(_BYTE **)(v27 + 24) == 4 )
          {
            v27 = *(_QWORD *)(v27 + 8);
            if ( !v27 )
              goto LABEL_30;
          }
LABEL_41:
          if ( !byte_4FF5308 )
          {
            if ( sub_250E810(v21) )
            {
              sub_AE6EC0((__int64)&v126, v21);
            }
            else if ( (*(_BYTE *)(v21 + 32) & 0xFu) - 7 > 1 && !(unsigned __int8)sub_B2D610(v21, 5) )
            {
              v68 = sub_BC1CD0(v65, &unk_4F8FAE8, v21);
              if ( (unsigned __int8)sub_266EEF0(*(_QWORD *)(v68 + 8)) )
              {
                sub_B179F0((__int64)v148, (__int64)"openmp-opt", (__int64)"OMP140", 6, v21);
                sub_B18290((__int64)v148, "Could not internalize function. ", 0x20u);
                sub_B18290((__int64)v148, "Some optimizations may not be possible. [OMP140]", 0x30u);
                sub_23FE290((__int64)&v143, (__int64)v148, v28, v29, (__int64)&v143, (__int64)v148);
                v147 = v150;
                v143.m128i_i64[0] = (__int64)&unk_49D9DE8;
                v148[0] = (__int64)&unk_49D9D40;
                sub_23FD590((__int64)v149);
                sub_1049740((__int64 *)(v68 + 8), (__int64)&v143);
                v143.m128i_i64[0] = (__int64)&unk_49D9D40;
                sub_23FD590((__int64)v146);
              }
            }
          }
        }
        goto LABEL_30;
      }
      if ( v22 == v25 )
        goto LABEL_37;
    }
LABEL_30:
    v20 = *(_QWORD *)(v20 + 8);
  }
  v66 = sub_25167D0((__int64)&v126, (__int64)&v72);
  if ( !v130 )
    _libc_free((unsigned __int64)v127);
LABEL_6:
  v9 = *(_QWORD *)(a3 + 32);
  v80 = 0;
  v84 = v86;
  v93 = v95;
  v94 = 0x1000000000LL;
  v81 = 0;
  v82 = 0;
  v83 = 0;
  v85 = 0;
  if ( v9 == v8 )
    goto LABEL_19;
  do
  {
    while ( 1 )
    {
      v10 = v9 - 56;
      if ( !v9 )
        v10 = 0;
      if ( sub_B2FC80(v10) )
        goto LABEL_8;
      if ( !v75 )
        goto LABEL_15;
      v13 = (v75 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
      v14 = (__int64 *)(v73 + 16LL * v13);
      v15 = *v14;
      if ( v10 != *v14 )
        break;
LABEL_14:
      if ( !v14[1] )
        goto LABEL_15;
LABEL_8:
      v9 = *(_QWORD *)(v9 + 8);
      if ( v9 == v8 )
        goto LABEL_18;
    }
    v49 = 1;
    while ( v15 != -4096 )
    {
      v11 = (unsigned int)(v49 + 1);
      v13 = (v75 - 1) & (v49 + v13);
      v14 = (__int64 *)(v73 + 16LL * v13);
      v15 = *v14;
      if ( v10 == *v14 )
        goto LABEL_14;
      v49 = v11;
    }
LABEL_15:
    v16 = (unsigned int)v94;
    v17 = (unsigned int)v94 + 1LL;
    if ( v17 > HIDWORD(v94) )
    {
      sub_C8D5F0((__int64)&v93, v95, v17, 8u, v11, v12);
      v16 = (unsigned int)v94;
    }
    *(_QWORD *)&v93[8 * v16] = v10;
    v148[0] = v10;
    LODWORD(v94) = v94 + 1;
    sub_2699F90((__int64)&v80, v148);
    v9 = *(_QWORD *)(v9 + 8);
  }
  while ( v9 != v8 );
LABEL_18:
  if ( !(_DWORD)v94 )
  {
LABEL_19:
    v18 = a1 + 32;
    v19 = a1 + 80;
    if ( v66 )
    {
      memset((void *)a1, 0, 0x60u);
      *(_QWORD *)(a1 + 8) = v18;
      *(_DWORD *)(a1 + 16) = 2;
      *(_BYTE *)(a1 + 28) = 1;
      *(_QWORD *)(a1 + 56) = v19;
      *(_DWORD *)(a1 + 64) = 2;
      *(_BYTE *)(a1 + 76) = 1;
    }
    else
    {
      *(_QWORD *)(a1 + 8) = v18;
      *(_QWORD *)(a1 + 16) = 0x100000002LL;
      *(_QWORD *)(a1 + 48) = 0;
      *(_QWORD *)(a1 + 56) = v19;
      *(_QWORD *)(a1 + 64) = 2;
      *(_DWORD *)(a1 + 72) = 0;
      *(_BYTE *)(a1 + 76) = 1;
      *(_DWORD *)(a1 + 24) = 0;
      *(_BYTE *)(a1 + 28) = 1;
      *(_QWORD *)(a1 + 32) = &qword_4F82400;
      *(_QWORD *)a1 = 1;
    }
    goto LABEL_21;
  }
  v70[1] = 0;
  v70[0] = v65;
  v69 = v65;
  v87 = (__int64 *)v89;
  v88 = 0x400000000LL;
  v90 = v92;
  v127 = v131;
  v132 = v134;
  v133 = 0x1000000000LL;
  v136 = 0x1000000000LL;
  v135 = v137;
  v71 = 0;
  v86[0] = 0;
  v86[1] = 0;
  v91 = 0;
  v92[0] = 0;
  v92[1] = 1;
  v126 = 0;
  v128 = 16;
  v129 = 0;
  v130 = 1;
  v138 = 0;
  v139 = 0;
  v140 = 0;
  v141 = 0;
  v142 = 0;
  sub_2695720((__int64)v148, a3, (__int64)v70, v86, 0, *a2 == 4 || (unsigned int)(*a2 - 1) <= 1);
  v30 = sub_2674830(a3);
  v31 = _mm_loadu_si128(&v97);
  v32 = !v30;
  v33 = 32;
  if ( !v32 )
    v33 = qword_4FF46C8;
  v96 = 0x100000101LL;
  LODWORD(v105) = v33;
  v106.m128i_i64[1] = (__int64)&v69;
  v107 = "openmp-opt";
  v143.m128i_i64[0] = (__int64)sub_26A1DB0;
  v34 = _mm_loadu_si128(&v143);
  v103 = &v126;
  v98 = (__int64 (__fastcall *)(__int64 *, __m128i *, int))sub_266DF20;
  v106.m128i_i64[0] = (__int64)sub_266DFB0;
  v145 = v99;
  v143 = v31;
  v99 = sub_266DF10;
  v97 = v34;
  v101 = 0;
  v104 = 0;
  v109 = 0;
  BYTE4(v105) = 1;
  v144 = 0;
  sub_A17130((__int64)&v143);
  v35 = _mm_loadu_si128(&v143);
  v36 = _mm_loadu_si128(&v108);
  v144 = v109;
  v109 = sub_266DF50;
  v143 = v36;
  v145 = v110;
  v110 = sub_266E030;
  v108 = v35;
  sub_A17130((__int64)&v143);
  v113 = 0;
  v111.m128i_i32[0] = v96;
  v111.m128i_i16[2] = WORD2(v96);
  if ( v98 )
  {
    v98(&v111.m128i_i64[1], &v97, 2);
    v114 = v99;
    v113 = v98;
  }
  v116 = 0;
  if ( v101 )
  {
    ((void (__fastcall *)(_QWORD *, _BYTE *, __int64))v101)(v115, v100, 2);
    v117 = v102;
    v116 = v101;
  }
  v37 = _mm_loadu_si128(&v106);
  v124 = 0;
  v118 = v103;
  v121 = v37;
  v119 = v104;
  v120 = v105;
  v122 = v107;
  if ( v109 )
  {
    v109(v123, (__int64)&v108, 2);
    v125 = v110;
    v124 = v109;
  }
  sub_250EFA0((__int64)&v143, (__int64)&v80, (__int64)v148, &v111);
  sub_A17130((__int64)v123);
  sub_A17130((__int64)v115);
  sub_A17130((__int64)&v111.m128i_i64[1]);
  v111 = 0u;
  v112 = 0;
  v113 = 0;
  v38 = *(__int64 (__fastcall **)(__int64 (__fastcall **)(__int64, __int64), __int64, __int64))(*(_QWORD *)v93 + 40LL);
  v115[1] = &v126;
  v116 = sub_266DFB0;
  v114 = v38;
  v115[0] = &v93;
  v117 = &v69;
  v118 = v148;
  v119 = &v143;
  if ( (_DWORD)v94 )
    v66 |= sub_26A50B0((__int64)&v111, 1);
  if ( (_BYTE)qword_4FF4888 )
  {
    if ( sub_2674830(a3) )
    {
      v56 = *(_QWORD *)(a3 + 32);
      if ( v56 != v8 )
      {
        while ( 2 )
        {
          v57 = v56 - 56;
          if ( !v56 )
            v57 = 0;
          if ( sub_B2FC80(v57) )
            goto LABEL_98;
          if ( v78 )
          {
            v58 = (v78 - 1) & (((unsigned int)v57 >> 9) ^ ((unsigned int)v57 >> 4));
            v59 = (__int64 *)(v77 + 8LL * v58);
            v60 = *v59;
            if ( v57 == *v59 )
            {
LABEL_104:
              if ( v59 == (__int64 *)(v77 + 8LL * v78) )
                break;
LABEL_98:
              v56 = *(_QWORD *)(v56 + 8);
              if ( v56 == v8 )
                goto LABEL_58;
              continue;
            }
            v61 = 1;
            while ( v60 != -4096 )
            {
              v62 = v61 + 1;
              v58 = (v78 - 1) & (v61 + v58);
              v59 = (__int64 *)(v77 + 8LL * v58);
              v60 = *v59;
              if ( v57 == *v59 )
                goto LABEL_104;
              v61 = v62;
            }
          }
          break;
        }
        if ( !(unsigned __int8)sub_B2D610(v57, 31) )
          sub_B2CD30(v57, 3);
        goto LABEL_98;
      }
    }
  }
LABEL_58:
  if ( v66 )
  {
    memset((void *)a1, 0, 0x60u);
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 8) = a1 + 32;
    *(_DWORD *)(a1 + 16) = 2;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_DWORD *)(a1 + 64) = 2;
    *(_BYTE *)(a1 + 76) = 1;
  }
  else
  {
    sub_232FC90(a1);
  }
  sub_C7D6A0(v111.m128i_i64[1], 24LL * (unsigned int)v113, 8);
  sub_250D880((__int64)&v143);
  sub_A17130((__int64)&v108);
  sub_A17130((__int64)v100);
  sub_A17130((__int64)&v97);
  sub_2673E20((__int64)v148);
  sub_29A2B10(&v126);
  if ( v135 != v137 )
    _libc_free((unsigned __int64)v135);
  if ( v132 != v134 )
    _libc_free((unsigned __int64)v132);
  if ( !v130 )
    _libc_free((unsigned __int64)v127);
  v39 = v87;
  v40 = &v87[(unsigned int)v88];
  if ( v87 != v40 )
  {
    for ( i = (unsigned __int64)v87; ; i = (unsigned __int64)v87 )
    {
      v42 = *v39;
      v43 = (unsigned int)((__int64)((__int64)v39 - i) >> 3) >> 7;
      v44 = 4096LL << v43;
      if ( v43 >= 0x1E )
        v44 = 0x40000000000LL;
      ++v39;
      sub_C7D6A0(v42, v44, 16);
      if ( v40 == v39 )
        break;
    }
  }
  v45 = v90;
  v46 = &v90[2 * (unsigned int)v91];
  if ( v90 != v46 )
  {
    do
    {
      v47 = v45[1];
      v48 = *v45;
      v45 += 2;
      sub_C7D6A0(v48, v47, 16);
    }
    while ( v46 != v45 );
    v46 = v90;
  }
  if ( v46 != v92 )
    _libc_free((unsigned __int64)v46);
  if ( v87 != (__int64 *)v89 )
    _libc_free((unsigned __int64)v87);
LABEL_21:
  if ( v93 != v95 )
    _libc_free((unsigned __int64)v93);
  if ( v84 != v86 )
    _libc_free((unsigned __int64)v84);
  sub_C7D6A0(v81, 8LL * (unsigned int)v83, 8);
  sub_C7D6A0(v73, 16LL * v75, 8);
  if ( v79 != &v80 )
    _libc_free((unsigned __int64)v79);
  sub_C7D6A0(v77, 8LL * v78, 8);
  return a1;
}
