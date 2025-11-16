// Function: sub_18708C0
// Address: 0x18708c0
//
__int64 __fastcall sub_18708C0(__int64 a1, __int64 a2, _QWORD *a3)
{
  int v4; // esi
  __int64 i; // r14
  __int64 v6; // rsi
  __int64 j; // r14
  __int64 v8; // rsi
  __int64 k; // r14
  __int64 v10; // rsi
  __int64 *v11; // rax
  __int64 *v12; // r13
  __int64 v13; // r14
  __int64 v14; // rdi
  __int64 *v15; // r15
  unsigned int v16; // r12d
  __int64 v17; // r13
  unsigned __int64 v18; // r14
  unsigned int v19; // eax
  unsigned __int8 v20; // dl
  _QWORD *v21; // r10
  _QWORD *v22; // rax
  _QWORD *v23; // r9
  __int64 v24; // rsi
  __int64 v25; // rcx
  unsigned int v26; // r15d
  __int64 *v27; // rdx
  unsigned int v28; // r15d
  __int64 *v29; // rdx
  unsigned int v30; // r15d
  __int64 *v31; // rdx
  unsigned int v32; // r15d
  __int64 *v33; // rdx
  unsigned int v34; // r15d
  __m128i **v35; // rdx
  unsigned int v36; // r15d
  __int64 *v37; // rdx
  unsigned int v38; // r15d
  __int64 *v39; // rdx
  __int64 v40; // r15
  int v41; // r13d
  __int64 v42; // rsi
  int v43; // eax
  __int64 v44; // r15
  __int64 v45; // r13
  int v46; // ebx
  __int64 v47; // rsi
  int v48; // eax
  size_t v50; // rdx
  size_t v51; // r12
  unsigned int v52; // r9d
  __int64 *v53; // r10
  __int64 *v54; // rax
  __int64 v55; // rax
  unsigned int v56; // r9d
  __int64 *v57; // r10
  __int64 v58; // r8
  __int64 v59; // rdi
  __int64 v60; // rax
  void *v61; // rax
  __int64 v62; // rax
  __int64 *v63; // rdx
  __int64 v64; // rax
  __int64 *v65; // rdx
  __m128i v66; // xmm0
  __int64 v67; // rax
  __int64 *v68; // rdx
  __m128i si128; // xmm0
  __int64 v70; // rax
  __int64 *v71; // rdx
  __m128i v72; // xmm0
  __int64 v73; // rax
  __int64 *v74; // rdx
  __m128i v75; // xmm0
  __int64 v76; // rax
  __m128i **v77; // rdx
  __m128i *v78; // r13
  __m128i v79; // xmm0
  __int64 v80; // rax
  __int64 *v81; // rdx
  __m128i v82; // xmm0
  __int64 v84; // [rsp+10h] [rbp-120h]
  __int64 *v85; // [rsp+18h] [rbp-118h]
  __int64 v86; // [rsp+18h] [rbp-118h]
  unsigned int v87; // [rsp+20h] [rbp-110h]
  __int64 v88; // [rsp+20h] [rbp-110h]
  __int64 *v89; // [rsp+20h] [rbp-110h]
  __int64 *v90; // [rsp+28h] [rbp-108h]
  unsigned int v91; // [rsp+28h] [rbp-108h]
  unsigned __int8 srca; // [rsp+30h] [rbp-100h]
  unsigned __int8 *src; // [rsp+30h] [rbp-100h]
  unsigned int v94; // [rsp+38h] [rbp-F8h]
  __int64 v96; // [rsp+48h] [rbp-E8h]
  __int64 v97; // [rsp+50h] [rbp-E0h]
  __int64 v98; // [rsp+58h] [rbp-D8h]
  __int64 *v99; // [rsp+58h] [rbp-D8h]
  __int64 *v100; // [rsp+58h] [rbp-D8h]
  __int64 *v101; // [rsp+58h] [rbp-D8h]
  __int64 *v102; // [rsp+58h] [rbp-D8h]
  __int64 *v103; // [rsp+58h] [rbp-D8h]
  __m128i **v104; // [rsp+58h] [rbp-D8h]
  __int64 *v105; // [rsp+58h] [rbp-D8h]
  __int64 v106; // [rsp+60h] [rbp-D0h] BYREF
  int v107; // [rsp+68h] [rbp-C8h] BYREF
  __int64 v108; // [rsp+70h] [rbp-C0h]
  int *v109; // [rsp+78h] [rbp-B8h]
  int *v110; // [rsp+80h] [rbp-B0h]
  __int64 v111; // [rsp+88h] [rbp-A8h]
  __int64 v112; // [rsp+90h] [rbp-A0h] BYREF
  __int64 *v113; // [rsp+98h] [rbp-98h]
  __int64 *v114; // [rsp+A0h] [rbp-90h]
  __int64 v115; // [rsp+A8h] [rbp-88h]
  int v116; // [rsp+B0h] [rbp-80h]
  _BYTE v117[120]; // [rsp+B8h] [rbp-78h] BYREF

  v84 = 0;
  if ( a3 )
    v84 = a3[7];
  v112 = 0;
  v113 = (__int64 *)v117;
  v114 = (__int64 *)v117;
  v115 = 8;
  v116 = 0;
  sub_1633E30(a2, (__int64)&v112, 0);
  v4 = *(_DWORD *)(a2 + 140);
  v98 = a2 + 24;
  v96 = a2 + 8;
  v107 = 0;
  v108 = 0;
  v109 = &v107;
  v110 = &v107;
  v111 = 0;
  v97 = a2 + 40;
  if ( v4 )
  {
    for ( i = *(_QWORD *)(a2 + 32); i != v98; i = *(_QWORD *)(i + 8) )
    {
      v6 = i - 56;
      if ( !i )
        v6 = 0;
      sub_1870860(a1, v6, &v106);
    }
    for ( j = *(_QWORD *)(a2 + 16); j != v96; j = *(_QWORD *)(j + 8) )
    {
      v8 = j - 56;
      if ( !j )
        v8 = 0;
      sub_1870860(a1, v8, &v106);
    }
    for ( k = *(_QWORD *)(a2 + 48); k != v97; k = *(_QWORD *)(k + 8) )
    {
      v10 = k - 48;
      if ( !k )
        v10 = 0;
      sub_1870860(a1, v10, &v106);
    }
  }
  v11 = v114;
  if ( v114 == v113 )
    v12 = &v114[HIDWORD(v115)];
  else
    v12 = &v114[(unsigned int)v115];
  v13 = a1 + 32;
  if ( v114 != v12 )
  {
    while ( 1 )
    {
      v14 = *v11;
      v15 = v11;
      if ( (unsigned __int64)*v11 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v12 == ++v11 )
        goto LABEL_21;
    }
    while ( v12 != v15 )
    {
      src = (unsigned __int8 *)sub_1649960(v14);
      v51 = v50;
      v52 = sub_16D19C0(a1 + 32, src, v50);
      v53 = (__int64 *)(*(_QWORD *)(a1 + 32) + 8LL * v52);
      if ( *v53 )
      {
        if ( *v53 != -8 )
          goto LABEL_74;
        --*(_DWORD *)(a1 + 48);
      }
      v85 = v53;
      v87 = v52;
      v55 = malloc(v51 + 17);
      v56 = v87;
      v57 = v85;
      v58 = v55;
      if ( v55 )
      {
        v59 = v55 + 16;
      }
      else
      {
        if ( v51 == -17 )
        {
          v60 = malloc(1u);
          v56 = v87;
          v57 = v85;
          v58 = 0;
          if ( v60 )
          {
            v59 = v60 + 16;
            v58 = v60;
LABEL_85:
            v88 = v58;
            v90 = v57;
            v94 = v56;
            v61 = memcpy((void *)v59, src, v51);
            v58 = v88;
            v57 = v90;
            v56 = v94;
            v59 = (__int64)v61;
            goto LABEL_82;
          }
        }
        v86 = v58;
        v89 = v57;
        v91 = v56;
        sub_16BD1C0("Allocation failed", 1u);
        v56 = v91;
        v59 = 16;
        v57 = v89;
        v58 = v86;
      }
      if ( v51 + 1 > 1 )
        goto LABEL_85;
LABEL_82:
      *(_BYTE *)(v59 + v51) = 0;
      *(_QWORD *)v58 = v51;
      *(_BYTE *)(v58 + 8) = 0;
      *v57 = v58;
      ++*(_DWORD *)(a1 + 44);
      sub_16D1CD0(a1 + 32, v56);
LABEL_74:
      v54 = v15 + 1;
      if ( v15 + 1 == v12 )
        break;
      while ( 1 )
      {
        v14 = *v54;
        v15 = v54;
        if ( (unsigned __int64)*v54 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v12 == ++v54 )
          goto LABEL_21;
      }
    }
  }
LABEL_21:
  v16 = 0;
  if ( *(_QWORD *)(a2 + 32) != v98 )
  {
    v17 = *(_QWORD *)(a2 + 32);
    do
    {
      v18 = 0;
      if ( v17 )
        v18 = v17 - 56;
      v19 = sub_1870150(a1, v18, (__int64)&v106);
      v20 = v19;
      if ( (_BYTE)v19 )
      {
        v16 = v19;
        if ( v84 )
        {
          v21 = a3 + 2;
          v22 = (_QWORD *)a3[3];
          if ( v22 )
          {
            v23 = a3 + 2;
            do
            {
              while ( 1 )
              {
                v24 = v22[2];
                v25 = v22[3];
                if ( v22[4] >= v18 )
                  break;
                v22 = (_QWORD *)v22[3];
                if ( !v25 )
                  goto LABEL_32;
              }
              v23 = v22;
              v22 = (_QWORD *)v22[2];
            }
            while ( v24 );
LABEL_32:
            if ( v21 != v23 && v23[4] <= v18 )
              v21 = v23;
          }
          srca = v20;
          sub_13985D0(v84, v21[5]);
          v16 = srca;
        }
      }
      v17 = *(_QWORD *)(v17 + 8);
    }
    while ( v17 != v98 );
    v13 = a1 + 32;
  }
  v26 = sub_16D19C0(v13, "llvm.used", 9u);
  v27 = (__int64 *)(*(_QWORD *)(a1 + 32) + 8LL * v26);
  if ( *v27 )
  {
    if ( *v27 != -8 )
      goto LABEL_40;
    --*(_DWORD *)(a1 + 48);
  }
  v99 = v27;
  v62 = malloc(0x1Au);
  v63 = v99;
  if ( !v62 )
  {
    sub_16BD1C0("Allocation failed", 1u);
    v62 = 0;
    v63 = v99;
  }
  strcpy((char *)(v62 + 16), "llvm.used");
  *(_QWORD *)v62 = 9;
  *(_BYTE *)(v62 + 8) = 0;
  *v63 = v62;
  ++*(_DWORD *)(a1 + 44);
  sub_16D1CD0(v13, v26);
LABEL_40:
  v28 = sub_16D19C0(v13, "llvm.compiler.used", 0x12u);
  v29 = (__int64 *)(*(_QWORD *)(a1 + 32) + 8LL * v28);
  if ( *v29 )
  {
    if ( *v29 != -8 )
      goto LABEL_42;
    --*(_DWORD *)(a1 + 48);
  }
  v101 = v29;
  v67 = malloc(0x23u);
  v68 = v101;
  if ( !v67 )
  {
    sub_16BD1C0("Allocation failed", 1u);
    v67 = 0;
    v68 = v101;
  }
  si128 = _mm_load_si128((const __m128i *)&xmmword_3F89B50);
  strcpy((char *)(v67 + 32), "ed");
  *(_QWORD *)v67 = 18;
  *(__m128i *)(v67 + 16) = si128;
  *(_BYTE *)(v67 + 8) = 0;
  *v68 = v67;
  ++*(_DWORD *)(a1 + 44);
  sub_16D1CD0(v13, v28);
LABEL_42:
  v30 = sub_16D19C0(v13, "llvm.global_ctors", 0x11u);
  v31 = (__int64 *)(*(_QWORD *)(a1 + 32) + 8LL * v30);
  if ( *v31 )
  {
    if ( *v31 != -8 )
      goto LABEL_44;
    --*(_DWORD *)(a1 + 48);
  }
  v100 = v31;
  v64 = malloc(0x22u);
  v65 = v100;
  if ( !v64 )
  {
    sub_16BD1C0("Allocation failed", 1u);
    v64 = 0;
    v65 = v100;
  }
  v66 = _mm_load_si128((const __m128i *)&xmmword_3F89B40);
  *(_BYTE *)(v64 + 32) = 115;
  *(_BYTE *)(v64 + 33) = 0;
  *(__m128i *)(v64 + 16) = v66;
  *(_QWORD *)v64 = 17;
  *(_BYTE *)(v64 + 8) = 0;
  *v65 = v64;
  ++*(_DWORD *)(a1 + 44);
  sub_16D1CD0(v13, v30);
LABEL_44:
  v32 = sub_16D19C0(v13, "llvm.global_dtors", 0x11u);
  v33 = (__int64 *)(*(_QWORD *)(a1 + 32) + 8LL * v32);
  if ( *v33 )
  {
    if ( *v33 != -8 )
      goto LABEL_46;
    --*(_DWORD *)(a1 + 48);
  }
  v105 = v33;
  v80 = malloc(0x22u);
  v81 = v105;
  if ( !v80 )
  {
    sub_16BD1C0("Allocation failed", 1u);
    v80 = 0;
    v81 = v105;
  }
  v82 = _mm_load_si128((const __m128i *)&xmmword_3F89B30);
  *(_BYTE *)(v80 + 32) = 115;
  *(_BYTE *)(v80 + 33) = 0;
  *(__m128i *)(v80 + 16) = v82;
  *(_QWORD *)v80 = 17;
  *(_BYTE *)(v80 + 8) = 0;
  *v81 = v80;
  ++*(_DWORD *)(a1 + 44);
  sub_16D1CD0(v13, v32);
LABEL_46:
  v34 = sub_16D19C0(v13, "llvm.global.annotations", 0x17u);
  v35 = (__m128i **)(*(_QWORD *)(a1 + 32) + 8LL * v34);
  if ( *v35 )
  {
    if ( *v35 != (__m128i *)-8LL )
      goto LABEL_48;
    --*(_DWORD *)(a1 + 48);
  }
  v104 = v35;
  v76 = malloc(0x28u);
  v77 = v104;
  v78 = (__m128i *)v76;
  if ( !v76 )
  {
    sub_16BD1C0("Allocation failed", 1u);
    v77 = v104;
  }
  v78[2].m128i_i8[6] = 115;
  v79 = _mm_load_si128((const __m128i *)&xmmword_3F89B20);
  v78[2].m128i_i32[0] = 1769234804;
  v78[2].m128i_i16[2] = 28271;
  v78[2].m128i_i8[7] = 0;
  v78->m128i_i64[0] = 23;
  v78->m128i_i8[8] = 0;
  v78[1] = v79;
  *v77 = v78;
  ++*(_DWORD *)(a1 + 44);
  sub_16D1CD0(v13, v34);
LABEL_48:
  v36 = sub_16D19C0(v13, "__stack_chk_fail", 0x10u);
  v37 = (__int64 *)(*(_QWORD *)(a1 + 32) + 8LL * v36);
  if ( *v37 )
  {
    if ( *v37 != -8 )
      goto LABEL_50;
    --*(_DWORD *)(a1 + 48);
  }
  v103 = v37;
  v73 = malloc(0x21u);
  v74 = v103;
  if ( !v73 )
  {
    sub_16BD1C0("Allocation failed", 1u);
    v73 = 0;
    v74 = v103;
  }
  v75 = _mm_load_si128((const __m128i *)&xmmword_3F89B10);
  *(_BYTE *)(v73 + 32) = 0;
  *(_QWORD *)v73 = 16;
  *(__m128i *)(v73 + 16) = v75;
  *(_BYTE *)(v73 + 8) = 0;
  *v74 = v73;
  ++*(_DWORD *)(a1 + 44);
  sub_16D1CD0(v13, v36);
LABEL_50:
  v38 = sub_16D19C0(v13, "__stack_chk_guard", 0x11u);
  v39 = (__int64 *)(*(_QWORD *)(a1 + 32) + 8LL * v38);
  if ( *v39 )
  {
    if ( *v39 != -8 )
      goto LABEL_52;
    --*(_DWORD *)(a1 + 48);
  }
  v102 = v39;
  v70 = malloc(0x22u);
  v71 = v102;
  if ( !v70 )
  {
    sub_16BD1C0("Allocation failed", 1u);
    v70 = 0;
    v71 = v102;
  }
  v72 = _mm_load_si128((const __m128i *)&xmmword_3F89B00);
  *(_BYTE *)(v70 + 32) = 100;
  *(_BYTE *)(v70 + 33) = 0;
  *(__m128i *)(v70 + 16) = v72;
  *(_QWORD *)v70 = 17;
  *(_BYTE *)(v70 + 8) = 0;
  *v71 = v70;
  ++*(_DWORD *)(a1 + 44);
  sub_16D1CD0(v13, v38);
LABEL_52:
  v40 = *(_QWORD *)(a2 + 16);
  if ( v40 != v96 )
  {
    v41 = v16;
    do
    {
      v42 = v40 - 56;
      if ( !v40 )
        v42 = 0;
      v43 = sub_1870150(a1, v42, (__int64)&v106);
      v40 = *(_QWORD *)(v40 + 8);
      if ( (_BYTE)v43 )
        v41 = v43;
    }
    while ( v40 != v96 );
    v16 = v41;
  }
  v44 = *(_QWORD *)(a2 + 48);
  if ( v44 != v97 )
  {
    v45 = a1;
    v46 = v16;
    do
    {
      v47 = v44 - 48;
      if ( !v44 )
        v47 = 0;
      v48 = sub_1870150(v45, v47, (__int64)&v106);
      v44 = *(_QWORD *)(v44 + 8);
      if ( (_BYTE)v48 )
        v46 = v48;
    }
    while ( v44 != v97 );
    v16 = v46;
  }
  sub_186F470(v108);
  if ( v114 != v113 )
    _libc_free((unsigned __int64)v114);
  return v16;
}
