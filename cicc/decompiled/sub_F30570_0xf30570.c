// Function: sub_F30570
// Address: 0xf30570
//
__int64 __fastcall sub_F30570(__int64 a1, __int64 a2)
{
  int v3; // esi
  __int64 i; // r15
  _BYTE *v5; // rsi
  __int64 j; // r15
  _BYTE *v7; // rsi
  __int64 k; // r15
  _BYTE *v9; // rsi
  __int64 *v10; // r13
  const char *v11; // r12
  size_t v12; // rdx
  size_t v13; // r14
  int v14; // eax
  unsigned int v15; // r8d
  _QWORD *v16; // r9
  __int64 v17; // rax
  unsigned int v18; // r8d
  _QWORD *v19; // r9
  _QWORD *v20; // rcx
  int v21; // eax
  unsigned int v22; // r12d
  __int64 *v23; // r13
  int v24; // eax
  unsigned int v25; // r12d
  __int64 *v26; // r13
  int v27; // eax
  unsigned int v28; // r12d
  __int64 *v29; // r13
  int v30; // eax
  unsigned int v31; // r12d
  __int64 *v32; // r13
  int v33; // eax
  unsigned int v34; // r12d
  __int64 *v35; // r13
  int v36; // eax
  unsigned int v37; // r12d
  __int64 *v38; // r13
  int v39; // eax
  unsigned int v40; // r12d
  __int64 *v41; // r13
  int v42; // eax
  unsigned int v43; // r12d
  __int64 *v44; // r13
  unsigned int v45; // r12d
  __int64 v46; // r15
  unsigned int v47; // r13d
  _BYTE *v48; // rsi
  unsigned int v49; // eax
  __int64 v50; // r15
  int v51; // r13d
  _BYTE *v52; // rsi
  int v53; // eax
  __int64 v54; // r15
  int v55; // ebx
  _BYTE *v56; // rsi
  int v57; // eax
  __int64 v58; // rsi
  __int64 v60; // rax
  __m128i v61; // xmm0
  int v62; // eax
  __int64 v63; // rax
  __m128i v64; // xmm0
  __int64 v65; // rax
  __m128i v66; // xmm0
  __int64 v67; // rax
  __m128i v68; // xmm0
  __int64 v69; // rax
  __m128i v70; // xmm0
  __int64 v71; // rax
  __m128i si128; // xmm0
  __int64 v73; // rax
  __int64 v74; // rax
  __m128i v75; // xmm0
  _QWORD *v76; // [rsp+10h] [rbp-C0h]
  __int64 *v78; // [rsp+20h] [rbp-B0h]
  _QWORD *v79; // [rsp+28h] [rbp-A8h]
  unsigned int v80; // [rsp+34h] [rbp-9Ch]
  __int64 v81; // [rsp+38h] [rbp-98h]
  __int64 v82; // [rsp+40h] [rbp-90h]
  __int64 v83; // [rsp+48h] [rbp-88h]
  __int64 v84; // [rsp+50h] [rbp-80h] BYREF
  __int64 v85; // [rsp+58h] [rbp-78h]
  __int64 v86; // [rsp+60h] [rbp-70h]
  unsigned int v87; // [rsp+68h] [rbp-68h]
  __int64 *v88; // [rsp+70h] [rbp-60h] BYREF
  __int64 v89; // [rsp+78h] [rbp-58h]
  _BYTE v90[80]; // [rsp+80h] [rbp-50h] BYREF

  v89 = 0x400000000LL;
  v88 = (__int64 *)v90;
  sub_BAA9B0(a2, (__int64)&v88, 0);
  v3 = *(_DWORD *)(a2 + 140);
  v83 = a2 + 40;
  v84 = 0;
  v85 = 0;
  v86 = 0;
  v87 = 0;
  v81 = a2 + 8;
  v82 = a2 + 24;
  if ( v3 )
  {
    for ( i = *(_QWORD *)(a2 + 32); i != v82; i = *(_QWORD *)(i + 8) )
    {
      v5 = (_BYTE *)(i - 56);
      if ( !i )
        v5 = 0;
      sub_F302F0(a1, v5, (__int64)&v84);
    }
    for ( j = *(_QWORD *)(a2 + 16); j != v81; j = *(_QWORD *)(j + 8) )
    {
      v7 = (_BYTE *)(j - 56);
      if ( !j )
        v7 = 0;
      sub_F302F0(a1, v7, (__int64)&v84);
    }
    for ( k = *(_QWORD *)(a2 + 48); k != v83; k = *(_QWORD *)(k + 8) )
    {
      v9 = (_BYTE *)(k - 48);
      if ( !k )
        v9 = 0;
      sub_F302F0(a1, v9, (__int64)&v84);
    }
  }
  v10 = v88;
  v78 = &v88[(unsigned int)v89];
  if ( v78 != v88 )
  {
    do
    {
      while ( 1 )
      {
        v11 = sub_BD5D20(*v10);
        v13 = v12;
        v14 = sub_C92610();
        v15 = sub_C92740(a1 + 40, v11, v13, v14);
        v16 = (_QWORD *)(*(_QWORD *)(a1 + 40) + 8LL * v15);
        if ( *v16 )
          break;
LABEL_20:
        v79 = v16;
        v80 = v15;
        v17 = sub_C7D670(v13 + 9, 8);
        v18 = v80;
        v19 = v79;
        v20 = (_QWORD *)v17;
        if ( v13 )
        {
          v76 = (_QWORD *)v17;
          memcpy((void *)(v17 + 8), v11, v13);
          v18 = v80;
          v19 = v79;
          v20 = v76;
        }
        *((_BYTE *)v20 + v13 + 8) = 0;
        ++v10;
        *v20 = v13;
        *v19 = v20;
        ++*(_DWORD *)(a1 + 52);
        sub_C929D0((__int64 *)(a1 + 40), v18);
        if ( v78 == v10 )
          goto LABEL_23;
      }
      if ( *v16 == -8 )
      {
        --*(_DWORD *)(a1 + 56);
        goto LABEL_20;
      }
      ++v10;
    }
    while ( v78 != v10 );
  }
LABEL_23:
  v21 = sub_C92610();
  v22 = sub_C92740(a1 + 40, "llvm.used", 9u, v21);
  v23 = (__int64 *)(*(_QWORD *)(a1 + 40) + 8LL * v22);
  if ( *v23 )
  {
    if ( *v23 != -8 )
      goto LABEL_25;
    --*(_DWORD *)(a1 + 56);
  }
  v73 = sub_C7D670(18, 8);
  strcpy((char *)(v73 + 8), "llvm.used");
  *(_QWORD *)v73 = 9;
  *v23 = v73;
  ++*(_DWORD *)(a1 + 52);
  sub_C929D0((__int64 *)(a1 + 40), v22);
LABEL_25:
  v24 = sub_C92610();
  v25 = sub_C92740(a1 + 40, "llvm.compiler.used", 0x12u, v24);
  v26 = (__int64 *)(*(_QWORD *)(a1 + 40) + 8LL * v25);
  if ( *v26 )
  {
    if ( *v26 != -8 )
      goto LABEL_27;
    --*(_DWORD *)(a1 + 56);
  }
  v71 = sub_C7D670(27, 8);
  si128 = _mm_load_si128((const __m128i *)&xmmword_3F89B50);
  strcpy((char *)(v71 + 24), "ed");
  *(__m128i *)(v71 + 8) = si128;
  *(_QWORD *)v71 = 18;
  *v26 = v71;
  ++*(_DWORD *)(a1 + 52);
  sub_C929D0((__int64 *)(a1 + 40), v25);
LABEL_27:
  v27 = sub_C92610();
  v28 = sub_C92740(a1 + 40, "llvm.global_ctors", 0x11u, v27);
  v29 = (__int64 *)(*(_QWORD *)(a1 + 40) + 8LL * v28);
  if ( *v29 )
  {
    if ( *v29 != -8 )
      goto LABEL_29;
    --*(_DWORD *)(a1 + 56);
  }
  v69 = sub_C7D670(26, 8);
  v70 = _mm_load_si128((const __m128i *)&xmmword_3F89B40);
  *(_BYTE *)(v69 + 24) = 115;
  *(__m128i *)(v69 + 8) = v70;
  *(_BYTE *)(v69 + 25) = 0;
  *(_QWORD *)v69 = 17;
  *v29 = v69;
  ++*(_DWORD *)(a1 + 52);
  sub_C929D0((__int64 *)(a1 + 40), v28);
LABEL_29:
  v30 = sub_C92610();
  v31 = sub_C92740(a1 + 40, "llvm.global_dtors", 0x11u, v30);
  v32 = (__int64 *)(*(_QWORD *)(a1 + 40) + 8LL * v31);
  if ( *v32 )
  {
    if ( *v32 != -8 )
      goto LABEL_31;
    --*(_DWORD *)(a1 + 56);
  }
  v67 = sub_C7D670(26, 8);
  v68 = _mm_load_si128((const __m128i *)&xmmword_3F89B30);
  *(_BYTE *)(v67 + 24) = 115;
  *(__m128i *)(v67 + 8) = v68;
  *(_BYTE *)(v67 + 25) = 0;
  *(_QWORD *)v67 = 17;
  *v32 = v67;
  ++*(_DWORD *)(a1 + 52);
  sub_C929D0((__int64 *)(a1 + 40), v31);
LABEL_31:
  v33 = sub_C92610();
  v34 = sub_C92740(a1 + 40, "llvm.global.annotations", 0x17u, v33);
  v35 = (__int64 *)(*(_QWORD *)(a1 + 40) + 8LL * v34);
  if ( *v35 )
  {
    if ( *v35 != -8 )
      goto LABEL_33;
    --*(_DWORD *)(a1 + 56);
  }
  v65 = sub_C7D670(32, 8);
  v66 = _mm_load_si128((const __m128i *)&xmmword_3F89B20);
  *(_WORD *)(v65 + 28) = 28271;
  *(_DWORD *)(v65 + 24) = 1769234804;
  *(__m128i *)(v65 + 8) = v66;
  *(_BYTE *)(v65 + 30) = 115;
  *(_BYTE *)(v65 + 31) = 0;
  *(_QWORD *)v65 = 23;
  *v35 = v65;
  ++*(_DWORD *)(a1 + 52);
  sub_C929D0((__int64 *)(a1 + 40), v34);
LABEL_33:
  v36 = sub_C92610();
  v37 = sub_C92740(a1 + 40, "__stack_chk_fail", 0x10u, v36);
  v38 = (__int64 *)(*(_QWORD *)(a1 + 40) + 8LL * v37);
  if ( !*v38 )
    goto LABEL_69;
  if ( *v38 == -8 )
  {
    --*(_DWORD *)(a1 + 56);
LABEL_69:
    v60 = sub_C7D670(25, 8);
    v61 = _mm_load_si128((const __m128i *)&xmmword_3F89B10);
    *(_BYTE *)(v60 + 24) = 0;
    *(_QWORD *)v60 = 16;
    *(__m128i *)(v60 + 8) = v61;
    *v38 = v60;
    ++*(_DWORD *)(a1 + 52);
    sub_C929D0((__int64 *)(a1 + 40), v37);
    if ( *(_DWORD *)(a2 + 276) != 19 )
      goto LABEL_36;
    goto LABEL_70;
  }
  if ( *(_DWORD *)(a2 + 276) != 19 )
  {
LABEL_36:
    v39 = sub_C92610();
    v40 = sub_C92740(a1 + 40, "__stack_chk_guard", 0x11u, v39);
    v41 = (__int64 *)(*(_QWORD *)(a1 + 40) + 8LL * v40);
    if ( *v41 )
    {
      if ( *v41 != -8 )
        goto LABEL_38;
      --*(_DWORD *)(a1 + 56);
    }
    v63 = sub_C7D670(26, 8);
    v64 = _mm_load_si128((const __m128i *)&xmmword_3F89B00);
    goto LABEL_86;
  }
LABEL_70:
  v62 = sub_C92610();
  v40 = sub_C92740(a1 + 40, "__ssp_canary_word", 0x11u, v62);
  v41 = (__int64 *)(*(_QWORD *)(a1 + 40) + 8LL * v40);
  if ( *v41 )
  {
    if ( *v41 != -8 )
      goto LABEL_38;
    --*(_DWORD *)(a1 + 56);
  }
  v63 = sub_C7D670(26, 8);
  v64 = _mm_load_si128((const __m128i *)&xmmword_3F89AF0);
LABEL_86:
  *(__m128i *)(v63 + 8) = v64;
  *(_BYTE *)(v63 + 24) = 100;
  *(_BYTE *)(v63 + 25) = 0;
  *(_QWORD *)v63 = 17;
  *v41 = v63;
  ++*(_DWORD *)(a1 + 52);
  sub_C929D0((__int64 *)(a1 + 40), v40);
LABEL_38:
  if ( (unsigned int)(*(_DWORD *)(a2 + 264) - 42) > 1 )
    goto LABEL_41;
  v42 = sub_C92610();
  v43 = sub_C92740(a1 + 40, "__llvm_rpc_client", 0x11u, v42);
  v44 = (__int64 *)(*(_QWORD *)(a1 + 40) + 8LL * v43);
  if ( *v44 )
  {
    if ( *v44 != -8 )
      goto LABEL_41;
    --*(_DWORD *)(a1 + 56);
  }
  v74 = sub_C7D670(26, 8);
  v75 = _mm_load_si128((const __m128i *)&xmmword_3F89AE0);
  *(_BYTE *)(v74 + 24) = 116;
  *(__m128i *)(v74 + 8) = v75;
  *(_BYTE *)(v74 + 25) = 0;
  *(_QWORD *)v74 = 17;
  *v44 = v74;
  ++*(_DWORD *)(a1 + 52);
  sub_C929D0((__int64 *)(a1 + 40), v43);
LABEL_41:
  *(_BYTE *)a1 = *(_DWORD *)(a2 + 284) == 7;
  v45 = 0;
  v46 = *(_QWORD *)(a2 + 32);
  if ( v46 != v82 )
  {
    v47 = 0;
    do
    {
      v48 = (_BYTE *)(v46 - 56);
      if ( !v46 )
        v48 = 0;
      v49 = sub_F2FEA0((_BYTE *)a1, v48, (__int64)&v84);
      v46 = *(_QWORD *)(v46 + 8);
      if ( (_BYTE)v49 )
        v47 = v49;
    }
    while ( v46 != v82 );
    v45 = v47;
  }
  v50 = *(_QWORD *)(a2 + 16);
  if ( v50 != v81 )
  {
    v51 = v45;
    do
    {
      v52 = (_BYTE *)(v50 - 56);
      if ( !v50 )
        v52 = 0;
      v53 = sub_F2FEA0((_BYTE *)a1, v52, (__int64)&v84);
      v50 = *(_QWORD *)(v50 + 8);
      if ( (_BYTE)v53 )
        v51 = v53;
    }
    while ( v50 != v81 );
    v45 = v51;
  }
  v54 = *(_QWORD *)(a2 + 48);
  if ( v54 != v83 )
  {
    v55 = v45;
    do
    {
      v56 = (_BYTE *)(v54 - 48);
      if ( !v54 )
        v56 = 0;
      v57 = sub_F2FEA0((_BYTE *)a1, v56, (__int64)&v84);
      v54 = *(_QWORD *)(v54 + 8);
      if ( (_BYTE)v57 )
        v55 = v57;
    }
    while ( v54 != v83 );
    v45 = v55;
  }
  v58 = 24LL * v87;
  sub_C7D6A0(v85, v58, 8);
  if ( v88 != (__int64 *)v90 )
    _libc_free(v88, v58);
  return v45;
}
