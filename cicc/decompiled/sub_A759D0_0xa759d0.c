// Function: sub_A759D0
// Address: 0xa759d0
//
__int64 __fastcall sub_A759D0(__int64 a1, __int64 *a2, char a3)
{
  bool v4; // zf
  int v6; // eax
  __int64 v7; // rdx
  char *v8; // rsi
  int v9; // eax
  char *v10; // rax
  __int64 v11; // rdx
  __int64 v12; // r14
  __m128i *v13; // rax
  unsigned __int64 v14; // r13
  unsigned __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // r14
  unsigned __int64 v19; // rax
  __int64 v20; // rax
  unsigned __int64 v22; // rax
  __int32 v23; // ecx
  int v24; // edx
  unsigned __int64 v25; // rax
  __int32 v26; // eax
  const char *v27; // rsi
  __int64 v28; // rcx
  char v29; // bl
  __int64 v30; // r8
  __int64 v31; // r9
  __int64 *v32; // rbx
  __int64 v33; // rax
  __m128i *v34; // r13
  __int64 v35; // rsi
  __m128i *v36; // rax
  __int64 v37; // rcx
  unsigned __int64 v38; // rdx
  __int64 v39; // rsi
  __int64 v40; // rcx
  __int64 v41; // r8
  __int64 v42; // r9
  unsigned int v43; // r14d
  bool v44; // dl
  char *v45; // rax
  int i; // r15d
  char *v47; // rax
  __int64 v48; // rax
  __int64 v49; // rax
  unsigned int v50; // eax
  unsigned int v51; // eax
  __int64 v52; // r14
  __int64 v53; // rax
  __int64 v54; // rax
  __int64 v55; // rax
  unsigned __int64 v56; // rax
  __int64 v57; // rdx
  __int64 v58; // rax
  __int64 v59; // r15
  __m128i *v60; // rbx
  __m128i *v61; // r15
  __int64 v62; // rdi
  __int64 v63; // r14
  const void *v64; // rax
  size_t v65; // rdx
  void *v66; // rdi
  __int64 v67; // r15
  __int64 v68; // rdx
  __int64 v69; // r14
  __int64 v70; // rax
  char v71; // [rsp+0h] [rbp-1C0h]
  size_t v72; // [rsp+0h] [rbp-1C0h]
  bool v73; // [rsp+1Bh] [rbp-1A5h]
  _BYTE v74[4]; // [rsp+1Ch] [rbp-1A4h] BYREF
  char *v75; // [rsp+20h] [rbp-1A0h] BYREF
  __int64 v76; // [rsp+28h] [rbp-198h]
  char v77; // [rsp+30h] [rbp-190h] BYREF
  __m128i v78[2]; // [rsp+40h] [rbp-180h] BYREF
  __int16 v79; // [rsp+60h] [rbp-160h]
  __m128i v80; // [rsp+70h] [rbp-150h] BYREF
  __int32 v81; // [rsp+80h] [rbp-140h]
  __int16 v82; // [rsp+90h] [rbp-130h]
  __m128i v83; // [rsp+A0h] [rbp-120h] BYREF
  char *v84; // [rsp+B0h] [rbp-110h]
  __int16 v85; // [rsp+C0h] [rbp-100h]
  __m128i v86; // [rsp+D0h] [rbp-F0h] BYREF
  __m128i v87; // [rsp+E0h] [rbp-E0h] BYREF
  __int16 v88; // [rsp+F0h] [rbp-D0h]
  __m128i v89; // [rsp+100h] [rbp-C0h] BYREF
  __m128i v90; // [rsp+110h] [rbp-B0h] BYREF
  __int64 v91; // [rsp+120h] [rbp-A0h]
  __int64 v92; // [rsp+128h] [rbp-98h]
  __m128i *v93; // [rsp+130h] [rbp-90h]
  __m128i *v94; // [rsp+140h] [rbp-80h] BYREF
  __int64 v95; // [rsp+148h] [rbp-78h]
  __m128i *v96; // [rsp+150h] [rbp-70h] BYREF
  __int64 v97; // [rsp+158h] [rbp-68h]
  __m128i *v98; // [rsp+160h] [rbp-60h]
  __int64 v99; // [rsp+168h] [rbp-58h]
  __m128i *v100; // [rsp+170h] [rbp-50h]

  v4 = *a2 == 0;
  v74[0] = a3;
  if ( v4 )
  {
    *(_QWORD *)(a1 + 8) = 0;
    *(_QWORD *)a1 = a1 + 16;
    *(_BYTE *)(a1 + 16) = 0;
    return a1;
  }
  if ( sub_A71800((__int64)a2) )
  {
    v6 = sub_A71AE0(a2);
    v8 = sub_A6FBB0(v6);
    *(_QWORD *)a1 = a1 + 16;
    if ( v8 )
    {
      sub_A6E150((__int64 *)a1, v8, (__int64)&v8[v7]);
    }
    else
    {
      *(_QWORD *)(a1 + 8) = 0;
      *(_BYTE *)(a1 + 16) = 0;
    }
    return a1;
  }
  if ( sub_A71860((__int64)a2) )
  {
    v9 = sub_A71AE0(a2);
    v10 = sub_A6FBB0(v9);
    if ( v10 )
    {
      v89.m128i_i64[0] = (__int64)&v90;
      sub_A6E150(v89.m128i_i64, v10, (__int64)&v10[v11]);
      v12 = v89.m128i_i64[1];
      v13 = (__m128i *)v89.m128i_i64[0];
      v14 = v89.m128i_i64[1] + 1;
      if ( (__m128i *)v89.m128i_i64[0] == &v90 )
        v15 = 15;
      else
        v15 = v90.m128i_i64[0];
      if ( v14 > v15 )
      {
        sub_2240BB0(&v89, v89.m128i_i64[1], 0, 0, 1);
        v13 = (__m128i *)v89.m128i_i64[0];
      }
    }
    else
    {
      v90.m128i_i8[0] = 0;
      v12 = 0;
      v89.m128i_i64[0] = (__int64)&v90;
      v14 = 1;
      v13 = &v90;
    }
    v13->m128i_i8[v12] = 40;
    v89.m128i_i64[1] = v14;
    *(_BYTE *)(v89.m128i_i64[0] + v14) = 0;
    v99 = 0x100000000LL;
    v95 = 0;
    v94 = (__m128i *)&unk_49DD210;
    v96 = 0;
    v100 = &v89;
    v97 = 0;
    v98 = 0;
    sub_CB5980(&v94, 0, 0, 0);
    v16 = sub_A72A60(a2);
    sub_A587F0(v16, (__int64)&v94, 0, 1);
    if ( v96 != v98 )
      sub_CB5AE0(&v94);
    v18 = v89.m128i_i64[1];
    v17 = v89.m128i_i64[0];
    v19 = 15;
    if ( (__m128i *)v89.m128i_i64[0] != &v90 )
      v19 = v90.m128i_i64[0];
    if ( v89.m128i_i64[1] + 1 > v19 )
    {
      sub_2240BB0(&v89, v89.m128i_i64[1], 0, 0, 1);
      v17 = v89.m128i_i64[0];
    }
    *(_BYTE *)(v17 + v18) = 41;
    v89.m128i_i64[1] = v18 + 1;
    *(_BYTE *)(v89.m128i_i64[0] + v18 + 1) = 0;
    *(_QWORD *)a1 = a1 + 16;
    if ( (__m128i *)v89.m128i_i64[0] == &v90 )
    {
      *(__m128i *)(a1 + 16) = _mm_load_si128(&v90);
    }
    else
    {
      *(_QWORD *)a1 = v89.m128i_i64[0];
      *(_QWORD *)(a1 + 16) = v90.m128i_i64[0];
    }
    v20 = v89.m128i_i64[1];
    v89.m128i_i64[0] = (__int64)&v90;
    v89.m128i_i64[1] = 0;
    *(_QWORD *)(a1 + 8) = v20;
    v90.m128i_i8[0] = 0;
    v94 = (__m128i *)&unk_49DD210;
    sub_CB5840(&v94);
    if ( (__m128i *)v89.m128i_i64[0] != &v90 )
      j_j___libc_free_0(v89.m128i_i64[0], v90.m128i_i64[0] + 1);
    return a1;
  }
  if ( sub_A71B30(a2, 86) )
  {
    if ( v74[0] )
    {
      v86.m128i_i64[0] = sub_A71B80(a2);
      v94 = (__m128i *)"align=";
      v96 = &v86;
    }
    else
    {
      v89.m128i_i64[0] = sub_A71B80(a2);
      v94 = (__m128i *)"align ";
      v96 = &v89;
    }
    LOWORD(v98) = 2819;
    goto LABEL_27;
  }
  if ( sub_A71B30(a2, 94) )
  {
    sub_A71BA0(a1, v74, a2, "alignstack");
    return a1;
  }
  if ( sub_A71B30(a2, 90) )
  {
    sub_A71BA0(a1, v74, a2, "dereferenceable");
    return a1;
  }
  if ( sub_A71B30(a2, 91) )
  {
    sub_A71BA0(a1, v74, a2, "dereferenceable_or_null");
    return a1;
  }
  if ( sub_A71B30(a2, 88) )
  {
    v22 = sub_A71E50(a2);
    v23 = v22;
    LODWORD(v95) = v24;
    v25 = HIDWORD(v22);
    if ( (_BYTE)v24 )
    {
      v87.m128i_i32[0] = v25;
      v80.m128i_i64[0] = (__int64)"allocsize(";
      v81 = v23;
      v83.m128i_i64[0] = (__int64)&v80;
      v94 = &v86;
      v82 = 2307;
      v84 = ",";
      v85 = 770;
      v86.m128i_i64[0] = (__int64)&v83;
      v88 = 2306;
      v96 = (__m128i *)")";
      LOWORD(v98) = 770;
LABEL_27:
      sub_CA0F50(a1, &v94);
      return a1;
    }
    v90.m128i_i32[0] = v23;
    v89.m128i_i64[0] = (__int64)"allocsize(";
    LOWORD(v91) = 2307;
LABEL_45:
    v94 = &v89;
    v96 = (__m128i *)")";
    LOWORD(v98) = 770;
    goto LABEL_27;
  }
  if ( sub_A71B30(a2, 96) )
  {
    LODWORD(v84) = sub_A71EB0(a2);
    v80.m128i_i64[0] = sub_A71ED0(a2);
    v26 = 0;
    v83.m128i_i64[0] = (__int64)"vscale_range(";
    if ( v80.m128i_i8[4] )
      v26 = v80.m128i_i32[0];
    v87.m128i_i64[0] = (__int64)",";
    v85 = 2307;
    v86.m128i_i64[0] = (__int64)&v83;
    v88 = 770;
    v89.m128i_i64[0] = (__int64)&v86;
    v90.m128i_i32[0] = v26;
    LOWORD(v91) = 2306;
    goto LABEL_45;
  }
  if ( sub_A71B30(a2, 95) )
  {
    v27 = "uwtable";
    if ( (unsigned int)sub_A71DF0(a2) != 2 )
      v27 = "uwtable(sync)";
    sub_A758C0((__int64 *)a1, v27);
    return a1;
  }
  if ( sub_A71B30(a2, 87) )
  {
    v29 = sub_A71E00(a2);
    v94 = (__m128i *)&v96;
    v95 = 0x300000000LL;
    if ( (v29 & 1) != 0 )
      sub_A75970((__int64)&v94, (__int64)"alloc", 5);
    if ( (v29 & 2) != 0 )
      sub_A75970((__int64)&v94, (__int64)"realloc", 7);
    if ( (v29 & 4) != 0 )
      sub_A75970((__int64)&v94, (__int64)"free", 4);
    if ( (v29 & 8) != 0 )
      sub_A75970((__int64)&v94, (__int64)"uninitialized", 13);
    if ( (v29 & 0x10) != 0 )
      sub_A75970((__int64)&v94, (__int64)"zeroed", 6);
    if ( (v29 & 0x20) != 0 )
      sub_A75970((__int64)&v94, (__int64)"aligned", 7);
    v32 = (__int64 *)v94;
    v86.m128i_i64[0] = (__int64)"\")";
    v88 = 259;
    v33 = (unsigned int)v95;
    v75 = &v77;
    v34 = &v94[v33];
    v77 = 0;
    v76 = 0;
    if ( v94 != &v94[v33] )
    {
      v35 = ((v33 * 16) >> 4) - 1;
      v36 = v94;
      do
      {
        v35 += v36->m128i_i64[1];
        ++v36;
      }
      while ( v34 != v36 );
      sub_2240E30(&v75, v35);
      v38 = v32[1];
      v39 = *v32;
      if ( v38 > 0x3FFFFFFFFFFFFFFFLL - v76 )
LABEL_104:
        sub_4262D8((__int64)"basic_string::append");
      while ( 1 )
      {
        v32 += 2;
        sub_2241490(&v75, v39, v38, v37);
        if ( v34 == (__m128i *)v32 )
          break;
        if ( v76 != 0x3FFFFFFFFFFFFFFFLL )
        {
          sub_2241490(&v75, ",", 1, v28);
          v38 = v32[1];
          v39 = *v32;
          if ( v38 <= 0x3FFFFFFFFFFFFFFFLL - v76 )
            continue;
        }
        goto LABEL_104;
      }
    }
    v79 = 260;
    v80.m128i_i64[0] = (__int64)"allockind(\"";
    v78[0].m128i_i64[0] = (__int64)&v75;
    v82 = 259;
    sub_9C6370(&v83, &v80, v78, v28, v30, v31);
    sub_9C6370(&v89, &v83, &v86, v40, v41, v42);
    sub_CA0F50(a1, &v89);
    sub_2240A30(&v75);
    if ( v94 != (__m128i *)&v96 )
      _libc_free(v94, &v89);
    return a1;
  }
  v73 = sub_A71B30(a2, 92);
  if ( v73 )
  {
    v89.m128i_i64[1] = 0;
    v100 = &v89;
    v99 = 0x100000000LL;
    v94 = (__m128i *)&unk_49DD210;
    v89.m128i_i64[0] = (__int64)&v90;
    v90.m128i_i8[0] = 0;
    v95 = 0;
    v96 = 0;
    v97 = 0;
    v98 = 0;
    sub_A6E5D0((__int64)&v94);
    sub_904010((__int64)&v94, "memory(");
    v43 = sub_A71E40(a2);
    v71 = (unsigned __int8)v43 >> 6;
    if ( (unsigned __int8)v43 >> 6
      || (v44 = v73, (((unsigned __int8)(v43 >> 4) | (unsigned __int8)(v43 | (v43 >> 2))) & 3) == 0) )
    {
      v45 = sub_A6DE70(v71);
      sub_904010((__int64)&v94, v45);
      v44 = 0;
    }
    for ( i = 0; i != 4; ++i )
    {
      if ( v71 != ((v43 >> (2 * i)) & 3) )
      {
        if ( !v44 )
          sub_904010((__int64)&v94, ", ");
        switch ( i )
        {
          case 2:
            sub_904010((__int64)&v94, "errnomem: ");
            break;
          case 3:
            goto LABEL_138;
          case 1:
            sub_904010((__int64)&v94, "inaccessiblemem: ");
            break;
          default:
            sub_904010((__int64)&v94, "argmem: ");
            break;
        }
        v47 = sub_A6DE70((v43 >> (2 * i)) & 3);
        sub_904010((__int64)&v94, v47);
        v44 = 0;
      }
    }
    sub_904010((__int64)&v94, ")");
    if ( v98 != v96 )
      sub_CB5AE0(&v94);
    *(_QWORD *)a1 = a1 + 16;
    v48 = v89.m128i_i64[0];
    if ( (__m128i *)v89.m128i_i64[0] != &v90 )
      goto LABEL_91;
LABEL_109:
    *(__m128i *)(a1 + 16) = _mm_load_si128(&v90);
    goto LABEL_92;
  }
  if ( sub_A71B30(a2, 89) )
  {
    v89.m128i_i64[1] = 0;
    v100 = &v89;
    v99 = 0x100000000LL;
    v94 = (__m128i *)&unk_49DD210;
    v89.m128i_i64[0] = (__int64)&v90;
    v90.m128i_i8[0] = 0;
    v95 = 0;
    v96 = 0;
    v97 = 0;
    v98 = 0;
    sub_A6E5D0((__int64)&v94);
    v50 = sub_A71E10(a2);
    sub_C7EF90(&v94, v50);
    *(_QWORD *)a1 = a1 + 16;
    v48 = v89.m128i_i64[0];
    if ( (__m128i *)v89.m128i_i64[0] == &v90 )
      goto LABEL_109;
LABEL_91:
    *(_QWORD *)a1 = v48;
    *(_QWORD *)(a1 + 16) = v90.m128i_i64[0];
LABEL_92:
    v49 = v89.m128i_i64[1];
    v89.m128i_i64[0] = (__int64)&v90;
    v89.m128i_i64[1] = 0;
    *(_QWORD *)(a1 + 8) = v49;
    v90.m128i_i8[0] = 0;
    v94 = (__m128i *)&unk_49DD210;
    sub_CB5840(&v94);
    sub_2240A30(&v89);
    return a1;
  }
  if ( sub_A71B30(a2, 93) )
  {
    sub_A758C0(v89.m128i_i64, "nofpclass");
    v100 = &v89;
    v95 = 0;
    v99 = 0x100000000LL;
    v96 = 0;
    v94 = (__m128i *)&unk_49DD210;
    v97 = 0;
    v98 = 0;
    sub_A6E5D0((__int64)&v94);
    v51 = sub_A71E30(a2);
    sub_C65140(&v94, v51);
    *(_QWORD *)a1 = a1 + 16;
    v48 = v89.m128i_i64[0];
    if ( (__m128i *)v89.m128i_i64[0] == &v90 )
      goto LABEL_109;
    goto LABEL_91;
  }
  if ( sub_A71B30(a2, 97) )
  {
    v89.m128i_i64[1] = 0;
    v100 = &v89;
    v99 = 0x100000000LL;
    v94 = (__m128i *)&unk_49DD210;
    v89.m128i_i64[0] = (__int64)&v90;
    v90.m128i_i8[0] = 0;
    v95 = 0;
    v96 = 0;
    v97 = 0;
    v98 = 0;
    sub_A6E5D0((__int64)&v94);
    v52 = sub_A72A90(a2);
    sub_904010((__int64)&v94, "range(");
    v53 = sub_904010((__int64)&v94, "i");
    v54 = sub_CB59D0(v53, *(unsigned int *)(v52 + 8));
    sub_904010(v54, " ");
    sub_C49420(v52, &v94, 1);
    v55 = sub_904010((__int64)&v94, ", ");
    sub_C49420(v52 + 16, v55, 1);
    sub_904010((__int64)&v94, ")");
    if ( v98 != v96 )
      sub_CB5AE0(&v94);
    *(_QWORD *)a1 = a1 + 16;
    v48 = v89.m128i_i64[0];
    if ( (__m128i *)v89.m128i_i64[0] == &v90 )
      goto LABEL_109;
    goto LABEL_91;
  }
  if ( sub_A71B30(a2, 98) )
  {
    v86.m128i_i64[1] = 0;
    v87.m128i_i8[0] = 0;
    v92 = 0x100000000LL;
    v86.m128i_i64[0] = (__int64)&v87;
    v89.m128i_i64[0] = (__int64)&unk_49DD210;
    v89.m128i_i64[1] = 0;
    v90 = 0u;
    v91 = 0;
    v93 = &v86;
    sub_A6E5D0((__int64)&v89);
    v56 = sub_A72F20(a2);
    sub_A6E600((__int64)&v94, v56, v57);
    sub_904010((__int64)&v89, "initializes(");
    sub_ABF230(&v94, &v89);
    sub_904010((__int64)&v89, ")");
    if ( v91 != v90.m128i_i64[0] )
      sub_CB5AE0(&v89);
    *(_QWORD *)a1 = a1 + 16;
    if ( (__m128i *)v86.m128i_i64[0] == &v87 )
    {
      *(__m128i *)(a1 + 16) = _mm_load_si128(&v87);
    }
    else
    {
      *(_QWORD *)a1 = v86.m128i_i64[0];
      *(_QWORD *)(a1 + 16) = v87.m128i_i64[0];
    }
    v58 = v86.m128i_i64[1];
    v59 = (unsigned int)v95;
    v86.m128i_i64[0] = (__int64)&v87;
    v86.m128i_i64[1] = 0;
    v60 = v94;
    *(_QWORD *)(a1 + 8) = v58;
    v87.m128i_i8[0] = 0;
    v61 = &v60[2 * v59];
    while ( v60 != v61 )
    {
      v61 -= 2;
      if ( v61[1].m128i_i32[2] > 0x40u )
      {
        v62 = v61[1].m128i_i64[0];
        if ( v62 )
          j_j___libc_free_0_0(v62);
      }
      if ( v61->m128i_i32[2] > 0x40u && v61->m128i_i64[0] )
        j_j___libc_free_0_0(v61->m128i_i64[0]);
    }
    if ( v94 != (__m128i *)&v96 )
      _libc_free(v94, ")");
    v89.m128i_i64[0] = (__int64)&unk_49DD210;
    sub_CB5840(&v89);
    sub_2240A30(&v86);
  }
  else
  {
    if ( !sub_A71840((__int64)a2) )
LABEL_138:
      BUG();
    v89.m128i_i64[1] = 0;
    v100 = &v89;
    v99 = 0x100000000LL;
    v94 = (__m128i *)&unk_49DD210;
    v89.m128i_i64[0] = (__int64)&v90;
    v90.m128i_i8[0] = 0;
    v95 = 0;
    v96 = 0;
    v97 = 0;
    v98 = 0;
    sub_A6E5D0((__int64)&v94);
    v63 = sub_A51310((__int64)&v94, 0x22u);
    v64 = (const void *)sub_A71FD0(a2);
    v66 = *(void **)(v63 + 32);
    if ( v65 > *(_QWORD *)(v63 + 24) - (_QWORD)v66 )
    {
      v63 = sub_CB6200(v63, v64, v65);
    }
    else if ( v65 )
    {
      v72 = v65;
      memcpy(v66, v64, v65);
      *(_QWORD *)(v63 + 32) += v72;
    }
    sub_A51310(v63, 0x22u);
    v67 = sub_A72230(*a2);
    v69 = v68;
    if ( v68 )
    {
      sub_904010((__int64)&v94, "=\"");
      sub_C92400(v67, v69, &v94);
      sub_904010((__int64)&v94, "\"");
    }
    v94 = (__m128i *)&unk_49DD210;
    sub_CB5840(&v94);
    *(_QWORD *)a1 = a1 + 16;
    if ( (__m128i *)v89.m128i_i64[0] == &v90 )
    {
      *(__m128i *)(a1 + 16) = _mm_load_si128(&v90);
    }
    else
    {
      *(_QWORD *)a1 = v89.m128i_i64[0];
      *(_QWORD *)(a1 + 16) = v90.m128i_i64[0];
    }
    v70 = v89.m128i_i64[1];
    v89.m128i_i64[0] = (__int64)&v90;
    v89.m128i_i64[1] = 0;
    *(_QWORD *)(a1 + 8) = v70;
    v90.m128i_i8[0] = 0;
    sub_2240A30(&v89);
  }
  return a1;
}
