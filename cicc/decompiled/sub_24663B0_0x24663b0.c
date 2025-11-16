// Function: sub_24663B0
// Address: 0x24663b0
//
void __fastcall sub_24663B0(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v4; // rdi
  int v5; // r8d
  char v6; // al
  const char *v7; // r13
  unsigned __int64 v9; // r12
  __int64 v10; // r15
  __int64 v11; // rax
  __int64 *v12; // rsi
  unsigned __int64 v13; // rax
  __int64 v14; // r13
  __int64 v15; // rdx
  __int64 v16; // r12
  __int64 *v17; // rax
  _BYTE *v18; // rax
  void *v19; // rcx
  _BYTE *v20; // rax
  _QWORD *v21; // rdi
  __int64 *v22; // rax
  _BYTE *v23; // rax
  __int64 *v24; // rdi
  _BYTE *v25; // rax
  _QWORD *v26; // rdi
  __int64 *v27; // rax
  _BYTE *v28; // rax
  __int64 *v29; // rdi
  _BYTE *v30; // rax
  _QWORD *v31; // rdi
  unsigned __int64 *v32; // rax
  __int64 v33; // r15
  __int64 v34; // rax
  __int64 *v35; // r11
  __int64 *v36; // rsi
  unsigned __int64 v37; // r8
  int v38; // ecx
  unsigned __int64 v39; // rax
  __int64 v40; // rdx
  __int64 v41; // r15
  __int64 v42; // rax
  __m128i *v43; // rax
  __m128i *v44; // rdi
  __int64 v45; // rsi
  __int64 v46; // r12
  __int64 v47; // r13
  __int64 v48; // rax
  __int64 *v49; // r15
  __int64 *v50; // r10
  unsigned __int64 v51; // r8
  int v52; // ecx
  unsigned __int64 v53; // r15
  unsigned __int64 v54; // rax
  __int64 v55; // r13
  __int64 v56; // rdx
  __int64 v57; // r12
  __m128i *v58; // rdi
  __int64 v59; // r12
  __int64 v60; // r13
  __int64 *v61; // rax
  unsigned __int64 v62; // rax
  __int64 v63; // r13
  __int64 v64; // rdx
  __int64 v65; // r12
  _QWORD *v66; // rdi
  __int64 v67; // r13
  __int64 v68; // r12
  __int64 *v69; // rax
  unsigned __int64 v70; // rax
  __int64 v71; // r13
  __int64 v72; // rdx
  __int64 v73; // r12
  _QWORD *v74; // rdi
  __int64 v75; // r13
  __int64 v76; // r12
  __int64 *v77; // rax
  unsigned __int64 v78; // rax
  __int64 v79; // r13
  __int64 v80; // rdx
  __int64 v81; // r12
  unsigned __int64 v82; // rax
  unsigned __int64 v83; // rax
  size_t v84; // rdx
  const char *v85; // r13
  unsigned __int64 v86; // r12
  __int64 *v87; // rax
  unsigned __int64 v88; // rax
  __int64 v89; // rdx
  __int64 v90; // [rsp+30h] [rbp-190h]
  __int64 *v91; // [rsp+30h] [rbp-190h]
  unsigned __int64 v92; // [rsp+38h] [rbp-188h]
  __int64 v93; // [rsp+38h] [rbp-188h]
  __int64 *v94; // [rsp+38h] [rbp-188h]
  int v95; // [rsp+38h] [rbp-188h]
  void *v96; // [rsp+40h] [rbp-180h]
  void *v97; // [rsp+40h] [rbp-180h]
  int v98; // [rsp+40h] [rbp-180h]
  __int64 v99; // [rsp+48h] [rbp-178h]
  __int64 v100; // [rsp+48h] [rbp-178h]
  __int64 v101; // [rsp+48h] [rbp-178h]
  __int64 *v103; // [rsp+78h] [rbp-148h]
  __int64 *v104; // [rsp+78h] [rbp-148h]
  unsigned __int64 v106; // [rsp+88h] [rbp-138h]
  __int64 i; // [rsp+88h] [rbp-138h]
  void *dest; // [rsp+90h] [rbp-130h] BYREF
  size_t v109; // [rsp+98h] [rbp-128h]
  __m128i v110; // [rsp+A0h] [rbp-120h] BYREF
  const char *v111; // [rsp+B0h] [rbp-110h] BYREF
  __int64 v112; // [rsp+B8h] [rbp-108h]
  _BYTE v113[4]; // [rsp+C0h] [rbp-100h] BYREF
  char v114; // [rsp+C4h] [rbp-FCh] BYREF
  _BYTE v115[3]; // [rsp+C5h] [rbp-FBh] BYREF
  __m128i *p_src; // [rsp+D0h] [rbp-F0h] BYREF
  size_t n; // [rsp+D8h] [rbp-E8h]
  __m128i src; // [rsp+E0h] [rbp-E0h] BYREF
  __int64 v119; // [rsp+F0h] [rbp-D0h]
  __int64 v120; // [rsp+F8h] [rbp-C8h]
  _QWORD v121[2]; // [rsp+100h] [rbp-C0h] BYREF
  char v122; // [rsp+110h] [rbp-B0h] BYREF
  __int64 v123; // [rsp+130h] [rbp-90h]
  __int64 v124; // [rsp+138h] [rbp-88h]
  __int16 v125; // [rsp+140h] [rbp-80h]
  _QWORD *v126; // [rsp+148h] [rbp-78h]
  void **v127; // [rsp+150h] [rbp-70h]
  void **v128; // [rsp+158h] [rbp-68h]
  __int64 v129; // [rsp+160h] [rbp-60h]
  int v130; // [rsp+168h] [rbp-58h]
  __int16 v131; // [rsp+16Ch] [rbp-54h]
  char v132; // [rsp+16Eh] [rbp-52h]
  __int64 v133; // [rsp+170h] [rbp-50h]
  __int64 v134; // [rsp+178h] [rbp-48h]
  void *v135; // [rsp+180h] [rbp-40h] BYREF
  void *v136; // [rsp+188h] [rbp-38h] BYREF

  v4 = *(_QWORD **)(a1 + 72);
  v121[0] = &v122;
  v121[1] = 0x200000000LL;
  v5 = *(_DWORD *)(a1 + 4);
  v127 = &v135;
  v128 = &v136;
  v135 = &unk_49DA100;
  v126 = v4;
  v129 = 0;
  v136 = &unk_49DA0B0;
  v6 = *(_BYTE *)(a1 + 8);
  v130 = 0;
  v131 = 512;
  v132 = 7;
  v133 = 0;
  v134 = 0;
  v123 = 0;
  v124 = 0;
  v125 = 0;
  if ( v5 )
  {
    v7 = "__msan_warning_with_origin";
    v9 = v6 == 0 ? 35LL : 26LL;
    if ( !v6 )
      v7 = "__msan_warning_with_origin_noreturn";
    v10 = sub_BCB2D0(v4);
    v11 = sub_BCB120(v126);
    v12 = *(__int64 **)(a1 + 72);
    v103 = (__int64 *)v11;
    LODWORD(v111) = 0;
    v106 = sub_24662B0(a3, v12, (int *)&v111, 1, 0, 0, 0);
    p_src = &src;
    src.m128i_i64[0] = v10;
    n = 0x100000001LL;
    v13 = sub_BCF480(v103, &src, 1, 0);
    v14 = sub_BA8C10(a2, (__int64)v7, v9, v13, v106);
    v16 = v15;
    if ( p_src != &src )
      _libc_free((unsigned __int64)p_src);
    *(_QWORD *)(a1 + 168) = v14;
    *(_QWORD *)(a1 + 176) = v16;
  }
  else
  {
    v85 = "__msan_warning";
    v86 = v6 == 0 ? 23LL : 14LL;
    if ( !v6 )
      v85 = "__msan_warning_noreturn";
    v87 = (__int64 *)sub_BCB120(v4);
    p_src = &src;
    n = 0;
    v88 = sub_BCF480(v87, &src, 0, 0);
    *(_QWORD *)(a1 + 168) = sub_BA8C10(a2, (__int64)v85, v86, v88, 0);
    *(_QWORD *)(a1 + 176) = v89;
  }
  v17 = (__int64 *)sub_BCB2E0(v126);
  dest = sub_BCD420(v17, 100);
  v111 = "__msan_retval_tls";
  v112 = 17;
  p_src = (__m128i *)a2;
  n = (size_t)&dest;
  src.m128i_i64[0] = (__int64)&v111;
  v18 = sub_BA8D20(
          a2,
          (__int64)"__msan_retval_tls",
          0x11u,
          (__int64)dest,
          (__int64 (__fastcall *)(__int64))sub_2461C20,
          (__int64)&p_src);
  v19 = *(void **)(a1 + 88);
  *(_QWORD *)(a1 + 120) = v18;
  dest = v19;
  v111 = "__msan_retval_origin_tls";
  v112 = 24;
  p_src = (__m128i *)a2;
  n = (size_t)&dest;
  src.m128i_i64[0] = (__int64)&v111;
  v20 = sub_BA8D20(
          a2,
          (__int64)"__msan_retval_origin_tls",
          0x18u,
          (__int64)v19,
          (__int64 (__fastcall *)(__int64))sub_2461C20,
          (__int64)&p_src);
  v21 = v126;
  *(_QWORD *)(a1 + 128) = v20;
  v22 = (__int64 *)sub_BCB2E0(v21);
  dest = sub_BCD420(v22, 100);
  v111 = "__msan_param_tls";
  v112 = 16;
  p_src = (__m128i *)a2;
  n = (size_t)&dest;
  src.m128i_i64[0] = (__int64)&v111;
  v23 = sub_BA8D20(
          a2,
          (__int64)"__msan_param_tls",
          0x10u,
          (__int64)dest,
          (__int64 (__fastcall *)(__int64))sub_2461C20,
          (__int64)&p_src);
  v24 = *(__int64 **)(a1 + 88);
  *(_QWORD *)(a1 + 104) = v23;
  dest = sub_BCD420(v24, 200);
  v111 = "__msan_param_origin_tls";
  v112 = 23;
  p_src = (__m128i *)a2;
  n = (size_t)&dest;
  src.m128i_i64[0] = (__int64)&v111;
  v25 = sub_BA8D20(
          a2,
          (__int64)"__msan_param_origin_tls",
          0x17u,
          (__int64)dest,
          (__int64 (__fastcall *)(__int64))sub_2461C20,
          (__int64)&p_src);
  v26 = v126;
  *(_QWORD *)(a1 + 112) = v25;
  v27 = (__int64 *)sub_BCB2E0(v26);
  dest = sub_BCD420(v27, 100);
  v111 = "__msan_va_arg_tls";
  v112 = 17;
  p_src = (__m128i *)a2;
  n = (size_t)&dest;
  src.m128i_i64[0] = (__int64)&v111;
  v28 = sub_BA8D20(
          a2,
          (__int64)"__msan_va_arg_tls",
          0x11u,
          (__int64)dest,
          (__int64 (__fastcall *)(__int64))sub_2461C20,
          (__int64)&p_src);
  v29 = *(__int64 **)(a1 + 88);
  *(_QWORD *)(a1 + 136) = v28;
  dest = sub_BCD420(v29, 200);
  v111 = "__msan_va_arg_origin_tls";
  v112 = 24;
  p_src = (__m128i *)a2;
  n = (size_t)&dest;
  src.m128i_i64[0] = (__int64)&v111;
  v30 = sub_BA8D20(
          a2,
          (__int64)"__msan_va_arg_origin_tls",
          0x18u,
          (__int64)dest,
          (__int64 (__fastcall *)(__int64))sub_2461C20,
          (__int64)&p_src);
  v31 = v126;
  *(_QWORD *)(a1 + 144) = v30;
  dest = (void *)sub_BCB2E0(v31);
  v111 = "__msan_va_arg_overflow_size_tls";
  v112 = 31;
  p_src = (__m128i *)a2;
  n = (size_t)&dest;
  src.m128i_i64[0] = (__int64)&v111;
  *(_QWORD *)(a1 + 152) = sub_BA8D20(
                            a2,
                            (__int64)"__msan_va_arg_overflow_size_tls",
                            0x1Fu,
                            (__int64)dest,
                            (__int64 (__fastcall *)(__int64))sub_2461C20,
                            (__int64)&p_src);
  v104 = (__int64 *)(a1 + 184);
  for ( i = 0; i != 4; ++i )
  {
    p_src = &src;
    v114 = (1 << i) + 48;
    sub_2462160((__int64 *)&p_src, &v114, (__int64)v115);
    v32 = sub_2241130((unsigned __int64 *)&p_src, 0, 0, "__msan_maybe_warning_", 0x15u);
    dest = &v110;
    if ( (unsigned __int64 *)*v32 == v32 + 2 )
    {
      v110 = _mm_loadu_si128((const __m128i *)v32 + 1);
    }
    else
    {
      dest = (void *)*v32;
      v110.m128i_i64[0] = v32[2];
    }
    v109 = v32[1];
    *v32 = (unsigned __int64)(v32 + 2);
    v32[1] = 0;
    *((_BYTE *)v32 + 16) = 0;
    if ( p_src != &src )
      j_j___libc_free_0((unsigned __int64)p_src);
    v99 = sub_BCB2D0(v126);
    v33 = sub_BCD140(v126, 8 << i);
    v34 = sub_BCB120(v126);
    p_src = 0;
    v35 = (__int64 *)v34;
    v36 = *(__int64 **)(a1 + 72);
    if ( *(_BYTE *)(*(_QWORD *)a3 + 168LL) )
    {
      v38 = 79;
    }
    else
    {
      v37 = 0;
      v38 = 54;
      if ( !*(_BYTE *)(*(_QWORD *)a3 + 170LL) )
        goto LABEL_14;
    }
    v91 = (__int64 *)v34;
    v95 = v38;
    p_src = (__m128i *)sub_A7A090((__int64 *)&p_src, v36, 1, v38);
    v83 = sub_A7A090((__int64 *)&p_src, v36, 2, v95);
    v35 = v91;
    v37 = v83;
LABEL_14:
    v90 = v37;
    src.m128i_i64[1] = v99;
    v92 = v109;
    v96 = dest;
    src.m128i_i64[0] = v33;
    p_src = &src;
    n = 0x200000002LL;
    v39 = sub_BCF480(v35, &src, 2, 0);
    v41 = sub_BA8C10(a2, (__int64)v96, v92, v39, v90);
    v42 = v40;
    if ( p_src != &src )
    {
      v100 = v40;
      _libc_free((unsigned __int64)p_src);
      v42 = v100;
    }
    src.m128i_i8[4] = (1 << i) + 48;
    *v104 = v41;
    v104[1] = v42;
    v111 = v113;
    sub_2462160((__int64 *)&v111, &src.m128i_i8[4], (__int64)src.m128i_i64 + 5);
    v43 = (__m128i *)sub_2241130((unsigned __int64 *)&v111, 0, 0, "__msan_maybe_store_origin_", 0x1Au);
    p_src = &src;
    if ( (__m128i *)v43->m128i_i64[0] == &v43[1] )
    {
      src = _mm_loadu_si128(v43 + 1);
    }
    else
    {
      p_src = (__m128i *)v43->m128i_i64[0];
      src.m128i_i64[0] = v43[1].m128i_i64[0];
    }
    n = v43->m128i_u64[1];
    v43->m128i_i64[0] = (__int64)v43[1].m128i_i64;
    v43->m128i_i64[1] = 0;
    v43[1].m128i_i8[0] = 0;
    v44 = (__m128i *)dest;
    if ( p_src == &src )
    {
      v84 = n;
      if ( n )
      {
        if ( n == 1 )
          *(_BYTE *)dest = src.m128i_i8[0];
        else
          memcpy(dest, &src, n);
        v84 = n;
        v44 = (__m128i *)dest;
      }
      v109 = v84;
      v44->m128i_i8[v84] = 0;
      v44 = p_src;
    }
    else
    {
      if ( dest == &v110 )
      {
        dest = p_src;
        v109 = n;
        v110.m128i_i64[0] = src.m128i_i64[0];
      }
      else
      {
        v45 = v110.m128i_i64[0];
        dest = p_src;
        v109 = n;
        v110.m128i_i64[0] = src.m128i_i64[0];
        if ( v44 )
        {
          p_src = v44;
          src.m128i_i64[0] = v45;
          goto LABEL_22;
        }
      }
      v44 = &src;
      p_src = &src;
    }
LABEL_22:
    n = 0;
    v44->m128i_i8[0] = 0;
    if ( p_src != &src )
      j_j___libc_free_0((unsigned __int64)p_src);
    if ( v111 != v113 )
      j_j___libc_free_0((unsigned __int64)v111);
    v46 = sub_BCB2D0(v126);
    v101 = *(_QWORD *)(a1 + 96);
    v47 = sub_BCD140(v126, 8 << i);
    v48 = sub_BCB120(v126);
    v49 = *(__int64 **)(a1 + 72);
    p_src = 0;
    v50 = (__int64 *)v48;
    if ( *(_BYTE *)(*(_QWORD *)a3 + 168LL) )
    {
      v52 = 79;
    }
    else
    {
      v51 = 0;
      v52 = 54;
      if ( !*(_BYTE *)(*(_QWORD *)a3 + 170LL) )
        goto LABEL_28;
    }
    v94 = (__int64 *)v48;
    v98 = v52;
    p_src = (__m128i *)sub_A7A090((__int64 *)&p_src, v49, 1, v52);
    v82 = sub_A7A090((__int64 *)&p_src, v49, 3, v98);
    v50 = v94;
    v51 = v82;
LABEL_28:
    v93 = v51;
    src.m128i_i64[1] = v101;
    v53 = v109;
    v97 = dest;
    src.m128i_i64[0] = v47;
    v119 = v46;
    p_src = &src;
    n = 0x300000003LL;
    v54 = sub_BCF480(v50, &src, 3, 0);
    v55 = sub_BA8C10(a2, (__int64)v97, v53, v54, v93);
    v57 = v56;
    if ( p_src != &src )
      _libc_free((unsigned __int64)p_src);
    v58 = (__m128i *)dest;
    v104[8] = v55;
    v104[9] = v57;
    if ( v58 != &v110 )
      j_j___libc_free_0((unsigned __int64)v58);
    v104 += 2;
  }
  v59 = *(_QWORD *)(a1 + 96);
  v60 = *(_QWORD *)(a1 + 80);
  v61 = (__int64 *)sub_BCB120(v126);
  src.m128i_i64[0] = v59;
  src.m128i_i64[1] = v60;
  v119 = v59;
  v120 = v59;
  p_src = &src;
  n = 0x400000004LL;
  v62 = sub_BCF480(v61, &src, 4, 0);
  v63 = sub_BA8C10(a2, (__int64)"__msan_set_alloca_origin_with_descr", 0x23u, v62, 0);
  v65 = v64;
  if ( p_src != &src )
    _libc_free((unsigned __int64)p_src);
  *(_QWORD *)(a1 + 312) = v63;
  v66 = v126;
  *(_QWORD *)(a1 + 320) = v65;
  v67 = *(_QWORD *)(a1 + 80);
  v68 = *(_QWORD *)(a1 + 96);
  v69 = (__int64 *)sub_BCB120(v66);
  src.m128i_i64[0] = v68;
  src.m128i_i64[1] = v67;
  v119 = v68;
  p_src = &src;
  n = 0x300000003LL;
  v70 = sub_BCF480(v69, &src, 3, 0);
  v71 = sub_BA8C10(a2, (__int64)"__msan_set_alloca_origin_no_descr", 0x21u, v70, 0);
  v73 = v72;
  if ( p_src != &src )
    _libc_free((unsigned __int64)p_src);
  *(_QWORD *)(a1 + 328) = v71;
  v74 = v126;
  *(_QWORD *)(a1 + 336) = v73;
  v75 = *(_QWORD *)(a1 + 96);
  v76 = *(_QWORD *)(a1 + 80);
  v77 = (__int64 *)sub_BCB120(v74);
  src.m128i_i64[0] = v75;
  src.m128i_i64[1] = v76;
  p_src = &src;
  n = 0x200000002LL;
  v78 = sub_BCF480(v77, &src, 2, 0);
  v79 = sub_BA8C10(a2, (__int64)"__msan_poison_stack", 0x13u, v78, 0);
  v81 = v80;
  if ( p_src != &src )
    _libc_free((unsigned __int64)p_src);
  *(_QWORD *)(a1 + 344) = v79;
  *(_QWORD *)(a1 + 352) = v81;
  sub_F94A20(v121, (__int64)"__msan_poison_stack");
}
