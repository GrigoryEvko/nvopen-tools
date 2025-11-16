// Function: sub_383EE00
// Address: 0x383ee00
//
unsigned __int8 *__fastcall sub_383EE00(__int64 *a1, __int64 a2)
{
  __int64 v4; // rsi
  __int64 v5; // rbx
  __int64 *v6; // rax
  unsigned int v7; // r13d
  __int64 v8; // rdx
  __m128i v9; // xmm0
  __m128i v10; // xmm1
  _QWORD *v11; // rax
  unsigned __int16 *v12; // rax
  int v13; // r14d
  __int64 v14; // r15
  unsigned int v15; // r14d
  __int64 v16; // rdx
  unsigned __int8 *v17; // rax
  __int64 v18; // rsi
  __int32 v19; // edx
  __int32 v20; // edx
  __int64 v21; // rax
  __int64 v22; // rax
  __int128 v23; // rax
  __int64 v24; // r15
  __int64 v25; // r14
  unsigned int v26; // r12d
  __int64 v27; // r9
  unsigned __int8 *v28; // rax
  __int32 v29; // edx
  __int64 v30; // r9
  __int64 v31; // r9
  __int32 v32; // edx
  __int64 v33; // rax
  __int64 v34; // rsi
  __int32 v35; // edx
  __int32 v36; // edx
  __int64 v37; // rax
  __int64 v38; // rax
  int v39; // eax
  __int128 v40; // rax
  __int64 v41; // r9
  _QWORD *v42; // rdi
  unsigned __int8 *v43; // rax
  __int32 v44; // edx
  unsigned __int8 *v45; // rax
  unsigned int v46; // edx
  unsigned __int8 *v47; // r12
  __int64 v49; // rdx
  __int64 v50; // rcx
  __int64 v51; // r8
  __int64 v52; // rdx
  __int64 v53; // rax
  __int64 v54; // rdx
  unsigned __int16 v55; // r13
  __int64 v56; // r15
  __int64 (__fastcall *v57)(__int64, __int64, unsigned int, __int64); // rax
  __int64 (*v58)(); // rax
  unsigned __int8 *v59; // rax
  __int64 v60; // rsi
  __int32 v61; // edx
  __int32 v62; // edx
  unsigned __int64 v63; // rax
  unsigned __int8 *v64; // rax
  __int64 v65; // rdx
  __int64 v66; // r13
  unsigned __int8 *v67; // r12
  __int64 v68; // r9
  unsigned __int8 *v69; // rax
  unsigned int v70; // edx
  unsigned __int64 v71; // rdi
  unsigned __int16 *v72; // rax
  __int64 v73; // r9
  unsigned __int8 *v74; // rax
  unsigned int v75; // r13d
  __int64 v76; // rbx
  unsigned __int64 v77; // rdi
  __int64 v78; // r8
  unsigned __int64 v79; // rax
  __int128 v80; // rax
  unsigned __int8 *v81; // rax
  __int64 v82; // rdx
  __int64 v83; // r15
  unsigned __int8 *v84; // r14
  __int64 v85; // r9
  __int128 v86; // rax
  __int64 v87; // r13
  __int64 v88; // r9
  unsigned int v89; // edx
  __int64 v90; // r9
  unsigned int v91; // ecx
  __int32 v92; // edx
  __int32 v93; // edx
  __int64 v94; // r9
  __int64 v95; // rdx
  __int128 v96; // [rsp-10h] [rbp-1E0h]
  __int128 v97; // [rsp-10h] [rbp-1E0h]
  __int128 v98; // [rsp-10h] [rbp-1E0h]
  __int128 v99; // [rsp+0h] [rbp-1D0h]
  __int128 v100; // [rsp+0h] [rbp-1D0h]
  __int128 v101; // [rsp+0h] [rbp-1D0h]
  __int128 v102; // [rsp+0h] [rbp-1D0h]
  __int64 v103; // [rsp+10h] [rbp-1C0h]
  unsigned int v104; // [rsp+20h] [rbp-1B0h]
  __int128 v105; // [rsp+20h] [rbp-1B0h]
  int v106; // [rsp+30h] [rbp-1A0h]
  unsigned int v107; // [rsp+30h] [rbp-1A0h]
  _QWORD *v108; // [rsp+38h] [rbp-198h]
  unsigned __int8 *v109; // [rsp+80h] [rbp-150h]
  __int64 v110; // [rsp+120h] [rbp-B0h] BYREF
  int v111; // [rsp+128h] [rbp-A8h]
  __m128i v112; // [rsp+130h] [rbp-A0h] BYREF
  __m128i v113; // [rsp+140h] [rbp-90h] BYREF
  unsigned int v114; // [rsp+150h] [rbp-80h] BYREF
  __int64 v115; // [rsp+158h] [rbp-78h]
  unsigned __int64 v116; // [rsp+160h] [rbp-70h] BYREF
  __int64 v117; // [rsp+168h] [rbp-68h]
  unsigned __int64 v118; // [rsp+170h] [rbp-60h] BYREF
  __int64 v119; // [rsp+178h] [rbp-58h]
  __int64 v120; // [rsp+180h] [rbp-50h] BYREF
  __int64 v121; // [rsp+188h] [rbp-48h]
  __int64 v122; // [rsp+190h] [rbp-40h]

  v4 = *(_QWORD *)(a2 + 80);
  v110 = v4;
  if ( v4 )
    sub_B96E90((__int64)&v110, v4, 1);
  v5 = *a1;
  v111 = *(_DWORD *)(a2 + 72);
  v6 = *(__int64 **)(a2 + 40);
  v7 = *(_DWORD *)(a2 + 24);
  v8 = *v6;
  v9 = _mm_loadu_si128((const __m128i *)v6);
  v10 = _mm_loadu_si128((const __m128i *)(v6 + 5));
  v11 = (_QWORD *)a1[1];
  v112 = v9;
  v108 = v11;
  v113 = v10;
  v12 = (unsigned __int16 *)(*(_QWORD *)(v8 + 48) + 16LL * v9.m128i_u32[2]);
  v13 = *v12;
  v14 = *((_QWORD *)v12 + 1);
  LOWORD(v118) = v13;
  v119 = v14;
  if ( (_WORD)v13 )
  {
    if ( (unsigned __int16)(v13 - 17) <= 0xD3u )
    {
      v14 = 0;
      LOWORD(v13) = word_4456580[v13 - 1];
    }
  }
  else if ( sub_30070B0((__int64)&v118) )
  {
    LOWORD(v13) = sub_3009970((__int64)&v118, v4, v49, v50, v51);
    v14 = v52;
  }
  v117 = v14;
  LOWORD(v116) = v13;
  v120 = sub_2D5B750((unsigned __int16 *)&v116);
  v15 = v120;
  v121 = v16;
  if ( v7 == 85 )
  {
    sub_383E4F0(a1, (__int64)&v112, (__int64)&v113, v9);
    v72 = (unsigned __int16 *)(*(_QWORD *)(v112.m128i_i64[0] + 48) + 16LL * v112.m128i_u32[2]);
    v74 = sub_3406EB0(v108, 0x55u, (__int64)&v110, *v72, *((_QWORD *)v72 + 1), v73, *(_OWORD *)&v112, *(_OWORD *)&v113);
LABEL_41:
    v47 = v74;
    goto LABEL_23;
  }
  if ( v7 != 83 )
  {
    if ( v7 - 86 <= 1 )
    {
      v33 = sub_37AE0F0((__int64)a1, v112.m128i_u64[0], v112.m128i_i64[1]);
      v34 = v113.m128i_i64[0];
      v112.m128i_i64[0] = v33;
      v112.m128i_i32[2] = v35;
      v113.m128i_i64[0] = (__int64)sub_37AF270((__int64)a1, v113.m128i_u64[0], v113.m128i_i64[1], v9);
      v113.m128i_i32[2] = v36;
      v37 = *(_QWORD *)(v112.m128i_i64[0] + 48) + 16LL * v112.m128i_u32[2];
      LOWORD(v36) = *(_WORD *)v37;
      v38 = *(_QWORD *)(v37 + 8);
      LOWORD(v114) = v36;
      v115 = v38;
      v39 = sub_32844A0((unsigned __int16 *)&v114, v34);
      *(_QWORD *)&v40 = sub_3400E40(a1[1], v39 - v15, v114, v115, (__int64)&v110, v9);
      v42 = (_QWORD *)a1[1];
      v24 = *((_QWORD *)&v40 + 1);
      v25 = v40;
      if ( v7 == 87 )
        v26 = 192;
      else
        v26 = 191;
      v43 = sub_3406EB0(v42, 0xBEu, (__int64)&v110, v114, v115, v41, *(_OWORD *)&v112, v40);
      v112.m128i_i64[0] = (__int64)v43;
      v112.m128i_i32[2] = v44;
LABEL_22:
      v45 = sub_3406EB0(v108, v7, (__int64)&v110, v114, v115, v31, *(_OWORD *)&v112, *(_OWORD *)&v113);
      *((_QWORD *)&v100 + 1) = v24;
      *(_QWORD *)&v100 = v25;
      *((_QWORD *)&v96 + 1) = v46;
      *(_QWORD *)&v96 = v45;
      v47 = sub_3406EB0(v108, v26, (__int64)&v110, v114, v115, (__int64)v45, v96, v100);
      goto LABEL_23;
    }
    v17 = sub_383B380((__int64)a1, v112.m128i_u64[0], v112.m128i_i64[1]);
    v18 = v113.m128i_i64[0];
    v112.m128i_i64[0] = (__int64)v17;
    v112.m128i_i32[2] = v19;
    v113.m128i_i64[0] = (__int64)sub_383B380((__int64)a1, v113.m128i_u64[0], v113.m128i_i64[1]);
    v113.m128i_i32[2] = v20;
    v21 = *(_QWORD *)(v112.m128i_i64[0] + 48) + 16LL * v112.m128i_u32[2];
    LOWORD(v20) = *(_WORD *)v21;
    v22 = *(_QWORD *)(v21 + 8);
    LOWORD(v114) = v20;
    v115 = v22;
    v104 = sub_32844A0((unsigned __int16 *)&v114, v18);
    if ( (_WORD)v114 == 1 || (_WORD)v114 && *(_QWORD *)(v5 + 8LL * (unsigned __int16)v114 + 112) )
    {
      if ( v7 > 0x1F3 )
      {
        v75 = 57;
LABEL_43:
        LODWORD(v121) = v15;
        v107 = v15 - 1;
        v76 = 1LL << ((unsigned __int8)v15 - 1);
        if ( v15 <= 0x40 )
        {
          v120 = 1LL << ((unsigned __int8)v15 - 1);
          sub_C44830((__int64)&v116, &v120, v104);
          if ( (unsigned int)v121 <= 0x40 )
          {
            LODWORD(v121) = v15;
            v78 = ~v76;
            goto LABEL_47;
          }
LABEL_45:
          v77 = v120;
          if ( !v120 )
            goto LABEL_46;
          goto LABEL_71;
        }
        sub_C43690((__int64)&v120, 0, 0);
        if ( (unsigned int)v121 <= 0x40 )
        {
          v120 |= v76;
          sub_C44830((__int64)&v116, &v120, v104);
          if ( (unsigned int)v121 > 0x40 )
            goto LABEL_45;
        }
        else
        {
          *(_QWORD *)(v120 + 8LL * (v107 >> 6)) |= v76;
          sub_C44830((__int64)&v116, &v120, v104);
          if ( (unsigned int)v121 > 0x40 )
          {
            v77 = v120;
            if ( v120 )
            {
LABEL_71:
              j_j___libc_free_0_0(v77);
LABEL_46:
              LODWORD(v121) = v15;
              v78 = ~v76;
              if ( v15 <= 0x40 )
              {
LABEL_47:
                v79 = 0xFFFFFFFFFFFFFFFFLL >> (63 - ((v15 - 1) & 0x3F));
                if ( !v15 )
                  v79 = 0;
                v120 = v79;
                goto LABEL_50;
              }
LABEL_63:
              v103 = v78;
              sub_C43690((__int64)&v120, -1, 1);
              v78 = v103;
              if ( (unsigned int)v121 > 0x40 )
              {
                *(_QWORD *)(v120 + 8LL * (v107 >> 6)) &= v103;
                goto LABEL_51;
              }
LABEL_50:
              v120 &= v78;
LABEL_51:
              sub_C44830((__int64)&v118, &v120, v104);
              if ( (unsigned int)v121 > 0x40 && v120 )
                j_j___libc_free_0_0(v120);
              *(_QWORD *)&v80 = sub_34007B0(a1[1], (__int64)&v116, (__int64)&v110, v114, v115, 0, v9, 0);
              v105 = v80;
              v81 = sub_34007B0(a1[1], (__int64)&v118, (__int64)&v110, v114, v115, 0, v9, 0);
              v83 = v82;
              v84 = v81;
              *(_QWORD *)&v86 = sub_3406EB0(
                                  v108,
                                  v75,
                                  (__int64)&v110,
                                  v114,
                                  v115,
                                  v85,
                                  *(_OWORD *)&v112,
                                  *(_OWORD *)&v113);
              *((_QWORD *)&v102 + 1) = v83;
              v87 = *((_QWORD *)&v86 + 1);
              *(_QWORD *)&v102 = v84;
              v109 = sub_3406EB0(v108, 0xB4u, (__int64)&v110, v114, v115, v88, v86, v102);
              *((_QWORD *)&v98 + 1) = v89 | v87 & 0xFFFFFFFF00000000LL;
              *(_QWORD *)&v98 = v109;
              v47 = sub_3406EB0(v108, 0xB5u, (__int64)&v110, v114, v115, v90, v98, v105);
              if ( (unsigned int)v119 > 0x40 && v118 )
                j_j___libc_free_0_0(v118);
              if ( (unsigned int)v117 <= 0x40 )
                goto LABEL_23;
              v71 = v116;
              if ( !v116 )
                goto LABEL_23;
LABEL_39:
              j_j___libc_free_0_0(v71);
              goto LABEL_23;
            }
          }
        }
        LODWORD(v121) = v15;
        v78 = ~v76;
        goto LABEL_63;
      }
      if ( !*(_BYTE *)(v7 + v5 + 500LL * (unsigned __int16)v114 + 6414) )
      {
        if ( (int)v7 > 87 || v7 != 84 && (v7 & 0xFFFFFFFB) != 0x52 )
          BUG();
        *(_QWORD *)&v23 = sub_3400E40(a1[1], v104 - v15, v114, v115, (__int64)&v110, v9);
        v24 = *((_QWORD *)&v23 + 1);
        v25 = v23;
        v26 = 191;
        v28 = sub_3406EB0((_QWORD *)a1[1], 0xBEu, (__int64)&v110, v114, v115, v27, *(_OWORD *)&v112, v23);
        *((_QWORD *)&v99 + 1) = v24;
        *(_QWORD *)&v99 = v25;
        v112.m128i_i64[0] = (__int64)v28;
        v112.m128i_i32[2] = v29;
        v113.m128i_i64[0] = (__int64)sub_3406EB0(v108, 0xBEu, (__int64)&v110, v114, v115, v30, *(_OWORD *)&v113, v99);
        v113.m128i_i32[2] = v32;
        goto LABEL_22;
      }
    }
    v75 = (v7 != 82) + 56;
    goto LABEL_43;
  }
  v53 = *(_QWORD *)(v112.m128i_i64[0] + 48) + 16LL * v112.m128i_u32[2];
  v54 = a1[1];
  v55 = *(_WORD *)v53;
  v56 = *(_QWORD *)(v53 + 8);
  v57 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  if ( v57 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v120, *a1, *(_QWORD *)(v54 + 64), v55, v56);
    LOWORD(v118) = v121;
    v119 = v122;
  }
  else
  {
    v106 = v55;
    LODWORD(v118) = v57(*a1, *(_QWORD *)(v54 + 64), v55, v56);
    v119 = v95;
  }
  v58 = *(__int64 (**)())(*(_QWORD *)*a1 + 1456LL);
  if ( v58 != sub_2D56680 )
  {
    HIWORD(v91) = HIWORD(v106);
    LOWORD(v91) = v55;
    if ( ((unsigned __int8 (__fastcall *)(__int64, _QWORD, __int64, _QWORD, __int64))v58)(
           *a1,
           v91,
           v56,
           (unsigned int)v118,
           v119) )
    {
      v112.m128i_i64[0] = (__int64)sub_383B380((__int64)a1, v112.m128i_u64[0], v112.m128i_i64[1]);
      v112.m128i_i32[2] = v92;
      v113.m128i_i64[0] = (__int64)sub_383B380((__int64)a1, v113.m128i_u64[0], v113.m128i_i64[1]);
      v113.m128i_i32[2] = v93;
      v74 = sub_3406EB0(v108, 0x53u, (__int64)&v110, (unsigned int)v118, v119, v94, *(_OWORD *)&v112, *(_OWORD *)&v113);
      goto LABEL_41;
    }
  }
  v59 = sub_37AF270((__int64)a1, v112.m128i_u64[0], v112.m128i_i64[1], v9);
  v60 = v113.m128i_i64[0];
  v112.m128i_i64[0] = (__int64)v59;
  v112.m128i_i32[2] = v61;
  v113.m128i_i64[0] = (__int64)sub_37AF270((__int64)a1, v113.m128i_u64[0], v113.m128i_i64[1], v9);
  v113.m128i_i32[2] = v62;
  LODWORD(v121) = sub_32844A0((unsigned __int16 *)&v118, v60);
  if ( (unsigned int)v121 > 0x40 )
    sub_C43690((__int64)&v120, 0, 0);
  else
    v120 = 0;
  if ( v15 )
  {
    if ( v15 > 0x40 )
    {
      sub_C43C90(&v120, 0, v15);
    }
    else
    {
      v63 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v15);
      if ( (unsigned int)v121 > 0x40 )
        *(_QWORD *)v120 |= v63;
      else
        v120 |= v63;
    }
  }
  v64 = sub_34007B0(a1[1], (__int64)&v120, (__int64)&v110, v118, v119, 0, v9, 0);
  v66 = v65;
  v67 = v64;
  v69 = sub_3406EB0(v108, 0x38u, (__int64)&v110, (unsigned int)v118, v119, v68, *(_OWORD *)&v112, *(_OWORD *)&v113);
  *((_QWORD *)&v101 + 1) = v66;
  *(_QWORD *)&v101 = v67;
  *((_QWORD *)&v97 + 1) = v70;
  *(_QWORD *)&v97 = v69;
  v47 = sub_3406EB0(v108, 0xB6u, (__int64)&v110, (unsigned int)v118, v119, (__int64)v69, v97, v101);
  if ( (unsigned int)v121 > 0x40 )
  {
    v71 = v120;
    if ( v120 )
      goto LABEL_39;
  }
LABEL_23:
  if ( v110 )
    sub_B91220((__int64)&v110, v110);
  return v47;
}
