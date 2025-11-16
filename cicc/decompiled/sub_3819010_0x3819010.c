// Function: sub_3819010
// Address: 0x3819010
//
__int64 __fastcall sub_3819010(__int64 *a1, __int64 a2, __int64 a3, __m128i *a4, __m128i a5)
{
  __int64 v7; // r10
  __int64 v8; // rax
  __int64 v9; // r15
  __int64 v10; // r12
  __int64 (__fastcall *v11)(__int64, __int64, unsigned int, __int64); // r11
  __int16 *v12; // rax
  unsigned __int16 v13; // di
  __int64 v14; // r8
  __int64 v15; // rax
  __int64 v16; // rsi
  __int64 v17; // r12
  __int64 v18; // rax
  unsigned int v19; // r12d
  unsigned int v20; // r15d
  unsigned int v21; // eax
  __int64 v22; // rsi
  unsigned int v23; // esi
  unsigned int v24; // eax
  unsigned int v25; // r14d
  unsigned __int64 v26; // r8
  __int64 v27; // r8
  unsigned __int64 v28; // r8
  bool v29; // r14
  unsigned int v30; // r14d
  unsigned int v31; // r13d
  _QWORD *v33; // r13
  __int128 v34; // rax
  __int64 v35; // r9
  __int64 v36; // rdx
  unsigned __int8 *v37; // r10
  unsigned int v38; // r12d
  unsigned int v39; // r15d
  _QWORD *v40; // r13
  __int128 v41; // rax
  __int64 v42; // r9
  __int128 v43; // rax
  __int64 v44; // r9
  unsigned __int8 *v45; // rax
  __int64 v46; // rdx
  __int64 v47; // r13
  unsigned __int8 *v48; // r12
  __int64 v49; // r9
  unsigned __int8 *v50; // rax
  __int64 v51; // r8
  int v52; // edx
  _QWORD *v53; // rbx
  __int64 v54; // r9
  __int128 v55; // rax
  __int64 v56; // r9
  __int32 v57; // edx
  __m128i v58; // xmm0
  unsigned int v59; // eax
  unsigned int v60; // edx
  __int64 v61; // rdx
  unsigned int v62; // eax
  _QWORD *v63; // r13
  unsigned __int64 v64; // rdx
  unsigned __int64 v65; // rdx
  __int128 v66; // rax
  __int64 v67; // r9
  unsigned int v68; // edx
  unsigned __int8 *v69; // rax
  __int64 v70; // r8
  __int32 v71; // edx
  __int64 v72; // r9
  int v73; // edx
  unsigned __int8 *v74; // rax
  __int64 v75; // r8
  int v76; // edx
  __int64 v77; // r9
  __int32 v78; // edx
  __int64 v79; // r13
  __int128 v80; // rax
  _QWORD *v81; // rdi
  __int64 v82; // r9
  unsigned __int8 *v83; // rax
  __int64 v84; // r8
  __int32 v85; // edx
  __int64 v86; // r9
  unsigned __int8 *v87; // rax
  int v88; // edx
  __int128 v89; // [rsp-10h] [rbp-1E0h]
  _QWORD *v90; // [rsp+0h] [rbp-1D0h]
  unsigned int v93; // [rsp+1Ch] [rbp-1B4h]
  __int64 v94; // [rsp+20h] [rbp-1B0h]
  unsigned __int64 v95; // [rsp+28h] [rbp-1A8h]
  unsigned int v96; // [rsp+30h] [rbp-1A0h]
  __int128 v97; // [rsp+30h] [rbp-1A0h]
  __m128i v98; // [rsp+40h] [rbp-190h]
  unsigned int v100; // [rsp+100h] [rbp-D0h] BYREF
  __int64 v101; // [rsp+108h] [rbp-C8h]
  unsigned int v102; // [rsp+110h] [rbp-C0h] BYREF
  __int64 v103; // [rsp+118h] [rbp-B8h]
  __int64 v104; // [rsp+120h] [rbp-B0h] BYREF
  int v105; // [rsp+128h] [rbp-A8h]
  unsigned __int64 v106; // [rsp+130h] [rbp-A0h] BYREF
  unsigned int v107; // [rsp+138h] [rbp-98h]
  __m128i v108; // [rsp+140h] [rbp-90h] BYREF
  __int128 v109; // [rsp+150h] [rbp-80h] BYREF
  _QWORD *v110; // [rsp+160h] [rbp-70h] BYREF
  unsigned int v111; // [rsp+168h] [rbp-68h]
  _QWORD *v112; // [rsp+170h] [rbp-60h] BYREF
  unsigned int v113; // [rsp+178h] [rbp-58h]
  unsigned __int64 v114; // [rsp+180h] [rbp-50h] BYREF
  unsigned int v115; // [rsp+188h] [rbp-48h]
  unsigned __int64 v116; // [rsp+190h] [rbp-40h] BYREF
  unsigned int v117; // [rsp+198h] [rbp-38h]

  v7 = *a1;
  v93 = *(_DWORD *)(a2 + 24);
  v8 = *(_QWORD *)(a2 + 40);
  v9 = *(_QWORD *)(v8 + 40);
  v10 = *(unsigned int *)(v8 + 48);
  v95 = *(_QWORD *)v8;
  v98 = _mm_loadu_si128((const __m128i *)(v8 + 40));
  v94 = *(_QWORD *)(v8 + 8);
  v11 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  v12 = *(__int16 **)(a2 + 48);
  v13 = *v12;
  v14 = *((_QWORD *)v12 + 1);
  v15 = a1[1];
  if ( v11 == sub_2D56A50 )
  {
    v16 = v7;
    sub_2FE6CC0((__int64)&v114, v7, *(_QWORD *)(v15 + 64), v13, v14);
    LOWORD(v100) = v115;
    v101 = v116;
  }
  else
  {
    v16 = *(_QWORD *)(v15 + 64);
    v100 = v11(v7, v16, v13, v14);
    v101 = v61;
  }
  v17 = *(_QWORD *)(v9 + 48) + 16 * v10;
  v18 = *(_QWORD *)(v17 + 8);
  LOWORD(v102) = *(_WORD *)v17;
  v103 = v18;
  v19 = sub_32844A0((unsigned __int16 *)&v102, v16);
  v20 = v19;
  v21 = sub_32844A0((unsigned __int16 *)&v100, v16);
  v22 = *(_QWORD *)(a2 + 80);
  v96 = v21;
  v104 = v22;
  if ( v22 )
    sub_B96E90((__int64)&v104, v22, 1);
  v23 = -1;
  v105 = *(_DWORD *)(a2 + 72);
  if ( v96 )
  {
    _BitScanReverse(&v24, v96);
    v23 = 31 - (v24 ^ 0x1F);
  }
  v107 = v19;
  v25 = v23 - v19;
  if ( v19 > 0x40 )
  {
    sub_C43690((__int64)&v106, 0, 0);
    v20 = v107;
    v23 = v25 + v107;
  }
  else
  {
    v106 = 0;
  }
  if ( v23 != v20 )
  {
    if ( v23 > 0x3F || v20 > 0x40 )
      sub_C43C90(&v106, v23, v20);
    else
      v106 |= 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v25 + 64) << v23;
  }
  sub_33DD090((__int64)&v114, a1[1], v98.m128i_i64[0], v98.m128i_i64[1], 0);
  DWORD2(v109) = v115;
  if ( v115 <= 0x40 )
  {
    v26 = v114;
LABEL_15:
    v27 = v116 | v26;
    DWORD2(v109) = 0;
    *(_QWORD *)&v109 = v27;
LABEL_16:
    v28 = v106 & v27;
LABEL_17:
    v29 = v28 == 0;
    goto LABEL_18;
  }
  sub_C43780((__int64)&v109, (const void **)&v114);
  if ( DWORD2(v109) <= 0x40 )
  {
    v26 = v109;
    goto LABEL_15;
  }
  sub_C43BD0(&v109, (__int64 *)&v116);
  v59 = DWORD2(v109);
  v27 = v109;
  DWORD2(v109) = 0;
  v111 = v59;
  v110 = (_QWORD *)v109;
  if ( v59 <= 0x40 )
    goto LABEL_16;
  sub_C43B90(&v110, (__int64 *)&v106);
  v60 = v111;
  v28 = (unsigned __int64)v110;
  v111 = 0;
  v113 = v60;
  v112 = v110;
  if ( v60 <= 0x40 )
    goto LABEL_17;
  v90 = v110;
  if ( v60 - (unsigned int)sub_C444A0((__int64)&v112) > 0x40 || *v90 )
  {
    if ( !v90 )
    {
      if ( DWORD2(v109) > 0x40 && (_QWORD)v109 )
        j_j___libc_free_0_0(v109);
      goto LABEL_35;
    }
    v29 = 0;
  }
  else
  {
    v29 = 1;
  }
  j_j___libc_free_0_0((unsigned __int64)v90);
  if ( v111 > 0x40 && v110 )
  {
    j_j___libc_free_0_0((unsigned __int64)v110);
    if ( DWORD2(v109) <= 0x40 )
      goto LABEL_19;
    goto LABEL_33;
  }
LABEL_18:
  if ( DWORD2(v109) <= 0x40 )
    goto LABEL_19;
LABEL_33:
  if ( (_QWORD)v109 )
  {
    j_j___libc_free_0_0(v109);
    if ( v29 )
      goto LABEL_20;
    goto LABEL_35;
  }
LABEL_19:
  if ( v29 )
  {
LABEL_20:
    v30 = v117;
    v31 = 0;
    goto LABEL_21;
  }
LABEL_35:
  *(_QWORD *)&v109 = 0;
  v108.m128i_i64[0] = 0;
  v108.m128i_i32[2] = 0;
  DWORD2(v109) = 0;
  sub_375E510((__int64)a1, v95, v94, (__int64)&v108, (__int64)&v109);
  v30 = v117;
  if ( v117 > 0x40 )
  {
    if ( !(unsigned __int8)sub_C446A0((__int64 *)&v116, (__int64 *)&v106) )
      goto LABEL_37;
LABEL_63:
    v62 = v107;
    v63 = (_QWORD *)a1[1];
    v111 = v107;
    if ( v107 > 0x40 )
    {
      sub_C43780((__int64)&v110, (const void **)&v106);
      v62 = v111;
      if ( v111 > 0x40 )
      {
        sub_C43D10((__int64)&v110);
        v62 = v111;
        v65 = (unsigned __int64)v110;
LABEL_68:
        v112 = (_QWORD *)v65;
        v113 = v62;
        v111 = 0;
        *(_QWORD *)&v66 = sub_34007B0((__int64)v63, (__int64)&v112, (__int64)&v104, v102, v103, 0, a5, 0);
        v98.m128i_i64[0] = (__int64)sub_3406EB0(v63, 0xBAu, (__int64)&v104, v102, v103, v67, *(_OWORD *)&v98, v66);
        v98.m128i_i64[1] = v68 | v98.m128i_i64[1] & 0xFFFFFFFF00000000LL;
        if ( v113 > 0x40 && v112 )
          j_j___libc_free_0_0((unsigned __int64)v112);
        if ( v111 > 0x40 && v110 )
          j_j___libc_free_0_0((unsigned __int64)v110);
        switch ( v93 )
        {
          case 0xBFu:
            v79 = a1[1];
            *(_QWORD *)&v80 = sub_3400BD0(v79, v96 - 1, (__int64)&v104, v102, v103, 0, a5, 0);
            v81 = (_QWORD *)v79;
            v31 = 1;
            v83 = sub_3406EB0(v81, 0xBFu, (__int64)&v104, v100, v101, v82, v109, v80);
            v84 = v101;
            a4->m128i_i64[0] = (__int64)v83;
            a4->m128i_i32[2] = v85;
            v87 = sub_3406EB0((_QWORD *)a1[1], 0xBFu, (__int64)&v104, v100, v84, v86, v109, *(_OWORD *)&v98);
            v30 = v117;
            *(_QWORD *)a3 = v87;
            *(_DWORD *)(a3 + 8) = v88;
            goto LABEL_21;
          case 0xC0u:
            v31 = 1;
            v69 = sub_3400BD0(a1[1], 0, (__int64)&v104, v100, v101, 0, a5, 0);
            v70 = v101;
            a4->m128i_i64[0] = (__int64)v69;
            a4->m128i_i32[2] = v71;
            *(_QWORD *)a3 = sub_3406EB0((_QWORD *)a1[1], 0xC0u, (__int64)&v104, v100, v70, v72, v109, *(_OWORD *)&v98);
            *(_DWORD *)(a3 + 8) = v73;
            v30 = v117;
            goto LABEL_21;
          case 0xBEu:
            v31 = 1;
            v74 = sub_3400BD0(a1[1], 0, (__int64)&v104, v100, v101, 0, a5, 0);
            v75 = v101;
            *(_QWORD *)a3 = v74;
            *(_DWORD *)(a3 + 8) = v76;
            a4->m128i_i64[0] = (__int64)sub_3406EB0(
                                          (_QWORD *)a1[1],
                                          0xBEu,
                                          (__int64)&v104,
                                          v100,
                                          v75,
                                          v77,
                                          *(_OWORD *)&v108,
                                          *(_OWORD *)&v98);
            a4->m128i_i32[2] = v78;
            v30 = v117;
            goto LABEL_21;
        }
        goto LABEL_88;
      }
      v64 = (unsigned __int64)v110;
    }
    else
    {
      v64 = v106;
    }
    v65 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v62) & ~v64;
    if ( !v62 )
      v65 = 0;
    v110 = (_QWORD *)v65;
    goto LABEL_68;
  }
  if ( (v106 & v116) != 0 )
    goto LABEL_63;
LABEL_37:
  if ( v107 <= 0x40 )
  {
    if ( (v106 & ~v114) != 0 )
    {
      v31 = 0;
      goto LABEL_21;
    }
  }
  else
  {
    v31 = sub_C446F0((__int64 *)&v106, (__int64 *)&v114);
    if ( !(_BYTE)v31 )
      goto LABEL_21;
  }
  v33 = (_QWORD *)a1[1];
  *(_QWORD *)&v34 = sub_3400BD0((__int64)v33, v96 - 1, (__int64)&v104, v102, v103, 0, a5, 0);
  v37 = sub_3406EB0(v33, 0xBCu, (__int64)&v104, v102, v103, v35, *(_OWORD *)&v98, v34);
  if ( v93 == 190 )
  {
    v38 = 192;
    v39 = 190;
    goto LABEL_42;
  }
  if ( v93 - 191 > 1 )
LABEL_88:
    BUG();
  a5 = _mm_loadu_si128(&v108);
  v38 = 190;
  v39 = 192;
  v108.m128i_i64[0] = v109;
  v108.m128i_i32[2] = DWORD2(v109);
  *(_QWORD *)&v109 = a5.m128i_i64[0];
  DWORD2(v109) = a5.m128i_i32[2];
LABEL_42:
  v40 = (_QWORD *)a1[1];
  *(_QWORD *)&v97 = v37;
  *((_QWORD *)&v97 + 1) = v36;
  *(_QWORD *)&v41 = sub_3400BD0((__int64)v40, 1, (__int64)&v104, v102, v103, 0, a5, 0);
  *(_QWORD *)&v43 = sub_3406EB0(v40, v38, (__int64)&v104, v100, v101, v42, *(_OWORD *)&v108, v41);
  v45 = sub_3406EB0((_QWORD *)a1[1], v38, (__int64)&v104, v100, v101, v44, v43, v97);
  v47 = v46;
  v48 = v45;
  v50 = sub_3406EB0((_QWORD *)a1[1], v93, (__int64)&v104, v100, v101, v49, *(_OWORD *)&v108, *(_OWORD *)&v98);
  v51 = v101;
  *(_QWORD *)a3 = v50;
  *(_DWORD *)(a3 + 8) = v52;
  v53 = (_QWORD *)a1[1];
  *(_QWORD *)&v55 = sub_3406EB0(v53, v39, (__int64)&v104, v100, v51, v54, v109, *(_OWORD *)&v98);
  *((_QWORD *)&v89 + 1) = v47;
  *(_QWORD *)&v89 = v48;
  a4->m128i_i64[0] = (__int64)sub_3406EB0(v53, 0xBBu, (__int64)&v104, v100, v101, v56, v55, v89);
  a4->m128i_i32[2] = v57;
  if ( v93 != 190 )
  {
    v58 = _mm_loadu_si128(a4);
    a4->m128i_i64[0] = *(_QWORD *)a3;
    a4->m128i_i32[2] = *(_DWORD *)(a3 + 8);
    *(_QWORD *)a3 = v58.m128i_i64[0];
    *(_DWORD *)(a3 + 8) = v58.m128i_i32[2];
  }
  v30 = v117;
  v31 = 1;
LABEL_21:
  if ( v30 > 0x40 && v116 )
    j_j___libc_free_0_0(v116);
  if ( v115 > 0x40 && v114 )
    j_j___libc_free_0_0(v114);
  if ( v107 > 0x40 && v106 )
    j_j___libc_free_0_0(v106);
  if ( v104 )
    sub_B91220((__int64)&v104, v104);
  return v31;
}
