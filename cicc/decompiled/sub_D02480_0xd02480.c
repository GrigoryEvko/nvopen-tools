// Function: sub_D02480
// Address: 0xd02480
//
__m128i *__fastcall sub_D02480(__m128i *a1, const __m128i *a2, int a3)
{
  _BYTE *v5; // r14
  char v6; // al
  unsigned __int32 v7; // edx
  __int32 v8; // eax
  __int32 v9; // eax
  __int32 v10; // eax
  __int64 v11; // rax
  _BYTE *v12; // rax
  __m128i v13; // rax
  unsigned __int32 v14; // eax
  __int64 v15; // rdx
  __m128i v16; // xmm0
  unsigned __int32 v17; // eax
  bool v18; // cc
  _QWORD *v19; // rdi
  __int64 v22; // rsi
  unsigned __int32 v23; // edx
  __int32 v24; // eax
  __int32 v25; // eax
  __int32 v26; // eax
  _QWORD *v27; // rax
  char v28; // cl
  char v29; // bl
  bool v30; // al
  unsigned int v31; // ebx
  bool v32; // al
  __int64 v33; // r15
  __int64 v34; // rdx
  __m128i v35; // rax
  __int8 v36; // al
  unsigned __int32 v37; // eax
  unsigned __int32 v38; // edx
  int v39; // ecx
  __m128i v40; // xmm1
  unsigned __int32 v41; // eax
  __int64 v42; // rbx
  unsigned int v43; // r15d
  __int64 v44; // rdx
  __m128i v45; // rax
  __int8 v46; // al
  unsigned __int32 v47; // eax
  unsigned __int32 v48; // edx
  __int8 v49; // cl
  __int64 v50; // rsi
  __int8 v51; // cl
  __int64 v52; // rsi
  char v53; // cl
  __int64 v54; // rdx
  __int64 v55; // rcx
  __int32 v56; // eax
  __m128i v57; // xmm3
  unsigned int v58; // eax
  unsigned int v59; // edx
  __int64 v60; // rcx
  __int8 v61; // bl
  __int8 v62; // r14
  unsigned int v63; // edx
  __m128i v64; // xmm2
  __int64 v65; // rsi
  __int64 v66; // rdx
  __int64 v67; // rdx
  __int64 v68; // rcx
  __int32 v69; // eax
  __m128i v70; // xmm4
  unsigned int v71; // eax
  __int64 v72; // r10
  __m128i v73; // rax
  int v74; // eax
  __int32 v75; // ecx
  __int32 v76; // edx
  __int64 v77; // r9
  char v78; // al
  __m128i v79; // xmm6
  unsigned int v80; // eax
  unsigned int v81; // r12d
  unsigned int v82; // r15d
  __int64 v83; // rsi
  __int64 v84; // rdx
  unsigned __int64 v85; // rax
  __int64 v86; // rsi
  __int64 v87; // rax
  unsigned __int64 v88; // rcx
  __int64 v89; // rdx
  __int64 v90; // rcx
  __int32 v91; // eax
  __m128i v92; // xmm5
  unsigned int v93; // eax
  __int32 v94; // ecx
  __int8 v95; // si
  unsigned __int32 v96; // ecx
  __int32 v97; // edx
  bool v98; // al
  int v99; // eax
  __m128i v100; // rax
  __int64 v101; // [rsp+8h] [rbp-138h]
  unsigned int v102; // [rsp+10h] [rbp-130h]
  _QWORD *v103; // [rsp+10h] [rbp-130h]
  bool v104; // [rsp+1Fh] [rbp-121h]
  char v105; // [rsp+1Fh] [rbp-121h]
  char v106; // [rsp+1Fh] [rbp-121h]
  _QWORD *v107; // [rsp+20h] [rbp-120h] BYREF
  unsigned int v108; // [rsp+28h] [rbp-118h]
  _QWORD v109[2]; // [rsp+30h] [rbp-110h] BYREF
  __int32 v110; // [rsp+40h] [rbp-100h]
  char v111; // [rsp+44h] [rbp-FCh]
  __m128i v112; // [rsp+50h] [rbp-F0h] BYREF
  __int64 v113; // [rsp+60h] [rbp-E0h]
  __int64 v114; // [rsp+68h] [rbp-D8h] BYREF
  unsigned int v115; // [rsp+70h] [rbp-D0h]
  __int64 v116; // [rsp+78h] [rbp-C8h] BYREF
  unsigned int v117; // [rsp+80h] [rbp-C0h]
  __int16 v118; // [rsp+88h] [rbp-B8h]
  __m128i v119; // [rsp+90h] [rbp-B0h] BYREF
  __int32 v120; // [rsp+A0h] [rbp-A0h]
  char v121; // [rsp+A4h] [rbp-9Ch]
  __int64 v122; // [rsp+A8h] [rbp-98h]
  unsigned int v123; // [rsp+B0h] [rbp-90h]
  __int64 v124; // [rsp+B8h] [rbp-88h]
  unsigned int v125; // [rsp+C0h] [rbp-80h]
  __m128i v126; // [rsp+D0h] [rbp-70h] BYREF
  unsigned __int32 v127; // [rsp+E0h] [rbp-60h]
  char v128; // [rsp+E4h] [rbp-5Ch]
  __int64 v129; // [rsp+E8h] [rbp-58h]
  unsigned int v130; // [rsp+F0h] [rbp-50h]
  __int64 v131; // [rsp+F8h] [rbp-48h]
  unsigned int v132; // [rsp+100h] [rbp-40h]
  __int16 v133; // [rsp+108h] [rbp-38h]

  if ( a3 == 6 )
    goto LABEL_42;
  v5 = (_BYTE *)a2->m128i_i64[0];
  v6 = *(_BYTE *)a2->m128i_i64[0];
  if ( v6 == 17 )
  {
    v7 = *((_DWORD *)v5 + 8);
    v112.m128i_i32[2] = v7;
    if ( v7 > 0x40 )
    {
      sub_C43780((__int64)&v112, (const void **)v5 + 3);
      v7 = v112.m128i_u32[2];
    }
    else
    {
      v112.m128i_i64[0] = *((_QWORD *)v5 + 3);
    }
    v8 = a2[1].m128i_i32[0];
    if ( v8 )
    {
      sub_C44740((__int64)&v126, (char **)&v112, v7 - v8);
      if ( v112.m128i_i32[2] > 0x40u && v112.m128i_i64[0] )
        j_j___libc_free_0_0(v112.m128i_i64[0]);
      v7 = v126.m128i_u32[2];
      v112.m128i_i64[0] = v126.m128i_i64[0];
      v112.m128i_i32[2] = v126.m128i_i32[2];
    }
    v9 = a2->m128i_i32[3];
    if ( v9 )
    {
      sub_C44830((__int64)&v126, &v112, v9 + v7);
      if ( v112.m128i_i32[2] > 0x40u && v112.m128i_i64[0] )
        j_j___libc_free_0_0(v112.m128i_i64[0]);
      v7 = v126.m128i_u32[2];
      v112.m128i_i64[0] = v126.m128i_i64[0];
      v112.m128i_i32[2] = v126.m128i_i32[2];
    }
    v10 = a2->m128i_i32[2];
    if ( v10 )
    {
      sub_C449B0((__int64)&v126, (const void **)&v112, v10 + v7);
      if ( v112.m128i_i32[2] > 0x40u && v112.m128i_i64[0] )
        j_j___libc_free_0_0(v112.m128i_i64[0]);
      v11 = v126.m128i_i64[0];
      v7 = v126.m128i_u32[2];
      v112.m128i_i64[0] = v126.m128i_i64[0];
    }
    else
    {
      v11 = v112.m128i_i64[0];
    }
    v119.m128i_i64[0] = v11;
    v12 = (_BYTE *)a2->m128i_i64[0];
    v119.m128i_i32[2] = v7;
    v112.m128i_i32[2] = 0;
    v13.m128i_i64[0] = sub_BCAE30(*((_QWORD *)v12 + 1));
    v126 = v13;
    v14 = sub_CA1930(&v126) + a2->m128i_i32[2] + a2->m128i_i32[3] - a2[1].m128i_i32[0];
    v126.m128i_i32[2] = v14;
    if ( v14 > 0x40 )
    {
      sub_C43690((__int64)&v126, 0, 0);
      v40 = _mm_loadu_si128(a2);
      v41 = v126.m128i_u32[2];
      a1[1].m128i_i64[0] = a2[1].m128i_i64[0];
      a1[2].m128i_i32[0] = v41;
      *a1 = v40;
      if ( v41 > 0x40 )
      {
        sub_C43780((__int64)&a1[1].m128i_i64[1], (const void **)&v126);
LABEL_12:
        v17 = v119.m128i_u32[2];
        a1[3].m128i_i32[0] = v119.m128i_i32[2];
        if ( v17 > 0x40 )
          sub_C43780((__int64)&a1[2].m128i_i64[1], (const void **)&v119);
        else
          a1[2].m128i_i64[1] = v119.m128i_i64[0];
        v18 = v126.m128i_i32[2] <= 0x40u;
        a1[3].m128i_i16[4] = 257;
        if ( !v18 && v126.m128i_i64[0] )
          j_j___libc_free_0_0(v126.m128i_i64[0]);
        if ( v119.m128i_i32[2] > 0x40u && v119.m128i_i64[0] )
          j_j___libc_free_0_0(v119.m128i_i64[0]);
        if ( v112.m128i_i32[2] > 0x40u )
        {
          v19 = (_QWORD *)v112.m128i_i64[0];
          if ( v112.m128i_i64[0] )
            goto LABEL_22;
        }
        return a1;
      }
    }
    else
    {
      v15 = a2[1].m128i_i64[0];
      v16 = _mm_loadu_si128(a2);
      v126.m128i_i64[0] = 0;
      a1[2].m128i_i32[0] = v14;
      a1[1].m128i_i64[0] = v15;
      *a1 = v16;
    }
    a1[1].m128i_i64[1] = v126.m128i_i64[0];
    goto LABEL_12;
  }
  if ( (unsigned __int8)(v6 - 42) > 0x11u )
  {
    if ( v6 == 68 )
    {
      v31 = a3 + 1;
      v32 = sub_B44910(a2->m128i_i64[0]);
      v33 = *((_QWORD *)v5 - 4);
      v104 = v32;
      v119.m128i_i64[0] = sub_BCAE30(*(_QWORD *)(v33 + 8));
      v119.m128i_i64[1] = v34;
      v35.m128i_i64[0] = sub_BCAE30(*((_QWORD *)v5 + 1));
      v112 = v35;
      v35.m128i_i64[1] = v35.m128i_i64[0];
      v36 = v112.m128i_i8[8];
      v126.m128i_i64[0] = v35.m128i_i64[1] - v119.m128i_i64[0];
      if ( v119.m128i_i64[0] )
        v36 = v119.m128i_i8[8];
      v126.m128i_i8[8] = v36;
      v37 = sub_CA1930(&v126);
      v38 = a2[1].m128i_u32[0];
      if ( v37 <= v38 )
      {
        v51 = a2[1].m128i_i8[4];
        v52 = a2->m128i_i64[1];
        v126.m128i_i64[0] = v33;
        v127 = v38 - v37;
        v126.m128i_i64[1] = v52;
        v128 = v51;
      }
      else
      {
        v39 = a2->m128i_i32[2] + a2->m128i_i32[3];
        v126.m128i_i64[0] = v33;
        v127 = 0;
        v126.m128i_i64[1] = v39 - v38 + v37;
        v128 = v104;
      }
      sub_D02480(a1, &v126, v31);
      return a1;
    }
    if ( v6 == 69 )
    {
      v42 = *((_QWORD *)v5 - 4);
      v43 = a3 + 1;
      v119.m128i_i64[0] = sub_BCAE30(*(_QWORD *)(v42 + 8));
      v119.m128i_i64[1] = v44;
      v45.m128i_i64[0] = sub_BCAE30(*((_QWORD *)v5 + 1));
      v112 = v45;
      v45.m128i_i64[1] = v45.m128i_i64[0];
      v46 = v112.m128i_i8[8];
      v126.m128i_i64[0] = v45.m128i_i64[1] - v119.m128i_i64[0];
      if ( v119.m128i_i64[0] )
        v46 = v119.m128i_i8[8];
      v126.m128i_i8[8] = v46;
      v47 = sub_CA1930(&v126);
      v48 = a2[1].m128i_u32[0];
      if ( v47 > v48 )
      {
        v94 = a2->m128i_i32[3];
        v95 = a2[1].m128i_i8[4];
        v126.m128i_i64[0] = v42;
        v127 = 0;
        v96 = v94 - v48;
        v97 = a2->m128i_i32[2];
        v128 = v95;
        v126.m128i_i32[2] = v97;
        v126.m128i_i32[3] = v96 + v47;
      }
      else
      {
        v49 = a2[1].m128i_i8[4];
        v50 = a2->m128i_i64[1];
        v126.m128i_i64[0] = v42;
        v127 = v48 - v47;
        v126.m128i_i64[1] = v50;
        v128 = v49;
      }
      sub_D02480(a1, &v126, v43);
      return a1;
    }
LABEL_42:
    sub_D01030(a1, a2);
    return a1;
  }
  v22 = *((_QWORD *)v5 - 4);
  if ( *(_BYTE *)v22 != 17 )
    goto LABEL_42;
  v23 = *(_DWORD *)(v22 + 32);
  v119.m128i_i32[2] = v23;
  if ( v23 > 0x40 )
  {
    sub_C43780((__int64)&v119, (const void **)(v22 + 24));
    v23 = v119.m128i_u32[2];
  }
  else
  {
    v119.m128i_i64[0] = *(_QWORD *)(v22 + 24);
  }
  v24 = a2[1].m128i_i32[0];
  if ( v24 )
  {
    sub_C44740((__int64)&v126, (char **)&v119, v23 - v24);
    if ( v119.m128i_i32[2] > 0x40u && v119.m128i_i64[0] )
      j_j___libc_free_0_0(v119.m128i_i64[0]);
    v23 = v126.m128i_u32[2];
    v119.m128i_i64[0] = v126.m128i_i64[0];
    v119.m128i_i32[2] = v126.m128i_i32[2];
  }
  v25 = a2->m128i_i32[3];
  if ( v25 )
  {
    sub_C44830((__int64)&v126, &v119, v25 + v23);
    if ( v119.m128i_i32[2] > 0x40u && v119.m128i_i64[0] )
      j_j___libc_free_0_0(v119.m128i_i64[0]);
    v23 = v126.m128i_u32[2];
    v119.m128i_i64[0] = v126.m128i_i64[0];
    v119.m128i_i32[2] = v126.m128i_i32[2];
  }
  v26 = a2->m128i_i32[2];
  if ( v26 )
  {
    sub_C449B0((__int64)&v126, (const void **)&v119, v26 + v23);
    if ( v119.m128i_i32[2] > 0x40u && v119.m128i_i64[0] )
      j_j___libc_free_0_0(v119.m128i_i64[0]);
    v27 = (_QWORD *)v126.m128i_i64[0];
    v23 = v126.m128i_u32[2];
  }
  else
  {
    v27 = (_QWORD *)v119.m128i_i64[0];
  }
  v108 = v23;
  v107 = v27;
  v28 = *v5;
  if ( *v5 > 0x36u )
  {
    v105 = 1;
    v29 = 1;
  }
  else
  {
    v29 = ((0x40540000000000uLL >> v28) & 1) == 0;
    if ( ((0x40540000000000uLL >> v28) & 1) != 0 )
    {
      v30 = sub_B448F0((__int64)v5);
      v29 = v30;
      if ( a2->m128i_i32[2] && !v30 || (v98 = sub_B44900((__int64)v5), v105 = v98, a2->m128i_i32[3]) && !v98 )
      {
        sub_D01030(a1, a2);
        goto LABEL_37;
      }
    }
    else
    {
      v105 = 1;
    }
  }
  v53 = v105;
  if ( a2[1].m128i_i32[0] )
  {
    v53 = 0;
    v29 = 0;
  }
  v106 = v53;
  sub_D01030(&v112, a2);
  switch ( *v5 )
  {
    case '*':
      goto LABEL_82;
    case ',':
      v67 = a2->m128i_i64[1];
      v68 = *((_QWORD *)v5 - 8);
      v69 = a2[1].m128i_i32[0];
      v121 = 0;
      v119.m128i_i64[1] = v67;
      v120 = v69;
      v119.m128i_i64[0] = v68;
      sub_D02480(&v126, &v119, (unsigned int)(a3 + 1));
      v70 = _mm_loadu_si128(&v126);
      LODWORD(v113) = v127;
      v112 = v70;
      BYTE4(v113) = v128;
      if ( v115 > 0x40 && v114 )
        j_j___libc_free_0_0(v114);
      v114 = v129;
      v71 = v130;
      v130 = 0;
      v115 = v71;
      if ( v117 > 0x40 && v116 )
      {
        j_j___libc_free_0_0(v116);
        v116 = v131;
        v117 = v132;
        v118 = v133;
        if ( v130 > 0x40 && v129 )
          j_j___libc_free_0_0(v129);
      }
      else
      {
        v116 = v131;
        v117 = v132;
        v118 = v133;
      }
      v61 = 0;
      sub_C46B40((__int64)&v116, (__int64 *)&v107);
      v59 = v115;
      v60 = v114;
      v62 = HIBYTE(v118) & v106;
      goto LABEL_91;
    case '.':
      v89 = a2->m128i_i64[1];
      v90 = *((_QWORD *)v5 - 8);
      v111 = 0;
      v91 = a2[1].m128i_i32[0];
      v109[1] = v89;
      v110 = v91;
      v109[0] = v90;
      sub_D02480(&v119, v109, (unsigned int)(a3 + 1));
      sub_D00E00(&v126, &v119, (__int64)&v107, v29, v106);
      v92 = _mm_loadu_si128(&v126);
      LODWORD(v113) = v127;
      v112 = v92;
      BYTE4(v113) = v128;
      if ( v115 > 0x40 && v114 )
        j_j___libc_free_0_0(v114);
      v114 = v129;
      v93 = v130;
      v130 = 0;
      v115 = v93;
      if ( v117 > 0x40 && v116 )
      {
        j_j___libc_free_0_0(v116);
        v116 = v131;
        v117 = v132;
        v118 = v133;
        if ( v130 > 0x40 && v129 )
          j_j___libc_free_0_0(v129);
      }
      else
      {
        v116 = v131;
        v117 = v132;
        v118 = v133;
      }
      if ( v125 > 0x40 && v124 )
        j_j___libc_free_0_0(v124);
      if ( v123 > 0x40 && v122 )
        j_j___libc_free_0_0(v122);
      v59 = v115;
      v60 = v114;
      v61 = v118;
      v62 = HIBYTE(v118);
      goto LABEL_91;
    case '6':
      v102 = v108;
      v72 = *(_QWORD *)(a2->m128i_i64[0] + 8);
      if ( v108 > 0x40 )
      {
        v101 = *(_QWORD *)(a2->m128i_i64[0] + 8);
        v99 = sub_C444A0((__int64)&v107);
        v72 = v101;
        if ( v102 - v99 > 0x40 )
        {
          v100.m128i_i64[0] = sub_BCAE30(v101);
          v126 = v100;
          sub_CA1930(&v126);
LABEL_76:
          sub_D01030(a1, a2);
          if ( v117 > 0x40 && v116 )
            j_j___libc_free_0_0(v116);
          if ( v115 > 0x40 && v114 )
            j_j___libc_free_0_0(v114);
          goto LABEL_37;
        }
        v103 = (_QWORD *)*v107;
      }
      else
      {
        v103 = v107;
      }
      v73.m128i_i64[0] = sub_BCAE30(v72);
      v126 = v73;
      v74 = sub_CA1930(&v126);
      v75 = a2->m128i_i32[3];
      v76 = a2[1].m128i_i32[0];
      if ( (unsigned int)(v74 + a2->m128i_i32[2] + v75 - v76) < (unsigned __int64)v103 )
        goto LABEL_76;
      v77 = *((_QWORD *)v5 - 8);
      v119.m128i_i32[2] = a2->m128i_i32[2];
      v120 = v76;
      v78 = a2[1].m128i_i8[4] & v106;
      v119.m128i_i64[0] = v77;
      v121 = v78;
      v119.m128i_i32[3] = v75;
      sub_D02480(&v126, &v119, (unsigned int)(a3 + 1));
      v79 = _mm_loadu_si128(&v126);
      LODWORD(v113) = v127;
      v112 = v79;
      BYTE4(v113) = v128;
      if ( v115 > 0x40 && v114 )
        j_j___libc_free_0_0(v114);
      v114 = v129;
      v80 = v130;
      v130 = 0;
      v115 = v80;
      if ( v117 > 0x40 && v116 )
      {
        j_j___libc_free_0_0(v116);
        v81 = v132;
        v116 = v131;
        v117 = v132;
        v118 = v133;
        if ( v130 > 0x40 )
        {
          if ( v129 )
            j_j___libc_free_0_0(v129);
          v81 = v117;
        }
      }
      else
      {
        v81 = v132;
        v116 = v131;
        v117 = v132;
        v118 = v133;
      }
      v82 = v108;
      if ( v108 > 0x40 )
      {
        LODWORD(v83) = -1;
        if ( v82 - (unsigned int)sub_C444A0((__int64)&v107) <= 0x40 )
          v83 = *v107;
      }
      else
      {
        LODWORD(v83) = (_DWORD)v107;
      }
      if ( v81 > 0x40 )
      {
        sub_C47690(&v116, v83);
        v82 = v108;
      }
      else
      {
        v84 = 0;
        if ( (_DWORD)v83 != v81 )
          v84 = v116 << v83;
        v85 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v81;
        if ( !v81 )
          v85 = 0;
        v116 = v84 & v85;
      }
      if ( v82 > 0x40 )
      {
        LODWORD(v86) = -1;
        if ( v82 - (unsigned int)sub_C444A0((__int64)&v107) <= 0x40 )
          v86 = *v107;
      }
      else
      {
        LODWORD(v86) = (_DWORD)v107;
      }
      v59 = v115;
      if ( v115 > 0x40 )
      {
        sub_C47690(&v114, v86);
        v59 = v115;
        v60 = v114;
      }
      else
      {
        v87 = 0;
        if ( (_DWORD)v86 != v115 )
          v87 = v114 << v86;
        v88 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v115;
        if ( !v115 )
          v88 = 0;
        v60 = v87 & v88;
      }
      v61 = v118 & v29;
      v62 = HIBYTE(v118) & v106;
LABEL_91:
      a1[2].m128i_i32[0] = v59;
      v63 = v117;
      v64 = _mm_loadu_si128(&v112);
      v65 = v113;
      a1[1].m128i_i64[1] = v60;
      a1[3].m128i_i32[0] = v63;
      v66 = v116;
      a1[1].m128i_i64[0] = v65;
      a1[2].m128i_i64[1] = v66;
      a1[3].m128i_i8[8] = v61;
      a1[3].m128i_i8[9] = v62;
      *a1 = v64;
LABEL_37:
      if ( v108 > 0x40 )
      {
        v19 = v107;
        if ( v107 )
LABEL_22:
          j_j___libc_free_0_0(v19);
      }
      return a1;
    case ':':
      if ( (v5[1] & 2) == 0 )
        goto LABEL_76;
LABEL_82:
      v54 = a2->m128i_i64[1];
      v55 = *((_QWORD *)v5 - 8);
      v56 = a2[1].m128i_i32[0];
      v121 = 0;
      v119.m128i_i64[1] = v54;
      v120 = v56;
      v119.m128i_i64[0] = v55;
      sub_D02480(&v126, &v119, (unsigned int)(a3 + 1));
      v57 = _mm_loadu_si128(&v126);
      LODWORD(v113) = v127;
      v112 = v57;
      BYTE4(v113) = v128;
      if ( v115 > 0x40 && v114 )
        j_j___libc_free_0_0(v114);
      v114 = v129;
      v58 = v130;
      v130 = 0;
      v115 = v58;
      if ( v117 > 0x40 && v116 )
      {
        j_j___libc_free_0_0(v116);
        v116 = v131;
        v117 = v132;
        v118 = v133;
        if ( v130 > 0x40 && v129 )
          j_j___libc_free_0_0(v129);
      }
      else
      {
        v116 = v131;
        v117 = v132;
        v118 = v133;
      }
      sub_C45EE0((__int64)&v116, (__int64 *)&v107);
      v59 = v115;
      v60 = v114;
      v61 = v118 & v29;
      v62 = HIBYTE(v118) & v106;
      goto LABEL_91;
    default:
      goto LABEL_76;
  }
}
