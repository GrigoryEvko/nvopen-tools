// Function: sub_3035F40
// Address: 0x3035f40
//
void __fastcall sub_3035F40(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  __int16 *v8; // rax
  __int64 v9; // rsi
  __int16 v10; // dx
  __int64 v11; // rax
  __int64 v12; // rdi
  int v13; // eax
  char v14; // r13
  __int64 v15; // rdi
  __int64 v16; // rax
  __int64 *v17; // rsi
  __int64 v18; // r15
  __int64 v19; // rax
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rdx
  int v25; // r9d
  __int64 v26; // r9
  int v27; // edx
  unsigned __int64 v28; // rax
  const __m128i *v29; // r15
  __int32 v30; // ecx
  unsigned __int64 v31; // rsi
  unsigned __int64 v32; // rdx
  __m128i *v33; // rax
  __int64 v34; // r8
  unsigned __int8 v35; // si
  __int64 v36; // r8
  __int64 v37; // r9
  __int64 v38; // r10
  __int64 v39; // rax
  __int64 v40; // rdx
  __int64 v41; // r11
  unsigned __int64 v42; // rdx
  __int64 *v43; // rax
  __int64 v44; // rsi
  __int64 v45; // r9
  __int64 v46; // rcx
  __int64 v47; // r8
  __int64 v48; // r9
  __int64 v49; // r14
  __int32 v50; // r11d
  unsigned __int32 v51; // eax
  __int64 v52; // rsi
  unsigned int v53; // ebx
  unsigned int v54; // edx
  unsigned int v55; // eax
  __int64 v56; // r8
  __int64 v57; // r9
  __int64 v58; // r12
  __int64 v59; // rax
  __int64 v60; // rdx
  __int64 v61; // r13
  unsigned __int32 v62; // r11d
  __int64 *v63; // rax
  __int64 v64; // r12
  unsigned __int64 v65; // rcx
  __int64 v66; // rax
  __int64 *v67; // rax
  _BYTE *v68; // rdi
  unsigned __int32 v69; // ebx
  char v70; // r15
  unsigned __int64 v71; // r13
  __int64 v72; // r12
  __int64 v73; // rax
  unsigned __int64 v74; // rdx
  __int64 *v75; // rax
  unsigned __int32 v76; // edx
  unsigned __int32 v77; // edx
  __int64 v78; // rax
  int v79; // r8d
  __int64 v80; // rax
  bool v81; // al
  __int32 v82; // r11d
  int v83; // edx
  __m128i v84; // xmm0
  int v85; // edx
  __m128i v86; // xmm0
  int v87; // edx
  __int64 v88; // [rsp-10h] [rbp-1F0h]
  __int64 v89; // [rsp-10h] [rbp-1F0h]
  __int128 v90; // [rsp-10h] [rbp-1F0h]
  __int128 v91; // [rsp-10h] [rbp-1F0h]
  __int64 v92; // [rsp-10h] [rbp-1F0h]
  __int64 v93; // [rsp-8h] [rbp-1E8h]
  __int64 v94; // [rsp+8h] [rbp-1D8h]
  unsigned __int64 v95; // [rsp+18h] [rbp-1C8h]
  __int64 v96; // [rsp+20h] [rbp-1C0h]
  __int64 v97; // [rsp+30h] [rbp-1B0h]
  int v98; // [rsp+30h] [rbp-1B0h]
  __int64 v99; // [rsp+30h] [rbp-1B0h]
  __int64 v100; // [rsp+38h] [rbp-1A8h]
  int v101; // [rsp+48h] [rbp-198h]
  int v102; // [rsp+50h] [rbp-190h]
  int v103; // [rsp+50h] [rbp-190h]
  __int32 v104; // [rsp+50h] [rbp-190h]
  unsigned __int8 v105; // [rsp+68h] [rbp-178h]
  unsigned int v106; // [rsp+68h] [rbp-178h]
  unsigned __int32 v107; // [rsp+68h] [rbp-178h]
  __int64 v108; // [rsp+68h] [rbp-178h]
  __int32 v109; // [rsp+68h] [rbp-178h]
  __int64 v110; // [rsp+68h] [rbp-178h]
  __int64 v111; // [rsp+68h] [rbp-178h]
  __int64 v112; // [rsp+70h] [rbp-170h] BYREF
  __int64 v113; // [rsp+78h] [rbp-168h]
  __int64 v114; // [rsp+80h] [rbp-160h] BYREF
  int v115; // [rsp+88h] [rbp-158h]
  __int64 v116; // [rsp+90h] [rbp-150h]
  __int64 v117; // [rsp+98h] [rbp-148h]
  __m128i v118; // [rsp+A0h] [rbp-140h] BYREF
  __int64 v119; // [rsp+B0h] [rbp-130h]
  __m128i v120; // [rsp+C0h] [rbp-120h] BYREF
  __int64 v121; // [rsp+D0h] [rbp-110h]
  char v122; // [rsp+D8h] [rbp-108h]
  _BYTE *v123; // [rsp+E0h] [rbp-100h] BYREF
  __int64 v124; // [rsp+E8h] [rbp-F8h]
  _BYTE v125[48]; // [rsp+F0h] [rbp-F0h] BYREF
  __m128i v126; // [rsp+120h] [rbp-C0h] BYREF
  __m128i v127; // [rsp+130h] [rbp-B0h] BYREF
  __m128i v128; // [rsp+140h] [rbp-A0h]
  __m128i v129; // [rsp+150h] [rbp-90h]
  __m128i v130; // [rsp+160h] [rbp-80h]
  __m128i v131; // [rsp+170h] [rbp-70h]
  __m128i v132; // [rsp+180h] [rbp-60h]
  __m128i v133; // [rsp+190h] [rbp-50h]
  __int16 v134; // [rsp+1A0h] [rbp-40h]
  __int64 v135; // [rsp+1A8h] [rbp-38h]

  v8 = *(__int16 **)(a1 + 48);
  v9 = *(_QWORD *)(a1 + 80);
  v10 = *v8;
  v11 = *((_QWORD *)v8 + 1);
  v114 = v9;
  LOWORD(v112) = v10;
  v113 = v11;
  if ( v9 )
    sub_B96E90((__int64)&v114, v9, 1);
  v12 = *(_QWORD *)(a1 + 112);
  v115 = *(_DWORD *)(a1 + 72);
  v13 = sub_2EAC1E0(v12);
  sub_3035DC0((__int64)&v120, v112, v113, a4 & (v13 == 1));
  v14 = v122;
  if ( !v122 )
    goto LABEL_4;
  v15 = *(_QWORD *)(a1 + 112);
  v118 = _mm_load_si128(&v120);
  v119 = v121;
  v105 = sub_2EAC4F0(v15);
  v16 = sub_2E79000(*(__int64 **)(a2 + 40));
  v17 = *(__int64 **)(a2 + 64);
  v18 = v16;
  v19 = *(_QWORD *)(a1 + 104);
  v126.m128i_i16[0] = *(_WORD *)(a1 + 96);
  v126.m128i_i64[1] = v19;
  v22 = sub_3007410((__int64)&v126, v17, v126.m128i_u16[0], (__int64)&v126, v20, v21);
  if ( (unsigned __int8)sub_AE5260(v18, v22) > v105 )
    goto LABEL_4;
  if ( v118.m128i_i16[4] )
  {
    if ( v118.m128i_i16[4] == 1 || (unsigned __int16)(v118.m128i_i16[4] - 504) <= 7u )
      BUG();
    v24 = 16LL * (v118.m128i_u16[4] - 1);
    v23 = *(_QWORD *)&byte_444C4A0[v24];
    LOBYTE(v24) = byte_444C4A0[v24 + 8];
  }
  else
  {
    v23 = sub_3007260((__int64)&v118.m128i_i64[1]);
    v116 = v23;
    v117 = v24;
  }
  v126.m128i_i64[0] = v23;
  v126.m128i_i8[8] = v24;
  if ( (unsigned __int64)sub_CA1930(&v126) > 0xF )
  {
    v14 = 0;
  }
  else
  {
    v119 = 0;
    v118.m128i_i16[4] = 6;
  }
  switch ( v118.m128i_i32[0] )
  {
    case 4:
      v86 = _mm_loadu_si128((const __m128i *)&v118.m128i_u64[1]);
      v130.m128i_i16[0] = 1;
      v130.m128i_i64[1] = 0;
      v126 = v86;
      v127 = v86;
      v128 = v86;
      v129 = v86;
      v106 = 549;
      v102 = sub_33E5830(a2, &v126);
      v101 = v87;
      break;
    case 8:
      v84 = _mm_loadu_si128((const __m128i *)&v118.m128i_u64[1]);
      v134 = 1;
      v135 = 0;
      v126 = v84;
      v127 = v84;
      v128 = v84;
      v129 = v84;
      v130 = v84;
      v131 = v84;
      v132 = v84;
      v133 = v84;
      v106 = 550;
      v102 = sub_33E5830(a2, &v126);
      v101 = v85;
      break;
    case 2:
      v106 = 548;
      v102 = sub_33E5B50(a2, v118.m128i_i32[2], v119, v118.m128i_i32[2], v119, v25, 1, 0);
      v101 = v27;
      break;
    default:
LABEL_4:
      if ( v114 )
        sub_B91220((__int64)&v114, v114);
      return;
  }
  v28 = *(unsigned int *)(a1 + 64);
  v29 = *(const __m128i **)(a1 + 40);
  v126.m128i_i64[1] = 0x800000000LL;
  v30 = 0;
  v31 = 5 * v28;
  v32 = v28;
  v33 = &v127;
  v34 = (__int64)&v29->m128i_i64[v31];
  v126.m128i_i64[0] = (__int64)&v127;
  if ( v31 > 40 )
  {
    v98 = v32;
    sub_C8D5F0((__int64)&v126, &v127, v32, 0x10u, v34, v26);
    v30 = v126.m128i_i32[2];
    v34 = (__int64)&v29->m128i_i64[v31];
    LODWORD(v32) = v98;
    v33 = (__m128i *)(v126.m128i_i64[0] + 16LL * v126.m128i_u32[2]);
  }
  if ( v29 != (const __m128i *)v34 )
  {
    do
    {
      if ( v33 )
        *v33 = _mm_loadu_si128(v29);
      v29 = (const __m128i *)((char *)v29 + 40);
      ++v33;
    }
    while ( (const __m128i *)v34 != v29 );
    v30 = v126.m128i_i32[2];
  }
  v35 = *(_BYTE *)(a1 + 33);
  v126.m128i_i32[2] = v30 + v32;
  v38 = sub_3400D50(a2, (v35 >> 2) & 3, &v114, 0);
  v39 = v126.m128i_u32[2];
  v41 = v40;
  v42 = v126.m128i_u32[2] + 1LL;
  if ( v42 > v126.m128i_u32[3] )
  {
    v99 = v38;
    v100 = v41;
    sub_C8D5F0((__int64)&v126, &v127, v42, 0x10u, v36, v37);
    v39 = v126.m128i_u32[2];
    v38 = v99;
    v41 = v100;
  }
  v43 = (__int64 *)(v126.m128i_i64[0] + 16 * v39);
  *v43 = v38;
  v43[1] = v41;
  v93 = *(_QWORD *)(a1 + 104);
  v88 = *(unsigned __int16 *)(a1 + 96);
  v44 = v106;
  v45 = *(_QWORD *)(a1 + 112);
  ++v126.m128i_i32[2];
  v49 = sub_33EA9D0(a2, v106, (unsigned int)&v114, v102, v101, v45, v126.m128i_i64[0], v126.m128i_u32[2], v88, v93);
  v123 = v125;
  v124 = 0x300000000LL;
  v107 = v118.m128i_i32[0];
  v50 = v118.m128i_i32[0];
  if ( (_WORD)v112 )
  {
    if ( (unsigned __int16)(v112 - 176) > 0x34u )
    {
LABEL_27:
      v51 = word_4456340[(unsigned __int16)v112 - 1];
      goto LABEL_28;
    }
  }
  else
  {
    v81 = sub_3007100((__int64)&v112);
    v82 = v107;
    if ( !v81 )
      goto LABEL_57;
  }
  sub_CA17B0(
    "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::ge"
    "tVectorElementCount() instead");
  if ( (_WORD)v112 )
  {
    if ( (unsigned __int16)(v112 - 176) <= 0x34u )
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
    v50 = v118.m128i_i32[0];
    goto LABEL_27;
  }
  v82 = v118.m128i_i32[0];
LABEL_57:
  v104 = v82;
  v51 = sub_3007130((__int64)&v112, v44);
  v50 = v104;
LABEL_28:
  if ( v107 >= v51 )
  {
    LODWORD(v48) = 0;
    if ( v50 )
    {
      v97 = a3;
      v69 = 0;
      v48 = v94;
      v70 = v14;
      v71 = v95;
      v103 = a2;
      do
      {
        v77 = v69;
        v78 = v49;
        if ( v70 )
        {
          if ( (_WORD)v112 )
          {
            v79 = 0;
            LOWORD(v80) = word_4456580[(unsigned __int16)v112 - 1];
          }
          else
          {
            v80 = sub_3009970((__int64)&v112, v44, v69, v46, v47);
            v48 = v80;
            v79 = v83;
          }
          LOWORD(v48) = v80;
          v44 = 216;
          v110 = v48;
          *((_QWORD *)&v91 + 1) = v69;
          v71 = v69;
          *(_QWORD *)&v91 = v49;
          v78 = sub_33FAF80(v103, 216, (unsigned int)&v114, v48, v79, v48, v91);
          v47 = v92;
          v48 = v110;
        }
        v72 = v78;
        v46 = HIDWORD(v124);
        v71 = v77 | v71 & 0xFFFFFFFF00000000LL;
        v73 = (unsigned int)v124;
        v74 = (unsigned int)v124 + 1LL;
        if ( v74 > HIDWORD(v124) )
        {
          v44 = (__int64)v125;
          v111 = v48;
          sub_C8D5F0((__int64)&v123, v125, v74, 0x10u, v47, v48);
          v73 = (unsigned int)v124;
          v48 = v111;
        }
        v75 = (__int64 *)&v123[16 * v73];
        ++v69;
        *v75 = v72;
        v76 = v118.m128i_i32[0];
        v75[1] = v71;
        v55 = v124 + 1;
        LODWORD(v124) = v124 + 1;
      }
      while ( v76 > v69 );
      LODWORD(a2) = v103;
      a3 = v97;
      v50 = v76;
      goto LABEL_33;
    }
  }
  else if ( v50 )
  {
    v52 = v96;
    v108 = a3;
    v53 = 0;
    do
    {
      LOWORD(v52) = 0;
      v54 = v53;
      v89 = v52;
      v52 = v49;
      ++v53;
      sub_3408690(a2, v49, v54, (unsigned int)&v123, 0, 0, v89, 0);
      v50 = v118.m128i_i32[0];
    }
    while ( v118.m128i_i32[0] > v53 );
    a3 = v108;
    v55 = v124;
    goto LABEL_33;
  }
  v55 = v124;
LABEL_33:
  *((_QWORD *)&v90 + 1) = v55;
  *(_QWORD *)&v90 = v123;
  v109 = v50;
  v58 = sub_33FC220(a2, 156, (unsigned int)&v114, v112, v113, v48, v90);
  v59 = *(unsigned int *)(a3 + 8);
  v61 = v60;
  v62 = v109;
  if ( v59 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
  {
    sub_C8D5F0(a3, (const void *)(a3 + 16), v59 + 1, 0x10u, v56, v57);
    v59 = *(unsigned int *)(a3 + 8);
    v62 = v109;
  }
  v63 = (__int64 *)(*(_QWORD *)a3 + 16 * v59);
  *v63 = v58;
  v64 = v62;
  v63[1] = v61;
  v65 = *(unsigned int *)(a3 + 12);
  v66 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
  *(_DWORD *)(a3 + 8) = v66;
  if ( v66 + 1 > v65 )
  {
    sub_C8D5F0(a3, (const void *)(a3 + 16), v66 + 1, 0x10u, v56, v57);
    v66 = *(unsigned int *)(a3 + 8);
  }
  v67 = (__int64 *)(*(_QWORD *)a3 + 16 * v66);
  *v67 = v49;
  v68 = v123;
  v67[1] = v64;
  ++*(_DWORD *)(a3 + 8);
  if ( v68 != v125 )
    _libc_free((unsigned __int64)v68);
  if ( (__m128i *)v126.m128i_i64[0] != &v127 )
    _libc_free(v126.m128i_u64[0]);
  if ( v114 )
    sub_B91220((__int64)&v114, v114);
}
