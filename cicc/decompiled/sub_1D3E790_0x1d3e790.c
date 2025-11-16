// Function: sub_1D3E790
// Address: 0x1d3e790
//
__int64 *__fastcall sub_1D3E790(
        __int64 *a1,
        unsigned __int64 a2,
        unsigned __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128 a7,
        __m128i a8,
        __m128i a9,
        __int128 a10,
        __int128 a11,
        unsigned int a12,
        char a13,
        unsigned __int8 a14,
        char a15,
        __int64 a16,
        __int64 a17,
        __int64 a18,
        __int64 a19,
        __int64 a20,
        __int64 a21)
{
  __int64 v25; // r14
  unsigned __int8 v26; // bl
  int v27; // eax
  __int64 v28; // rdi
  __int64 (*v29)(); // rax
  __int64 v30; // rdx
  _QWORD *v31; // rax
  __int64 *result; // rax
  __int64 v33; // rcx
  int v34; // eax
  unsigned __int64 v35; // rax
  char v36; // dl
  unsigned int v37; // eax
  unsigned int v38; // eax
  __int64 v39; // rdi
  __int64 v40; // rax
  __int64 v41; // rax
  __m128i v42; // xmm0
  __m128i v43; // xmm2
  __m128i *v44; // rsi
  __int64 v45; // rsi
  __int64 v46; // r10
  int v47; // eax
  __int64 v48; // rdi
  __int64 v49; // rax
  unsigned int v50; // edx
  unsigned __int8 v51; // al
  __int64 v52; // rdx
  __int64 v53; // rax
  char v54; // si
  __int64 v55; // rax
  __int64 v56; // rsi
  __int64 v57; // rax
  __int64 v58; // rsi
  unsigned int v59; // r15d
  const __m128i *v60; // rdi
  signed __int64 v61; // rax
  const __m128i *v62; // rsi
  const __m128i *v63; // rax
  void (***v64)(); // rdi
  void (*v65)(); // rax
  __int64 v66; // rsi
  __m128i v67; // xmm5
  __m128i v68; // xmm4
  __m128i *v69; // rsi
  __m128i *v70; // rax
  __int64 v71; // [rsp+8h] [rbp-1038h]
  __int64 v72; // [rsp+8h] [rbp-1038h]
  int v73; // [rsp+10h] [rbp-1030h]
  __int64 v74; // [rsp+10h] [rbp-1030h]
  __int64 v75; // [rsp+10h] [rbp-1030h]
  __int64 v76; // [rsp+18h] [rbp-1028h]
  __int64 v77; // [rsp+18h] [rbp-1028h]
  __int64 v78; // [rsp+18h] [rbp-1028h]
  char v79; // [rsp+20h] [rbp-1020h]
  unsigned __int8 v80; // [rsp+28h] [rbp-1018h]
  __int64 v81; // [rsp+28h] [rbp-1018h]
  __int64 v82; // [rsp+28h] [rbp-1018h]
  __m128i v83; // [rsp+30h] [rbp-1010h] BYREF
  __int64 v84; // [rsp+40h] [rbp-1000h]
  __int64 v85; // [rsp+48h] [rbp-FF8h]
  unsigned __int64 v86; // [rsp+50h] [rbp-FF0h]
  unsigned __int64 v87; // [rsp+58h] [rbp-FE8h]
  __m128i v88; // [rsp+60h] [rbp-FE0h]
  __m128i v89; // [rsp+70h] [rbp-FD0h]
  __m128i v90; // [rsp+80h] [rbp-FC0h]
  __m128i v91; // [rsp+A0h] [rbp-FA0h]
  const __m128i *v92; // [rsp+B0h] [rbp-F90h] BYREF
  __m128i *v93; // [rsp+B8h] [rbp-F88h]
  const __m128i *v94; // [rsp+C0h] [rbp-F80h]
  char v95[8]; // [rsp+D0h] [rbp-F70h] BYREF
  __int64 v96; // [rsp+D8h] [rbp-F68h]
  __int64 v97; // [rsp+E0h] [rbp-F60h]
  __m128i v98; // [rsp+F0h] [rbp-F50h] BYREF
  __m128i v99; // [rsp+100h] [rbp-F40h] BYREF
  __int64 v100; // [rsp+110h] [rbp-F30h]
  unsigned __int64 v101; // [rsp+120h] [rbp-F20h] BYREF
  __int64 v102; // [rsp+128h] [rbp-F18h]
  __int64 v103; // [rsp+130h] [rbp-F10h]
  unsigned __int64 v104; // [rsp+138h] [rbp-F08h]
  __int64 v105; // [rsp+140h] [rbp-F00h]
  __int64 v106; // [rsp+148h] [rbp-EF8h]
  __int64 v107; // [rsp+150h] [rbp-EF0h]
  const __m128i *v108; // [rsp+158h] [rbp-EE8h] BYREF
  __m128i *v109; // [rsp+160h] [rbp-EE0h]
  const __m128i *v110; // [rsp+168h] [rbp-ED8h]
  __int64 *v111; // [rsp+170h] [rbp-ED0h]
  __int64 v112; // [rsp+178h] [rbp-EC8h] BYREF
  int v113; // [rsp+180h] [rbp-EC0h]
  __int64 v114; // [rsp+188h] [rbp-EB8h]
  _BYTE *v115; // [rsp+190h] [rbp-EB0h]
  __int64 v116; // [rsp+198h] [rbp-EA8h]
  _BYTE v117[1536]; // [rsp+1A0h] [rbp-EA0h] BYREF
  _BYTE *v118; // [rsp+7A0h] [rbp-8A0h]
  __int64 v119; // [rsp+7A8h] [rbp-898h]
  _BYTE v120[512]; // [rsp+7B0h] [rbp-890h] BYREF
  _BYTE *v121; // [rsp+9B0h] [rbp-690h]
  __int64 v122; // [rsp+9B8h] [rbp-688h]
  _BYTE v123[1536]; // [rsp+9C0h] [rbp-680h] BYREF
  _BYTE *v124; // [rsp+FC0h] [rbp-80h]
  __int64 v125; // [rsp+FC8h] [rbp-78h]
  _BYTE v126[112]; // [rsp+FD0h] [rbp-70h] BYREF

  v25 = a11;
  v83.m128i_i64[0] = a5;
  v83.m128i_i64[1] = a6;
  v26 = a13;
  v80 = a14;
  v79 = a15;
  v27 = *(unsigned __int16 *)(a11 + 24);
  if ( v27 != 32 && v27 != 10 )
  {
    v25 = 0;
    goto LABEL_4;
  }
  v33 = *(_QWORD *)(a11 + 88);
  if ( *(_DWORD *)(v33 + 32) <= 0x40u )
  {
    v35 = *(_QWORD *)(v33 + 24);
    if ( !v35 )
      return (__int64 *)a2;
    v36 = a13;
  }
  else
  {
    v71 = a4;
    v73 = *(_DWORD *)(v33 + 32);
    v76 = *(_QWORD *)(a11 + 88);
    v34 = sub_16A57B0(v33 + 24);
    a4 = v71;
    if ( v73 == v34 )
      return (__int64 *)a2;
    v36 = v26;
    v35 = **(_QWORD **)(v76 + 24);
  }
  v78 = a4;
  result = sub_1D3D2D0(
             a1,
             a4,
             a2,
             a3,
             v83.m128i_i64[0],
             v83.m128i_u64[1],
             a7,
             a8,
             a9,
             a10,
             *((unsigned __int64 *)&a10 + 1),
             v35,
             a12,
             v36,
             0,
             a16,
             a17,
             a18,
             a19,
             a20,
             a21);
  a4 = v78;
  if ( result )
    return result;
LABEL_4:
  v28 = a1[1];
  if ( v28 )
  {
    v29 = *(__int64 (**)())(*(_QWORD *)v28 + 16LL);
    if ( v29 != sub_1D12E50 )
    {
      v77 = a4;
      result = (__int64 *)((__int64 (__fastcall *)(__int64, __int64 *, __int64, unsigned __int64, unsigned __int64, _QWORD, __int64, __int64, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, __int64, __int64, __int64, __int64, __int64, __int64))v29)(
                            v28,
                            a1,
                            a4,
                            a2,
                            a3,
                            a12,
                            v83.m128i_i64[0],
                            v83.m128i_i64[1],
                            a10,
                            *((_QWORD *)&a10 + 1),
                            a11,
                            *((_QWORD *)&a11 + 1),
                            v26,
                            v80,
                            a16,
                            a17,
                            a18,
                            a19,
                            a20,
                            a21);
      a4 = v77;
      if ( result )
        return result;
    }
  }
  if ( v80 )
  {
    v30 = *(_QWORD *)(v25 + 88);
    v31 = *(_QWORD **)(v30 + 24);
    if ( *(_DWORD *)(v30 + 32) > 0x40u )
      v31 = (_QWORD *)*v31;
    return sub_1D3D2D0(
             a1,
             a4,
             a2,
             a3,
             v83.m128i_i64[0],
             v83.m128i_u64[1],
             a7,
             a8,
             a9,
             a10,
             *((unsigned __int64 *)&a10 + 1),
             (unsigned __int64)v31,
             a12,
             v26,
             1,
             a16,
             a17,
             a18,
             a19,
             a20,
             a21);
  }
  v81 = a4;
  v37 = sub_1E340A0(&a16);
  sub_1D13830(a1[2], v37);
  v38 = sub_1E340A0(&a19);
  sub_1D13830(a1[2], v38);
  v39 = a1[4];
  v92 = 0;
  v93 = 0;
  v94 = 0;
  v98 = 0u;
  v99 = 0u;
  LODWORD(v100) = 0;
  v40 = sub_1E0A0C0(v39);
  v41 = sub_15A9620(v40, a1[6], 0);
  v42 = _mm_load_si128(&v83);
  v99.m128i_i64[1] = v41;
  v91 = v42;
  v98.m128i_i64[1] = v83.m128i_i64[0];
  v99.m128i_i32[0] = v42.m128i_i32[2];
  sub_1D27190(&v92, 0, &v98);
  v68 = _mm_loadu_si128((const __m128i *)&a10);
  v69 = v93;
  v70 = (__m128i *)v94;
  v98.m128i_i64[1] = a10;
  v46 = v81;
  v90 = v68;
  v99.m128i_i32[0] = v68.m128i_i32[2];
  if ( v94 == v93 )
  {
    sub_1D27190(&v92, v93, &v98);
    v67 = _mm_loadu_si128((const __m128i *)&a11);
    v44 = v93;
    v98.m128i_i64[1] = a11;
    v46 = v81;
    v88 = v67;
    v99.m128i_i32[0] = v67.m128i_i32[2];
    if ( v94 != v93 )
    {
      if ( !v93 )
        goto LABEL_24;
      goto LABEL_23;
    }
LABEL_53:
    v82 = v46;
    sub_1D27190(&v92, v44, &v98);
    v46 = v82;
    goto LABEL_25;
  }
  if ( v93 )
  {
    *v93 = _mm_loadu_si128(&v98);
    v69[1] = _mm_loadu_si128(&v99);
    v69[2].m128i_i64[0] = v100;
    v69 = v93;
    v70 = (__m128i *)v94;
  }
  v43 = _mm_loadu_si128((const __m128i *)&a11);
  v44 = (__m128i *)((char *)v69 + 40);
  v93 = v44;
  v98.m128i_i64[1] = a11;
  v89 = v43;
  v99.m128i_i32[0] = v43.m128i_i32[2];
  if ( v44 == v70 )
    goto LABEL_53;
LABEL_23:
  *v44 = _mm_loadu_si128(&v98);
  v44[1] = _mm_loadu_si128(&v99);
  v44[2].m128i_i64[0] = v100;
  v44 = v93;
LABEL_24:
  v93 = (__m128i *)((char *)v44 + 40);
LABEL_25:
  v45 = *(_QWORD *)v46;
  v104 = 0xFFFFFFFF00000020LL;
  v116 = 0x2000000000LL;
  v119 = 0x2000000000LL;
  v122 = 0x2000000000LL;
  v124 = v126;
  v101 = 0;
  v102 = 0;
  v103 = 0;
  v105 = 0;
  v106 = 0;
  v107 = 0;
  v108 = 0;
  v109 = 0;
  v110 = 0;
  v111 = a1;
  v113 = 0;
  v114 = 0;
  v115 = v117;
  v118 = v120;
  v121 = v123;
  v125 = 0x400000000LL;
  v112 = v45;
  if ( v45 )
  {
    v74 = v46;
    sub_1623A60((__int64)&v112, v45, 2);
    v46 = v74;
  }
  v47 = *(_DWORD *)(v46 + 8);
  v48 = a1[4];
  v86 = a2;
  v87 = a3;
  v113 = v47;
  v101 = a2;
  LODWORD(v102) = a3;
  v49 = sub_1E0A0C0(v48);
  v50 = 8 * sub_15A9520(v49, 0);
  if ( v50 == 32 )
  {
    v51 = 5;
  }
  else if ( v50 > 0x20 )
  {
    v51 = 6;
    if ( v50 != 64 )
    {
      v51 = 0;
      if ( v50 == 128 )
        v51 = 7;
    }
  }
  else
  {
    v51 = 3;
    if ( v50 != 8 )
      v51 = 4 * (v50 == 16);
  }
  v75 = sub_1D27640((__int64)a1, *(char **)(a1[2] + 76704), v51, 0);
  v72 = v52;
  v53 = *(_QWORD *)(v83.m128i_i64[0] + 40) + 16LL * v83.m128i_u32[2];
  v54 = *(_BYTE *)v53;
  v55 = *(_QWORD *)(v53 + 8);
  v95[0] = v54;
  v56 = a1[6];
  v96 = v55;
  v57 = sub_1F58E60(v95, v56);
  v58 = a1[2];
  v103 = v57;
  v59 = *(_DWORD *)(v58 + 80952);
  v84 = v75;
  v106 = v75;
  v60 = v108;
  LODWORD(v105) = v59;
  v85 = v72;
  v109 = v93;
  LODWORD(v107) = v72;
  v61 = (char *)v93 - (char *)v92;
  v108 = v92;
  v92 = 0;
  v93 = 0;
  v62 = v110;
  HIDWORD(v104) = -858993459 * (v61 >> 3);
  v63 = v94;
  v94 = 0;
  v110 = v63;
  if ( v60 )
    j_j___libc_free_0(v60, (char *)v62 - (char *)v60);
  v64 = (void (***)())v111[2];
  v65 = **v64;
  if ( v65 != nullsub_684 )
    ((void (__fastcall *)(void (***)(), __int64, _QWORD, const __m128i **))v65)(v64, v111[4], v59, &v108);
  v66 = a1[2];
  LOBYTE(v104) = v104 & 0xDF;
  BYTE1(v104) = v79;
  sub_2056920(v95, v66, &v101);
  result = (__int64 *)v97;
  if ( v124 != v126 )
  {
    v83.m128i_i64[0] = v97;
    _libc_free((unsigned __int64)v124);
    result = (__int64 *)v83.m128i_i64[0];
  }
  if ( v121 != v123 )
  {
    v83.m128i_i64[0] = (__int64)result;
    _libc_free((unsigned __int64)v121);
    result = (__int64 *)v83.m128i_i64[0];
  }
  if ( v118 != v120 )
  {
    v83.m128i_i64[0] = (__int64)result;
    _libc_free((unsigned __int64)v118);
    result = (__int64 *)v83.m128i_i64[0];
  }
  if ( v115 != v117 )
  {
    v83.m128i_i64[0] = (__int64)result;
    _libc_free((unsigned __int64)v115);
    result = (__int64 *)v83.m128i_i64[0];
  }
  if ( v112 )
  {
    v83.m128i_i64[0] = (__int64)result;
    sub_161E7C0((__int64)&v112, v112);
    result = (__int64 *)v83.m128i_i64[0];
  }
  if ( v108 )
  {
    v83.m128i_i64[0] = (__int64)result;
    j_j___libc_free_0(v108, (char *)v110 - (char *)v108);
    result = (__int64 *)v83.m128i_i64[0];
  }
  if ( v92 )
  {
    v83.m128i_i64[0] = (__int64)result;
    j_j___libc_free_0(v92, (char *)v94 - (char *)v92);
    return (__int64 *)v83.m128i_i64[0];
  }
  return result;
}
