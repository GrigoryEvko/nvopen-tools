// Function: sub_202F990
// Address: 0x202f990
//
__int64 __fastcall sub_202F990(
        __int64 *a1,
        __int64 a2,
        __m128i a3,
        double a4,
        __m128i a5,
        __int64 a6,
        __int64 a7,
        const void **a8,
        int a9,
        __int64 a10,
        const void **a11)
{
  __int64 v13; // rax
  __int64 v14; // r14
  const __m128i *v15; // r13
  __int64 v16; // rax
  _BYTE *v17; // rdx
  __int64 v18; // r14
  __int64 i; // r12
  _BYTE *v20; // rdx
  __int64 v21; // rsi
  __int64 *v22; // r10
  _BYTE *v23; // r12
  __int64 v24; // r13
  __int64 v25; // rsi
  __int64 *v26; // r12
  unsigned __int64 v27; // rdx
  unsigned __int64 v28; // r13
  __int64 v29; // rbx
  char v30; // di
  const void **v31; // rdx
  char v32; // di
  const void **v33; // rdx
  unsigned int v34; // eax
  bool v35; // cc
  unsigned __int8 *v36; // rax
  __int64 v37; // r9
  __int64 v38; // r8
  unsigned int v39; // ecx
  char v40; // al
  unsigned int v41; // eax
  bool v42; // cc
  unsigned int v44; // eax
  __int64 v45; // rax
  unsigned int v46; // edx
  unsigned __int8 v47; // al
  __int64 v48; // r10
  __int64 v49; // rcx
  __int64 v50; // rdx
  __int64 v51; // rsi
  __int64 *v52; // r15
  unsigned int v53; // eax
  unsigned int v54; // ecx
  unsigned int v55; // eax
  __int64 v56; // rdx
  __int64 v57; // rax
  const void **v58; // rdx
  __int64 *v59; // r10
  __int64 v60; // rcx
  const void **v61; // r8
  unsigned int v62; // edx
  _QWORD *v63; // rdi
  _QWORD *v64; // rax
  __int32 v65; // edx
  __int64 v66; // rcx
  int v67; // r8d
  int v68; // r9d
  __int64 v69; // rax
  __int64 *v70; // r14
  __int128 v71; // rax
  unsigned int v72; // ecx
  unsigned int v73; // eax
  __int64 v74; // rdx
  __int64 v75; // rax
  const void **v76; // rdx
  __int64 *v77; // r10
  __int64 v78; // rcx
  const void **v79; // r8
  __int128 v80; // [rsp-10h] [rbp-240h]
  __int128 v81; // [rsp-10h] [rbp-240h]
  __int128 v82; // [rsp-10h] [rbp-240h]
  unsigned int v83; // [rsp+0h] [rbp-230h]
  __int64 v84; // [rsp+0h] [rbp-230h]
  __int32 v85; // [rsp+0h] [rbp-230h]
  __int128 v86; // [rsp+0h] [rbp-230h]
  __int64 v87; // [rsp+0h] [rbp-230h]
  __int64 *v88; // [rsp+10h] [rbp-220h]
  unsigned int v89; // [rsp+10h] [rbp-220h]
  unsigned __int8 v90; // [rsp+10h] [rbp-220h]
  __int64 (__fastcall *v91)(__int64, __int64); // [rsp+10h] [rbp-220h]
  __int64 v92; // [rsp+10h] [rbp-220h]
  __int128 v93; // [rsp+10h] [rbp-220h]
  unsigned int v94; // [rsp+10h] [rbp-220h]
  __int64 *v95; // [rsp+10h] [rbp-220h]
  unsigned int v96; // [rsp+10h] [rbp-220h]
  unsigned int v97; // [rsp+10h] [rbp-220h]
  __int64 *v98; // [rsp+10h] [rbp-220h]
  _QWORD *v99; // [rsp+20h] [rbp-210h]
  __int64 v100; // [rsp+20h] [rbp-210h]
  __int64 v101; // [rsp+20h] [rbp-210h]
  __int64 v102; // [rsp+20h] [rbp-210h]
  __int64 v103; // [rsp+20h] [rbp-210h]
  const void **v104; // [rsp+20h] [rbp-210h]
  _QWORD *v105; // [rsp+20h] [rbp-210h]
  const void **v106; // [rsp+20h] [rbp-210h]
  __int64 v107; // [rsp+80h] [rbp-1B0h] BYREF
  const void **v108; // [rsp+88h] [rbp-1A8h]
  __m128i v109; // [rsp+90h] [rbp-1A0h] BYREF
  _BYTE *v110; // [rsp+A0h] [rbp-190h] BYREF
  __int64 v111; // [rsp+A8h] [rbp-188h]
  _BYTE v112[64]; // [rsp+B0h] [rbp-180h] BYREF
  _BYTE *v113; // [rsp+F0h] [rbp-140h] BYREF
  __int64 v114; // [rsp+F8h] [rbp-138h]
  _BYTE v115[304]; // [rsp+100h] [rbp-130h] BYREF

  v110 = v112;
  v111 = 0x400000000LL;
  v13 = *(unsigned int *)(a2 + 56);
  v107 = a7;
  v108 = a8;
  if ( (_DWORD)v13 )
  {
    v14 = 5 * v13;
    v15 = *(const __m128i **)(a2 + 32);
    v16 = 0;
    v17 = v112;
    v18 = 8 * v14;
    for ( i = 40; ; i += 40 )
    {
      a3 = _mm_loadu_si128(v15);
      *(__m128i *)&v17[16 * v16] = a3;
      v16 = (unsigned int)(v111 + 1);
      LODWORD(v111) = v111 + 1;
      if ( v18 == i )
        break;
      v15 = (const __m128i *)(i + *(_QWORD *)(a2 + 32));
      if ( HIDWORD(v111) <= (unsigned int)v16 )
      {
        sub_16CD150((__int64)&v110, v112, 0, 16, (int)a8, a9);
        v16 = (unsigned int)v111;
      }
      v17 = v110;
    }
    v20 = v110;
  }
  else
  {
    v20 = v112;
    v16 = 0;
  }
  v21 = *(_QWORD *)(a2 + 72);
  v22 = (__int64 *)a1[1];
  v23 = v20;
  v24 = v16;
  v113 = (_BYTE *)v21;
  if ( v21 )
  {
    v88 = v22;
    sub_1623A60((__int64)&v113, v21, 2);
    v22 = v88;
  }
  v25 = *(unsigned __int16 *)(a2 + 24);
  *((_QWORD *)&v80 + 1) = v24;
  *(_QWORD *)&v80 = v23;
  LODWORD(v114) = *(_DWORD *)(a2 + 64);
  v26 = sub_1D359D0(v22, v25, (__int64)&v113, (unsigned int)v107, v108, 0, *(double *)a3.m128i_i64, a4, a5, v80);
  v28 = v27;
  v29 = (__int64)v26;
  if ( v113 )
    sub_161E7C0((__int64)&v113, (__int64)v113);
  v99 = *(_QWORD **)(a1[1] + 48);
  if ( (_BYTE)v107 )
  {
    if ( (unsigned __int8)(v107 - 14) > 0x5Fu )
    {
LABEL_14:
      v30 = v107;
      v31 = v108;
      goto LABEL_15;
    }
  }
  else if ( !sub_1F58D20((__int64)&v107) )
  {
    goto LABEL_14;
  }
  v30 = sub_1F7E0F0((__int64)&v107);
LABEL_15:
  LOBYTE(v113) = v30;
  v114 = (__int64)v31;
  if ( v30 )
    v89 = sub_2021900(v30);
  else
    v89 = sub_1F58D40((__int64)&v113);
  if ( (_BYTE)a10 )
  {
    if ( (unsigned __int8)(a10 - 14) > 0x5Fu )
    {
LABEL_19:
      v32 = a10;
      v33 = a11;
      goto LABEL_20;
    }
  }
  else if ( !sub_1F58D20((__int64)&a10) )
  {
    goto LABEL_19;
  }
  v32 = sub_1F7E0F0((__int64)&a10);
LABEL_20:
  LOBYTE(v113) = v32;
  v114 = (__int64)v33;
  if ( !v32 )
  {
    v53 = sub_1F58D40((__int64)&v113);
    v35 = v89 <= v53;
    if ( v89 >= v53 )
      goto LABEL_22;
LABEL_46:
    if ( (_BYTE)v107 )
      v54 = word_4305480[(unsigned __int8)(v107 - 14)];
    else
      v54 = sub_1F58D30((__int64)&v107);
    v94 = v54;
    LOBYTE(v55) = sub_1F7E0F0((__int64)&a10);
    v57 = sub_1F7DEB0(v99, v55, v56, v94, 0);
    v59 = (__int64 *)a1[1];
    v60 = v57;
    v61 = v58;
    v113 = (_BYTE *)v26[9];
    if ( v113 )
    {
      v84 = v57;
      v95 = v59;
      v104 = v58;
      sub_20219D0((__int64 *)&v113);
      v60 = v84;
      v61 = v104;
      v59 = v95;
    }
    *((_QWORD *)&v81 + 1) = v28;
    *(_QWORD *)&v81 = v26;
    LODWORD(v114) = *((_DWORD *)v26 + 16);
    v29 = sub_1D309E0(v59, 142, (__int64)&v113, v60, v61, 0, *(double *)a3.m128i_i64, a4, *(double *)a5.m128i_i64, v81);
    goto LABEL_51;
  }
  v34 = sub_2021900(v32);
  v35 = v89 <= v34;
  if ( v89 < v34 )
    goto LABEL_46;
LABEL_22:
  if ( v35 )
    goto LABEL_23;
  if ( (_BYTE)v107 )
    v72 = word_4305480[(unsigned __int8)(v107 - 14)];
  else
    v72 = sub_1F58D30((__int64)&v107);
  v97 = v72;
  LOBYTE(v73) = sub_1F7E0F0((__int64)&a10);
  v75 = sub_1F7DEB0(v99, v73, v74, v97, 0);
  v77 = (__int64 *)a1[1];
  v78 = v75;
  v79 = v76;
  v113 = (_BYTE *)v26[9];
  if ( v113 )
  {
    v87 = v75;
    v98 = v77;
    v106 = v76;
    sub_20219D0((__int64 *)&v113);
    v78 = v87;
    v79 = v106;
    v77 = v98;
  }
  *((_QWORD *)&v82 + 1) = v28;
  *(_QWORD *)&v82 = v26;
  LODWORD(v114) = *((_DWORD *)v26 + 16);
  v29 = sub_1D309E0(v77, 145, (__int64)&v113, v78, v79, 0, *(double *)a3.m128i_i64, a4, *(double *)a5.m128i_i64, v82);
LABEL_51:
  v28 = v62 | v28 & 0xFFFFFFFF00000000LL;
  if ( v113 )
    sub_161E7C0((__int64)&v113, (__int64)v113);
LABEL_23:
  v36 = *(unsigned __int8 **)(v29 + 40);
  v37 = *v36;
  v38 = *((_QWORD *)v36 + 1);
  LOBYTE(v113) = v37;
  v114 = v38;
  if ( (_BYTE)v37 )
  {
    v39 = word_4305480[(unsigned __int8)(v37 - 14)];
    v40 = a10;
    if ( (_BYTE)a10 )
      goto LABEL_25;
  }
  else
  {
    v100 = v38;
    v44 = sub_1F58D30((__int64)&v113);
    v37 = 0;
    v38 = v100;
    v39 = v44;
    v40 = a10;
    if ( (_BYTE)a10 )
    {
LABEL_25:
      v41 = word_4305480[(unsigned __int8)(v40 - 14)];
      v42 = v41 <= v39;
      if ( v41 >= v39 )
        goto LABEL_26;
      goto LABEL_32;
    }
  }
  v83 = v39;
  v101 = v38;
  v90 = v37;
  v41 = sub_1F58D30((__int64)&a10);
  v39 = v83;
  v38 = v101;
  v37 = v90;
  v42 = v41 <= v83;
  if ( v41 >= v83 )
  {
LABEL_26:
    if ( !v42 )
    {
      v63 = (_QWORD *)a1[1];
      v113 = 0;
      LODWORD(v114) = 0;
      v96 = v41 / v39;
      v64 = sub_1D2B300(v63, 0x30u, (__int64)&v113, (unsigned __int8)v37, v38, v37);
      if ( v113 )
      {
        v85 = v65;
        v105 = v64;
        sub_161E7C0((__int64)&v113, (__int64)v113);
        v65 = v85;
        v64 = v105;
      }
      v109.m128i_i64[0] = (__int64)v64;
      v109.m128i_i32[2] = v65;
      v113 = v115;
      v114 = 0x1000000000LL;
      sub_202F910((__int64)&v113, v96, &v109, v66, v67, v68);
      v69 = (__int64)v113;
      *(_QWORD *)v113 = v29;
      *(_DWORD *)(v69 + 8) = v28;
      v70 = (__int64 *)a1[1];
      *(_QWORD *)&v71 = v113;
      v109.m128i_i64[0] = *(_QWORD *)(v29 + 72);
      *((_QWORD *)&v71 + 1) = (unsigned int)v114;
      if ( v109.m128i_i64[0] )
      {
        *(_QWORD *)&v86 = v113;
        *((_QWORD *)&v86 + 1) = (unsigned int)v114;
        sub_20219D0(v109.m128i_i64);
        v71 = v86;
      }
      v109.m128i_i32[2] = *(_DWORD *)(v29 + 64);
      v29 = (__int64)sub_1D359D0(
                       v70,
                       107,
                       (__int64)&v109,
                       (unsigned int)a10,
                       a11,
                       0,
                       *(double *)a3.m128i_i64,
                       a4,
                       a5,
                       v71);
      if ( v109.m128i_i64[0] )
        sub_161E7C0((__int64)&v109, v109.m128i_i64[0]);
      if ( v113 != v115 )
        _libc_free((unsigned __int64)v113);
    }
    goto LABEL_27;
  }
LABEL_32:
  v102 = *a1;
  v91 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)*a1 + 48LL);
  v45 = sub_1E0A0C0(*(_QWORD *)(a1[1] + 32));
  if ( v91 == sub_1D13A20 )
  {
    v46 = 8 * sub_15A9520(v45, 0);
    if ( v46 == 32 )
    {
      v47 = 5;
    }
    else if ( v46 > 0x20 )
    {
      v47 = 6;
      if ( v46 != 64 )
      {
        v47 = 0;
        if ( v46 == 128 )
          v47 = 7;
      }
    }
    else
    {
      v47 = 3;
      if ( v46 != 8 )
        v47 = 4 * (v46 == 16);
    }
  }
  else
  {
    v47 = v91(v102, v45);
  }
  v48 = a1[1];
  v49 = v47;
  v113 = *(_BYTE **)(v29 + 72);
  if ( v113 )
  {
    v103 = v47;
    v92 = v48;
    sub_20219D0((__int64 *)&v113);
    v49 = v103;
    v48 = v92;
  }
  LODWORD(v114) = *(_DWORD *)(v29 + 64);
  *(_QWORD *)&v93 = sub_1D38BB0(v48, 0, (__int64)&v113, v49, 0, 0, a3, a4, a5, 0);
  *((_QWORD *)&v93 + 1) = v50;
  if ( v113 )
    sub_161E7C0((__int64)&v113, (__int64)v113);
  v51 = *(_QWORD *)(v29 + 72);
  v52 = (__int64 *)a1[1];
  v113 = (_BYTE *)v51;
  if ( v51 )
    sub_1623A60((__int64)&v113, v51, 2);
  LODWORD(v114) = *(_DWORD *)(v29 + 64);
  v29 = (__int64)sub_1D332F0(
                   v52,
                   109,
                   (__int64)&v113,
                   (unsigned int)a10,
                   a11,
                   0,
                   *(double *)a3.m128i_i64,
                   a4,
                   a5,
                   v29,
                   v28,
                   v93);
  if ( v113 )
    sub_161E7C0((__int64)&v113, (__int64)v113);
LABEL_27:
  if ( v110 != v112 )
    _libc_free((unsigned __int64)v110);
  return v29;
}
