// Function: sub_32F3B40
// Address: 0x32f3b40
//
__int64 __fastcall sub_32F3B40(__int64 *a1, __int64 a2)
{
  __int64 *v4; // r15
  __int64 v5; // rax
  __int64 v6; // rdx
  __m128i v7; // xmm0
  __m128i v8; // xmm1
  __int64 v9; // rax
  __int16 *v10; // rax
  __int16 v11; // cx
  __int64 v12; // rax
  __int64 (__fastcall *v13)(__int64 *, __int64, __int64, __int64, __int64); // rbx
  __int64 v14; // rax
  __int16 v15; // ax
  __int64 v16; // rsi
  __int16 v17; // bx
  __int64 v18; // rdx
  __int64 v19; // rdi
  __m128i v20; // xmm2
  __m128i v21; // xmm3
  __int64 v22; // rax
  __int64 v23; // r14
  char v25; // bl
  char v26; // al
  __int128 v27; // rax
  __int64 v28; // rdi
  int v29; // r13d
  int v30; // ebx
  __int128 v31; // rax
  int v32; // r9d
  __int64 v33; // rax
  __int64 v34; // r14
  __int64 v35; // r13
  unsigned int v36; // edx
  unsigned int v37; // ebx
  __m128i v38; // rax
  int v39; // ecx
  __int64 v40; // r10
  int v41; // r8d
  __int64 v42; // rbx
  __int64 v43; // r11
  __int16 v44; // ax
  int v45; // esi
  __int64 v46; // rbx
  __int64 v47; // r8
  __int64 v48; // r9
  __int64 v49; // rdi
  __int64 v50; // r8
  __int64 v51; // r9
  __int64 v52; // rbx
  __int64 v53; // rdi
  __int64 (*v54)(); // rax
  __int64 v55; // rax
  __int64 v56; // r10
  __int64 v57; // rdx
  __int64 v58; // r11
  __int64 v59; // rbx
  __int64 v60; // rsi
  __int64 v61; // rdi
  __int64 v62; // rdx
  __m128i v63; // xmm4
  __m128i v64; // xmm5
  __int64 v65; // rcx
  __int64 v66; // rax
  int v67; // r9d
  __int64 v68; // r11
  __int128 v69; // rax
  __int64 v70; // rdi
  int v71; // r9d
  __int64 v72; // r15
  __int64 v73; // r8
  __int64 v74; // r9
  __int64 v75; // rdi
  __int64 v76; // r8
  __int64 v77; // r9
  bool v78; // al
  __int64 v79; // rbx
  __int64 v80; // rax
  int v81; // r9d
  __int64 v82; // rax
  __int128 v83; // rax
  int v84; // r9d
  __int64 v85; // rdx
  __int64 v86; // rax
  __int64 v87; // rdx
  __int32 v88; // ebx
  __int64 v89; // r9
  __int64 v90; // rdi
  __int64 v91; // rdx
  __int64 v92; // rcx
  __int64 v93; // rax
  __int64 v94; // rax
  unsigned int v95; // ebx
  __int64 v96; // rdx
  int v97; // eax
  __int64 v98; // rdi
  __int64 v99; // rcx
  __int64 v100; // r15
  __int64 v101; // r13
  __int64 v102; // r8
  __int64 v103; // r9
  __int64 v104; // rax
  __int128 v105; // [rsp-20h] [rbp-140h]
  __int64 v106; // [rsp+0h] [rbp-120h]
  __int16 v107; // [rsp+Eh] [rbp-112h]
  __int64 v108; // [rsp+10h] [rbp-110h]
  int v109; // [rsp+10h] [rbp-110h]
  __int32 v110; // [rsp+10h] [rbp-110h]
  int v111; // [rsp+18h] [rbp-108h]
  char v112; // [rsp+18h] [rbp-108h]
  __int64 v113; // [rsp+18h] [rbp-108h]
  __int64 v114; // [rsp+18h] [rbp-108h]
  __m128i v115; // [rsp+20h] [rbp-100h] BYREF
  __m128i v116; // [rsp+30h] [rbp-F0h] BYREF
  __int128 v117; // [rsp+40h] [rbp-E0h]
  __int64 v118; // [rsp+50h] [rbp-D0h] BYREF
  __int64 v119; // [rsp+58h] [rbp-C8h]
  __int64 v120; // [rsp+60h] [rbp-C0h] BYREF
  __int64 v121; // [rsp+68h] [rbp-B8h]
  __int64 v122; // [rsp+70h] [rbp-B0h] BYREF
  int v123; // [rsp+78h] [rbp-A8h]
  _QWORD v124[2]; // [rsp+80h] [rbp-A0h] BYREF
  __int128 v125; // [rsp+90h] [rbp-90h]
  __m128i v126; // [rsp+A0h] [rbp-80h] BYREF
  __int128 v127; // [rsp+B0h] [rbp-70h] BYREF

  v4 = (__int64 *)a1[1];
  LODWORD(v117) = *(_DWORD *)(a2 + 24);
  v5 = *(_QWORD *)(a2 + 40);
  v6 = *v4;
  v7 = _mm_loadu_si128((const __m128i *)v5);
  v8 = _mm_loadu_si128((const __m128i *)(v5 + 40));
  v9 = *(_QWORD *)(v5 + 40);
  v115 = v7;
  v106 = v9;
  v10 = *(__int16 **)(a2 + 48);
  v116 = v8;
  v11 = *v10;
  v119 = *((_QWORD *)v10 + 1);
  v12 = *a1;
  LOWORD(v118) = v11;
  v107 = v11;
  v13 = *(__int64 (__fastcall **)(__int64 *, __int64, __int64, __int64, __int64))(v6 + 528);
  v108 = *(_QWORD *)(v12 + 64);
  v14 = sub_2E79000(*(__int64 **)(v12 + 40));
  v15 = v13(v4, v14, v108, v118, v119);
  v16 = *(_QWORD *)(a2 + 80);
  LOWORD(v120) = v15;
  v17 = v15;
  v121 = v18;
  v122 = v16;
  if ( v16 )
    sub_B96E90((__int64)&v122, v16, 1);
  v19 = *a1;
  v20 = _mm_load_si128(&v115);
  v21 = _mm_load_si128(&v116);
  v123 = *(_DWORD *)(a2 + 72);
  v126 = v20;
  v127 = (__int128)v21;
  v22 = sub_3402EA0(v19, v117, (unsigned int)&v122, v118, v119, 0, (__int64)&v126, 2);
  if ( v22 )
  {
    v23 = v22;
    goto LABEL_5;
  }
  if ( (_DWORD)v117 != 61 && (unsigned __int8)sub_33E07E0(v116.m128i_i64[0], v116.m128i_i64[1], 0) )
  {
    v25 = v17 ? (unsigned __int16)(v17 - 17) <= 0xD3u : sub_30070B0((__int64)&v120);
    v26 = v107 ? (unsigned __int16)(v107 - 17) <= 0xD3u : sub_30070B0((__int64)&v118);
    if ( v26 == v25 )
    {
      *(_QWORD *)&v27 = sub_33FB960(*a1, v115.m128i_i64[0], v115.m128i_i64[1]);
      v28 = *a1;
      v29 = v120;
      v30 = v121;
      v117 = v27;
      v115.m128i_i64[0] = v28;
      *(_QWORD *)&v31 = sub_33ED040(v28, 17);
      v33 = sub_340F900(v115.m128i_i32[0], 208, (unsigned int)&v122, v29, v30, v32, v117, *(_OWORD *)&v116, v31);
      v34 = *a1;
      v35 = v33;
      v37 = v36;
      v38.m128i_i64[0] = sub_3400BD0(v34, 0, (unsigned int)&v122, v118, v119, 0, 0);
      v39 = v118;
      v40 = v35;
      v116 = v38;
      v41 = v119;
      v38.m128i_i64[0] = v37;
      v42 = *(_QWORD *)(v35 + 48) + 16LL * v37;
      v43 = v38.m128i_i64[0];
      v44 = *(_WORD *)v42;
      v126.m128i_i64[1] = *(_QWORD *)(v42 + 8);
      v126.m128i_i16[0] = v44;
      if ( v44 )
      {
        v45 = ((unsigned __int16)(v44 - 17) < 0xD4u) + 205;
      }
      else
      {
        v109 = v119;
        v111 = v118;
        v115.m128i_i64[1] = v43;
        v115.m128i_i64[0] = v35;
        v78 = sub_30070B0((__int64)&v126);
        v41 = v109;
        v39 = v111;
        v43 = v115.m128i_i64[1];
        v40 = v115.m128i_i64[0];
        v45 = 205 - (!v78 - 1);
      }
      v23 = sub_340EC60(v34, v45, (unsigned int)&v122, v39, v41, 0, v40, v43, *(_OWORD *)&v116, v117);
      goto LABEL_5;
    }
  }
  v46 = sub_3269740(a2, *a1);
  if ( v46 || (v46 = sub_329BF20(a1, a2)) != 0 )
  {
    v23 = v46;
    goto LABEL_5;
  }
  v49 = *a1;
  if ( (_DWORD)v117 == 61 )
  {
    if ( (unsigned __int8)sub_33DD2A0(v49, v116.m128i_i64[0], v116.m128i_i64[1], 0)
      && (unsigned __int8)sub_33DD2A0(*a1, v115.m128i_i64[0], v115.m128i_i64[1], 0) )
    {
      v82 = sub_3406EB0(*a1, 62, (unsigned int)&v122, v118, v119, v81, *(_OWORD *)&v115, *(_OWORD *)&v116);
LABEL_42:
      v23 = v82;
      goto LABEL_5;
    }
  }
  else if ( (unsigned __int8)sub_33E0A10(v49, v116.m128i_i64[0], v116.m128i_i64[1], 0, v47, v48)
         || ((*(_DWORD *)(v106 + 24) - 190) & 0xFFFFFFFD) == 0
         && (unsigned __int8)sub_33E0A10(
                               *a1,
                               **(_QWORD **)(v106 + 40),
                               *(_QWORD *)(*(_QWORD *)(v106 + 40) + 8LL),
                               0,
                               v50,
                               v51) )
  {
    *(_QWORD *)&v83 = sub_34015B0(*a1, &v122, (unsigned int)v118, v119, 0, 0);
    *(_QWORD *)&v117 = sub_3406EB0(*a1, 56, (unsigned int)&v122, v118, v119, v84, *(_OWORD *)&v116, v83);
    *((_QWORD *)&v117 + 1) = v85;
    sub_32B3E80((__int64)a1, v117, 1, 0, v117, v85);
    v82 = sub_3406EB0(*a1, 186, (unsigned int)&v122, v118, v119, DWORD2(v117), *(_OWORD *)&v115, v117);
    goto LABEL_42;
  }
  v52 = *(_QWORD *)(**(_QWORD **)(*a1 + 40) + 120LL);
  if ( !(unsigned __int8)sub_33DE9F0(*a1, v116.m128i_i64[0], v116.m128i_i64[1], 0) )
    goto LABEL_36;
  v53 = a1[1];
  v54 = *(__int64 (**)())(*(_QWORD *)v53 + 200LL);
  if ( v54 != sub_2FE2F30 )
  {
    if ( ((unsigned __int8 (__fastcall *)(__int64, _QWORD, __int64, __int64))v54)(v53, (unsigned int)v118, v119, v52) )
      goto LABEL_36;
  }
  if ( (_DWORD)v117 != 61 )
  {
    v55 = sub_32CA770(a1, v115.m128i_i64[0], v115.m128i_i64[1], v116.m128i_i64[0], v116.m128i_i64[1], a2);
    v56 = v55;
    v58 = v57;
    v59 = v55;
    if ( v55 )
    {
      v60 = 60;
      if ( a2 != v55 )
      {
LABEL_31:
        v61 = *a1;
        v62 = *(_QWORD *)(a2 + 48);
        v63 = _mm_load_si128(&v115);
        v64 = _mm_load_si128(&v116);
        *((_QWORD *)&v117 + 1) = v58;
        v65 = *(unsigned int *)(a2 + 68);
        *(_QWORD *)&v117 = v56;
        v126 = v63;
        v127 = (__int128)v64;
        v66 = sub_33D01C0(v61, v60, v62, v65, &v126, 2);
        v68 = *((_QWORD *)&v117 + 1);
        if ( v66 )
        {
          v126.m128i_i64[1] = *((_QWORD *)&v117 + 1);
          v126.m128i_i64[0] = v59;
          *(_QWORD *)&v117 = v59;
          sub_32EB790((__int64)a1, v66, v126.m128i_i64, 1, 1);
          v68 = *((_QWORD *)&v117 + 1);
        }
        *((_QWORD *)&v105 + 1) = v68;
        *(_QWORD *)&v105 = v59;
        *(_QWORD *)&v69 = sub_3406EB0(*a1, 58, (unsigned int)&v122, v118, v119, v67, v105, *(_OWORD *)&v116);
        v70 = *a1;
        v117 = v69;
        v72 = sub_3406EB0(v70, 57, (unsigned int)&v122, v118, v119, v71, *(_OWORD *)&v115, v69);
        sub_32B3E80((__int64)a1, v59, 1, 0, v73, v74);
        v75 = (__int64)a1;
        v23 = v72;
        sub_32B3E80(v75, v117, 1, 0, v76, v77);
        goto LABEL_5;
      }
    }
LABEL_36:
    v79 = 0;
    v80 = sub_32EC020(a1, a2);
    if ( v80 )
      v79 = v80;
    v23 = v79;
    goto LABEL_5;
  }
  if ( (*(_BYTE *)(a2 + 28) & 4) != 0 )
    goto LABEL_46;
  *((_QWORD *)&v125 + 1) = sub_3260EB0;
  *(_QWORD *)&v125 = sub_325D4D0;
  v110 = v115.m128i_i32[2];
  *(_QWORD *)&v127 = 0;
  v88 = v116.m128i_i32[2];
  *(_QWORD *)&v117 = v124;
  sub_325D4D0(&v126, (__int64)v124, 2);
  v127 = v125;
  v112 = sub_33CA8D0(v106, v89, &v126);
  sub_A17130((__int64)&v126);
  sub_A17130(v117);
  if ( !v112 )
    goto LABEL_46;
  v90 = *a1;
  DWORD2(v127) = v88;
  v91 = *(_QWORD *)(a2 + 48);
  v92 = *(unsigned int *)(a2 + 68);
  v126.m128i_i64[0] = v115.m128i_i64[0];
  v126.m128i_i32[2] = v110;
  *(_QWORD *)&v127 = v106;
  if ( (unsigned __int8)sub_33CEDC0(v90, 59, v91, v92, &v126, 2)
    || (v93 = sub_33DFBC0(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL), 0, 0)) == 0 )
  {
LABEL_46:
    v86 = sub_32CEA50(a1, v115.m128i_i64[0], v115.m128i_i64[1], v116.m128i_i64[0], v116.m128i_i64[1], a2);
    v56 = v86;
    v58 = v87;
    v59 = v86;
    if ( v86 )
    {
      v60 = 59;
      if ( a2 != v86 )
        goto LABEL_31;
    }
    goto LABEL_36;
  }
  v94 = *(_QWORD *)(v93 + 96);
  v95 = *(_DWORD *)(v94 + 32);
  v96 = v94 + 24;
  if ( v95 <= 0x40 )
  {
    if ( !*(_QWORD *)(v94 + 24) )
      goto LABEL_46;
  }
  else
  {
    v113 = v94 + 24;
    v97 = sub_C444A0(v94 + 24);
    v96 = v113;
    if ( v95 == v97 )
      goto LABEL_46;
  }
  v98 = a1[1];
  v99 = *a1;
  v126.m128i_i64[0] = (__int64)&v127;
  v126.m128i_i64[1] = 0x800000000LL;
  v114 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __m128i *))(*(_QWORD *)v98 + 2552LL))(
           v98,
           a2,
           v96,
           v99,
           &v126);
  if ( !v114 )
  {
    if ( (__int128 *)v126.m128i_i64[0] != &v127 )
      _libc_free(v126.m128i_u64[0]);
    goto LABEL_46;
  }
  v100 = v126.m128i_i64[0];
  v116.m128i_i64[0] = v126.m128i_i64[0] + 8LL * v126.m128i_u32[2];
  v115.m128i_i64[0] = (__int64)(a1 + 71);
  while ( v116.m128i_i64[0] != v100 )
  {
    v101 = *(_QWORD *)v100;
    if ( *(_DWORD *)(*(_QWORD *)v100 + 24LL) != 328 )
    {
      v124[0] = *(_QWORD *)v100;
      sub_32B3B20(v115.m128i_i64[0], (__int64 *)v117);
      if ( *(int *)(v101 + 88) < 0 )
      {
        *(_DWORD *)(v101 + 88) = *((_DWORD *)a1 + 12);
        v104 = *((unsigned int *)a1 + 12);
        if ( v104 + 1 > (unsigned __int64)*((unsigned int *)a1 + 13) )
        {
          sub_C8D5F0((__int64)(a1 + 5), a1 + 7, v104 + 1, 8u, v102, v103);
          v104 = *((unsigned int *)a1 + 12);
        }
        *(_QWORD *)(a1[5] + 8 * v104) = v101;
        ++*((_DWORD *)a1 + 12);
      }
    }
    v100 += 8;
  }
  if ( (__int128 *)v126.m128i_i64[0] != &v127 )
    _libc_free(v126.m128i_u64[0]);
  v23 = v114;
LABEL_5:
  if ( v122 )
    sub_B91220((__int64)&v122, v122);
  return v23;
}
