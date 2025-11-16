// Function: sub_12C8DD0
// Address: 0x12c8dd0
//
_BYTE *__fastcall sub_12C8DD0(__int64 a1, char a2)
{
  _BYTE *v2; // rax
  _BYTE *result; // rax
  bool v4; // zf
  _BYTE *v5; // r13
  _QWORD *v6; // r14
  __m128i *v7; // rax
  __m128i *v8; // rax
  __m128i *v9; // rax
  __m128i *v10; // rax
  __m128i *v11; // rax
  __m128i *v12; // rax
  __m128i *v13; // rax
  __m128i *v14; // rax
  __m128i *v15; // rax
  __m128i *v16; // rax
  __m128i *v17; // rax
  __m128i *v18; // rax
  __m128i *v19; // rax
  __m128i *v20; // rax
  __m128i *v21; // rax
  __m128i *v22; // rax
  __m128i *v23; // rax
  __m128i *v24; // rax
  __m128i *v25; // rax
  __m128i *v26; // rax
  __m128i *v27; // rax
  __m128i *v28; // rax
  __m128i *v29; // rax
  __m128i *v30; // rax
  __m128i *v31; // rax
  __m128i *v32; // rax
  __m128i *v33; // rax
  __m128i *v34; // rax
  __m128i *v35; // rax
  __m128i *v36; // rax
  __m128i *v37; // rax
  __m128i *v38; // rax
  __m128i *v39; // rax
  __m128i *v40; // rax
  __m128i *v41; // rax
  __m128i *v42; // rax
  __m128i *v43; // rax
  __m128i *v44; // rax
  __m128i *v45; // rax
  const char *v46; // rsi
  __m128i *v47; // rax
  __m128i *v48; // rax
  __m128i *v49; // rax
  __m128i *v50; // rax
  __m128i *v51; // rax
  __m128i *v52; // rax
  __m128i *v53; // rax
  __m128i *v54; // rax
  __m128i *v55; // rax
  __m128i *v56; // rax
  __m128i *v57; // rax
  __m128i *v58; // rax
  __m128i *v59; // rax
  __m128i *v60; // rax
  __m128i *v61; // rax
  __m128i *v62; // rax
  __m128i *v63; // rax
  __m128i *v64; // rax
  __m128i *v65; // rax
  __m128i *v66; // rax
  __m128i *v67; // rax
  __m128i *v68; // rax
  __m128i *v69; // rax
  __m128i *v70; // rax
  __m128i *v71; // rax
  unsigned __int64 v72; // rax
  unsigned __int64 v73; // rdi
  unsigned __int64 v74; // rcx
  __m128i *v75; // rax
  __m128i *v76; // rdx
  __m128i *v77; // rcx
  __int64 v78; // r12
  __int64 v79; // r13
  size_t v80; // r15
  size_t v81; // rdx
  int v82; // eax
  __int64 v83; // r15
  __int64 v84; // r15
  __int64 v85; // r13
  __int64 v86; // rax
  __int64 v87; // rdx
  __int64 v88; // r14
  __int64 v89; // rdi
  __m128i *v90; // rax
  __int64 v92; // [rsp+8h] [rbp-188h]
  __int64 v93; // [rsp+10h] [rbp-180h]
  _QWORD *v95; // [rsp+38h] [rbp-158h]
  __int64 v96; // [rsp+38h] [rbp-158h]
  __m128i *s2; // [rsp+40h] [rbp-150h]
  size_t v98; // [rsp+48h] [rbp-148h]
  __m128i v99; // [rsp+50h] [rbp-140h] BYREF
  _QWORD *v100; // [rsp+60h] [rbp-130h] BYREF
  __int64 v101; // [rsp+68h] [rbp-128h]
  _QWORD v102[2]; // [rsp+70h] [rbp-120h] BYREF
  _BYTE *v103[2]; // [rsp+80h] [rbp-110h] BYREF
  char v104; // [rsp+90h] [rbp-100h] BYREF
  char *v105; // [rsp+A0h] [rbp-F0h] BYREF
  __int64 v106; // [rsp+A8h] [rbp-E8h]
  char v107; // [rsp+B0h] [rbp-E0h] BYREF
  char *v108; // [rsp+C0h] [rbp-D0h] BYREF
  __int64 v109; // [rsp+C8h] [rbp-C8h]
  char v110; // [rsp+D0h] [rbp-C0h] BYREF
  __int64 v111; // [rsp+E0h] [rbp-B0h]
  _QWORD *v112; // [rsp+F0h] [rbp-A0h] BYREF
  __int64 v113; // [rsp+F8h] [rbp-98h]
  _QWORD v114[2]; // [rsp+100h] [rbp-90h] BYREF
  _BYTE *v115; // [rsp+110h] [rbp-80h] BYREF
  __int64 v116; // [rsp+118h] [rbp-78h]
  _BYTE v117[16]; // [rsp+120h] [rbp-70h] BYREF
  _BYTE *v118; // [rsp+130h] [rbp-60h] BYREF
  __int64 v119; // [rsp+138h] [rbp-58h]
  _BYTE v120[16]; // [rsp+140h] [rbp-50h] BYREF
  __int64 v121; // [rsp+150h] [rbp-40h]

  v2 = *(_BYTE **)(a1 + 1200);
  *(_QWORD *)(a1 + 1208) = 0;
  *v2 = 0;
  result = *(_BYTE **)(a1 + 1488);
  *(_QWORD *)(a1 + 1496) = 0;
  *result = 0;
  v4 = *(_QWORD *)(a1 + 296) == 0;
  *(_BYTE *)(a1 + 1652) = 0;
  v92 = a1 + 264;
  if ( !v4 )
  {
    v5 = *(_BYTE **)(a1 + 280);
    if ( v5 != (_BYTE *)(a1 + 264) )
    {
      do
      {
        v112 = v114;
        v113 = 0;
        LOBYTE(v114[0]) = 0;
        sub_2240AE0(*((_QWORD *)v5 + 20), &v112);
        if ( v112 != v114 )
          j_j___libc_free_0(v112, v114[0] + 1LL);
        result = (_BYTE *)sub_220EEE0(v5);
        v5 = result;
      }
      while ( result != (_BYTE *)(a1 + 264) );
    }
    return result;
  }
  v6 = (_QWORD *)(a1 + 256);
  v103[1] = 0;
  v104 = 0;
  v103[0] = &v104;
  v105 = &v107;
  v111 = a1 + 304;
  v106 = 0;
  v107 = 0;
  v108 = &v110;
  v109 = 0;
  v110 = 0;
  sub_2241130(&v105, 0, 0, "-debug-compile", 14);
  sub_2241130(&v108, 0, v109, "-debug-compile", 14);
  v112 = v114;
  sub_12C6440((__int64 *)&v112, v103[0], (__int64)v103[0]);
  v115 = v117;
  sub_12C6440((__int64 *)&v115, v105, (__int64)&v105[v106]);
  v118 = v120;
  sub_12C6440((__int64 *)&v118, v108, (__int64)&v108[v109]);
  v121 = v111;
  sub_12C7330(&v112);
  v121 = a1 + 304;
  v118 = v120;
  v112 = v114;
  v113 = 0;
  LOBYTE(v114[0]) = 0;
  v115 = v117;
  v116 = 0;
  v117[0] = 0;
  v119 = 0;
  v120[0] = 0;
  sub_2241130(&v115, 0, 0, "-debug-compile", 14);
  sub_2241130(&v118, 0, v119, "-debug-compile", 14);
  sub_12C65C0((__int64 *)&v100, "-g");
  v7 = sub_12C8B40((_QWORD *)(a1 + 256), (__int64)&v100);
  sub_12C6670((__int64)v7, (__int64)&v112);
  if ( v100 != v102 )
    j_j___libc_free_0(v100, v102[0] + 1LL);
  sub_12C7330(&v112);
  v112 = v114;
  v115 = v117;
  v113 = 0;
  v118 = v120;
  LOBYTE(v114[0]) = 0;
  v116 = 0;
  v117[0] = 0;
  v119 = 0;
  v120[0] = 0;
  v121 = a1 + 336;
  sub_2241130(&v118, 0, 0, "-generate-line-info", 19);
  sub_12C65C0((__int64 *)&v100, "-generate-line-info");
  v8 = sub_12C8B40(v6, (__int64)&v100);
  sub_12C6670((__int64)v8, (__int64)&v112);
  if ( v100 != v102 )
    j_j___libc_free_0(v100, v102[0] + 1LL);
  sub_12C7330(&v112);
  v112 = v114;
  v115 = v117;
  v113 = 0;
  v118 = v120;
  LOBYTE(v114[0]) = 0;
  v116 = 0;
  v117[0] = 0;
  v119 = 0;
  v120[0] = 0;
  v121 = a1 + 368;
  sub_2241130(&v118, 0, 0, "-line-info-inlined-at=0", 23);
  sub_12C65C0((__int64 *)&v100, "-no-lineinfo-inlined-at");
  v9 = sub_12C8B40(v6, (__int64)&v100);
  sub_12C6670((__int64)v9, (__int64)&v112);
  if ( v100 != v102 )
    j_j___libc_free_0(v100, v102[0] + 1LL);
  sub_12C7330(&v112);
  v112 = v114;
  v115 = v117;
  v113 = 0;
  v118 = v120;
  LOBYTE(v114[0]) = 0;
  v116 = 0;
  v117[0] = 0;
  v119 = 0;
  v120[0] = 0;
  v121 = a1 + 432;
  sub_2241130(&v112, 0, 0, "-lnk-disable-allopts", 20);
  sub_2241130(&v115, 0, v116, "-opt-disable-allopts", 20);
  sub_2241130(&v118, 0, v119, "-llc-disable-allopts", 20);
  sub_12C65C0((__int64 *)&v100, "-disable-allopts");
  v10 = sub_12C8B40(v6, (__int64)&v100);
  sub_12C6670((__int64)v10, (__int64)&v112);
  if ( v100 != v102 )
    j_j___libc_free_0(v100, v102[0] + 1LL);
  sub_12C7330(&v112);
  v117[0] = 0;
  v120[0] = 0;
  v115 = v117;
  v112 = v114;
  v118 = v120;
  v121 = a1 + 400;
  v113 = 0;
  LOBYTE(v114[0]) = 0;
  v116 = 0;
  v119 = 0;
  sub_12C65C0((__int64 *)&v100, "-opt=0");
  v11 = sub_12C8B40(v6, (__int64)&v100);
  sub_12C6670((__int64)v11, (__int64)&v112);
  if ( v100 != v102 )
    j_j___libc_free_0(v100, v102[0] + 1LL);
  sub_12C7330(&v112);
  v112 = v114;
  v115 = v117;
  v113 = 0;
  v118 = v120;
  LOBYTE(v114[0]) = 0;
  v121 = a1 + 400;
  v116 = 0;
  v117[0] = 0;
  v119 = 0;
  v120[0] = 0;
  sub_12C65C0((__int64 *)&v100, "-opt=1");
  v12 = sub_12C8B40(v6, (__int64)&v100);
  sub_12C6670((__int64)v12, (__int64)&v112);
  if ( v100 != v102 )
    j_j___libc_free_0(v100, v102[0] + 1LL);
  sub_12C7330(&v112);
  v112 = v114;
  v115 = v117;
  v113 = 0;
  v118 = v120;
  LOBYTE(v114[0]) = 0;
  v121 = a1 + 400;
  v116 = 0;
  v117[0] = 0;
  v119 = 0;
  v120[0] = 0;
  sub_12C65C0((__int64 *)&v100, "-opt=2");
  v13 = sub_12C8B40(v6, (__int64)&v100);
  sub_12C6670((__int64)v13, (__int64)&v112);
  if ( v100 != v102 )
    j_j___libc_free_0(v100, v102[0] + 1LL);
  sub_12C7330(&v112);
  v112 = v114;
  v115 = v117;
  v113 = 0;
  v118 = v120;
  LOBYTE(v114[0]) = 0;
  v121 = a1 + 400;
  v116 = 0;
  v117[0] = 0;
  v119 = 0;
  v120[0] = 0;
  sub_12C65C0((__int64 *)&v100, "-opt=3");
  v14 = sub_12C8B40(v6, (__int64)&v100);
  sub_12C6670((__int64)v14, (__int64)&v112);
  if ( v100 != v102 )
    j_j___libc_free_0(v100, v102[0] + 1LL);
  sub_12C7330(&v112);
  v112 = v114;
  v115 = v117;
  v113 = 0;
  v118 = v120;
  LOBYTE(v114[0]) = 0;
  v116 = 0;
  v117[0] = 0;
  v119 = 0;
  v120[0] = 0;
  v121 = a1 + 496;
  sub_2241130(&v115, 0, 0, "-Osize", 6);
  sub_2241130(&v118, 0, v119, "-Osize", 6);
  sub_12C65C0((__int64 *)&v100, "-Osize");
  v15 = sub_12C8B40(v6, (__int64)&v100);
  sub_12C6670((__int64)v15, (__int64)&v112);
  if ( v100 != v102 )
    j_j___libc_free_0(v100, v102[0] + 1LL);
  sub_12C7330(&v112);
  v112 = v114;
  v115 = v117;
  v113 = 0;
  v118 = v120;
  LOBYTE(v114[0]) = 0;
  v116 = 0;
  v117[0] = 0;
  v119 = 0;
  v120[0] = 0;
  v121 = a1 + 528;
  sub_2241130(&v115, 0, 0, "-Om", 3);
  sub_2241130(&v118, 0, v119, "-Om", 3);
  sub_12C65C0((__int64 *)&v100, "-Om");
  v16 = sub_12C8B40(v6, (__int64)&v100);
  sub_12C6670((__int64)v16, (__int64)&v112);
  if ( v100 != v102 )
    j_j___libc_free_0(v100, v102[0] + 1LL);
  sub_12C7330(&v112);
  v112 = v114;
  v115 = v117;
  LOBYTE(v114[0]) = 0;
  v118 = v120;
  v93 = a1 + 560;
  v113 = 0;
  v116 = 0;
  v117[0] = 0;
  v119 = 0;
  v120[0] = 0;
  v121 = a1 + 560;
  sub_2241130(&v112, 0, 0, "-R __CUDA_ARCH=750", 18);
  sub_2241130(&v115, 0, v116, "-opt-arch=sm_75", 15);
  sub_2241130(&v118, 0, v119, "-mcpu=sm_75", 11);
  sub_12C65C0((__int64 *)&v100, "-arch=compute_75");
  v17 = sub_12C8B40(v6, (__int64)&v100);
  sub_12C6670((__int64)v17, (__int64)&v112);
  if ( v100 != v102 )
    j_j___libc_free_0(v100, v102[0] + 1LL);
  sub_12C7330(&v112);
  v112 = v114;
  v115 = v117;
  v113 = 0;
  v118 = v120;
  LOBYTE(v114[0]) = 0;
  v116 = 0;
  v117[0] = 0;
  v119 = 0;
  v120[0] = 0;
  v121 = a1 + 560;
  sub_2241130(&v112, 0, 0, "-R __CUDA_ARCH=800", 18);
  sub_2241130(&v115, 0, v116, "-opt-arch=sm_80", 15);
  sub_2241130(&v118, 0, v119, "-mcpu=sm_80", 11);
  sub_12C65C0((__int64 *)&v100, "-arch=compute_80");
  v18 = sub_12C8B40(v6, (__int64)&v100);
  sub_12C6670((__int64)v18, (__int64)&v112);
  if ( v100 != v102 )
    j_j___libc_free_0(v100, v102[0] + 1LL);
  sub_12C7330(&v112);
  sub_12C7250((__int64)&v112, "-R __CUDA_ARCH=860", "-opt-arch=sm_86", "-mcpu=sm_86", (__int64)"-mcpu=sm_86", v93);
  sub_12C65C0((__int64 *)&v100, "-arch=compute_86");
  v19 = sub_12C8B40(v6, (__int64)&v100);
  sub_12C6670((__int64)v19, (__int64)&v112);
  if ( v100 != v102 )
    j_j___libc_free_0(v100, v102[0] + 1LL);
  sub_12C7330(&v112);
  sub_12C7250((__int64)&v112, "-R __CUDA_ARCH=870", "-opt-arch=sm_87", "-mcpu=sm_87", (__int64)"-mcpu=sm_87", v93);
  sub_12C65C0((__int64 *)&v100, "-arch=compute_87");
  v20 = sub_12C8B40(v6, (__int64)&v100);
  sub_12C6670((__int64)v20, (__int64)&v112);
  if ( v100 != v102 )
    j_j___libc_free_0(v100, v102[0] + 1LL);
  sub_12C7330(&v112);
  sub_12C7250((__int64)&v112, "-R __CUDA_ARCH=880", "-opt-arch=sm_88", "-mcpu=sm_88", (__int64)"-mcpu=sm_88", v93);
  sub_12C65C0((__int64 *)&v100, "-arch=compute_88");
  v21 = sub_12C8B40(v6, (__int64)&v100);
  sub_12C6670((__int64)v21, (__int64)&v112);
  if ( v100 != v102 )
    j_j___libc_free_0(v100, v102[0] + 1LL);
  sub_12C7330(&v112);
  v112 = v114;
  v115 = v117;
  v113 = 0;
  v118 = v120;
  LOBYTE(v114[0]) = 0;
  v116 = 0;
  v117[0] = 0;
  v119 = 0;
  v120[0] = 0;
  v121 = a1 + 560;
  sub_2241130(&v112, 0, 0, "-R __CUDA_ARCH=890", 18);
  sub_2241130(&v115, 0, v116, "-opt-arch=sm_89", 15);
  sub_2241130(&v118, 0, v119, "-mcpu=sm_89", 11);
  sub_12C65C0((__int64 *)&v100, "-arch=compute_89");
  v22 = sub_12C8B40(v6, (__int64)&v100);
  sub_12C6670((__int64)v22, (__int64)&v112);
  if ( v100 != v102 )
    j_j___libc_free_0(v100, v102[0] + 1LL);
  sub_12C7330(&v112);
  v112 = v114;
  v115 = v117;
  v113 = 0;
  v118 = v120;
  LOBYTE(v114[0]) = 0;
  v116 = 0;
  v117[0] = 0;
  v119 = 0;
  v120[0] = 0;
  v121 = a1 + 560;
  sub_2241130(&v112, 0, 0, "-R __CUDA_ARCH=900", 18);
  sub_2241130(&v115, 0, v116, "-opt-arch=sm_90", 15);
  sub_2241130(&v118, 0, v119, "-mcpu=sm_90", 11);
  sub_12C65C0((__int64 *)&v100, "-arch=compute_90");
  v23 = sub_12C8B40(v6, (__int64)&v100);
  sub_12C6670((__int64)v23, (__int64)&v112);
  if ( v100 != v102 )
    j_j___libc_free_0(v100, v102[0] + 1LL);
  sub_12C7330(&v112);
  v112 = v114;
  v115 = v117;
  v113 = 0;
  v118 = v120;
  LOBYTE(v114[0]) = 0;
  v116 = 0;
  v117[0] = 0;
  v119 = 0;
  v120[0] = 0;
  v121 = a1 + 560;
  sub_2241130(&v112, 0, 0, "-R __CUDA_ARCH=900", 18);
  sub_2241130(&v115, 0, v116, "-opt-arch=sm_90a", 16);
  sub_2241130(&v118, 0, v119, "-mcpu=sm_90a", 12);
  sub_12C65C0((__int64 *)&v100, "-arch=compute_90a");
  v24 = sub_12C8B40(v6, (__int64)&v100);
  sub_12C6670((__int64)v24, (__int64)&v112);
  if ( v100 != v102 )
    j_j___libc_free_0(v100, v102[0] + 1LL);
  sub_12C7330(&v112);
  v112 = v114;
  v115 = v117;
  v113 = 0;
  v118 = v120;
  LOBYTE(v114[0]) = 0;
  v116 = 0;
  v117[0] = 0;
  v119 = 0;
  v120[0] = 0;
  v121 = a1 + 560;
  sub_2241130(&v112, 0, 0, "-R __CUDA_ARCH=1000", 19);
  sub_2241130(&v115, 0, v116, "-opt-arch=sm_100", 16);
  sub_2241130(&v118, 0, v119, "-mcpu=sm_100", 12);
  sub_12C65C0((__int64 *)&v100, "-arch=compute_100");
  v25 = sub_12C8B40(v6, (__int64)&v100);
  sub_12C6670((__int64)v25, (__int64)&v112);
  if ( v100 != v102 )
    j_j___libc_free_0(v100, v102[0] + 1LL);
  sub_12C7330(&v112);
  sub_12C7250(
    (__int64)&v112,
    "-R __CUDA_ARCH=1000",
    "-opt-arch=sm_100a",
    "-mcpu=sm_100a",
    (__int64)"-mcpu=sm_100a",
    v93);
  sub_12C65C0((__int64 *)&v100, "-arch=compute_100a");
  v26 = sub_12C8B40(v6, (__int64)&v100);
  sub_12C6670((__int64)v26, (__int64)&v112);
  if ( v100 != v102 )
    j_j___libc_free_0(v100, v102[0] + 1LL);
  sub_12C7330(&v112);
  sub_12C7250(
    (__int64)&v112,
    "-R __CUDA_ARCH=1000",
    "-opt-arch=sm_100f",
    "-mcpu=sm_100f",
    (__int64)"-mcpu=sm_100f",
    v93);
  sub_12C65C0((__int64 *)&v100, "-arch=compute_100f");
  v27 = sub_12C8B40(v6, (__int64)&v100);
  sub_12C6670((__int64)v27, (__int64)&v112);
  if ( v100 != v102 )
    j_j___libc_free_0(v100, v102[0] + 1LL);
  sub_12C7330(&v112);
  v112 = v114;
  v115 = v117;
  v113 = 0;
  v118 = v120;
  LOBYTE(v114[0]) = 0;
  v116 = 0;
  v117[0] = 0;
  v119 = 0;
  v120[0] = 0;
  v121 = a1 + 560;
  sub_2241130(&v112, 0, 0, "-R __CUDA_ARCH=1030", 19);
  sub_2241130(&v115, 0, v116, "-opt-arch=sm_103", 16);
  sub_2241130(&v118, 0, v119, "-mcpu=sm_103", 12);
  sub_12C65C0((__int64 *)&v100, "-arch=compute_103");
  v28 = sub_12C8B40(v6, (__int64)&v100);
  sub_12C6670((__int64)v28, (__int64)&v112);
  if ( v100 != v102 )
    j_j___libc_free_0(v100, v102[0] + 1LL);
  sub_12C7330(&v112);
  v112 = v114;
  v115 = v117;
  v113 = 0;
  v118 = v120;
  LOBYTE(v114[0]) = 0;
  v116 = 0;
  v117[0] = 0;
  v119 = 0;
  v120[0] = 0;
  v121 = a1 + 560;
  sub_2241130(&v112, 0, 0, "-R __CUDA_ARCH=1030", 19);
  sub_2241130(&v115, 0, v116, "-opt-arch=sm_103a", 17);
  sub_2241130(&v118, 0, v119, "-mcpu=sm_103a", 13);
  sub_12C65C0((__int64 *)&v100, "-arch=compute_103a");
  v29 = sub_12C8B40(v6, (__int64)&v100);
  sub_12C6670((__int64)v29, (__int64)&v112);
  if ( v100 != v102 )
    j_j___libc_free_0(v100, v102[0] + 1LL);
  sub_12C7330(&v112);
  v112 = v114;
  v115 = v117;
  v113 = 0;
  v118 = v120;
  LOBYTE(v114[0]) = 0;
  v116 = 0;
  v117[0] = 0;
  v119 = 0;
  v120[0] = 0;
  v121 = a1 + 560;
  sub_2241130(&v112, 0, 0, "-R __CUDA_ARCH=1030", 19);
  sub_2241130(&v115, 0, v116, "-opt-arch=sm_103f", 17);
  sub_2241130(&v118, 0, v119, "-mcpu=sm_103f", 13);
  sub_12C65C0((__int64 *)&v100, "-arch=compute_103f");
  v30 = sub_12C8B40(v6, (__int64)&v100);
  sub_12C6670((__int64)v30, (__int64)&v112);
  if ( v100 != v102 )
    j_j___libc_free_0(v100, v102[0] + 1LL);
  sub_12C7330(&v112);
  v112 = v114;
  v115 = v117;
  v113 = 0;
  v118 = v120;
  LOBYTE(v114[0]) = 0;
  v116 = 0;
  v117[0] = 0;
  v119 = 0;
  v120[0] = 0;
  v121 = a1 + 560;
  sub_2241130(&v112, 0, 0, "-R __CUDA_ARCH=1100", 19);
  sub_2241130(&v115, 0, v116, "-opt-arch=sm_110", 16);
  sub_2241130(&v118, 0, v119, "-mcpu=sm_110", 12);
  sub_12C65C0((__int64 *)&v100, "-arch=compute_110");
  v31 = sub_12C8B40(v6, (__int64)&v100);
  sub_12C6670((__int64)v31, (__int64)&v112);
  if ( v100 != v102 )
    j_j___libc_free_0(v100, v102[0] + 1LL);
  sub_12C7330(&v112);
  v112 = v114;
  v115 = v117;
  v113 = 0;
  v118 = v120;
  LOBYTE(v114[0]) = 0;
  v116 = 0;
  v117[0] = 0;
  v119 = 0;
  v120[0] = 0;
  v121 = a1 + 560;
  sub_2241130(&v112, 0, 0, "-R __CUDA_ARCH=1100", 19);
  sub_2241130(&v115, 0, v116, "-opt-arch=sm_110a", 17);
  sub_2241130(&v118, 0, v119, "-mcpu=sm_110a", 13);
  sub_12C65C0((__int64 *)&v100, "-arch=compute_110a");
  v32 = sub_12C8B40(v6, (__int64)&v100);
  sub_12C6670((__int64)v32, (__int64)&v112);
  if ( v100 != v102 )
    j_j___libc_free_0(v100, v102[0] + 1LL);
  sub_12C7330(&v112);
  v112 = v114;
  v115 = v117;
  v113 = 0;
  v118 = v120;
  LOBYTE(v114[0]) = 0;
  v116 = 0;
  v117[0] = 0;
  v119 = 0;
  v120[0] = 0;
  v121 = a1 + 560;
  sub_2241130(&v112, 0, 0, "-R __CUDA_ARCH=1100", 19);
  sub_2241130(&v115, 0, v116, "-opt-arch=sm_110f", 17);
  sub_2241130(&v118, 0, v119, "-mcpu=sm_110f", 13);
  sub_12C65C0((__int64 *)&v100, "-arch=compute_110f");
  v33 = sub_12C8B40(v6, (__int64)&v100);
  sub_12C6670((__int64)v33, (__int64)&v112);
  if ( v100 != v102 )
    j_j___libc_free_0(v100, v102[0] + 1LL);
  sub_12C7330(&v112);
  v112 = v114;
  v115 = v117;
  v113 = 0;
  v118 = v120;
  LOBYTE(v114[0]) = 0;
  v116 = 0;
  v117[0] = 0;
  v119 = 0;
  v120[0] = 0;
  v121 = a1 + 560;
  sub_2241130(&v112, 0, 0, "-R __CUDA_ARCH=1200", 19);
  sub_2241130(&v115, 0, v116, "-opt-arch=sm_120", 16);
  sub_2241130(&v118, 0, v119, "-mcpu=sm_120", 12);
  sub_12C65C0((__int64 *)&v100, "-arch=compute_120");
  v34 = sub_12C8B40(v6, (__int64)&v100);
  sub_12C6670((__int64)v34, (__int64)&v112);
  if ( v100 != v102 )
    j_j___libc_free_0(v100, v102[0] + 1LL);
  sub_12C7330(&v112);
  v112 = v114;
  v115 = v117;
  v113 = 0;
  v118 = v120;
  LOBYTE(v114[0]) = 0;
  v116 = 0;
  v117[0] = 0;
  v119 = 0;
  v120[0] = 0;
  v121 = a1 + 560;
  sub_2241130(&v112, 0, 0, "-R __CUDA_ARCH=1200", 19);
  sub_2241130(&v115, 0, v116, "-opt-arch=sm_120a", 17);
  sub_2241130(&v118, 0, v119, "-mcpu=sm_120a", 13);
  sub_12C65C0((__int64 *)&v100, "-arch=compute_120a");
  v35 = sub_12C8B40(v6, (__int64)&v100);
  sub_12C6670((__int64)v35, (__int64)&v112);
  if ( v100 != v102 )
    j_j___libc_free_0(v100, v102[0] + 1LL);
  sub_12C7330(&v112);
  sub_12C7250(
    (__int64)&v112,
    "-R __CUDA_ARCH=1200",
    "-opt-arch=sm_120f",
    "-mcpu=sm_120f",
    (__int64)"-mcpu=sm_120f",
    v93);
  sub_12C65C0((__int64 *)&v100, "-arch=compute_120f");
  v36 = sub_12C8B40(v6, (__int64)&v100);
  sub_12C6670((__int64)v36, (__int64)&v112);
  if ( v100 != v102 )
    j_j___libc_free_0(v100, v102[0] + 1LL);
  sub_12C7330(&v112);
  v112 = v114;
  v115 = v117;
  v113 = 0;
  v118 = v120;
  LOBYTE(v114[0]) = 0;
  v116 = 0;
  v117[0] = 0;
  v119 = 0;
  v120[0] = 0;
  v121 = a1 + 560;
  sub_2241130(&v112, 0, 0, "-R __CUDA_ARCH=1210", 19);
  sub_2241130(&v115, 0, v116, "-opt-arch=sm_121", 16);
  sub_2241130(&v118, 0, v119, "-mcpu=sm_121", 12);
  sub_12C65C0((__int64 *)&v100, "-arch=compute_121");
  v37 = sub_12C8B40(v6, (__int64)&v100);
  sub_12C6670((__int64)v37, (__int64)&v112);
  if ( v100 != v102 )
    j_j___libc_free_0(v100, v102[0] + 1LL);
  sub_12C7330(&v112);
  v112 = v114;
  v115 = v117;
  v113 = 0;
  v118 = v120;
  LOBYTE(v114[0]) = 0;
  v116 = 0;
  v117[0] = 0;
  v119 = 0;
  v120[0] = 0;
  v121 = a1 + 560;
  sub_2241130(&v112, 0, 0, "-R __CUDA_ARCH=1210", 19);
  sub_2241130(&v115, 0, v116, "-opt-arch=sm_121a", 17);
  sub_2241130(&v118, 0, v119, "-mcpu=sm_121a", 13);
  sub_12C65C0((__int64 *)&v100, "-arch=compute_121a");
  v38 = sub_12C8B40(v6, (__int64)&v100);
  sub_12C6670((__int64)v38, (__int64)&v112);
  if ( v100 != v102 )
    j_j___libc_free_0(v100, v102[0] + 1LL);
  sub_12C7330(&v112);
  sub_12C7250(
    (__int64)&v112,
    "-R __CUDA_ARCH=1210",
    "-opt-arch=sm_121f",
    "-mcpu=sm_121f",
    (__int64)"-mcpu=sm_121f",
    v93);
  sub_12C65C0((__int64 *)&v100, "-arch=compute_121f");
  v39 = sub_12C8B40(v6, (__int64)&v100);
  sub_12C6670((__int64)v39, (__int64)&v112);
  if ( v100 != v102 )
    j_j___libc_free_0(v100, v102[0] + 1LL);
  sub_12C7330(&v112);
  v117[0] = 0;
  v120[0] = 0;
  v115 = v117;
  v112 = v114;
  v118 = v120;
  v121 = a1 + 592;
  v113 = 0;
  LOBYTE(v114[0]) = 0;
  v116 = 0;
  v119 = 0;
  sub_12C65C0((__int64 *)&v100, "-ftz=0");
  v40 = sub_12C8B40(v6, (__int64)&v100);
  sub_12C6670((__int64)v40, (__int64)&v112);
  if ( v100 != v102 )
    j_j___libc_free_0(v100, v102[0] + 1LL);
  sub_12C7330(&v112);
  v112 = v114;
  v115 = v117;
  v113 = 0;
  v118 = v120;
  LOBYTE(v114[0]) = 0;
  v116 = 0;
  v117[0] = 0;
  v119 = 0;
  v120[0] = 0;
  v121 = a1 + 592;
  sub_2241130(&v112, 0, 0, "-R __CUDA_FTZ=1", 15);
  sub_2241130(&v115, 0, v116, "-nvptx-f32ftz", 13);
  sub_2241130(&v118, 0, v119, "-nvptx-f32ftz", 13);
  sub_12C65C0((__int64 *)&v100, "-ftz=1");
  v41 = sub_12C8B40(v6, (__int64)&v100);
  sub_12C6670((__int64)v41, (__int64)&v112);
  if ( v100 != v102 )
    j_j___libc_free_0(v100, v102[0] + 1LL);
  sub_12C7330(&v112);
  v112 = v114;
  v115 = v117;
  LOBYTE(v114[0]) = 0;
  v118 = v120;
  v113 = 0;
  v116 = 0;
  v117[0] = 0;
  v119 = 0;
  v120[0] = 0;
  v121 = a1 + 624;
  sub_2241130(&v118, 0, 0, "-nvptx-prec-sqrtf32=0", 21);
  sub_12C65C0((__int64 *)&v100, "-prec-sqrt=0");
  v42 = sub_12C8B40(v6, (__int64)&v100);
  sub_12C6670((__int64)v42, (__int64)&v112);
  if ( v100 != v102 )
    j_j___libc_free_0(v100, v102[0] + 1LL);
  sub_12C7330(&v112);
  v112 = v114;
  v115 = v117;
  v113 = 0;
  v118 = v120;
  LOBYTE(v114[0]) = 0;
  v116 = 0;
  v117[0] = 0;
  v119 = 0;
  v120[0] = 0;
  v121 = a1 + 624;
  sub_2241130(&v112, 0, 0, "-R __CUDA_PREC_SQRT=1", 21);
  sub_2241130(&v118, 0, v119, "-nvptx-prec-sqrtf32=1", 21);
  sub_12C65C0((__int64 *)&v100, "-prec-sqrt=1");
  v43 = sub_12C8B40(v6, (__int64)&v100);
  sub_12C6670((__int64)v43, (__int64)&v112);
  if ( v100 != v102 )
    j_j___libc_free_0(v100, v102[0] + 1LL);
  sub_12C7330(&v112);
  v112 = v114;
  v113 = 0;
  LOBYTE(v114[0]) = 0;
  if ( a2 )
  {
    v117[0] = 0;
    v120[0] = 0;
    v115 = v117;
    v116 = 0;
    v118 = v120;
    v119 = 0;
    v121 = a1 + 656;
    sub_2241130(&v115, 0, 0, "-opt-use-prec-div=false", 23);
    sub_2241130(&v118, 0, v119, "-nvptx-prec-divf32=0", 20);
    sub_12C65C0((__int64 *)&v100, "-prec-div=0");
    v44 = sub_12C8B40(v6, (__int64)&v100);
    sub_12C6670((__int64)v44, (__int64)&v112);
    if ( v100 != v102 )
      j_j___libc_free_0(v100, v102[0] + 1LL);
    sub_12C7330(&v112);
    v112 = v114;
    v118 = v120;
    v121 = a1 + 656;
    v113 = 0;
    LOBYTE(v114[0]) = 0;
    v115 = v117;
    v116 = 0;
    v117[0] = 0;
    v119 = 0;
    v120[0] = 0;
    sub_2241130(&v115, 0, 0, "-opt-use-prec-div=true", 22);
    sub_2241130(&v118, 0, v119, "-nvptx-prec-divf32=1", 20);
    sub_12C65C0((__int64 *)&v100, "-prec-div=1");
    v45 = sub_12C8B40(v6, (__int64)&v100);
    sub_12C6670((__int64)v45, (__int64)&v112);
    sub_2240A30(&v100);
    sub_12C7330(&v112);
    v112 = v114;
    LOBYTE(v114[0]) = 0;
    v118 = v120;
    v121 = a1 + 656;
    v113 = 0;
    v115 = v117;
    v116 = 0;
    v117[0] = 0;
    v119 = 0;
    v120[0] = 0;
    sub_2241130(&v118, 0, 0, "-nvptx-prec-divf32=3", 20);
    v46 = "-prec-div=2";
  }
  else
  {
    v117[0] = 0;
    v116 = 0;
    v115 = v117;
    v119 = 0;
    v118 = v120;
    v120[0] = 0;
    v121 = a1 + 656;
    sub_2241130(&v115, 0, 0, "-opt-use-prec-div=false", 23);
    sub_2241130(&v118, 0, v119, "-nvptx-prec-divf32=1", 20);
    sub_12C65C0((__int64 *)&v100, "-prec-div=0");
    v90 = sub_12C8B40(v6, (__int64)&v100);
    sub_12C6670((__int64)v90, (__int64)&v112);
    sub_2240A30(&v100);
    sub_12C7330(&v112);
    sub_12C7170((__int64)&v112, "-R __CUDA_PREC_DIV=1", "-opt-use-prec-div=true", "-nvptx-prec-divf32=2", a1 + 656);
    v46 = "-prec-div=1";
  }
  sub_12C65C0((__int64 *)&v100, v46);
  v47 = sub_12C8B40(v6, (__int64)&v100);
  sub_12C6670((__int64)v47, (__int64)&v112);
  sub_2240A30(&v100);
  sub_12C7330(&v112);
  v112 = v114;
  v115 = v117;
  LOBYTE(v114[0]) = 0;
  v118 = v120;
  v113 = 0;
  v116 = 0;
  v117[0] = 0;
  v119 = 0;
  v120[0] = 0;
  v121 = a1 + 688;
  sub_2241130(&v118, 0, 0, "-nvptx-fma-level=0", 18);
  sub_12C65C0((__int64 *)&v100, "-fma=0");
  v48 = sub_12C8B40(v6, (__int64)&v100);
  sub_12C6670((__int64)v48, (__int64)&v112);
  if ( v100 != v102 )
    j_j___libc_free_0(v100, v102[0] + 1LL);
  sub_12C7330(&v112);
  v112 = v114;
  v115 = v117;
  v113 = 0;
  v118 = v120;
  LOBYTE(v114[0]) = 0;
  v116 = 0;
  v117[0] = 0;
  v119 = 0;
  v120[0] = 0;
  v121 = a1 + 688;
  sub_2241130(&v118, 0, 0, "-nvptx-fma-level=1 ", 19);
  sub_12C65C0((__int64 *)&v100, "-fma=1");
  v49 = sub_12C8B40(v6, (__int64)&v100);
  sub_12C6670((__int64)v49, (__int64)&v112);
  if ( v100 != v102 )
    j_j___libc_free_0(v100, v102[0] + 1LL);
  sub_12C7330(&v112);
  v112 = v114;
  v115 = v117;
  v113 = 0;
  v118 = v120;
  LOBYTE(v114[0]) = 0;
  v116 = 0;
  v117[0] = 0;
  v119 = 0;
  v120[0] = 0;
  v121 = a1 + 720;
  sub_2241130(&v118, 0, 0, "-nvptx-fma-level=1 ", 19);
  sub_12C65C0((__int64 *)&v100, "-enable-mad");
  v50 = sub_12C8B40(v6, (__int64)&v100);
  sub_12C6670((__int64)v50, (__int64)&v112);
  if ( v100 != v102 )
    j_j___libc_free_0(v100, v102[0] + 1LL);
  sub_12C7330(&v112);
  v112 = v114;
  v115 = v117;
  v113 = 0;
  v118 = v120;
  LOBYTE(v114[0]) = 0;
  v116 = 0;
  v117[0] = 0;
  v119 = 0;
  v120[0] = 0;
  v121 = a1 + 752;
  sub_2241130(&v112, 0, 0, "-R FAST_RELAXED_MATH=1 -R __CUDA_FTZ=1", 38);
  sub_2241130(&v115, 0, v116, "-opt-use-fast-math -nvptx-f32ftz", 32);
  sub_2241130(&v118, 0, v119, "-nvptx-fma-level=1 -nvptx-f32ftz", 32);
  sub_12C65C0((__int64 *)&v100, "-unsafe-math");
  v51 = sub_12C8B40(v6, (__int64)&v100);
  sub_12C6670((__int64)v51, (__int64)&v112);
  if ( v100 != v102 )
    j_j___libc_free_0(v100, v102[0] + 1LL);
  sub_12C7330(&v112);
  if ( a2 )
  {
    sub_12C7170(
      (__int64)&v112,
      "-R FAST_RELAXED_MATH=1 -R __CUDA_FTZ=1",
      "-opt-use-fast-math -nvptx-f32ftz",
      "-nvptx-f32ftz",
      a1 + 784);
  }
  else
  {
    v121 = a1 + 784;
    v112 = v114;
    v115 = v117;
    v113 = 0;
    LOBYTE(v114[0]) = 0;
    v116 = 0;
    v117[0] = 0;
    v118 = v120;
    v119 = 0;
    v120[0] = 0;
    sub_2241130(&v112, 0, 0, "-R __CUDA_USE_FAST_MATH=1", 25);
    sub_2241130(&v115, 0, v116, "-opt-use-fast-math ", 19);
  }
  sub_12C65C0((__int64 *)&v100, "-fast-math");
  v52 = sub_12C8B40(v6, (__int64)&v100);
  sub_12C6670((__int64)v52, (__int64)&v112);
  sub_2240A30(&v100);
  sub_12C7330(&v112);
  v112 = v114;
  v115 = v117;
  v113 = 0;
  v118 = v120;
  LOBYTE(v114[0]) = 0;
  v116 = 0;
  v117[0] = 0;
  v119 = 0;
  v120[0] = 0;
  v121 = a1 + 816;
  sub_2241130(&v118, 0, 0, "-nvptx-emit-src", 15);
  sub_12C65C0((__int64 *)&v100, "-show-src");
  v53 = sub_12C8B40(v6, (__int64)&v100);
  sub_12C6670((__int64)v53, (__int64)&v112);
  if ( v100 != v102 )
    j_j___libc_free_0(v100, v102[0] + 1LL);
  sub_12C7330(&v112);
  v112 = v114;
  v115 = v117;
  v113 = 0;
  v118 = v120;
  LOBYTE(v114[0]) = 0;
  v116 = 0;
  v117[0] = 0;
  v119 = 0;
  v120[0] = 0;
  v121 = a1 + 1040;
  sub_2241130(&v115, 0, 0, "-enable-opt-byval", 17);
  sub_12C65C0((__int64 *)&v100, "-enable-opt-byval");
  v54 = sub_12C8B40(v6, (__int64)&v100);
  sub_12C6670((__int64)v54, (__int64)&v112);
  if ( v100 != v102 )
    j_j___libc_free_0(v100, v102[0] + 1LL);
  sub_12C7330(&v112);
  v112 = v114;
  v115 = v117;
  v113 = 0;
  v118 = v120;
  LOBYTE(v114[0]) = 0;
  v116 = 0;
  v121 = a1 + 848;
  v117[0] = 0;
  v119 = 0;
  v120[0] = 0;
  sub_12C65C0((__int64 *)&v100, "-disable-llc-opts");
  v55 = sub_12C8B40(v6, (__int64)&v100);
  sub_12C6670((__int64)v55, (__int64)&v112);
  if ( v100 != v102 )
    j_j___libc_free_0(v100, v102[0] + 1LL);
  sub_12C7330(&v112);
  v112 = v114;
  v115 = v117;
  v113 = 0;
  v118 = v120;
  LOBYTE(v114[0]) = 0;
  v116 = 0;
  v117[0] = 0;
  v119 = 0;
  v120[0] = 0;
  v121 = a1 + 880;
  sub_2241130(&v115, 0, 0, "-w", 2);
  sub_2241130(&v118, 0, v119, "-w", 2);
  sub_12C65C0((__int64 *)&v100, "-w");
  v56 = sub_12C8B40(v6, (__int64)&v100);
  sub_12C6670((__int64)v56, (__int64)&v112);
  if ( v100 != v102 )
    j_j___libc_free_0(v100, v102[0] + 1LL);
  sub_12C7330(&v112);
  v112 = v114;
  v115 = v117;
  v113 = 0;
  v118 = v120;
  LOBYTE(v114[0]) = 0;
  v116 = 0;
  v117[0] = 0;
  v119 = 0;
  v120[0] = 0;
  v121 = a1 + 912;
  sub_2241130(&v115, 0, 0, "-Werror", 7);
  sub_2241130(&v118, 0, v119, "-Werror", 7);
  sub_12C65C0((__int64 *)&v100, "-Werror");
  v57 = sub_12C8B40(v6, (__int64)&v100);
  sub_12C6670((__int64)v57, (__int64)&v112);
  if ( v100 != v102 )
    j_j___libc_free_0(v100, v102[0] + 1LL);
  sub_12C7330(&v112);
  v112 = v114;
  v115 = v117;
  v113 = 0;
  v118 = v120;
  LOBYTE(v114[0]) = 0;
  v116 = 0;
  v117[0] = 0;
  v119 = 0;
  v120[0] = 0;
  v121 = a1 + 1072;
  sub_2241130(&v115, 0, 0, "-disable-inlining", 17);
  sub_12C65C0((__int64 *)&v100, "-disable-inlining");
  v58 = sub_12C8B40(v6, (__int64)&v100);
  sub_12C6670((__int64)v58, (__int64)&v112);
  if ( v100 != v102 )
    j_j___libc_free_0(v100, v102[0] + 1LL);
  sub_12C7330(&v112);
  v112 = v114;
  v115 = v117;
  v113 = 0;
  v118 = v120;
  LOBYTE(v114[0]) = 0;
  v116 = 0;
  v117[0] = 0;
  v119 = 0;
  v120[0] = 0;
  v121 = a1 + 1616;
  sub_2241130(&v115, 0, 0, "-inline-budget=40000", 20);
  sub_12C65C0((__int64 *)&v100, "-aggressive-inline");
  v59 = sub_12C8B40(v6, (__int64)&v100);
  sub_12C6670((__int64)v59, (__int64)&v112);
  if ( v100 != v102 )
    j_j___libc_free_0(v100, v102[0] + 1LL);
  sub_12C7330(&v112);
  v112 = v114;
  v115 = v117;
  v113 = 0;
  v118 = v120;
  LOBYTE(v114[0]) = 0;
  v116 = 0;
  v117[0] = 0;
  v119 = 0;
  v120[0] = 0;
  v121 = a1 + 1104;
  sub_2241130(&v118, 0, 0, "-nvptx-kernel-params-restrict", 29);
  sub_12C65C0((__int64 *)&v100, "-restrict");
  v60 = sub_12C8B40(v6, (__int64)&v100);
  sub_12C6670((__int64)v60, (__int64)&v112);
  if ( v100 != v102 )
    j_j___libc_free_0(v100, v102[0] + 1LL);
  sub_12C7330(&v112);
  v112 = v114;
  v115 = v117;
  v113 = 0;
  v118 = v120;
  LOBYTE(v114[0]) = 0;
  v116 = 0;
  v117[0] = 0;
  v119 = 0;
  v120[0] = 0;
  v121 = a1 + 1136;
  sub_2241130(&v115, 0, 0, "-allow-restrict-in-struct", 25);
  sub_2241130(&v118, 0, v119, "-allow-restrict-in-struct", 25);
  sub_12C65C0((__int64 *)&v100, "-allow-restrict-in-struct");
  v61 = sub_12C8B40(v6, (__int64)&v100);
  sub_12C6670((__int64)v61, (__int64)&v112);
  if ( v100 != v102 )
    j_j___libc_free_0(v100, v102[0] + 1LL);
  sub_12C7330(&v112);
  v112 = v114;
  v115 = v117;
  v113 = 0;
  v118 = v120;
  LOBYTE(v114[0]) = 0;
  v116 = 0;
  v117[0] = 0;
  v119 = 0;
  v120[0] = 0;
  v121 = a1 + 1168;
  sub_2241130(&v115, 0, 0, "-opt-no-signed-zeros", 20);
  sub_12C65C0((__int64 *)&v100, "-no-signed-zeros");
  v62 = sub_12C8B40(v6, (__int64)&v100);
  sub_12C6670((__int64)v62, (__int64)&v112);
  if ( v100 != v102 )
    j_j___libc_free_0(v100, v102[0] + 1LL);
  sub_12C7330(&v112);
  v112 = v114;
  v115 = v117;
  v113 = 0;
  v118 = v120;
  LOBYTE(v114[0]) = 0;
  v116 = 0;
  v117[0] = 0;
  v119 = 0;
  v120[0] = 0;
  v121 = a1 + 1232;
  sub_2241130(&v118, 0, 0, "-asm-verbose", 12);
  sub_12C65C0((__int64 *)&v100, "-enable-verbose-asm");
  v63 = sub_12C8B40(v6, (__int64)&v100);
  sub_12C6670((__int64)v63, (__int64)&v112);
  if ( v100 != v102 )
    j_j___libc_free_0(v100, v102[0] + 1LL);
  sub_12C7330(&v112);
  v112 = v114;
  v115 = v117;
  LOBYTE(v114[0]) = 0;
  v118 = v120;
  v113 = 0;
  v116 = 0;
  v117[0] = 0;
  v119 = 0;
  v120[0] = 0;
  v121 = a1 + 464;
  sub_2241130(&v115, 0, 0, "-opt-fdiv=0", 11);
  sub_12C65C0((__int64 *)&v100, "-opt-fdiv=0");
  v64 = sub_12C8B40(v6, (__int64)&v100);
  sub_12C6670((__int64)v64, (__int64)&v112);
  if ( v100 != v102 )
    j_j___libc_free_0(v100, v102[0] + 1LL);
  sub_12C7330(&v112);
  v112 = v114;
  v115 = v117;
  v113 = 0;
  v118 = v120;
  LOBYTE(v114[0]) = 0;
  v116 = 0;
  v117[0] = 0;
  v119 = 0;
  v120[0] = 0;
  v121 = a1 + 464;
  sub_2241130(&v115, 0, 0, "-opt-fdiv=1", 11);
  sub_12C65C0((__int64 *)&v100, "-opt-fdiv=1");
  v65 = sub_12C8B40(v6, (__int64)&v100);
  sub_12C6670((__int64)v65, (__int64)&v112);
  if ( v100 != v102 )
    j_j___libc_free_0(v100, v102[0] + 1LL);
  sub_12C7330(&v112);
  v112 = v114;
  v115 = v117;
  v113 = 0;
  v118 = v120;
  LOBYTE(v114[0]) = 0;
  v116 = 0;
  v117[0] = 0;
  v119 = 0;
  v120[0] = 0;
  v121 = a1 + 1360;
  sub_2241130(&v118, 0, 0, "-vasp-fix1=true -vasp-fix2=true", 31);
  sub_12C65C0((__int64 *)&v100, "-vasp-fix");
  v66 = sub_12C8B40(v6, (__int64)&v100);
  sub_12C6670((__int64)v66, (__int64)&v112);
  if ( v100 != v102 )
    j_j___libc_free_0(v100, v102[0] + 1LL);
  sub_12C7330(&v112);
  v112 = v114;
  v115 = v117;
  v113 = 0;
  v118 = v120;
  LOBYTE(v114[0]) = 0;
  v116 = 0;
  v117[0] = 0;
  v119 = 0;
  v120[0] = 0;
  v121 = a1 + 1392;
  sub_2241130(
    &v118,
    0,
    0,
    "-enable-new-nvvm-remat=true                                             -nv-disable-remat=true                      "
    "                       -rp-aware-mcse=true",
    158);
  sub_12C65C0((__int64 *)&v100, "-new-nvvm-remat");
  v67 = sub_12C8B40(v6, (__int64)&v100);
  sub_12C6670((__int64)v67, (__int64)&v112);
  if ( v100 != v102 )
    j_j___libc_free_0(v100, v102[0] + 1LL);
  sub_12C7330(&v112);
  v112 = v114;
  v115 = v117;
  v113 = 0;
  v118 = v120;
  LOBYTE(v114[0]) = 0;
  v116 = 0;
  v117[0] = 0;
  v119 = 0;
  v120[0] = 0;
  v121 = a1 + 1424;
  sub_2241130(
    &v118,
    0,
    0,
    "-enable-new-nvvm-remat=false                                             -nv-disable-remat=false                    "
    "                         -rp-aware-mcse=false",
    161);
  sub_12C65C0((__int64 *)&v100, "-disable-new-nvvm-remat");
  v68 = sub_12C8B40(v6, (__int64)&v100);
  sub_12C6670((__int64)v68, (__int64)&v112);
  if ( v100 != v102 )
    j_j___libc_free_0(v100, v102[0] + 1LL);
  sub_12C7330(&v112);
  v112 = v114;
  v115 = v117;
  v113 = 0;
  v118 = v120;
  LOBYTE(v114[0]) = 0;
  v116 = 0;
  v117[0] = 0;
  v119 = 0;
  v120[0] = 0;
  v121 = a1 + 1456;
  sub_2241130(
    &v118,
    0,
    0,
    "-enable-new-nvvm-remat=false                                             -nv-disable-remat=true                     "
    "                        -rp-aware-mcse=false",
    160);
  sub_12C65C0((__int64 *)&v100, "-disable-nvvm-remat");
  v69 = sub_12C8B40(v6, (__int64)&v100);
  sub_12C6670((__int64)v69, (__int64)&v112);
  if ( v100 != v102 )
    j_j___libc_free_0(v100, v102[0] + 1LL);
  sub_12C7330(&v112);
  v112 = v114;
  v115 = v117;
  v113 = 0;
  v118 = v120;
  LOBYTE(v114[0]) = 0;
  v116 = 0;
  v117[0] = 0;
  v119 = 0;
  v120[0] = 0;
  v121 = a1 + 1552;
  sub_2241130(&v115, 0, 0, "-aggressive-positive-stride-analysis=false", 42);
  sub_12C65C0((__int64 *)&v100, "-no-aggressive-positive-stride-analysis");
  v70 = sub_12C8B40(v6, (__int64)&v100);
  sub_12C6670((__int64)v70, (__int64)&v112);
  if ( v100 != v102 )
    j_j___libc_free_0(v100, v102[0] + 1LL);
  sub_12C7330(&v112);
  v112 = v114;
  v115 = v117;
  v113 = 0;
  v118 = v120;
  LOBYTE(v114[0]) = 0;
  v116 = 0;
  v117[0] = 0;
  v119 = 0;
  v120[0] = 0;
  v121 = a1 + 1584;
  sub_2241130(&v115, 0, 0, "-disable-load-select-transform=true", 35);
  sub_12C65C0((__int64 *)&v100, "disable-load-select-transform");
  v71 = sub_12C8B40(v6, (__int64)&v100);
  sub_12C6670((__int64)v71, (__int64)&v112);
  if ( v100 != v102 )
    j_j___libc_free_0(v100, v102[0] + 1LL);
  sub_12C7330(&v112);
  v112 = v114;
  sub_12C6230((__int64 *)&v112, (char)&byte_4281A03[-4], &byte_4281A03[-4], byte_4281A03);
  sub_12C65C0((__int64 *)&v100, "-");
  v72 = 15;
  v73 = 15;
  if ( v100 != v102 )
    v73 = v102[0];
  v74 = v101 + v113;
  if ( v101 + v113 <= v73 )
    goto LABEL_139;
  if ( v112 != v114 )
    v72 = v114[0];
  if ( v74 <= v72 )
  {
    v75 = (__m128i *)sub_2241130(&v112, 0, 0, v100, v101);
    v76 = v75 + 1;
    s2 = &v99;
    v77 = (__m128i *)v75->m128i_i64[0];
    if ( (__m128i *)v75->m128i_i64[0] != &v75[1] )
      goto LABEL_140;
  }
  else
  {
LABEL_139:
    v75 = (__m128i *)sub_2241490(&v100, v112, v113, v74);
    v76 = v75 + 1;
    s2 = &v99;
    v77 = (__m128i *)v75->m128i_i64[0];
    if ( (__m128i *)v75->m128i_i64[0] != &v75[1] )
    {
LABEL_140:
      s2 = v77;
      v99.m128i_i64[0] = v75[1].m128i_i64[0];
      goto LABEL_141;
    }
  }
  v99 = _mm_loadu_si128(v75 + 1);
LABEL_141:
  v98 = v75->m128i_u64[1];
  v75->m128i_i64[0] = (__int64)v76;
  v75->m128i_i64[1] = 0;
  v75[1].m128i_i8[0] = 0;
  if ( v100 != v102 )
    j_j___libc_free_0(v100, v102[0] + 1LL);
  if ( v112 != v114 )
    j_j___libc_free_0(v112, v114[0] + 1LL);
  sub_12C7250((__int64)&v112, s2->m128i_i8, s2->m128i_i8, s2->m128i_i8, (__int64)s2, a1 + 1264);
  if ( *(_QWORD *)(a1 + 272) )
  {
    v78 = *(_QWORD *)(a1 + 272);
    v79 = v92;
    while ( 1 )
    {
      v80 = *(_QWORD *)(v78 + 40);
      v81 = v98;
      if ( v80 <= v98 )
        v81 = *(_QWORD *)(v78 + 40);
      if ( v81 )
      {
        v82 = memcmp(*(const void **)(v78 + 32), s2, v81);
        if ( v82 )
          goto LABEL_155;
      }
      v83 = v80 - v98;
      if ( v83 >= 0x80000000LL )
      {
LABEL_156:
        v79 = v78;
        v78 = *(_QWORD *)(v78 + 16);
        if ( !v78 )
        {
LABEL_157:
          if ( v92 != v79 )
          {
            v84 = v79 + 64;
            if ( sub_12C6100(s2, v98, *(const void **)(v79 + 32), *(_QWORD *)(v79 + 40)) >= 0 )
              goto LABEL_159;
          }
          goto LABEL_163;
        }
      }
      else
      {
        if ( v83 <= (__int64)0xFFFFFFFF7FFFFFFFLL )
          goto LABEL_147;
        v82 = v83;
LABEL_155:
        if ( v82 >= 0 )
          goto LABEL_156;
LABEL_147:
        v78 = *(_QWORD *)(v78 + 24);
        if ( !v78 )
          goto LABEL_157;
      }
    }
  }
  v79 = v92;
LABEL_163:
  v95 = (_QWORD *)v79;
  v85 = sub_22077B0(168);
  *(_QWORD *)(v85 + 32) = v85 + 48;
  v84 = v85 + 64;
  sub_12C6440((__int64 *)(v85 + 32), s2, (__int64)s2->m128i_i64 + v98);
  *(_QWORD *)(v85 + 64) = v85 + 80;
  *(_QWORD *)(v85 + 96) = v85 + 112;
  *(_QWORD *)(v85 + 72) = 0;
  *(_BYTE *)(v85 + 80) = 0;
  *(_QWORD *)(v85 + 104) = 0;
  *(_BYTE *)(v85 + 112) = 0;
  *(_QWORD *)(v85 + 128) = v85 + 144;
  *(_QWORD *)(v85 + 136) = 0;
  *(_BYTE *)(v85 + 144) = 0;
  *(_QWORD *)(v85 + 160) = 0;
  v86 = sub_12C88B0(v6, v95, v85 + 32);
  v88 = v87;
  if ( v87 )
  {
    if ( v86 || v92 == v87 )
      v89 = 1;
    else
      v89 = (unsigned int)sub_12C6100(
                            *(const void **)(v85 + 32),
                            *(_QWORD *)(v85 + 40),
                            *(const void **)(v87 + 32),
                            *(_QWORD *)(v87 + 40)) >> 31;
    sub_220F040(v89, v85, v88, v92);
    ++*(_QWORD *)(a1 + 296);
  }
  else
  {
    v96 = v86;
    sub_12C7330((_QWORD *)(v85 + 64));
    sub_2240A30(v85 + 32);
    j_j___libc_free_0(v85, 168);
    v84 = v96 + 64;
  }
LABEL_159:
  sub_12C6670(v84, (__int64)&v112);
  sub_12C7330(&v112);
  if ( s2 != &v99 )
    j_j___libc_free_0(s2, v99.m128i_i64[0] + 1);
  return (_BYTE *)sub_12C7330(v103);
}
