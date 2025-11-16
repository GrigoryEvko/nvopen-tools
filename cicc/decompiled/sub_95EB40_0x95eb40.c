// Function: sub_95EB40
// Address: 0x95eb40
//
_BYTE *__fastcall sub_95EB40(__int64 a1, char a2)
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
  __m128i *v46; // rax
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
  unsigned __int64 v71; // rax
  unsigned __int64 v72; // rdi
  unsigned __int64 v73; // rcx
  __m128i *v74; // rax
  __m128i *v75; // rdx
  __m128i *v76; // rcx
  __int64 v77; // r12
  __int64 v78; // r13
  size_t v79; // r15
  size_t v80; // rdx
  int v81; // eax
  __int64 v82; // r15
  __int64 v83; // r15
  __int64 v84; // r13
  __int64 v85; // rax
  __int64 v86; // rdx
  __int64 v87; // r14
  __int64 v88; // rdi
  __m128i *v89; // rax
  __m128i *v90; // rax
  __int64 v91; // [rsp+8h] [rbp-188h]
  __int64 v92; // [rsp+10h] [rbp-180h]
  _QWORD *v94; // [rsp+38h] [rbp-158h]
  __int64 v95; // [rsp+38h] [rbp-158h]
  __m128i *s2; // [rsp+40h] [rbp-150h]
  size_t v97; // [rsp+48h] [rbp-148h]
  __m128i v98; // [rsp+50h] [rbp-140h] BYREF
  _QWORD *v99; // [rsp+60h] [rbp-130h] BYREF
  __int64 v100; // [rsp+68h] [rbp-128h]
  _QWORD v101[2]; // [rsp+70h] [rbp-120h] BYREF
  _BYTE *v102[2]; // [rsp+80h] [rbp-110h] BYREF
  char v103; // [rsp+90h] [rbp-100h] BYREF
  char *v104; // [rsp+A0h] [rbp-F0h] BYREF
  __int64 v105; // [rsp+A8h] [rbp-E8h]
  char v106; // [rsp+B0h] [rbp-E0h] BYREF
  char *v107; // [rsp+C0h] [rbp-D0h] BYREF
  __int64 v108; // [rsp+C8h] [rbp-C8h]
  char v109; // [rsp+D0h] [rbp-C0h] BYREF
  __int64 v110; // [rsp+E0h] [rbp-B0h]
  _QWORD *v111; // [rsp+F0h] [rbp-A0h] BYREF
  __int64 v112; // [rsp+F8h] [rbp-98h]
  _QWORD v113[2]; // [rsp+100h] [rbp-90h] BYREF
  _BYTE *v114; // [rsp+110h] [rbp-80h] BYREF
  __int64 v115; // [rsp+118h] [rbp-78h]
  _BYTE v116[16]; // [rsp+120h] [rbp-70h] BYREF
  _BYTE *v117; // [rsp+130h] [rbp-60h] BYREF
  __int64 v118; // [rsp+138h] [rbp-58h]
  _BYTE v119[16]; // [rsp+140h] [rbp-50h] BYREF
  __int64 v120; // [rsp+150h] [rbp-40h]

  v2 = *(_BYTE **)(a1 + 1192);
  *(_QWORD *)(a1 + 1200) = 0;
  *v2 = 0;
  result = *(_BYTE **)(a1 + 1480);
  *(_QWORD *)(a1 + 1488) = 0;
  *result = 0;
  v4 = *(_QWORD *)(a1 + 288) == 0;
  *(_BYTE *)(a1 + 1644) = 0;
  v91 = a1 + 256;
  if ( !v4 )
  {
    v5 = *(_BYTE **)(a1 + 272);
    if ( v5 != (_BYTE *)(a1 + 256) )
    {
      do
      {
        v111 = v113;
        v112 = 0;
        LOBYTE(v113[0]) = 0;
        sub_2240AE0(*((_QWORD *)v5 + 20), &v111);
        if ( v111 != v113 )
          j_j___libc_free_0(v111, v113[0] + 1LL);
        result = (_BYTE *)sub_220EEE0(v5);
        v5 = result;
      }
      while ( result != (_BYTE *)(a1 + 256) );
    }
    return result;
  }
  v6 = (_QWORD *)(a1 + 248);
  v102[1] = 0;
  v103 = 0;
  v102[0] = &v103;
  v104 = &v106;
  v110 = a1 + 296;
  v105 = 0;
  v106 = 0;
  v107 = &v109;
  v108 = 0;
  v109 = 0;
  sub_2241130(&v104, 0, 0, "-debug-compile", 14);
  sub_2241130(&v107, 0, v108, "-debug-compile", 14);
  v111 = v113;
  sub_95BD60((__int64 *)&v111, v102[0], (__int64)v102[0]);
  v114 = v116;
  sub_95BD60((__int64 *)&v114, v104, (__int64)&v104[v105]);
  v117 = v119;
  sub_95BD60((__int64 *)&v117, v107, (__int64)&v107[v108]);
  v120 = v110;
  sub_95CD70(&v111);
  v120 = a1 + 296;
  v117 = v119;
  v111 = v113;
  v112 = 0;
  LOBYTE(v113[0]) = 0;
  v114 = v116;
  v115 = 0;
  v116[0] = 0;
  v118 = 0;
  v119[0] = 0;
  sub_2241130(&v114, 0, 0, "-debug-compile", 14);
  sub_2241130(&v117, 0, v118, "-debug-compile", 14);
  sub_95BEE0((__int64 *)&v99, "-g");
  v7 = sub_95E8B0((_QWORD *)(a1 + 248), (__int64)&v99);
  sub_95BF90((__int64)v7, (__int64)&v111);
  if ( v99 != v101 )
    j_j___libc_free_0(v99, v101[0] + 1LL);
  sub_95CD70(&v111);
  v111 = v113;
  v114 = v116;
  v112 = 0;
  v117 = v119;
  LOBYTE(v113[0]) = 0;
  v115 = 0;
  v116[0] = 0;
  v118 = 0;
  v119[0] = 0;
  v120 = a1 + 328;
  sub_2241130(&v117, 0, 0, "-generate-line-info", 19);
  sub_95BEE0((__int64 *)&v99, "-generate-line-info");
  v8 = sub_95E8B0(v6, (__int64)&v99);
  sub_95BF90((__int64)v8, (__int64)&v111);
  if ( v99 != v101 )
    j_j___libc_free_0(v99, v101[0] + 1LL);
  sub_95CD70(&v111);
  v111 = v113;
  v114 = v116;
  v112 = 0;
  v117 = v119;
  LOBYTE(v113[0]) = 0;
  v115 = 0;
  v116[0] = 0;
  v118 = 0;
  v119[0] = 0;
  v120 = a1 + 360;
  sub_2241130(&v117, 0, 0, "-line-info-inlined-at=0", 23);
  sub_95BEE0((__int64 *)&v99, "-no-lineinfo-inlined-at");
  v9 = sub_95E8B0(v6, (__int64)&v99);
  sub_95BF90((__int64)v9, (__int64)&v111);
  if ( v99 != v101 )
    j_j___libc_free_0(v99, v101[0] + 1LL);
  sub_95CD70(&v111);
  v111 = v113;
  v114 = v116;
  v112 = 0;
  v117 = v119;
  LOBYTE(v113[0]) = 0;
  v115 = 0;
  v116[0] = 0;
  v118 = 0;
  v119[0] = 0;
  v120 = a1 + 424;
  sub_2241130(&v111, 0, 0, "-lnk-disable-allopts", 20);
  sub_2241130(&v114, 0, v115, "-opt-disable-allopts", 20);
  sub_2241130(&v117, 0, v118, "-llc-disable-allopts", 20);
  sub_95BEE0((__int64 *)&v99, "-disable-allopts");
  v10 = sub_95E8B0(v6, (__int64)&v99);
  sub_95BF90((__int64)v10, (__int64)&v111);
  if ( v99 != v101 )
    j_j___libc_free_0(v99, v101[0] + 1LL);
  sub_95CD70(&v111);
  v116[0] = 0;
  v119[0] = 0;
  v114 = v116;
  v111 = v113;
  v117 = v119;
  v120 = a1 + 392;
  v112 = 0;
  LOBYTE(v113[0]) = 0;
  v115 = 0;
  v118 = 0;
  sub_95BEE0((__int64 *)&v99, "-opt=0");
  v11 = sub_95E8B0(v6, (__int64)&v99);
  sub_95BF90((__int64)v11, (__int64)&v111);
  if ( v99 != v101 )
    j_j___libc_free_0(v99, v101[0] + 1LL);
  sub_95CD70(&v111);
  v111 = v113;
  v114 = v116;
  v112 = 0;
  v117 = v119;
  LOBYTE(v113[0]) = 0;
  v120 = a1 + 392;
  v115 = 0;
  v116[0] = 0;
  v118 = 0;
  v119[0] = 0;
  sub_95BEE0((__int64 *)&v99, "-opt=1");
  v12 = sub_95E8B0(v6, (__int64)&v99);
  sub_95BF90((__int64)v12, (__int64)&v111);
  if ( v99 != v101 )
    j_j___libc_free_0(v99, v101[0] + 1LL);
  sub_95CD70(&v111);
  v111 = v113;
  v114 = v116;
  v112 = 0;
  v117 = v119;
  LOBYTE(v113[0]) = 0;
  v120 = a1 + 392;
  v115 = 0;
  v116[0] = 0;
  v118 = 0;
  v119[0] = 0;
  sub_95BEE0((__int64 *)&v99, "-opt=2");
  v13 = sub_95E8B0(v6, (__int64)&v99);
  sub_95BF90((__int64)v13, (__int64)&v111);
  if ( v99 != v101 )
    j_j___libc_free_0(v99, v101[0] + 1LL);
  sub_95CD70(&v111);
  v111 = v113;
  v114 = v116;
  v112 = 0;
  v117 = v119;
  LOBYTE(v113[0]) = 0;
  v120 = a1 + 392;
  v115 = 0;
  v116[0] = 0;
  v118 = 0;
  v119[0] = 0;
  sub_95BEE0((__int64 *)&v99, "-opt=3");
  v14 = sub_95E8B0(v6, (__int64)&v99);
  sub_95BF90((__int64)v14, (__int64)&v111);
  if ( v99 != v101 )
    j_j___libc_free_0(v99, v101[0] + 1LL);
  sub_95CD70(&v111);
  v111 = v113;
  v114 = v116;
  v112 = 0;
  v117 = v119;
  LOBYTE(v113[0]) = 0;
  v115 = 0;
  v116[0] = 0;
  v118 = 0;
  v119[0] = 0;
  v120 = a1 + 488;
  sub_2241130(&v114, 0, 0, "-Osize", 6);
  sub_2241130(&v117, 0, v118, "-Osize", 6);
  sub_95BEE0((__int64 *)&v99, "-Osize");
  v15 = sub_95E8B0(v6, (__int64)&v99);
  sub_95BF90((__int64)v15, (__int64)&v111);
  if ( v99 != v101 )
    j_j___libc_free_0(v99, v101[0] + 1LL);
  sub_95CD70(&v111);
  v111 = v113;
  v114 = v116;
  v112 = 0;
  v117 = v119;
  LOBYTE(v113[0]) = 0;
  v115 = 0;
  v116[0] = 0;
  v118 = 0;
  v119[0] = 0;
  v120 = a1 + 520;
  sub_2241130(&v114, 0, 0, "-Om", 3);
  sub_2241130(&v117, 0, v118, "-Om", 3);
  sub_95BEE0((__int64 *)&v99, "-Om");
  v16 = sub_95E8B0(v6, (__int64)&v99);
  sub_95BF90((__int64)v16, (__int64)&v111);
  if ( v99 != v101 )
    j_j___libc_free_0(v99, v101[0] + 1LL);
  sub_95CD70(&v111);
  v111 = v113;
  v114 = v116;
  LOBYTE(v113[0]) = 0;
  v117 = v119;
  v92 = a1 + 552;
  v112 = 0;
  v115 = 0;
  v116[0] = 0;
  v118 = 0;
  v119[0] = 0;
  v120 = a1 + 552;
  sub_2241130(&v111, 0, 0, "-R __CUDA_ARCH=750", 18);
  sub_2241130(&v114, 0, v115, "-opt-arch=sm_75", 15);
  sub_2241130(&v117, 0, v118, "-mcpu=sm_75", 11);
  sub_95BEE0((__int64 *)&v99, "-arch=compute_75");
  v17 = sub_95E8B0(v6, (__int64)&v99);
  sub_95BF90((__int64)v17, (__int64)&v111);
  if ( v99 != v101 )
    j_j___libc_free_0(v99, v101[0] + 1LL);
  sub_95CD70(&v111);
  v111 = v113;
  v114 = v116;
  v112 = 0;
  v117 = v119;
  LOBYTE(v113[0]) = 0;
  v115 = 0;
  v116[0] = 0;
  v118 = 0;
  v119[0] = 0;
  v120 = a1 + 552;
  sub_2241130(&v111, 0, 0, "-R __CUDA_ARCH=800", 18);
  sub_2241130(&v114, 0, v115, "-opt-arch=sm_80", 15);
  sub_2241130(&v117, 0, v118, "-mcpu=sm_80", 11);
  sub_95BEE0((__int64 *)&v99, "-arch=compute_80");
  v18 = sub_95E8B0(v6, (__int64)&v99);
  sub_95BF90((__int64)v18, (__int64)&v111);
  if ( v99 != v101 )
    j_j___libc_free_0(v99, v101[0] + 1LL);
  sub_95CD70(&v111);
  v111 = v113;
  v114 = v116;
  v112 = 0;
  v117 = v119;
  LOBYTE(v113[0]) = 0;
  v115 = 0;
  v116[0] = 0;
  v118 = 0;
  v119[0] = 0;
  v120 = a1 + 552;
  sub_2241130(&v111, 0, 0, "-R __CUDA_ARCH=860", 18);
  sub_2241130(&v114, 0, v115, "-opt-arch=sm_86", 15);
  sub_2241130(&v117, 0, v118, "-mcpu=sm_86", 11);
  sub_95BEE0((__int64 *)&v99, "-arch=compute_86");
  v19 = sub_95E8B0(v6, (__int64)&v99);
  sub_95BF90((__int64)v19, (__int64)&v111);
  if ( v99 != v101 )
    j_j___libc_free_0(v99, v101[0] + 1LL);
  sub_95CD70(&v111);
  v111 = v113;
  v114 = v116;
  v112 = 0;
  v117 = v119;
  LOBYTE(v113[0]) = 0;
  v115 = 0;
  v116[0] = 0;
  v118 = 0;
  v119[0] = 0;
  v120 = a1 + 552;
  sub_2241130(&v111, 0, 0, "-R __CUDA_ARCH=870", 18);
  sub_2241130(&v114, 0, v115, "-opt-arch=sm_87", 15);
  sub_2241130(&v117, 0, v118, "-mcpu=sm_87", 11);
  sub_95BEE0((__int64 *)&v99, "-arch=compute_87");
  v20 = sub_95E8B0(v6, (__int64)&v99);
  sub_95BF90((__int64)v20, (__int64)&v111);
  if ( v99 != v101 )
    j_j___libc_free_0(v99, v101[0] + 1LL);
  sub_95CD70(&v111);
  v111 = v113;
  v114 = v116;
  v112 = 0;
  v117 = v119;
  LOBYTE(v113[0]) = 0;
  v115 = 0;
  v116[0] = 0;
  v118 = 0;
  v119[0] = 0;
  v120 = a1 + 552;
  sub_2241130(&v111, 0, 0, "-R __CUDA_ARCH=880", 18);
  sub_2241130(&v114, 0, v115, "-opt-arch=sm_88", 15);
  sub_2241130(&v117, 0, v118, "-mcpu=sm_88", 11);
  sub_95BEE0((__int64 *)&v99, "-arch=compute_88");
  v21 = sub_95E8B0(v6, (__int64)&v99);
  sub_95BF90((__int64)v21, (__int64)&v111);
  if ( v99 != v101 )
    j_j___libc_free_0(v99, v101[0] + 1LL);
  sub_95CD70(&v111);
  v111 = v113;
  v114 = v116;
  v112 = 0;
  v117 = v119;
  LOBYTE(v113[0]) = 0;
  v115 = 0;
  v116[0] = 0;
  v118 = 0;
  v119[0] = 0;
  v120 = a1 + 552;
  sub_2241130(&v111, 0, 0, "-R __CUDA_ARCH=890", 18);
  sub_2241130(&v114, 0, v115, "-opt-arch=sm_89", 15);
  sub_2241130(&v117, 0, v118, "-mcpu=sm_89", 11);
  sub_95BEE0((__int64 *)&v99, "-arch=compute_89");
  v22 = sub_95E8B0(v6, (__int64)&v99);
  sub_95BF90((__int64)v22, (__int64)&v111);
  if ( v99 != v101 )
    j_j___libc_free_0(v99, v101[0] + 1LL);
  sub_95CD70(&v111);
  sub_95CC90((__int64)&v111, "-R __CUDA_ARCH=900", "-opt-arch=sm_90", "-mcpu=sm_90", (__int64)"-mcpu=sm_90", v92);
  sub_95BEE0((__int64 *)&v99, "-arch=compute_90");
  v23 = sub_95E8B0(v6, (__int64)&v99);
  sub_95BF90((__int64)v23, (__int64)&v111);
  if ( v99 != v101 )
    j_j___libc_free_0(v99, v101[0] + 1LL);
  sub_95CD70(&v111);
  sub_95CC90((__int64)&v111, "-R __CUDA_ARCH=900", "-opt-arch=sm_90a", "-mcpu=sm_90a", (__int64)"-mcpu=sm_90a", v92);
  sub_95BEE0((__int64 *)&v99, "-arch=compute_90a");
  v24 = sub_95E8B0(v6, (__int64)&v99);
  sub_95BF90((__int64)v24, (__int64)&v111);
  if ( v99 != v101 )
    j_j___libc_free_0(v99, v101[0] + 1LL);
  sub_95CD70(&v111);
  v111 = v113;
  v114 = v116;
  v112 = 0;
  v117 = v119;
  LOBYTE(v113[0]) = 0;
  v115 = 0;
  v116[0] = 0;
  v118 = 0;
  v119[0] = 0;
  v120 = a1 + 552;
  sub_2241130(&v111, 0, 0, "-R __CUDA_ARCH=1000", 19);
  sub_2241130(&v114, 0, v115, "-opt-arch=sm_100", 16);
  sub_2241130(&v117, 0, v118, "-mcpu=sm_100", 12);
  sub_95BEE0((__int64 *)&v99, "-arch=compute_100");
  v25 = sub_95E8B0(v6, (__int64)&v99);
  sub_95BF90((__int64)v25, (__int64)&v111);
  if ( v99 != v101 )
    j_j___libc_free_0(v99, v101[0] + 1LL);
  sub_95CD70(&v111);
  sub_95CC90((__int64)&v111, "-R __CUDA_ARCH=1000", "-opt-arch=sm_100a", "-mcpu=sm_100a", (__int64)"-mcpu=sm_100a", v92);
  sub_95BEE0((__int64 *)&v99, "-arch=compute_100a");
  v26 = sub_95E8B0(v6, (__int64)&v99);
  sub_95BF90((__int64)v26, (__int64)&v111);
  if ( v99 != v101 )
    j_j___libc_free_0(v99, v101[0] + 1LL);
  sub_95CD70(&v111);
  v111 = v113;
  v114 = v116;
  v112 = 0;
  v117 = v119;
  LOBYTE(v113[0]) = 0;
  v115 = 0;
  v116[0] = 0;
  v118 = 0;
  v119[0] = 0;
  v120 = a1 + 552;
  sub_2241130(&v111, 0, 0, "-R __CUDA_ARCH=1000", 19);
  sub_2241130(&v114, 0, v115, "-opt-arch=sm_100f", 17);
  sub_2241130(&v117, 0, v118, "-mcpu=sm_100f", 13);
  sub_95BEE0((__int64 *)&v99, "-arch=compute_100f");
  v27 = sub_95E8B0(v6, (__int64)&v99);
  sub_95BF90((__int64)v27, (__int64)&v111);
  if ( v99 != v101 )
    j_j___libc_free_0(v99, v101[0] + 1LL);
  sub_95CD70(&v111);
  sub_95CC90((__int64)&v111, "-R __CUDA_ARCH=1030", "-opt-arch=sm_103", "-mcpu=sm_103", (__int64)"-mcpu=sm_103", v92);
  sub_95BEE0((__int64 *)&v99, "-arch=compute_103");
  v28 = sub_95E8B0(v6, (__int64)&v99);
  sub_95BF90((__int64)v28, (__int64)&v111);
  if ( v99 != v101 )
    j_j___libc_free_0(v99, v101[0] + 1LL);
  sub_95CD70(&v111);
  v111 = v113;
  v114 = v116;
  v112 = 0;
  v117 = v119;
  LOBYTE(v113[0]) = 0;
  v115 = 0;
  v116[0] = 0;
  v118 = 0;
  v119[0] = 0;
  v120 = a1 + 552;
  sub_2241130(&v111, 0, 0, "-R __CUDA_ARCH=1030", 19);
  sub_2241130(&v114, 0, v115, "-opt-arch=sm_103a", 17);
  sub_2241130(&v117, 0, v118, "-mcpu=sm_103a", 13);
  sub_95BEE0((__int64 *)&v99, "-arch=compute_103a");
  v29 = sub_95E8B0(v6, (__int64)&v99);
  sub_95BF90((__int64)v29, (__int64)&v111);
  if ( v99 != v101 )
    j_j___libc_free_0(v99, v101[0] + 1LL);
  sub_95CD70(&v111);
  v111 = v113;
  v114 = v116;
  v112 = 0;
  v117 = v119;
  LOBYTE(v113[0]) = 0;
  v115 = 0;
  v116[0] = 0;
  v118 = 0;
  v119[0] = 0;
  v120 = a1 + 552;
  sub_2241130(&v111, 0, 0, "-R __CUDA_ARCH=1030", 19);
  sub_2241130(&v114, 0, v115, "-opt-arch=sm_103f", 17);
  sub_2241130(&v117, 0, v118, "-mcpu=sm_103f", 13);
  sub_95BEE0((__int64 *)&v99, "-arch=compute_103f");
  v30 = sub_95E8B0(v6, (__int64)&v99);
  sub_95BF90((__int64)v30, (__int64)&v111);
  if ( v99 != v101 )
    j_j___libc_free_0(v99, v101[0] + 1LL);
  sub_95CD70(&v111);
  v111 = v113;
  v114 = v116;
  v112 = 0;
  v117 = v119;
  LOBYTE(v113[0]) = 0;
  v115 = 0;
  v116[0] = 0;
  v118 = 0;
  v119[0] = 0;
  v120 = a1 + 552;
  sub_2241130(&v111, 0, 0, "-R __CUDA_ARCH=1100", 19);
  sub_2241130(&v114, 0, v115, "-opt-arch=sm_110", 16);
  sub_2241130(&v117, 0, v118, "-mcpu=sm_110", 12);
  sub_95BEE0((__int64 *)&v99, "-arch=compute_110");
  v31 = sub_95E8B0(v6, (__int64)&v99);
  sub_95BF90((__int64)v31, (__int64)&v111);
  if ( v99 != v101 )
    j_j___libc_free_0(v99, v101[0] + 1LL);
  sub_95CD70(&v111);
  v111 = v113;
  v114 = v116;
  v112 = 0;
  v117 = v119;
  LOBYTE(v113[0]) = 0;
  v115 = 0;
  v116[0] = 0;
  v118 = 0;
  v119[0] = 0;
  v120 = a1 + 552;
  sub_2241130(&v111, 0, 0, "-R __CUDA_ARCH=1100", 19);
  sub_2241130(&v114, 0, v115, "-opt-arch=sm_110a", 17);
  sub_2241130(&v117, 0, v118, "-mcpu=sm_110a", 13);
  sub_95BEE0((__int64 *)&v99, "-arch=compute_110a");
  v32 = sub_95E8B0(v6, (__int64)&v99);
  sub_95BF90((__int64)v32, (__int64)&v111);
  if ( v99 != v101 )
    j_j___libc_free_0(v99, v101[0] + 1LL);
  sub_95CD70(&v111);
  v111 = v113;
  v114 = v116;
  v112 = 0;
  v117 = v119;
  LOBYTE(v113[0]) = 0;
  v115 = 0;
  v116[0] = 0;
  v118 = 0;
  v119[0] = 0;
  v120 = a1 + 552;
  sub_2241130(&v111, 0, 0, "-R __CUDA_ARCH=1100", 19);
  sub_2241130(&v114, 0, v115, "-opt-arch=sm_110f", 17);
  sub_2241130(&v117, 0, v118, "-mcpu=sm_110f", 13);
  sub_95BEE0((__int64 *)&v99, "-arch=compute_110f");
  v33 = sub_95E8B0(v6, (__int64)&v99);
  sub_95BF90((__int64)v33, (__int64)&v111);
  if ( v99 != v101 )
    j_j___libc_free_0(v99, v101[0] + 1LL);
  sub_95CD70(&v111);
  v111 = v113;
  v114 = v116;
  v112 = 0;
  v117 = v119;
  LOBYTE(v113[0]) = 0;
  v115 = 0;
  v116[0] = 0;
  v118 = 0;
  v119[0] = 0;
  v120 = a1 + 552;
  sub_2241130(&v111, 0, 0, "-R __CUDA_ARCH=1200", 19);
  sub_2241130(&v114, 0, v115, "-opt-arch=sm_120", 16);
  sub_2241130(&v117, 0, v118, "-mcpu=sm_120", 12);
  sub_95BEE0((__int64 *)&v99, "-arch=compute_120");
  v34 = sub_95E8B0(v6, (__int64)&v99);
  sub_95BF90((__int64)v34, (__int64)&v111);
  if ( v99 != v101 )
    j_j___libc_free_0(v99, v101[0] + 1LL);
  sub_95CD70(&v111);
  sub_95CC90((__int64)&v111, "-R __CUDA_ARCH=1200", "-opt-arch=sm_120a", "-mcpu=sm_120a", (__int64)"-mcpu=sm_120a", v92);
  sub_95BEE0((__int64 *)&v99, "-arch=compute_120a");
  v35 = sub_95E8B0(v6, (__int64)&v99);
  sub_95BF90((__int64)v35, (__int64)&v111);
  if ( v99 != v101 )
    j_j___libc_free_0(v99, v101[0] + 1LL);
  sub_95CD70(&v111);
  sub_95CC90((__int64)&v111, "-R __CUDA_ARCH=1200", "-opt-arch=sm_120f", "-mcpu=sm_120f", (__int64)"-mcpu=sm_120f", v92);
  sub_95BEE0((__int64 *)&v99, "-arch=compute_120f");
  v36 = sub_95E8B0(v6, (__int64)&v99);
  sub_95BF90((__int64)v36, (__int64)&v111);
  if ( v99 != v101 )
    j_j___libc_free_0(v99, v101[0] + 1LL);
  sub_95CD70(&v111);
  v111 = v113;
  v114 = v116;
  v112 = 0;
  v117 = v119;
  LOBYTE(v113[0]) = 0;
  v115 = 0;
  v116[0] = 0;
  v118 = 0;
  v119[0] = 0;
  v120 = a1 + 552;
  sub_2241130(&v111, 0, 0, "-R __CUDA_ARCH=1210", 19);
  sub_2241130(&v114, 0, v115, "-opt-arch=sm_121", 16);
  sub_2241130(&v117, 0, v118, "-mcpu=sm_121", 12);
  sub_95BEE0((__int64 *)&v99, "-arch=compute_121");
  v37 = sub_95E8B0(v6, (__int64)&v99);
  sub_95BF90((__int64)v37, (__int64)&v111);
  if ( v99 != v101 )
    j_j___libc_free_0(v99, v101[0] + 1LL);
  sub_95CD70(&v111);
  v111 = v113;
  v114 = v116;
  v112 = 0;
  v117 = v119;
  LOBYTE(v113[0]) = 0;
  v115 = 0;
  v116[0] = 0;
  v118 = 0;
  v119[0] = 0;
  v120 = a1 + 552;
  sub_2241130(&v111, 0, 0, "-R __CUDA_ARCH=1210", 19);
  sub_2241130(&v114, 0, v115, "-opt-arch=sm_121a", 17);
  sub_2241130(&v117, 0, v118, "-mcpu=sm_121a", 13);
  sub_95BEE0((__int64 *)&v99, "-arch=compute_121a");
  v38 = sub_95E8B0(v6, (__int64)&v99);
  sub_95BF90((__int64)v38, (__int64)&v111);
  if ( v99 != v101 )
    j_j___libc_free_0(v99, v101[0] + 1LL);
  sub_95CD70(&v111);
  sub_95CC90((__int64)&v111, "-R __CUDA_ARCH=1210", "-opt-arch=sm_121f", "-mcpu=sm_121f", (__int64)"-mcpu=sm_121f", v92);
  sub_95BEE0((__int64 *)&v99, "-arch=compute_121f");
  v39 = sub_95E8B0(v6, (__int64)&v99);
  sub_95BF90((__int64)v39, (__int64)&v111);
  if ( v99 != v101 )
    j_j___libc_free_0(v99, v101[0] + 1LL);
  sub_95CD70(&v111);
  v116[0] = 0;
  v119[0] = 0;
  v114 = v116;
  v111 = v113;
  v117 = v119;
  v120 = a1 + 584;
  v112 = 0;
  LOBYTE(v113[0]) = 0;
  v115 = 0;
  v118 = 0;
  sub_95BEE0((__int64 *)&v99, "-ftz=0");
  v40 = sub_95E8B0(v6, (__int64)&v99);
  sub_95BF90((__int64)v40, (__int64)&v111);
  if ( v99 != v101 )
    j_j___libc_free_0(v99, v101[0] + 1LL);
  sub_95CD70(&v111);
  v111 = v113;
  v114 = v116;
  v112 = 0;
  v117 = v119;
  LOBYTE(v113[0]) = 0;
  v115 = 0;
  v116[0] = 0;
  v118 = 0;
  v119[0] = 0;
  v120 = a1 + 584;
  sub_2241130(&v111, 0, 0, "-R __CUDA_FTZ=1", 15);
  sub_2241130(&v114, 0, v115, "-nvptx-f32ftz", 13);
  sub_2241130(&v117, 0, v118, "-nvptx-f32ftz", 13);
  sub_95BEE0((__int64 *)&v99, "-ftz=1");
  v41 = sub_95E8B0(v6, (__int64)&v99);
  sub_95BF90((__int64)v41, (__int64)&v111);
  if ( v99 != v101 )
    j_j___libc_free_0(v99, v101[0] + 1LL);
  sub_95CD70(&v111);
  v111 = v113;
  v114 = v116;
  LOBYTE(v113[0]) = 0;
  v117 = v119;
  v112 = 0;
  v115 = 0;
  v116[0] = 0;
  v118 = 0;
  v119[0] = 0;
  v120 = a1 + 616;
  sub_2241130(&v117, 0, 0, "-nvptx-prec-sqrtf32=0", 21);
  sub_95BEE0((__int64 *)&v99, "-prec-sqrt=0");
  v42 = sub_95E8B0(v6, (__int64)&v99);
  sub_95BF90((__int64)v42, (__int64)&v111);
  if ( v99 != v101 )
    j_j___libc_free_0(v99, v101[0] + 1LL);
  sub_95CD70(&v111);
  v111 = v113;
  v114 = v116;
  v112 = 0;
  v117 = v119;
  LOBYTE(v113[0]) = 0;
  v115 = 0;
  v116[0] = 0;
  v118 = 0;
  v119[0] = 0;
  v120 = a1 + 616;
  sub_2241130(&v111, 0, 0, "-R __CUDA_PREC_SQRT=1", 21);
  sub_2241130(&v117, 0, v118, "-nvptx-prec-sqrtf32=1", 21);
  sub_95BEE0((__int64 *)&v99, "-prec-sqrt=1");
  v43 = sub_95E8B0(v6, (__int64)&v99);
  sub_95BF90((__int64)v43, (__int64)&v111);
  if ( v99 != v101 )
    j_j___libc_free_0(v99, v101[0] + 1LL);
  sub_95CD70(&v111);
  v116[0] = 0;
  v111 = v113;
  v112 = 0;
  LOBYTE(v113[0]) = 0;
  v114 = v116;
  v115 = 0;
  if ( a2 )
  {
    v120 = a1 + 648;
    v119[0] = 0;
    v117 = v119;
    v118 = 0;
    sub_2241130(&v114, 0, 0, "-opt-use-prec-div=false", 23);
    sub_2241130(&v117, 0, v118, "-nvptx-prec-divf32=0", 20);
    sub_95BEE0((__int64 *)&v99, "-prec-div=0");
    v44 = sub_95E8B0(v6, (__int64)&v99);
    sub_95BF90((__int64)v44, (__int64)&v111);
    sub_2240A30(&v99);
    sub_95CD70(&v111);
    v111 = v113;
    v117 = v119;
    v120 = a1 + 648;
    v112 = 0;
    LOBYTE(v113[0]) = 0;
    v114 = v116;
    v115 = 0;
    v116[0] = 0;
    v118 = 0;
    v119[0] = 0;
    sub_2241130(&v114, 0, 0, "-opt-use-prec-div=true", 22);
    sub_2241130(&v117, 0, v118, "-nvptx-prec-divf32=1", 20);
    sub_95BEE0((__int64 *)&v99, "-prec-div=1");
    v45 = sub_95E8B0(v6, (__int64)&v99);
    sub_95BF90((__int64)v45, (__int64)&v111);
    sub_2240A30(&v99);
    sub_95CD70(&v111);
    v116[0] = 0;
    v117 = v119;
    v120 = a1 + 648;
    v111 = v113;
    v112 = 0;
    LOBYTE(v113[0]) = 0;
    v114 = v116;
    v115 = 0;
    v118 = 0;
    v119[0] = 0;
    sub_2241130(&v117, 0, 0, "-nvptx-prec-divf32=3", 20);
    sub_95BEE0((__int64 *)&v99, "-prec-div=2");
    v46 = sub_95E8B0(v6, (__int64)&v99);
    sub_95BF90((__int64)v46, (__int64)&v111);
    if ( v99 != v101 )
      j_j___libc_free_0(v99, v101[0] + 1LL);
  }
  else
  {
    v120 = a1 + 648;
    v117 = v119;
    v118 = 0;
    v119[0] = 0;
    sub_2241130(&v114, 0, 0, "-opt-use-prec-div=false", 23);
    sub_2241130(&v117, 0, v118, "-nvptx-prec-divf32=1", 20);
    sub_95BEE0((__int64 *)&v99, "-prec-div=0");
    v89 = sub_95E8B0(v6, (__int64)&v99);
    sub_95BF90((__int64)v89, (__int64)&v111);
    sub_2240A30(&v99);
    sub_95CD70(&v111);
    sub_95CBB0((__int64)&v111, "-R __CUDA_PREC_DIV=1", "-opt-use-prec-div=true", "-nvptx-prec-divf32=2", a1 + 648);
    sub_95BEE0((__int64 *)&v99, "-prec-div=1");
    v90 = sub_95E8B0(v6, (__int64)&v99);
    sub_95BF90((__int64)v90, (__int64)&v111);
    sub_2240A30(&v99);
  }
  sub_95CD70(&v111);
  v111 = v113;
  v114 = v116;
  LOBYTE(v113[0]) = 0;
  v117 = v119;
  v112 = 0;
  v115 = 0;
  v116[0] = 0;
  v118 = 0;
  v119[0] = 0;
  v120 = a1 + 680;
  sub_2241130(&v117, 0, 0, "-nvptx-fma-level=0", 18);
  sub_95BEE0((__int64 *)&v99, "-fma=0");
  v47 = sub_95E8B0(v6, (__int64)&v99);
  sub_95BF90((__int64)v47, (__int64)&v111);
  if ( v99 != v101 )
    j_j___libc_free_0(v99, v101[0] + 1LL);
  sub_95CD70(&v111);
  v111 = v113;
  v114 = v116;
  v112 = 0;
  v117 = v119;
  LOBYTE(v113[0]) = 0;
  v115 = 0;
  v116[0] = 0;
  v118 = 0;
  v119[0] = 0;
  v120 = a1 + 680;
  sub_2241130(&v117, 0, 0, "-nvptx-fma-level=1 ", 19);
  sub_95BEE0((__int64 *)&v99, "-fma=1");
  v48 = sub_95E8B0(v6, (__int64)&v99);
  sub_95BF90((__int64)v48, (__int64)&v111);
  if ( v99 != v101 )
    j_j___libc_free_0(v99, v101[0] + 1LL);
  sub_95CD70(&v111);
  v111 = v113;
  v114 = v116;
  v112 = 0;
  v117 = v119;
  LOBYTE(v113[0]) = 0;
  v115 = 0;
  v116[0] = 0;
  v118 = 0;
  v119[0] = 0;
  v120 = a1 + 712;
  sub_2241130(&v117, 0, 0, "-nvptx-fma-level=1 ", 19);
  sub_95BEE0((__int64 *)&v99, "-enable-mad");
  v49 = sub_95E8B0(v6, (__int64)&v99);
  sub_95BF90((__int64)v49, (__int64)&v111);
  if ( v99 != v101 )
    j_j___libc_free_0(v99, v101[0] + 1LL);
  sub_95CD70(&v111);
  v111 = v113;
  v114 = v116;
  v112 = 0;
  v117 = v119;
  LOBYTE(v113[0]) = 0;
  v115 = 0;
  v116[0] = 0;
  v118 = 0;
  v119[0] = 0;
  v120 = a1 + 744;
  sub_2241130(&v111, 0, 0, "-R FAST_RELAXED_MATH=1 -R __CUDA_FTZ=1", 38);
  sub_2241130(&v114, 0, v115, "-opt-use-fast-math -nvptx-f32ftz", 32);
  sub_2241130(&v117, 0, v118, "-nvptx-fma-level=1 -nvptx-f32ftz", 32);
  sub_95BEE0((__int64 *)&v99, "-unsafe-math");
  v50 = sub_95E8B0(v6, (__int64)&v99);
  sub_95BF90((__int64)v50, (__int64)&v111);
  if ( v99 != v101 )
    j_j___libc_free_0(v99, v101[0] + 1LL);
  sub_95CD70(&v111);
  if ( a2 )
  {
    sub_95CBB0(
      (__int64)&v111,
      "-R FAST_RELAXED_MATH=1 -R __CUDA_FTZ=1",
      "-opt-use-fast-math -nvptx-f32ftz",
      "-nvptx-f32ftz",
      a1 + 776);
  }
  else
  {
    v120 = a1 + 776;
    v111 = v113;
    v114 = v116;
    v112 = 0;
    LOBYTE(v113[0]) = 0;
    v115 = 0;
    v116[0] = 0;
    v117 = v119;
    v118 = 0;
    v119[0] = 0;
    sub_2241130(&v111, 0, 0, "-R __CUDA_USE_FAST_MATH=1", 25);
    sub_2241130(&v114, 0, v115, "-opt-use-fast-math ", 19);
  }
  sub_95BEE0((__int64 *)&v99, "-fast-math");
  v51 = sub_95E8B0(v6, (__int64)&v99);
  sub_95BF90((__int64)v51, (__int64)&v111);
  sub_2240A30(&v99);
  sub_95CD70(&v111);
  v111 = v113;
  v114 = v116;
  v112 = 0;
  v117 = v119;
  LOBYTE(v113[0]) = 0;
  v115 = 0;
  v116[0] = 0;
  v118 = 0;
  v119[0] = 0;
  v120 = a1 + 808;
  sub_2241130(&v117, 0, 0, "-nvptx-emit-src", 15);
  sub_95BEE0((__int64 *)&v99, "-show-src");
  v52 = sub_95E8B0(v6, (__int64)&v99);
  sub_95BF90((__int64)v52, (__int64)&v111);
  if ( v99 != v101 )
    j_j___libc_free_0(v99, v101[0] + 1LL);
  sub_95CD70(&v111);
  v111 = v113;
  v114 = v116;
  v112 = 0;
  v117 = v119;
  LOBYTE(v113[0]) = 0;
  v115 = 0;
  v116[0] = 0;
  v118 = 0;
  v119[0] = 0;
  v120 = a1 + 1032;
  sub_2241130(&v114, 0, 0, "-enable-opt-byval", 17);
  sub_95BEE0((__int64 *)&v99, "-enable-opt-byval");
  v53 = sub_95E8B0(v6, (__int64)&v99);
  sub_95BF90((__int64)v53, (__int64)&v111);
  if ( v99 != v101 )
    j_j___libc_free_0(v99, v101[0] + 1LL);
  sub_95CD70(&v111);
  v111 = v113;
  v114 = v116;
  v112 = 0;
  v117 = v119;
  LOBYTE(v113[0]) = 0;
  v115 = 0;
  v120 = a1 + 840;
  v116[0] = 0;
  v118 = 0;
  v119[0] = 0;
  sub_95BEE0((__int64 *)&v99, "-disable-llc-opts");
  v54 = sub_95E8B0(v6, (__int64)&v99);
  sub_95BF90((__int64)v54, (__int64)&v111);
  if ( v99 != v101 )
    j_j___libc_free_0(v99, v101[0] + 1LL);
  sub_95CD70(&v111);
  v111 = v113;
  v114 = v116;
  v112 = 0;
  v117 = v119;
  LOBYTE(v113[0]) = 0;
  v115 = 0;
  v116[0] = 0;
  v118 = 0;
  v119[0] = 0;
  v120 = a1 + 872;
  sub_2241130(&v114, 0, 0, "-w", 2);
  sub_2241130(&v117, 0, v118, "-w", 2);
  sub_95BEE0((__int64 *)&v99, "-w");
  v55 = sub_95E8B0(v6, (__int64)&v99);
  sub_95BF90((__int64)v55, (__int64)&v111);
  if ( v99 != v101 )
    j_j___libc_free_0(v99, v101[0] + 1LL);
  sub_95CD70(&v111);
  v111 = v113;
  v114 = v116;
  v112 = 0;
  v117 = v119;
  LOBYTE(v113[0]) = 0;
  v115 = 0;
  v116[0] = 0;
  v118 = 0;
  v119[0] = 0;
  v120 = a1 + 904;
  sub_2241130(&v114, 0, 0, "-Werror", 7);
  sub_2241130(&v117, 0, v118, "-Werror", 7);
  sub_95BEE0((__int64 *)&v99, "-Werror");
  v56 = sub_95E8B0(v6, (__int64)&v99);
  sub_95BF90((__int64)v56, (__int64)&v111);
  if ( v99 != v101 )
    j_j___libc_free_0(v99, v101[0] + 1LL);
  sub_95CD70(&v111);
  v111 = v113;
  v114 = v116;
  v112 = 0;
  v117 = v119;
  LOBYTE(v113[0]) = 0;
  v115 = 0;
  v116[0] = 0;
  v118 = 0;
  v119[0] = 0;
  v120 = a1 + 1064;
  sub_2241130(&v114, 0, 0, "-disable-inlining", 17);
  sub_95BEE0((__int64 *)&v99, "-disable-inlining");
  v57 = sub_95E8B0(v6, (__int64)&v99);
  sub_95BF90((__int64)v57, (__int64)&v111);
  if ( v99 != v101 )
    j_j___libc_free_0(v99, v101[0] + 1LL);
  sub_95CD70(&v111);
  v111 = v113;
  v114 = v116;
  v112 = 0;
  v117 = v119;
  LOBYTE(v113[0]) = 0;
  v115 = 0;
  v116[0] = 0;
  v118 = 0;
  v119[0] = 0;
  v120 = a1 + 1608;
  sub_2241130(&v114, 0, 0, "-inline-budget=40000", 20);
  sub_95BEE0((__int64 *)&v99, "-aggressive-inline");
  v58 = sub_95E8B0(v6, (__int64)&v99);
  sub_95BF90((__int64)v58, (__int64)&v111);
  if ( v99 != v101 )
    j_j___libc_free_0(v99, v101[0] + 1LL);
  sub_95CD70(&v111);
  v111 = v113;
  v114 = v116;
  v112 = 0;
  v117 = v119;
  LOBYTE(v113[0]) = 0;
  v115 = 0;
  v116[0] = 0;
  v118 = 0;
  v119[0] = 0;
  v120 = a1 + 1096;
  sub_2241130(&v117, 0, 0, "-nvptx-kernel-params-restrict", 29);
  sub_95BEE0((__int64 *)&v99, "-restrict");
  v59 = sub_95E8B0(v6, (__int64)&v99);
  sub_95BF90((__int64)v59, (__int64)&v111);
  if ( v99 != v101 )
    j_j___libc_free_0(v99, v101[0] + 1LL);
  sub_95CD70(&v111);
  v111 = v113;
  v114 = v116;
  v112 = 0;
  v117 = v119;
  LOBYTE(v113[0]) = 0;
  v115 = 0;
  v116[0] = 0;
  v118 = 0;
  v119[0] = 0;
  v120 = a1 + 1128;
  sub_2241130(&v114, 0, 0, "-allow-restrict-in-struct", 25);
  sub_2241130(&v117, 0, v118, "-allow-restrict-in-struct", 25);
  sub_95BEE0((__int64 *)&v99, "-allow-restrict-in-struct");
  v60 = sub_95E8B0(v6, (__int64)&v99);
  sub_95BF90((__int64)v60, (__int64)&v111);
  if ( v99 != v101 )
    j_j___libc_free_0(v99, v101[0] + 1LL);
  sub_95CD70(&v111);
  v111 = v113;
  v114 = v116;
  v112 = 0;
  v117 = v119;
  LOBYTE(v113[0]) = 0;
  v115 = 0;
  v116[0] = 0;
  v118 = 0;
  v119[0] = 0;
  v120 = a1 + 1160;
  sub_2241130(&v114, 0, 0, "-opt-no-signed-zeros", 20);
  sub_95BEE0((__int64 *)&v99, "-no-signed-zeros");
  v61 = sub_95E8B0(v6, (__int64)&v99);
  sub_95BF90((__int64)v61, (__int64)&v111);
  if ( v99 != v101 )
    j_j___libc_free_0(v99, v101[0] + 1LL);
  sub_95CD70(&v111);
  v111 = v113;
  v114 = v116;
  v112 = 0;
  v117 = v119;
  LOBYTE(v113[0]) = 0;
  v115 = 0;
  v116[0] = 0;
  v118 = 0;
  v119[0] = 0;
  v120 = a1 + 1224;
  sub_2241130(&v117, 0, 0, "-asm-verbose", 12);
  sub_95BEE0((__int64 *)&v99, "-enable-verbose-asm");
  v62 = sub_95E8B0(v6, (__int64)&v99);
  sub_95BF90((__int64)v62, (__int64)&v111);
  if ( v99 != v101 )
    j_j___libc_free_0(v99, v101[0] + 1LL);
  sub_95CD70(&v111);
  v111 = v113;
  v114 = v116;
  LOBYTE(v113[0]) = 0;
  v117 = v119;
  v112 = 0;
  v115 = 0;
  v116[0] = 0;
  v118 = 0;
  v119[0] = 0;
  v120 = a1 + 456;
  sub_2241130(&v114, 0, 0, "-opt-fdiv=0", 11);
  sub_95BEE0((__int64 *)&v99, "-opt-fdiv=0");
  v63 = sub_95E8B0(v6, (__int64)&v99);
  sub_95BF90((__int64)v63, (__int64)&v111);
  if ( v99 != v101 )
    j_j___libc_free_0(v99, v101[0] + 1LL);
  sub_95CD70(&v111);
  v111 = v113;
  v114 = v116;
  v112 = 0;
  v117 = v119;
  LOBYTE(v113[0]) = 0;
  v115 = 0;
  v116[0] = 0;
  v118 = 0;
  v119[0] = 0;
  v120 = a1 + 456;
  sub_2241130(&v114, 0, 0, "-opt-fdiv=1", 11);
  sub_95BEE0((__int64 *)&v99, "-opt-fdiv=1");
  v64 = sub_95E8B0(v6, (__int64)&v99);
  sub_95BF90((__int64)v64, (__int64)&v111);
  if ( v99 != v101 )
    j_j___libc_free_0(v99, v101[0] + 1LL);
  sub_95CD70(&v111);
  v111 = v113;
  v114 = v116;
  v112 = 0;
  v117 = v119;
  LOBYTE(v113[0]) = 0;
  v115 = 0;
  v116[0] = 0;
  v118 = 0;
  v119[0] = 0;
  v120 = a1 + 1352;
  sub_2241130(&v117, 0, 0, "-vasp-fix1=true -vasp-fix2=true", 31);
  sub_95BEE0((__int64 *)&v99, "-vasp-fix");
  v65 = sub_95E8B0(v6, (__int64)&v99);
  sub_95BF90((__int64)v65, (__int64)&v111);
  if ( v99 != v101 )
    j_j___libc_free_0(v99, v101[0] + 1LL);
  sub_95CD70(&v111);
  v111 = v113;
  v114 = v116;
  v112 = 0;
  v117 = v119;
  LOBYTE(v113[0]) = 0;
  v115 = 0;
  v116[0] = 0;
  v118 = 0;
  v119[0] = 0;
  v120 = a1 + 1384;
  sub_2241130(
    &v117,
    0,
    0,
    "-enable-new-nvvm-remat=true                                             -nv-disable-remat=true                      "
    "                       -rp-aware-mcse=true",
    158);
  sub_95BEE0((__int64 *)&v99, "-new-nvvm-remat");
  v66 = sub_95E8B0(v6, (__int64)&v99);
  sub_95BF90((__int64)v66, (__int64)&v111);
  if ( v99 != v101 )
    j_j___libc_free_0(v99, v101[0] + 1LL);
  sub_95CD70(&v111);
  v111 = v113;
  v114 = v116;
  v112 = 0;
  v117 = v119;
  LOBYTE(v113[0]) = 0;
  v115 = 0;
  v116[0] = 0;
  v118 = 0;
  v119[0] = 0;
  v120 = a1 + 1416;
  sub_2241130(
    &v117,
    0,
    0,
    "-enable-new-nvvm-remat=false                                             -nv-disable-remat=false                    "
    "                         -rp-aware-mcse=false",
    161);
  sub_95BEE0((__int64 *)&v99, "-disable-new-nvvm-remat");
  v67 = sub_95E8B0(v6, (__int64)&v99);
  sub_95BF90((__int64)v67, (__int64)&v111);
  if ( v99 != v101 )
    j_j___libc_free_0(v99, v101[0] + 1LL);
  sub_95CD70(&v111);
  v111 = v113;
  v114 = v116;
  v112 = 0;
  v117 = v119;
  LOBYTE(v113[0]) = 0;
  v115 = 0;
  v116[0] = 0;
  v118 = 0;
  v119[0] = 0;
  v120 = a1 + 1448;
  sub_2241130(
    &v117,
    0,
    0,
    "-enable-new-nvvm-remat=false                                             -nv-disable-remat=true                     "
    "                        -rp-aware-mcse=false",
    160);
  sub_95BEE0((__int64 *)&v99, "-disable-nvvm-remat");
  v68 = sub_95E8B0(v6, (__int64)&v99);
  sub_95BF90((__int64)v68, (__int64)&v111);
  if ( v99 != v101 )
    j_j___libc_free_0(v99, v101[0] + 1LL);
  sub_95CD70(&v111);
  v111 = v113;
  v114 = v116;
  v112 = 0;
  v117 = v119;
  LOBYTE(v113[0]) = 0;
  v115 = 0;
  v116[0] = 0;
  v118 = 0;
  v119[0] = 0;
  v120 = a1 + 1544;
  sub_2241130(&v114, 0, 0, "-aggressive-positive-stride-analysis=false", 42);
  sub_95BEE0((__int64 *)&v99, "-no-aggressive-positive-stride-analysis");
  v69 = sub_95E8B0(v6, (__int64)&v99);
  sub_95BF90((__int64)v69, (__int64)&v111);
  if ( v99 != v101 )
    j_j___libc_free_0(v99, v101[0] + 1LL);
  sub_95CD70(&v111);
  v111 = v113;
  v114 = v116;
  v112 = 0;
  v117 = v119;
  LOBYTE(v113[0]) = 0;
  v115 = 0;
  v116[0] = 0;
  v118 = 0;
  v119[0] = 0;
  v120 = a1 + 1576;
  sub_2241130(&v114, 0, 0, "-disable-load-select-transform=true", 35);
  sub_95BEE0((__int64 *)&v99, "disable-load-select-transform");
  v70 = sub_95E8B0(v6, (__int64)&v99);
  sub_95BF90((__int64)v70, (__int64)&v111);
  if ( v99 != v101 )
    j_j___libc_free_0(v99, v101[0] + 1LL);
  sub_95CD70(&v111);
  v111 = v113;
  sub_95BB50((__int64 *)&v111, (char)&byte_3F157FB[-4], &byte_3F157FB[-4], byte_3F157FB);
  sub_95BEE0((__int64 *)&v99, "-");
  v71 = 15;
  v72 = 15;
  if ( v99 != v101 )
    v72 = v101[0];
  v73 = v100 + v112;
  if ( v100 + v112 <= v72 )
    goto LABEL_138;
  if ( v111 != v113 )
    v71 = v113[0];
  if ( v73 <= v71 )
  {
    v74 = (__m128i *)sub_2241130(&v111, 0, 0, v99, v100);
    v75 = v74 + 1;
    s2 = &v98;
    v76 = (__m128i *)v74->m128i_i64[0];
    if ( (__m128i *)v74->m128i_i64[0] != &v74[1] )
      goto LABEL_139;
  }
  else
  {
LABEL_138:
    v74 = (__m128i *)sub_2241490(&v99, v111, v112, v73);
    v75 = v74 + 1;
    s2 = &v98;
    v76 = (__m128i *)v74->m128i_i64[0];
    if ( (__m128i *)v74->m128i_i64[0] != &v74[1] )
    {
LABEL_139:
      s2 = v76;
      v98.m128i_i64[0] = v74[1].m128i_i64[0];
      goto LABEL_140;
    }
  }
  v98 = _mm_loadu_si128(v74 + 1);
LABEL_140:
  v97 = v74->m128i_u64[1];
  v74->m128i_i64[0] = (__int64)v75;
  v74->m128i_i64[1] = 0;
  v74[1].m128i_i8[0] = 0;
  if ( v99 != v101 )
    j_j___libc_free_0(v99, v101[0] + 1LL);
  if ( v111 != v113 )
    j_j___libc_free_0(v111, v113[0] + 1LL);
  sub_95CC90((__int64)&v111, s2->m128i_i8, s2->m128i_i8, s2->m128i_i8, (__int64)s2, a1 + 1256);
  if ( *(_QWORD *)(a1 + 264) )
  {
    v77 = *(_QWORD *)(a1 + 264);
    v78 = v91;
    while ( 1 )
    {
      v79 = *(_QWORD *)(v77 + 40);
      v80 = v97;
      if ( v79 <= v97 )
        v80 = *(_QWORD *)(v77 + 40);
      if ( v80 )
      {
        v81 = memcmp(*(const void **)(v77 + 32), s2, v80);
        if ( v81 )
          goto LABEL_154;
      }
      v82 = v79 - v97;
      if ( v82 >= 0x80000000LL )
      {
LABEL_155:
        v78 = v77;
        v77 = *(_QWORD *)(v77 + 16);
        if ( !v77 )
        {
LABEL_156:
          if ( v91 != v78 )
          {
            v83 = v78 + 64;
            if ( sub_95B9E0(s2, v97, *(const void **)(v78 + 32), *(_QWORD *)(v78 + 40)) >= 0 )
              goto LABEL_158;
          }
          goto LABEL_162;
        }
      }
      else
      {
        if ( v82 <= (__int64)0xFFFFFFFF7FFFFFFFLL )
          goto LABEL_146;
        v81 = v82;
LABEL_154:
        if ( v81 >= 0 )
          goto LABEL_155;
LABEL_146:
        v77 = *(_QWORD *)(v77 + 24);
        if ( !v77 )
          goto LABEL_156;
      }
    }
  }
  v78 = v91;
LABEL_162:
  v94 = (_QWORD *)v78;
  v84 = sub_22077B0(168);
  *(_QWORD *)(v84 + 32) = v84 + 48;
  v83 = v84 + 64;
  sub_95BD60((__int64 *)(v84 + 32), s2, (__int64)s2->m128i_i64 + v97);
  *(_QWORD *)(v84 + 64) = v84 + 80;
  *(_QWORD *)(v84 + 96) = v84 + 112;
  *(_QWORD *)(v84 + 72) = 0;
  *(_BYTE *)(v84 + 80) = 0;
  *(_QWORD *)(v84 + 104) = 0;
  *(_BYTE *)(v84 + 112) = 0;
  *(_QWORD *)(v84 + 128) = v84 + 144;
  *(_QWORD *)(v84 + 136) = 0;
  *(_BYTE *)(v84 + 144) = 0;
  *(_QWORD *)(v84 + 160) = 0;
  v85 = sub_95E620(v6, v94, v84 + 32);
  v87 = v86;
  if ( v86 )
  {
    if ( v85 || v91 == v86 )
      v88 = 1;
    else
      v88 = (unsigned int)sub_95B9E0(
                            *(const void **)(v84 + 32),
                            *(_QWORD *)(v84 + 40),
                            *(const void **)(v86 + 32),
                            *(_QWORD *)(v86 + 40)) >> 31;
    sub_220F040(v88, v84, v87, v91);
    ++*(_QWORD *)(a1 + 288);
  }
  else
  {
    v95 = v85;
    sub_95CD70((_QWORD *)(v84 + 64));
    sub_2240A30(v84 + 32);
    j_j___libc_free_0(v84, 168);
    v83 = v95 + 64;
  }
LABEL_158:
  sub_95BF90(v83, (__int64)&v111);
  sub_95CD70(&v111);
  if ( s2 != &v98 )
    j_j___libc_free_0(s2, v98.m128i_i64[0] + 1);
  return (_BYTE *)sub_95CD70(v102);
}
