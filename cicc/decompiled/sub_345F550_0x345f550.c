// Function: sub_345F550
// Address: 0x345f550
//
unsigned __int8 *__fastcall sub_345F550(__int64 a1, __int64 a2, _QWORD *a3)
{
  __int64 v6; // rsi
  __int64 *v7; // rdi
  __int16 *v8; // rax
  __int16 v9; // dx
  __int64 v10; // rax
  __int128 v11; // xmm0
  unsigned int v12; // ebx
  __int64 v13; // rax
  __int64 v14; // rsi
  unsigned int v15; // eax
  unsigned __int16 v16; // r13
  __int64 v17; // rdx
  __int64 v18; // rdx
  __int64 v19; // rsi
  unsigned __int8 *v20; // r13
  __int64 v22; // r9
  __int128 v23; // rax
  __int64 v24; // r9
  unsigned int v25; // edx
  __int64 v26; // r15
  __int128 v27; // rax
  __int64 v28; // r9
  unsigned int v29; // edx
  unsigned __int64 v30; // r15
  __int128 v31; // rax
  __int64 v32; // r9
  unsigned int v33; // edx
  __int128 v34; // rax
  __int64 v35; // r9
  unsigned int v36; // edx
  __int64 v37; // r9
  unsigned int v38; // edx
  __int128 v39; // rax
  __int64 v40; // r9
  unsigned int v41; // edx
  unsigned __int64 v42; // r15
  __int128 v43; // rax
  __int64 v44; // r9
  unsigned int v45; // edx
  unsigned __int64 v46; // r15
  __int128 v47; // rax
  __int64 v48; // r9
  unsigned int v49; // edx
  __int128 v50; // rax
  __int64 v51; // r9
  unsigned int v52; // edx
  __int64 v53; // r9
  unsigned int v54; // edx
  __int128 v55; // rax
  __int64 v56; // r9
  unsigned int v57; // edx
  unsigned __int64 v58; // r15
  __int128 v59; // rax
  __int64 v60; // r9
  unsigned int v61; // edx
  unsigned __int64 v62; // r15
  __int128 v63; // rax
  __int64 v64; // r9
  unsigned int v65; // edx
  __int128 v66; // rax
  __int64 v67; // r9
  unsigned int v68; // edx
  __int64 v69; // r9
  __int64 v70; // rdx
  __int64 v71; // rcx
  __int64 v72; // r8
  unsigned int v73; // edx
  __int128 v74; // [rsp-50h] [rbp-240h]
  __int128 v75; // [rsp-50h] [rbp-240h]
  __int128 v76; // [rsp-50h] [rbp-240h]
  __int128 v77; // [rsp-50h] [rbp-240h]
  __int128 v78; // [rsp-40h] [rbp-230h]
  __int128 v79; // [rsp-40h] [rbp-230h]
  __int128 v80; // [rsp-40h] [rbp-230h]
  __int128 v81; // [rsp-30h] [rbp-220h]
  __int128 v82; // [rsp-30h] [rbp-220h]
  __int128 v83; // [rsp-30h] [rbp-220h]
  __int64 v84; // [rsp+18h] [rbp-1D8h]
  unsigned int v85; // [rsp+20h] [rbp-1D0h]
  __int64 v86; // [rsp+28h] [rbp-1C8h]
  __int128 v87; // [rsp+30h] [rbp-1C0h]
  __int128 v88; // [rsp+40h] [rbp-1B0h]
  __int128 v89; // [rsp+50h] [rbp-1A0h]
  __int128 v90; // [rsp+60h] [rbp-190h]
  unsigned __int8 *v91; // [rsp+70h] [rbp-180h]
  unsigned __int8 *v92; // [rsp+90h] [rbp-160h]
  unsigned __int8 *v93; // [rsp+A0h] [rbp-150h]
  unsigned __int8 *v94; // [rsp+C0h] [rbp-130h]
  unsigned __int8 *v95; // [rsp+E0h] [rbp-110h]
  unsigned __int8 *v96; // [rsp+F0h] [rbp-100h]
  unsigned __int8 *v97; // [rsp+110h] [rbp-E0h]
  unsigned __int8 *v98; // [rsp+130h] [rbp-C0h]
  unsigned __int8 *v99; // [rsp+140h] [rbp-B0h]
  __int64 v100; // [rsp+150h] [rbp-A0h] BYREF
  int v101; // [rsp+158h] [rbp-98h]
  unsigned int v102; // [rsp+160h] [rbp-90h] BYREF
  __int64 v103; // [rsp+168h] [rbp-88h]
  unsigned __int64 v104; // [rsp+170h] [rbp-80h] BYREF
  unsigned int v105; // [rsp+178h] [rbp-78h]
  unsigned __int64 v106; // [rsp+180h] [rbp-70h] BYREF
  unsigned int v107; // [rsp+188h] [rbp-68h]
  unsigned __int64 v108; // [rsp+190h] [rbp-60h] BYREF
  unsigned int v109; // [rsp+198h] [rbp-58h]
  unsigned __int64 v110; // [rsp+1A0h] [rbp-50h] BYREF
  __int64 v111; // [rsp+1A8h] [rbp-48h]

  v6 = *(_QWORD *)(a2 + 80);
  v100 = v6;
  if ( v6 )
    sub_B96E90((__int64)&v100, v6, 1);
  v7 = (__int64 *)a3[5];
  v101 = *(_DWORD *)(a2 + 72);
  v8 = *(__int16 **)(a2 + 48);
  v9 = *v8;
  v103 = *((_QWORD *)v8 + 1);
  v10 = *(_QWORD *)(a2 + 40);
  LOWORD(v102) = v9;
  v11 = (__int128)_mm_loadu_si128((const __m128i *)v10);
  v12 = *(_DWORD *)(v10 + 8);
  v84 = *(_QWORD *)v10;
  v89 = (__int128)_mm_loadu_si128((const __m128i *)(v10 + 40));
  v88 = (__int128)_mm_loadu_si128((const __m128i *)(v10 + 80));
  v13 = sub_2E79000(v7);
  v14 = v102;
  v15 = sub_2FE6750(a1, v102, v103, v13);
  v16 = v102;
  v86 = v17;
  v85 = v15;
  if ( (_WORD)v102 )
  {
    if ( (unsigned __int16)(v102 - 17) <= 0xD3u )
    {
      v111 = 0;
      v16 = word_4456580[(unsigned __int16)v102 - 1];
      LOWORD(v110) = v16;
      if ( !v16 )
        goto LABEL_7;
      goto LABEL_35;
    }
    goto LABEL_5;
  }
  if ( !sub_30070B0((__int64)&v102) )
  {
LABEL_5:
    v18 = v103;
    goto LABEL_6;
  }
  v16 = sub_3009970((__int64)&v102, v14, v70, v71, v72);
LABEL_6:
  LOWORD(v110) = v16;
  v111 = v18;
  if ( !v16 )
  {
LABEL_7:
    LODWORD(v19) = sub_3007260((__int64)&v110);
    goto LABEL_8;
  }
LABEL_35:
  if ( v16 == 1 || (unsigned __int16)(v16 - 504) <= 7u )
    BUG();
  v19 = *(_QWORD *)&byte_444C4A0[16 * v16 - 16];
LABEL_8:
  if ( (unsigned int)v19 <= 7 || ((unsigned int)v19 & ((_DWORD)v19 - 1)) != 0 )
  {
    v20 = 0;
  }
  else
  {
    LODWORD(v111) = 8;
    v110 = 15;
    sub_C47700((__int64)&v104, v19, (__int64)&v110);
    if ( (unsigned int)v111 > 0x40 && v110 )
      j_j___libc_free_0_0(v110);
    LODWORD(v111) = 8;
    v110 = 51;
    sub_C47700((__int64)&v106, v19, (__int64)&v110);
    if ( (unsigned int)v111 > 0x40 && v110 )
      j_j___libc_free_0_0(v110);
    LODWORD(v111) = 8;
    v110 = 85;
    sub_C47700((__int64)&v108, v19, (__int64)&v110);
    if ( (unsigned int)v111 > 0x40 && v110 )
      j_j___libc_free_0_0(v110);
    if ( (_DWORD)v19 != 8 )
    {
      v84 = sub_340F900(a3, 0x19Du, (__int64)&v100, v102, v103, v22, v11, v89, v88);
      v12 = v73;
    }
    *(_QWORD *)&v23 = sub_3400BD0((__int64)a3, 4, (__int64)&v100, v85, v86, 0, (__m128i)v11, 0);
    *(_QWORD *)&v87 = v84;
    *((_QWORD *)&v87 + 1) = v12;
    *((_QWORD *)&v74 + 1) = v12;
    *(_QWORD *)&v74 = v84;
    v99 = sub_33FC130(a3, 398, (__int64)&v100, v102, v103, v24, v74, v23, v89, v88);
    v26 = v25;
    *(_QWORD *)&v27 = sub_34007B0((__int64)a3, (__int64)&v104, (__int64)&v100, v102, v103, 0, (__m128i)v11, 0);
    *((_QWORD *)&v75 + 1) = v26;
    *(_QWORD *)&v75 = v99;
    v98 = sub_33FC130(a3, 396, (__int64)&v100, v102, v103, v28, v75, v27, v89, v88);
    v30 = v29 | v26 & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v31 = sub_34007B0((__int64)a3, (__int64)&v104, (__int64)&v100, v102, v103, 0, (__m128i)v11, 0);
    *(_QWORD *)&v90 = sub_33FC130(a3, 396, (__int64)&v100, v102, v103, v32, v87, v31, v89, v88);
    *((_QWORD *)&v90 + 1) = v33;
    *(_QWORD *)&v34 = sub_3400BD0((__int64)a3, 4, (__int64)&v100, v85, v86, 0, (__m128i)v11, 0);
    v97 = sub_33FC130(a3, 402, (__int64)&v100, v102, v103, v35, v90, v34, v89, v88);
    *((_QWORD *)&v90 + 1) = v36 | *((_QWORD *)&v90 + 1) & 0xFFFFFFFF00000000LL;
    *((_QWORD *)&v81 + 1) = *((_QWORD *)&v90 + 1);
    *(_QWORD *)&v81 = v97;
    *((_QWORD *)&v78 + 1) = v30;
    *(_QWORD *)&v78 = v98;
    *(_QWORD *)&v87 = sub_33FC130(a3, 400, (__int64)&v100, v102, v103, v37, v78, v81, v89, v88);
    *((_QWORD *)&v87 + 1) = v38;
    *(_QWORD *)&v39 = sub_3400BD0((__int64)a3, 2, (__int64)&v100, v85, v86, 0, (__m128i)v11, 0);
    v96 = sub_33FC130(a3, 398, (__int64)&v100, v102, v103, v40, v87, v39, v89, v88);
    v42 = v41 | v30 & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v43 = sub_34007B0((__int64)a3, (__int64)&v106, (__int64)&v100, v102, v103, 0, (__m128i)v11, 0);
    *((_QWORD *)&v76 + 1) = v42;
    *(_QWORD *)&v76 = v96;
    v95 = sub_33FC130(a3, 396, (__int64)&v100, v102, v103, v44, v76, v43, v89, v88);
    v46 = v45 | v42 & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v47 = sub_34007B0((__int64)a3, (__int64)&v106, (__int64)&v100, v102, v103, 0, (__m128i)v11, 0);
    *(_QWORD *)&v90 = sub_33FC130(a3, 396, (__int64)&v100, v102, v103, v48, v87, v47, v89, v88);
    *((_QWORD *)&v90 + 1) = v49 | *((_QWORD *)&v90 + 1) & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v50 = sub_3400BD0((__int64)a3, 2, (__int64)&v100, v85, v86, 0, (__m128i)v11, 0);
    v94 = sub_33FC130(a3, 402, (__int64)&v100, v102, v103, v51, v90, v50, v89, v88);
    *((_QWORD *)&v90 + 1) = v52 | *((_QWORD *)&v90 + 1) & 0xFFFFFFFF00000000LL;
    *((_QWORD *)&v82 + 1) = *((_QWORD *)&v90 + 1);
    *(_QWORD *)&v82 = v94;
    *((_QWORD *)&v79 + 1) = v46;
    *(_QWORD *)&v79 = v95;
    *(_QWORD *)&v87 = sub_33FC130(a3, 400, (__int64)&v100, v102, v103, v53, v79, v82, v89, v88);
    *((_QWORD *)&v87 + 1) = v54 | *((_QWORD *)&v87 + 1) & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v55 = sub_3400BD0((__int64)a3, 1, (__int64)&v100, v85, v86, 0, (__m128i)v11, 0);
    v93 = sub_33FC130(a3, 398, (__int64)&v100, v102, v103, v56, v87, v55, v89, v88);
    v58 = v57 | v46 & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v59 = sub_34007B0((__int64)a3, (__int64)&v108, (__int64)&v100, v102, v103, 0, (__m128i)v11, 0);
    *((_QWORD *)&v77 + 1) = v58;
    *(_QWORD *)&v77 = v93;
    v92 = sub_33FC130(a3, 396, (__int64)&v100, v102, v103, v60, v77, v59, v89, v88);
    v62 = v61 | v58 & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v63 = sub_34007B0((__int64)a3, (__int64)&v108, (__int64)&v100, v102, v103, 0, (__m128i)v11, 0);
    *(_QWORD *)&v90 = sub_33FC130(a3, 396, (__int64)&v100, v102, v103, v64, v87, v63, v89, v88);
    *((_QWORD *)&v90 + 1) = v65 | *((_QWORD *)&v90 + 1) & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v66 = sub_3400BD0((__int64)a3, 1, (__int64)&v100, v85, v86, 0, (__m128i)v11, 0);
    v91 = sub_33FC130(a3, 402, (__int64)&v100, v102, v103, v67, v90, v66, v89, v88);
    *((_QWORD *)&v83 + 1) = v68 | *((_QWORD *)&v90 + 1) & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v83 = v91;
    *((_QWORD *)&v80 + 1) = v62;
    *(_QWORD *)&v80 = v92;
    v20 = sub_33FC130(a3, 400, (__int64)&v100, v102, v103, v69, v80, v83, v89, v88);
    if ( v109 > 0x40 && v108 )
      j_j___libc_free_0_0(v108);
    if ( v107 > 0x40 && v106 )
      j_j___libc_free_0_0(v106);
    if ( v105 > 0x40 && v104 )
      j_j___libc_free_0_0(v104);
  }
  if ( v100 )
    sub_B91220((__int64)&v100, v100);
  return v20;
}
