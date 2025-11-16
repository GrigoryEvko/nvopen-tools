// Function: sub_3263260
// Address: 0x3263260
//
__int64 __fastcall sub_3263260(
        __int64 a1,
        __int64 a2,
        int a3,
        __int64 a4,
        int a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9,
        __int64 a10,
        __int64 a11,
        __int64 a12)
{
  __int64 *v13; // rax
  __int64 v14; // r14
  __int64 v15; // rax
  __int64 v16; // rdi
  __int64 v17; // rax
  const __m128i *v18; // rbx
  __int64 v19; // r14
  __int64 *v20; // rax
  int v21; // edx
  __int64 v22; // r12
  __int64 v23; // rax
  __int64 v24; // rdi
  int v25; // r9d
  __int64 v26; // rax
  __int64 v27; // r12
  __int64 v28; // r14
  int v29; // edx
  int v30; // r15d
  __int64 *v31; // rax
  __int64 v32; // rbx
  __int64 v33; // rax
  __int64 v34; // rdi
  __int64 v35; // rax
  const __m128i *v36; // rbx
  __int64 v37; // r15
  __int64 v38; // r14
  __int64 *v39; // rax
  __int32 v40; // edx
  __int64 v41; // r12
  __int64 v42; // rax
  __int64 v43; // rdi
  int v44; // r9d
  __int64 v45; // rax
  const __m128i *v46; // rbx
  __int64 v47; // r12
  __int64 *v48; // rax
  int v49; // edx
  __int64 v50; // rax
  __int64 v51; // rdi
  int v52; // r9d
  __int64 v53; // r12
  int v54; // edx
  __int64 v55; // rdi
  __int64 *v56; // rax
  __int64 v57; // r13
  __int64 v58; // rax
  __int64 v59; // rdi
  __int128 v61; // [rsp-20h] [rbp-100h]
  __int128 v62; // [rsp-20h] [rbp-100h]
  __int128 v63; // [rsp-20h] [rbp-100h]
  __int128 v64; // [rsp-10h] [rbp-F0h]
  __int128 v65; // [rsp-10h] [rbp-F0h]
  __int128 v66; // [rsp-10h] [rbp-F0h]
  __int64 v67; // [rsp+8h] [rbp-D8h]
  __int64 v68; // [rsp+10h] [rbp-D0h]
  __int64 v69; // [rsp+18h] [rbp-C8h]
  __int64 v70; // [rsp+18h] [rbp-C8h]
  __int32 v71; // [rsp+28h] [rbp-B8h]
  __int64 v72; // [rsp+30h] [rbp-B0h]
  __int64 v73; // [rsp+30h] [rbp-B0h]
  __int64 v74; // [rsp+30h] [rbp-B0h]
  __int64 v76; // [rsp+38h] [rbp-A8h]
  __int64 v78; // [rsp+40h] [rbp-A0h]
  __int64 v80; // [rsp+48h] [rbp-98h]
  __int64 v81; // [rsp+50h] [rbp-90h]
  const __m128i *v82; // [rsp+58h] [rbp-88h]
  __int64 v83; // [rsp+60h] [rbp-80h] BYREF
  int v84; // [rsp+68h] [rbp-78h]
  __m128i v85; // [rsp+70h] [rbp-70h]
  __m128i v86; // [rsp+80h] [rbp-60h]
  __m128i v87; // [rsp+90h] [rbp-50h]
  __m128i v88; // [rsp+A0h] [rbp-40h]

  v13 = *(__int64 **)(a1 + 24);
  v82 = *(const __m128i **)a1;
  v14 = *(_QWORD *)(a1 + 16);
  v69 = *v13;
  v68 = v13[1];
  v15 = sub_33CB7C0(233);
  v83 = v15;
  v16 = v82->m128i_i64[0];
  v83 = a9;
  v84 = a10;
  v85 = _mm_loadu_si128(v82 + 1);
  *((_QWORD *)&v64 + 1) = 3;
  *(_QWORD *)&v64 = &v83;
  v86 = _mm_loadu_si128(v82 + 2);
  v17 = sub_33FC220(v16, v15, v14, v69, v68, (_DWORD)v82, v64);
  v18 = *(const __m128i **)a1;
  v19 = *(_QWORD *)(a1 + 16);
  v67 = v17;
  v20 = *(__int64 **)(a1 + 24);
  LODWORD(v68) = v21;
  v22 = *v20;
  v70 = v20[1];
  v23 = sub_33CB7C0(233);
  v83 = v23;
  v24 = v18->m128i_i64[0];
  v84 = a8;
  v83 = a7;
  v85 = _mm_loadu_si128(v18 + 1);
  *((_QWORD *)&v61 + 1) = 3;
  *(_QWORD *)&v61 = &v83;
  v86 = _mm_loadu_si128(v18 + 2);
  v26 = sub_33FC220(v24, v23, v19, v22, v70, v25, v61);
  v27 = *(_QWORD *)(a1 + 16);
  v28 = v26;
  v30 = v29;
  v31 = *(__int64 **)(a1 + 24);
  v32 = v31[1];
  v72 = *v31;
  v33 = sub_33CB7C0(**(unsigned int **)(a1 + 8));
  v83 = v33;
  v34 = v82->m128i_i64[0];
  v84 = v30;
  v85.m128i_i32[2] = v68;
  v85.m128i_i64[0] = v67;
  v83 = v28;
  v86.m128i_i64[0] = a11;
  v86.m128i_i32[2] = a12;
  v87 = _mm_loadu_si128(v82 + 1);
  *((_QWORD *)&v65 + 1) = 5;
  *(_QWORD *)&v65 = &v83;
  v88 = _mm_loadu_si128(v82 + 2);
  v35 = sub_33FC220(v34, v33, v27, v72, v32, (_DWORD)v82, v65);
  v36 = *(const __m128i **)a1;
  v37 = *(_QWORD *)(a1 + 16);
  v38 = v35;
  v39 = *(__int64 **)(a1 + 24);
  v71 = v40;
  v41 = *v39;
  v73 = v39[1];
  v42 = sub_33CB7C0(233);
  v83 = v42;
  v43 = v36->m128i_i64[0];
  v83 = a4;
  v84 = a5;
  v85 = _mm_loadu_si128(v36 + 1);
  *((_QWORD *)&v62 + 1) = 3;
  *(_QWORD *)&v62 = &v83;
  v86 = _mm_loadu_si128(v36 + 2);
  v45 = sub_33FC220(v43, v42, v37, v41, v73, v44, v62);
  v46 = *(const __m128i **)a1;
  v47 = *(_QWORD *)(a1 + 16);
  v74 = v45;
  v48 = *(__int64 **)(a1 + 24);
  LODWORD(v37) = v49;
  v76 = v48[1];
  v78 = *v48;
  v50 = sub_33CB7C0(233);
  v83 = v50;
  v51 = v46->m128i_i64[0];
  v83 = a2;
  v84 = a3;
  v85 = _mm_loadu_si128(v46 + 1);
  *((_QWORD *)&v66 + 1) = 3;
  *(_QWORD *)&v66 = &v83;
  v86 = _mm_loadu_si128(v46 + 2);
  v53 = sub_33FC220(v51, v50, v47, v78, v76, v52, v66);
  LODWORD(v46) = v54;
  v55 = **(unsigned int **)(a1 + 8);
  v56 = *(__int64 **)(a1 + 24);
  v80 = *(_QWORD *)(a1 + 16);
  v57 = v56[1];
  v81 = *v56;
  v58 = sub_33CB7C0(v55);
  v83 = v58;
  v59 = v82->m128i_i64[0];
  v83 = v53;
  v84 = (int)v46;
  v85.m128i_i32[2] = v37;
  v86.m128i_i64[0] = v38;
  v85.m128i_i64[0] = v74;
  v86.m128i_i32[2] = v71;
  v87 = _mm_loadu_si128(v82 + 1);
  *((_QWORD *)&v63 + 1) = 5;
  *(_QWORD *)&v63 = &v83;
  v88 = _mm_loadu_si128(v82 + 2);
  return sub_33FC220(v59, v58, v80, v81, v57, (_DWORD)v82, v63);
}
