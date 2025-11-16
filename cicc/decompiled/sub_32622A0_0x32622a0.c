// Function: sub_32622A0
// Address: 0x32622a0
//
__int64 __fastcall sub_32622A0(
        __int64 a1,
        __int64 a2,
        int a3,
        __int64 a4,
        __int32 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9,
        __int64 a10,
        __int64 a11,
        __int64 a12)
{
  const __m128i *v13; // r12
  __int64 *v14; // rax
  __int64 v15; // r14
  __int64 v16; // rax
  __int64 v17; // rdi
  int v18; // r9d
  __int64 v19; // rax
  const __m128i *v20; // r15
  __int64 v21; // r14
  __int64 *v22; // rax
  int v23; // edx
  __int64 v24; // rax
  __int64 v25; // rdi
  __int64 v26; // rax
  __int64 v27; // r14
  __int64 v28; // r15
  int v29; // edx
  __int64 *v30; // rax
  __int64 v31; // rax
  __int64 v32; // rdi
  int v33; // r9d
  __int64 v34; // rax
  __int64 v35; // r15
  __int64 v36; // r14
  int v37; // edx
  __int64 v38; // rdi
  __int64 *v39; // rax
  __int64 v40; // rbx
  __int64 v41; // rax
  __int64 v42; // rdi
  int v43; // r9d
  __int128 v45; // [rsp-20h] [rbp-100h]
  __int128 v46; // [rsp-20h] [rbp-100h]
  __int128 v47; // [rsp-10h] [rbp-F0h]
  __int128 v48; // [rsp-10h] [rbp-F0h]
  int v49; // [rsp+8h] [rbp-D8h]
  __int64 v50; // [rsp+10h] [rbp-D0h]
  __int64 v51; // [rsp+10h] [rbp-D0h]
  __int64 v52; // [rsp+18h] [rbp-C8h]
  __int64 v53; // [rsp+18h] [rbp-C8h]
  __int64 v54; // [rsp+30h] [rbp-B0h]
  __int64 v55; // [rsp+38h] [rbp-A8h]
  __int64 v56; // [rsp+38h] [rbp-A8h]
  __int64 v57; // [rsp+38h] [rbp-A8h]
  __int64 v61; // [rsp+60h] [rbp-80h] BYREF
  int v62; // [rsp+68h] [rbp-78h]
  __m128i v63; // [rsp+70h] [rbp-70h]
  __m128i v64; // [rsp+80h] [rbp-60h]
  __m128i v65; // [rsp+90h] [rbp-50h]
  __m128i v66; // [rsp+A0h] [rbp-40h]

  v13 = *(const __m128i **)a1;
  v14 = *(__int64 **)(a1 + 24);
  v15 = *(_QWORD *)(a1 + 16);
  v54 = v14[1];
  v55 = *v14;
  v16 = sub_33CB7C0(233);
  v61 = v16;
  v17 = v13->m128i_i64[0];
  v61 = a9;
  v62 = a10;
  v63 = _mm_loadu_si128(v13 + 1);
  *((_QWORD *)&v47 + 1) = 3;
  *(_QWORD *)&v47 = &v61;
  v64 = _mm_loadu_si128(v13 + 2);
  v19 = sub_33FC220(v17, v16, v15, v55, v54, v18, v47);
  v20 = *(const __m128i **)a1;
  v21 = *(_QWORD *)(a1 + 16);
  v56 = v19;
  v22 = *(__int64 **)(a1 + 24);
  LODWORD(v54) = v23;
  v52 = *v22;
  v50 = v22[1];
  v24 = sub_33CB7C0(233);
  v61 = v24;
  v25 = v20->m128i_i64[0];
  v62 = a8;
  v61 = a7;
  v63 = _mm_loadu_si128(v20 + 1);
  *((_QWORD *)&v45 + 1) = 3;
  *(_QWORD *)&v45 = &v61;
  v64 = _mm_loadu_si128(v20 + 2);
  v26 = sub_33FC220(v25, v24, v21, v52, v50, a7, v45);
  v27 = *(_QWORD *)(a1 + 16);
  v28 = v26;
  v49 = v29;
  v30 = *(__int64 **)(a1 + 24);
  v53 = *v30;
  v51 = v30[1];
  v31 = sub_33CB7C0(**(unsigned int **)(a1 + 8));
  v61 = v31;
  v32 = v13->m128i_i64[0];
  v63.m128i_i64[0] = v56;
  v64.m128i_i64[0] = a11;
  v61 = v28;
  v63.m128i_i32[2] = v54;
  v62 = v49;
  v64.m128i_i32[2] = a12;
  v65 = _mm_loadu_si128(v13 + 1);
  *((_QWORD *)&v48 + 1) = 5;
  *(_QWORD *)&v48 = &v61;
  v66 = _mm_loadu_si128(v13 + 2);
  v34 = sub_33FC220(v32, v31, v27, v53, v51, v33, v48);
  v35 = *(_QWORD *)(a1 + 16);
  v36 = v34;
  LODWORD(v54) = v37;
  v38 = **(unsigned int **)(a1 + 8);
  v39 = *(__int64 **)(a1 + 24);
  v57 = *v39;
  v40 = v39[1];
  v41 = sub_33CB7C0(v38);
  v61 = v41;
  v42 = v13->m128i_i64[0];
  v64.m128i_i64[0] = v36;
  v61 = a2;
  v64.m128i_i32[2] = v54;
  v62 = a3;
  v63.m128i_i64[0] = a4;
  v63.m128i_i32[2] = a5;
  v65 = _mm_loadu_si128(v13 + 1);
  *((_QWORD *)&v46 + 1) = 5;
  *(_QWORD *)&v46 = &v61;
  v66 = _mm_loadu_si128(v13 + 2);
  return sub_33FC220(v42, v41, v35, v57, v40, v43, v46);
}
