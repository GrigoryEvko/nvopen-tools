// Function: sub_19389B0
// Address: 0x19389b0
//
__int64 __fastcall sub_19389B0(
        __int64 a1,
        __m128 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128 a9)
{
  __int64 v9; // rbx
  _QWORD *v10; // rdi
  const __m128i *v11; // rsi
  const __m128i *v12; // r9
  unsigned __int64 v13; // rdx
  __int64 v14; // rax
  __m128 *v15; // r8
  __m128 *v16; // rdx
  const __m128i *v17; // rax
  __m128 *v18; // rax
  __m128 *v19; // rax
  char *v20; // rax
  const __m128i *v21; // rcx
  const __m128i *v22; // r9
  unsigned __int64 v23; // r13
  __int64 v24; // rax
  __m128 *v25; // rdi
  __m128 *v26; // rdx
  const __m128i *v27; // rax
  __m128 *v28; // rax
  __m128 *v29; // rax
  char *v30; // rax
  double v31; // xmm4_8
  double v32; // xmm5_8
  __int64 v33; // r14
  __int64 v34; // r12
  int v35; // ebx
  __int64 *v36; // r13
  __int64 v37; // rsi
  unsigned __int64 *v38; // rbx
  unsigned __int64 *v39; // r12
  unsigned __int64 v40; // rdi
  unsigned __int64 *v41; // rbx
  unsigned __int64 *v42; // r12
  unsigned __int64 v43; // rdi
  unsigned __int64 v45; // [rsp+8h] [rbp-488h]
  __int64 v46; // [rsp+20h] [rbp-470h] BYREF
  __int64 v47; // [rsp+28h] [rbp-468h]
  __int64 v48; // [rsp+30h] [rbp-460h]
  _QWORD v49[16]; // [rsp+40h] [rbp-450h] BYREF
  __int64 v50; // [rsp+C0h] [rbp-3D0h] BYREF
  _QWORD *v51; // [rsp+C8h] [rbp-3C8h]
  _QWORD *v52; // [rsp+D0h] [rbp-3C0h]
  __int64 v53; // [rsp+D8h] [rbp-3B8h]
  int v54; // [rsp+E0h] [rbp-3B0h]
  _QWORD v55[8]; // [rsp+E8h] [rbp-3A8h] BYREF
  const __m128i *v56; // [rsp+128h] [rbp-368h] BYREF
  const __m128i *v57; // [rsp+130h] [rbp-360h]
  __int64 v58; // [rsp+138h] [rbp-358h]
  _QWORD v59[2]; // [rsp+140h] [rbp-350h] BYREF
  unsigned __int64 v60; // [rsp+150h] [rbp-340h]
  _BYTE v61[64]; // [rsp+168h] [rbp-328h] BYREF
  __m128 *v62; // [rsp+1A8h] [rbp-2E8h]
  __m128 *v63; // [rsp+1B0h] [rbp-2E0h]
  char *v64; // [rsp+1B8h] [rbp-2D8h]
  _QWORD v65[2]; // [rsp+1C0h] [rbp-2D0h] BYREF
  unsigned __int64 v66; // [rsp+1D0h] [rbp-2C0h]
  char v67[64]; // [rsp+1E8h] [rbp-2A8h] BYREF
  __m128 *v68; // [rsp+228h] [rbp-268h]
  __m128 *v69; // [rsp+230h] [rbp-260h]
  char *v70; // [rsp+238h] [rbp-258h]
  _QWORD v71[2]; // [rsp+240h] [rbp-250h] BYREF
  unsigned __int64 v72; // [rsp+250h] [rbp-240h]
  char v73[64]; // [rsp+268h] [rbp-228h] BYREF
  __m128 *v74; // [rsp+2A8h] [rbp-1E8h]
  __m128 *v75; // [rsp+2B0h] [rbp-1E0h]
  char *v76; // [rsp+2B8h] [rbp-1D8h]
  __m128i v77; // [rsp+2C0h] [rbp-1D0h] BYREF
  unsigned __int64 v78; // [rsp+2D0h] [rbp-1C0h]
  char v79[64]; // [rsp+2E8h] [rbp-1A8h] BYREF
  __m128 *v80; // [rsp+328h] [rbp-168h]
  __m128 *v81; // [rsp+330h] [rbp-160h]
  char *v82; // [rsp+338h] [rbp-158h]
  __int64 v83; // [rsp+340h] [rbp-150h] BYREF
  __int64 v84; // [rsp+348h] [rbp-148h]
  __int64 v85; // [rsp+350h] [rbp-140h]
  int v86; // [rsp+358h] [rbp-138h]
  __int64 v87; // [rsp+360h] [rbp-130h]
  __int64 v88; // [rsp+368h] [rbp-128h]
  __int64 v89; // [rsp+370h] [rbp-120h]
  int v90; // [rsp+378h] [rbp-118h]
  __int64 v91; // [rsp+380h] [rbp-110h]
  __int64 v92; // [rsp+388h] [rbp-108h]
  __int64 v93; // [rsp+390h] [rbp-100h]
  int v94; // [rsp+398h] [rbp-F8h]
  __int64 v95; // [rsp+3A0h] [rbp-F0h]
  __int64 v96; // [rsp+3A8h] [rbp-E8h]
  unsigned __int64 *v97; // [rsp+3B0h] [rbp-E0h]
  __int64 v98; // [rsp+3B8h] [rbp-D8h]
  _BYTE v99[32]; // [rsp+3C0h] [rbp-D0h] BYREF
  unsigned __int64 *v100; // [rsp+3E0h] [rbp-B0h]
  __int64 v101; // [rsp+3E8h] [rbp-A8h]
  _QWORD v102[3]; // [rsp+3F0h] [rbp-A0h] BYREF
  _BYTE *v103; // [rsp+408h] [rbp-88h]
  __int64 v104; // [rsp+410h] [rbp-80h]
  _BYTE v105[64]; // [rsp+418h] [rbp-78h] BYREF
  int v106; // [rsp+458h] [rbp-38h]

  v97 = (unsigned __int64 *)v99;
  v98 = 0x400000000LL;
  v100 = v102;
  v104 = 0x800000000LL;
  v83 = 0;
  v84 = 0;
  v85 = 0;
  v86 = 0;
  v87 = 0;
  v88 = 0;
  v89 = 0;
  v90 = 0;
  v91 = 0;
  v92 = 0;
  v93 = 0;
  v94 = 0;
  v95 = 0;
  v96 = 0;
  v101 = 0;
  v102[0] = 0;
  v102[1] = 1;
  v103 = v105;
  v106 = 1;
  v46 = 0;
  v9 = *(_QWORD *)(a1 + 80);
  v47 = 0;
  v48 = 0;
  if ( v9 )
    v9 -= 24;
  memset(v49, 0, sizeof(v49));
  LODWORD(v49[3]) = 8;
  v49[1] = &v49[5];
  v49[2] = &v49[5];
  v51 = v55;
  v52 = v55;
  v55[0] = v9;
  v56 = 0;
  v57 = 0;
  v58 = 0;
  v53 = 0x100000008LL;
  v54 = 0;
  v50 = 1;
  v77.m128i_i64[1] = sub_157EBA0(v9);
  v77.m128i_i64[0] = v9;
  LODWORD(v78) = 0;
  sub_13FDF40(&v56, 0, &v77);
  sub_13FE0F0((__int64)&v50);
  v10 = v71;
  sub_16CCCB0(v71, (__int64)v73, (__int64)v49);
  v11 = (const __m128i *)v49[14];
  v12 = (const __m128i *)v49[13];
  v74 = 0;
  v75 = 0;
  v76 = 0;
  v13 = v49[14] - v49[13];
  if ( v49[14] == v49[13] )
  {
    v15 = 0;
  }
  else
  {
    if ( v13 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_66;
    v45 = v49[14] - v49[13];
    v14 = sub_22077B0(v49[14] - v49[13]);
    v11 = (const __m128i *)v49[14];
    v12 = (const __m128i *)v49[13];
    v13 = v45;
    v15 = (__m128 *)v14;
  }
  v74 = v15;
  v75 = v15;
  v76 = (char *)v15 + v13;
  if ( v11 != v12 )
  {
    v16 = v15;
    v17 = v12;
    do
    {
      if ( v16 )
      {
        a2 = (__m128)_mm_loadu_si128(v17);
        *v16 = a2;
        v16[1].m128_u64[0] = v17[1].m128i_u64[0];
      }
      v17 = (const __m128i *)((char *)v17 + 24);
      v16 = (__m128 *)((char *)v16 + 24);
    }
    while ( v17 != v11 );
    v15 = (__m128 *)((char *)v15 + 8 * ((unsigned __int64)((char *)&v17[-2].m128i_u64[1] - (char *)v12) >> 3) + 24);
  }
  v75 = v15;
  sub_16CCEE0(&v77, (__int64)v79, 8, (__int64)v71);
  v18 = v74;
  v10 = v59;
  v11 = (const __m128i *)v61;
  v74 = 0;
  v80 = v18;
  v19 = v75;
  v75 = 0;
  v81 = v19;
  v20 = v76;
  v76 = 0;
  v82 = v20;
  sub_16CCCB0(v59, (__int64)v61, (__int64)&v50);
  v21 = v57;
  v22 = v56;
  v62 = 0;
  v63 = 0;
  v64 = 0;
  v23 = (char *)v57 - (char *)v56;
  if ( v57 != v56 )
  {
    if ( v23 <= 0x7FFFFFFFFFFFFFF8LL )
    {
      v24 = sub_22077B0((char *)v57 - (char *)v56);
      v21 = v57;
      v22 = v56;
      v25 = (__m128 *)v24;
      goto LABEL_15;
    }
LABEL_66:
    sub_4261EA(v10, v11, v13);
  }
  v25 = 0;
LABEL_15:
  v62 = v25;
  v63 = v25;
  v64 = (char *)v25 + v23;
  if ( v22 != v21 )
  {
    v26 = v25;
    v27 = v22;
    do
    {
      if ( v26 )
      {
        a3 = (__m128)_mm_loadu_si128(v27);
        *v26 = a3;
        v26[1].m128_u64[0] = v27[1].m128i_u64[0];
      }
      v27 = (const __m128i *)((char *)v27 + 24);
      v26 = (__m128 *)((char *)v26 + 24);
    }
    while ( v21 != v27 );
    v25 = (__m128 *)((char *)v25 + 8 * ((unsigned __int64)((char *)&v21[-2].m128i_u64[1] - (char *)v22) >> 3) + 24);
  }
  v63 = v25;
  sub_16CCEE0(v65, (__int64)v67, 8, (__int64)v59);
  v28 = v62;
  v62 = 0;
  v68 = v28;
  v29 = v63;
  v63 = 0;
  v69 = v29;
  v30 = v64;
  v64 = 0;
  v70 = v30;
  sub_19380F0((__int64)v65, (__int64)&v77, (__int64)&v46);
  if ( v68 )
    j_j___libc_free_0(v68, v70 - (char *)v68);
  if ( v66 != v65[1] )
    _libc_free(v66);
  if ( v62 )
    j_j___libc_free_0(v62, v64 - (char *)v62);
  if ( v60 != v59[1] )
    _libc_free(v60);
  if ( v80 )
    j_j___libc_free_0(v80, v82 - (char *)v80);
  if ( v78 != v77.m128i_i64[1] )
    _libc_free(v78);
  if ( v74 )
    j_j___libc_free_0(v74, v76 - (char *)v74);
  if ( v72 != v71[1] )
    _libc_free(v72);
  if ( v56 )
    j_j___libc_free_0(v56, v58 - (_QWORD)v56);
  if ( v52 != v51 )
    _libc_free((unsigned __int64)v52);
  if ( v49[13] )
    j_j___libc_free_0(v49[13], v49[15] - v49[13]);
  if ( v49[2] != v49[1] )
    _libc_free(v49[2]);
  v33 = v47;
  v34 = v46;
  if ( v47 == v46 )
  {
    LODWORD(v36) = 0;
  }
  else
  {
    v35 = 0;
    v36 = &v83;
    do
    {
      v37 = *(_QWORD *)(v33 - 8);
      v33 -= 8;
      v35 += sub_1935AA0((__int64)&v83, v37, a2, *(double *)a3.m128_u64, a4, a5, v31, v32, a8, a9);
    }
    while ( v34 != v33 );
    v34 = v46;
    LOBYTE(v36) = v35 != 0;
  }
  if ( v34 )
    j_j___libc_free_0(v34, v48 - v34);
  if ( v103 != v105 )
    _libc_free((unsigned __int64)v103);
  v38 = v97;
  v39 = &v97[(unsigned int)v98];
  if ( v97 != v39 )
  {
    do
    {
      v40 = *v38++;
      _libc_free(v40);
    }
    while ( v39 != v38 );
  }
  v41 = v100;
  v42 = &v100[2 * (unsigned int)v101];
  if ( v100 != v42 )
  {
    do
    {
      v43 = *v41;
      v41 += 2;
      _libc_free(v43);
    }
    while ( v42 != v41 );
    v42 = v100;
  }
  if ( v42 != v102 )
    _libc_free((unsigned __int64)v42);
  if ( v97 != (unsigned __int64 *)v99 )
    _libc_free((unsigned __int64)v97);
  j___libc_free_0(v92);
  j___libc_free_0(v88);
  j___libc_free_0(v84);
  return (unsigned int)v36;
}
