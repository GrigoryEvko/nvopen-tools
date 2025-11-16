// Function: sub_2175A80
// Address: 0x2175a80
//
__int64 __fastcall sub_2175A80(
        __int64 *a1,
        __m128i *a2,
        __int64 a3,
        __m128i a4,
        double a5,
        __m128i a6,
        __int64 a7,
        __int64 a8,
        const void **a9,
        __m128i *a10,
        __int64 a11)
{
  unsigned int v11; // ebx
  unsigned int v12; // r12d
  char v13; // r8
  unsigned int v14; // eax
  __int64 v15; // rdx
  const void **v16; // rbx
  __int64 v17; // r13
  __int64 v18; // r8
  __int64 v19; // rdx
  __int64 v20; // r9
  __int64 v21; // rdx
  __int64 *v22; // rdx
  __int64 v23; // r8
  __int64 v24; // rdx
  __int64 v25; // r9
  __int64 v26; // rdx
  __int64 *v27; // rdx
  __int64 v28; // rax
  __int64 v29; // rax
  int v30; // edx
  __int128 v31; // rax
  __int128 v32; // rax
  unsigned int v33; // edx
  unsigned int v34; // edx
  unsigned int v35; // edx
  unsigned int v36; // edx
  __int64 v38; // rcx
  __int128 v39; // rax
  __int128 v40; // [rsp+0h] [rbp-220h]
  __int128 v41; // [rsp+0h] [rbp-220h]
  __int64 v44; // [rsp+30h] [rbp-1F0h]
  __int128 v45; // [rsp+40h] [rbp-1E0h]
  __int128 v46; // [rsp+50h] [rbp-1D0h]
  __int64 v47; // [rsp+50h] [rbp-1D0h]
  __int64 v48; // [rsp+50h] [rbp-1D0h]
  __int64 v49; // [rsp+58h] [rbp-1C8h]
  __int64 v50; // [rsp+58h] [rbp-1C8h]
  unsigned int v51; // [rsp+60h] [rbp-1C0h]
  char v52; // [rsp+66h] [rbp-1BAh]
  char v53; // [rsp+67h] [rbp-1B9h]
  __int64 *v54; // [rsp+90h] [rbp-190h]
  __int64 v55; // [rsp+98h] [rbp-188h]
  __int128 v56; // [rsp+A0h] [rbp-180h]
  __int64 *v57; // [rsp+D0h] [rbp-150h]
  __int64 *v58; // [rsp+E0h] [rbp-140h]
  __int64 v59; // [rsp+F0h] [rbp-130h]
  __int64 v60; // [rsp+100h] [rbp-120h] BYREF
  const void **v61; // [rsp+108h] [rbp-118h]
  __int128 v62; // [rsp+110h] [rbp-110h]
  __int64 v63; // [rsp+120h] [rbp-100h]
  _QWORD v64[4]; // [rsp+130h] [rbp-F0h] BYREF
  unsigned __int8 *v65; // [rsp+150h] [rbp-D0h]
  __int64 v66; // [rsp+158h] [rbp-C8h]
  _QWORD v67[8]; // [rsp+160h] [rbp-C0h] BYREF
  __int64 *v68; // [rsp+1A0h] [rbp-80h] BYREF
  __int64 v69; // [rsp+1A8h] [rbp-78h]
  _OWORD v70[7]; // [rsp+1B0h] [rbp-70h] BYREF

  v11 = a8;
  v52 = a8;
  v60 = a8;
  v61 = a9;
  if ( !(_BYTE)a8 )
  {
    v12 = a8;
    v51 = (unsigned int)sub_1F58D40((__int64)&v60) >> 3;
    goto LABEL_24;
  }
  v12 = a8;
  v51 = (unsigned int)sub_216FFF0(a8) >> 3;
  if ( v13 != 9 )
  {
    if ( v52 == 10 )
    {
      v53 = 6;
      goto LABEL_4;
    }
LABEL_24:
    v38 = v11;
    v16 = a9;
    *(_QWORD *)&v39 = sub_1D38BB0((__int64)a1, 0, a11, v38, a9, 0, a4, a5, a6, 0);
    v56 = v39;
    if ( !v51 )
      return v56;
    v53 = v52;
    goto LABEL_6;
  }
  v53 = 5;
LABEL_4:
  v14 = v12;
  LOBYTE(v14) = v53;
  v12 = v14;
  *(_QWORD *)&v56 = sub_1D38BB0((__int64)a1, 0, a11, v14, 0, 0, a4, a5, a6, 0);
  *((_QWORD *)&v56 + 1) = v15;
  if ( !v51 )
  {
LABEL_21:
    *(_QWORD *)&v56 = sub_1D309E0(
                        a1,
                        158,
                        a11,
                        (unsigned int)v60,
                        v61,
                        0,
                        *(double *)a4.m128i_i64,
                        a5,
                        *(double *)a6.m128i_i64,
                        v56);
    return v56;
  }
  v16 = 0;
LABEL_6:
  v17 = 0;
  do
  {
    v65 = (unsigned __int8 *)v67;
    v66 = 0x400000003LL;
    v67[0] = 4;
    v68 = (__int64 *)v70;
    v67[1] = 0;
    v67[2] = 1;
    a4 = _mm_loadu_si128(a2);
    v67[3] = 0;
    v67[4] = 111;
    v67[5] = 0;
    v69 = 0x400000001LL;
    v70[0] = a4;
    v18 = sub_1D38BB0((__int64)a1, 1, a11, 5, 0, 0, a4, a5, a6, 0);
    v20 = v19;
    v21 = (unsigned int)v69;
    if ( (unsigned int)v69 >= HIDWORD(v69) )
    {
      v48 = v18;
      v50 = v20;
      sub_16CD150((__int64)&v68, v70, 0, 16, v18, v20);
      v21 = (unsigned int)v69;
      v20 = v50;
      v18 = v48;
    }
    v22 = &v68[2 * v21];
    *v22 = v18;
    v22[1] = v20;
    LODWORD(v69) = v69 + 1;
    v23 = sub_1D38BB0((__int64)a1, a3 + v17, a11, 5, 0, 0, a4, a5, a6, 0);
    v25 = v24;
    v26 = (unsigned int)v69;
    if ( (unsigned int)v69 >= HIDWORD(v69) )
    {
      v47 = v23;
      v49 = v25;
      sub_16CD150((__int64)&v68, v70, 0, 16, v23, v25);
      v26 = (unsigned int)v69;
      v25 = v49;
      v23 = v47;
    }
    v27 = &v68[2 * v26];
    *v27 = v23;
    v27[1] = v25;
    v28 = (unsigned int)(v69 + 1);
    LODWORD(v69) = v28;
    if ( HIDWORD(v69) <= (unsigned int)v28 )
    {
      sub_16CD150((__int64)&v68, v70, 0, 16, v23, v25);
      v28 = (unsigned int)v69;
    }
    v62 = 0u;
    a6 = _mm_loadu_si128(a10);
    v63 = 0;
    *(__m128i *)&v68[2 * v28] = a6;
    memset(v64, 0, 24);
    v54 = v68;
    LODWORD(v69) = v69 + 1;
    v55 = (unsigned int)v69;
    v29 = sub_1D25C30((__int64)a1, v65, (unsigned int)v66);
    LOBYTE(v12) = v53;
    v44 = sub_1D251C0(a1, 667, a11, v29, v30, 1, v54, v55, 3, 0, v62, v63, 3u, 0, (__int64)v64);
    *(_QWORD *)&v31 = sub_1D38BB0((__int64)a1, 255, a11, v12, v16, 0, a4, a5, a6, 0);
    v46 = v31;
    *(_QWORD *)&v32 = sub_1D38BB0((__int64)a1, 8 * v17, a11, 5, 0, 0, a4, a5, a6, 0);
    v45 = v32;
    a2->m128i_i64[0] = v44;
    a2->m128i_i32[2] = 1;
    a10->m128i_i64[0] = v44;
    a10->m128i_i32[2] = 2;
    v59 = sub_1D309E0(
            a1,
            143,
            a11,
            v12,
            v16,
            0,
            *(double *)a4.m128i_i64,
            a5,
            *(double *)a6.m128i_i64,
            (unsigned __int64)v44);
    v40 = v46;
    *((_QWORD *)&v46 + 1) = v33;
    v58 = sub_1D332F0(a1, 118, a11, v12, v16, 0, *(double *)a4.m128i_i64, a5, a6, v59, v33, v40);
    *((_QWORD *)&v46 + 1) = v34 | *((_QWORD *)&v46 + 1) & 0xFFFFFFFF00000000LL;
    v57 = sub_1D332F0(
            a1,
            122,
            a11,
            v12,
            v16,
            0,
            *(double *)a4.m128i_i64,
            a5,
            a6,
            (__int64)v58,
            *((unsigned __int64 *)&v46 + 1),
            v45);
    *((_QWORD *)&v41 + 1) = v35 | *((_QWORD *)&v46 + 1) & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v41 = v57;
    *(_QWORD *)&v56 = sub_1D332F0(
                        a1,
                        119,
                        a11,
                        v12,
                        v16,
                        0,
                        *(double *)a4.m128i_i64,
                        a5,
                        a6,
                        v56,
                        *((unsigned __int64 *)&v56 + 1),
                        v41);
    *((_QWORD *)&v56 + 1) = v36 | *((_QWORD *)&v56 + 1) & 0xFFFFFFFF00000000LL;
    if ( v68 != (__int64 *)v70 )
      _libc_free((unsigned __int64)v68);
    if ( v65 != (unsigned __int8 *)v67 )
      _libc_free((unsigned __int64)v65);
    ++v17;
  }
  while ( v51 > (unsigned int)v17 );
  if ( v52 != v53 || v16 != a9 && !v52 )
    goto LABEL_21;
  return v56;
}
