// Function: sub_33F3F90
// Address: 0x33f3f90
//
__m128i *__fastcall sub_33F3F90(
        _QWORD *a1,
        unsigned __int64 a2,
        unsigned __int64 a3,
        __int64 a4,
        unsigned __int64 a5,
        unsigned __int64 a6,
        unsigned __int64 a7,
        unsigned __int64 a8,
        const __m128i *a9)
{
  __int64 v11; // rax
  unsigned __int16 *v12; // rax
  __int32 v13; // edx
  __int64 v14; // r8
  unsigned int v15; // ecx
  _QWORD *v16; // rax
  int v17; // edx
  __int64 v18; // r9
  __int64 v19; // r9
  unsigned __int64 v20; // r14
  __int64 v21; // rax
  __int64 v22; // r8
  unsigned __int64 v23; // r14
  __int64 v24; // rax
  __int64 v25; // r9
  int v26; // eax
  __int64 v27; // rdx
  unsigned __int64 v28; // r8
  int v29; // eax
  __int64 v30; // r9
  __int64 v31; // rdx
  unsigned __int64 v32; // r8
  __int64 v33; // r8
  __int64 v34; // rax
  __m128i *v35; // rax
  __m128i *v36; // r13
  __int32 v38; // r14d
  __int64 v39; // rcx
  unsigned __int64 v40; // rax
  __int128 v41; // [rsp-20h] [rbp-200h]
  __int128 v42; // [rsp-20h] [rbp-200h]
  int v43; // [rsp+0h] [rbp-1E0h]
  _QWORD *v44; // [rsp+8h] [rbp-1D8h]
  __int32 v45; // [rsp+18h] [rbp-1C8h]
  __int16 v47; // [rsp+28h] [rbp-1B8h]
  __int16 v48; // [rsp+28h] [rbp-1B8h]
  int v49; // [rsp+28h] [rbp-1B8h]
  int v50; // [rsp+28h] [rbp-1B8h]
  int v51; // [rsp+28h] [rbp-1B8h]
  __m128i *v52; // [rsp+30h] [rbp-1B0h]
  unsigned __int64 v53; // [rsp+38h] [rbp-1A8h]
  unsigned __int16 v54; // [rsp+46h] [rbp-19Ah]
  unsigned __int8 *v56; // [rsp+58h] [rbp-188h] BYREF
  unsigned __int64 v57[7]; // [rsp+60h] [rbp-180h] BYREF
  int v58; // [rsp+98h] [rbp-148h]
  __m128i v59[2]; // [rsp+A0h] [rbp-140h] BYREF
  __int16 v60; // [rsp+C0h] [rbp-120h]
  __int64 v61[6]; // [rsp+F0h] [rbp-F0h] BYREF
  _BYTE *v62; // [rsp+120h] [rbp-C0h] BYREF
  __int64 v63; // [rsp+128h] [rbp-B8h]
  _BYTE v64[176]; // [rsp+130h] [rbp-B0h] BYREF

  v11 = *(_QWORD *)(a5 + 48) + 16LL * (unsigned int)a6;
  v54 = *(_WORD *)v11;
  v53 = *(_QWORD *)(v11 + 8);
  v52 = sub_33ED250((__int64)a1, 1, 0);
  v12 = (unsigned __int16 *)(*(_QWORD *)(a7 + 48) + 16LL * (unsigned int)a8);
  v45 = v13;
  v14 = *((_QWORD *)v12 + 1);
  v15 = *v12;
  v62 = 0;
  LODWORD(v63) = 0;
  v16 = sub_33F17F0(a1, 51, (__int64)&v62, v15, v14);
  if ( v62 )
  {
    v43 = v17;
    v44 = v16;
    sub_B91220((__int64)&v62, (__int64)v62);
    v17 = v43;
    v16 = v44;
  }
  v57[3] = a6;
  v57[4] = a7;
  v58 = v17;
  v57[0] = a2;
  v57[1] = a3;
  v57[5] = a8;
  v57[6] = (unsigned __int64)v16;
  v57[2] = a5;
  v63 = 0x2000000000LL;
  v62 = v64;
  sub_33C9670((__int64)&v62, 299, (unsigned __int64)v52, v57, 4, v18);
  v20 = v54;
  v21 = (unsigned int)v63;
  if ( !v54 )
    v20 = v53;
  v22 = (unsigned int)v20;
  if ( (unsigned __int64)(unsigned int)v63 + 1 > HIDWORD(v63) )
  {
    sub_C8D5F0((__int64)&v62, v64, (unsigned int)v63 + 1LL, 4u, (unsigned int)v20, v19);
    v21 = (unsigned int)v63;
    v22 = (unsigned int)v20;
  }
  v23 = HIDWORD(v20);
  *(_DWORD *)&v62[4 * v21] = v22;
  LODWORD(v63) = v63 + 1;
  v24 = (unsigned int)v63;
  if ( (unsigned __int64)(unsigned int)v63 + 1 > HIDWORD(v63) )
  {
    sub_C8D5F0((__int64)&v62, v64, (unsigned int)v63 + 1LL, 4u, v22, v19);
    v24 = (unsigned int)v63;
  }
  *(_DWORD *)&v62[4 * v24] = v23;
  LODWORD(v63) = v63 + 1;
  v56 = 0;
  *((_QWORD *)&v41 + 1) = v53;
  *(_QWORD *)&v41 = v54;
  sub_33CF750(v59, 299, *(_DWORD *)(a4 + 8), &v56, (__int64)v52, v45, v41, (__int64)a9);
  LOWORD(v26) = v60 & 0xF87F;
  v60 = v26;
  LOBYTE(v26) = v26 & 0x7A;
  if ( v61[0] )
  {
    v47 = v26;
    sub_B91220((__int64)v61, v61[0]);
    LOWORD(v26) = v47;
  }
  if ( v56 )
  {
    v48 = v26;
    sub_B91220((__int64)&v56, (__int64)v56);
    LOWORD(v26) = v48;
  }
  v27 = (unsigned int)v63;
  v26 = (unsigned __int16)v26;
  v28 = (unsigned int)v63 + 1LL;
  if ( v28 > HIDWORD(v63) )
  {
    v50 = (unsigned __int16)v26;
    sub_C8D5F0((__int64)&v62, v64, (unsigned int)v63 + 1LL, 4u, v28, v25);
    v27 = (unsigned int)v63;
    v26 = v50;
  }
  *(_DWORD *)&v62[4 * v27] = v26;
  LODWORD(v63) = v63 + 1;
  v29 = sub_2EAC1E0((__int64)a9);
  v31 = (unsigned int)v63;
  v32 = (unsigned int)v63 + 1LL;
  if ( v32 > HIDWORD(v63) )
  {
    v51 = v29;
    sub_C8D5F0((__int64)&v62, v64, (unsigned int)v63 + 1LL, 4u, v32, v30);
    v31 = (unsigned int)v63;
    v29 = v51;
  }
  *(_DWORD *)&v62[4 * v31] = v29;
  v33 = a9[2].m128i_u16[0];
  LODWORD(v63) = v63 + 1;
  v34 = (unsigned int)v63;
  if ( (unsigned __int64)(unsigned int)v63 + 1 > HIDWORD(v63) )
  {
    v49 = v33;
    sub_C8D5F0((__int64)&v62, v64, (unsigned int)v63 + 1LL, 4u, v33, v30);
    v34 = (unsigned int)v63;
    LODWORD(v33) = v49;
  }
  *(_DWORD *)&v62[4 * v34] = v33;
  LODWORD(v63) = v63 + 1;
  v59[0].m128i_i64[0] = 0;
  v35 = (__m128i *)sub_33CCCF0((__int64)a1, (__int64)&v62, a4, v59[0].m128i_i64);
  v36 = v35;
  if ( v35 )
  {
    sub_2EAC4C0((__m128i *)v35[7].m128i_i64[0], a9);
    goto LABEL_21;
  }
  v36 = (__m128i *)a1[52];
  v38 = *(_DWORD *)(a4 + 8);
  if ( v36 )
  {
    a1[52] = v36->m128i_i64[0];
  }
  else
  {
    v39 = a1[53];
    a1[63] += 120LL;
    v40 = (v39 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    if ( a1[54] >= v40 + 120 && v39 )
    {
      a1[53] = v40 + 120;
      if ( !v40 )
        goto LABEL_27;
    }
    else
    {
      v40 = sub_9D1E70((__int64)(a1 + 53), 120, 120, 3);
    }
    v36 = (__m128i *)v40;
  }
  *((_QWORD *)&v42 + 1) = v53;
  *(_QWORD *)&v42 = v54;
  sub_33CF750(v36, 299, v38, (unsigned __int8 **)a4, (__int64)v52, v45, v42, (__int64)a9);
  v36[2].m128i_i16[0] &= 0xF87Fu;
LABEL_27:
  sub_33E4EC0((__int64)a1, (__int64)v36, (__int64)v57, 4);
  sub_C657C0(a1 + 65, v36->m128i_i64, (__int64 *)v59[0].m128i_i64[0], (__int64)off_4A367D0);
  sub_33CC420((__int64)a1, (__int64)v36);
LABEL_21:
  if ( v62 != v64 )
    _libc_free((unsigned __int64)v62);
  return v36;
}
