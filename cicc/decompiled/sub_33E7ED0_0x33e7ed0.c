// Function: sub_33E7ED0
// Address: 0x33e7ed0
//
__m128i *__fastcall sub_33E7ED0(
        _QWORD *a1,
        unsigned __int64 a2,
        __int32 a3,
        unsigned __int16 a4,
        unsigned __int64 a5,
        __int64 a6,
        unsigned __int64 *a7,
        __int64 a8,
        const __m128i *a9,
        __int16 a10,
        char a11)
{
  __int64 v13; // r9
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // r15
  __int64 v16; // r8
  __int64 v17; // rax
  unsigned __int64 v18; // rdx
  unsigned __int64 v19; // r15
  __int64 v20; // rax
  __int32 v21; // r10d
  __int64 v22; // r9
  int v23; // eax
  __int64 v24; // rdx
  unsigned __int64 v25; // r8
  int v26; // eax
  __int64 v27; // r9
  __int64 v28; // rdx
  unsigned __int64 v29; // r8
  __int64 v30; // r8
  __int64 v31; // rax
  __m128i *v32; // rax
  __m128i *v33; // r14
  __int32 v35; // r15d
  __int16 v36; // ax
  __int64 v37; // rcx
  unsigned __int64 v38; // rax
  __int128 v39; // [rsp-20h] [rbp-1B0h]
  __int128 v40; // [rsp-20h] [rbp-1B0h]
  __int32 v41; // [rsp+8h] [rbp-188h]
  __int16 v43; // [rsp+20h] [rbp-170h]
  __int16 v44; // [rsp+20h] [rbp-170h]
  int v45; // [rsp+20h] [rbp-170h]
  int v46; // [rsp+20h] [rbp-170h]
  int v47; // [rsp+20h] [rbp-170h]
  unsigned __int8 *v50; // [rsp+48h] [rbp-148h] BYREF
  __m128i v51[2]; // [rsp+50h] [rbp-140h] BYREF
  __int16 v52; // [rsp+70h] [rbp-120h]
  __int64 v53[6]; // [rsp+A0h] [rbp-F0h] BYREF
  _BYTE *v54; // [rsp+D0h] [rbp-C0h] BYREF
  __int64 v55; // [rsp+D8h] [rbp-B8h]
  _BYTE v56[176]; // [rsp+E0h] [rbp-B0h] BYREF

  v54 = v56;
  v55 = 0x2000000000LL;
  sub_33C9670((__int64)&v54, 365, a2, a7, a8, a6);
  v14 = a4;
  if ( !a4 )
    v14 = a5;
  v15 = v14;
  v16 = (unsigned int)v14;
  v17 = (unsigned int)v55;
  v18 = (unsigned int)v55 + 1LL;
  if ( v18 > HIDWORD(v55) )
  {
    sub_C8D5F0((__int64)&v54, v56, v18, 4u, v16, v13);
    v17 = (unsigned int)v55;
    v16 = (unsigned int)v15;
  }
  v19 = HIDWORD(v15);
  *(_DWORD *)&v54[4 * v17] = v16;
  LODWORD(v55) = v55 + 1;
  v20 = (unsigned int)v55;
  if ( (unsigned __int64)(unsigned int)v55 + 1 > HIDWORD(v55) )
  {
    sub_C8D5F0((__int64)&v54, v56, (unsigned int)v55 + 1LL, 4u, v16, v13);
    v20 = (unsigned int)v55;
  }
  *(_DWORD *)&v54[4 * v20] = v19;
  v21 = *(_DWORD *)(a6 + 8);
  v41 = a3;
  *((_QWORD *)&v39 + 1) = a5;
  LODWORD(v55) = v55 + 1;
  *(_QWORD *)&v39 = a4;
  v50 = 0;
  sub_33CF750(v51, 365, v21, &v50, a2, a3, v39, (__int64)a9);
  LOWORD(v23) = ((a10 & 7) << 7) | v52 & 0xFC7F;
  LOBYTE(v52) = ((a10 & 7) << 7) | v52 & 0x7F;
  HIBYTE(v52) = (4 * (a11 & 1)) | BYTE1(v23) & 0xFB;
  LOWORD(v23) = v52 & 0xFFFA;
  if ( v53[0] )
  {
    v43 = v52 & 0xFFFA;
    sub_B91220((__int64)v53, v53[0]);
    LOWORD(v23) = v43;
  }
  if ( v50 )
  {
    v44 = v23;
    sub_B91220((__int64)&v50, (__int64)v50);
    LOWORD(v23) = v44;
  }
  v24 = (unsigned int)v55;
  v23 = (unsigned __int16)v23;
  v25 = (unsigned int)v55 + 1LL;
  if ( v25 > HIDWORD(v55) )
  {
    v46 = (unsigned __int16)v23;
    sub_C8D5F0((__int64)&v54, v56, (unsigned int)v55 + 1LL, 4u, v25, v22);
    v24 = (unsigned int)v55;
    v23 = v46;
  }
  *(_DWORD *)&v54[4 * v24] = v23;
  LODWORD(v55) = v55 + 1;
  v26 = sub_2EAC1E0((__int64)a9);
  v28 = (unsigned int)v55;
  v29 = (unsigned int)v55 + 1LL;
  if ( v29 > HIDWORD(v55) )
  {
    v47 = v26;
    sub_C8D5F0((__int64)&v54, v56, (unsigned int)v55 + 1LL, 4u, v29, v27);
    v28 = (unsigned int)v55;
    v26 = v47;
  }
  *(_DWORD *)&v54[4 * v28] = v26;
  v30 = a9[2].m128i_u16[0];
  LODWORD(v55) = v55 + 1;
  v31 = (unsigned int)v55;
  if ( (unsigned __int64)(unsigned int)v55 + 1 > HIDWORD(v55) )
  {
    v45 = v30;
    sub_C8D5F0((__int64)&v54, v56, (unsigned int)v55 + 1LL, 4u, v30, v27);
    v31 = (unsigned int)v55;
    LODWORD(v30) = v45;
  }
  *(_DWORD *)&v54[4 * v31] = v30;
  LODWORD(v55) = v55 + 1;
  v51[0].m128i_i64[0] = 0;
  v32 = (__m128i *)sub_33CCCF0((__int64)a1, (__int64)&v54, a6, v51[0].m128i_i64);
  v33 = v32;
  if ( v32 )
  {
    sub_2EAC4C0((__m128i *)v32[7].m128i_i64[0], a9);
    goto LABEL_19;
  }
  v33 = (__m128i *)a1[52];
  v35 = *(_DWORD *)(a6 + 8);
  if ( v33 )
  {
    a1[52] = v33->m128i_i64[0];
  }
  else
  {
    v37 = a1[53];
    a1[63] += 120LL;
    v38 = (v37 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    if ( a1[54] >= v38 + 120 && v37 )
    {
      a1[53] = v38 + 120;
      if ( !v38 )
        goto LABEL_25;
    }
    else
    {
      v38 = sub_9D1E70((__int64)(a1 + 53), 120, 120, 3);
    }
    v33 = (__m128i *)v38;
  }
  *((_QWORD *)&v40 + 1) = a5;
  *(_QWORD *)&v40 = a4;
  sub_33CF750(v33, 365, v35, (unsigned __int8 **)a6, a2, v41, v40, (__int64)a9);
  v36 = v33[2].m128i_i16[0] & 0xFC7F | ((a10 & 7) << 7);
  v33[2].m128i_i16[0] = v36;
  v33[2].m128i_i8[1] = HIBYTE(v36) & 0xFB | (4 * (a11 & 1));
LABEL_25:
  sub_33E4EC0((__int64)a1, (__int64)v33, (__int64)a7, a8);
  sub_C657C0(a1 + 65, v33->m128i_i64, (__int64 *)v51[0].m128i_i64[0], (__int64)off_4A367D0);
  sub_33CC420((__int64)a1, (__int64)v33);
LABEL_19:
  if ( v54 != v56 )
    _libc_free((unsigned __int64)v54);
  return v33;
}
