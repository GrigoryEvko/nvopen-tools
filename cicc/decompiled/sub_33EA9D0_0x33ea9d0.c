// Function: sub_33EA9D0
// Address: 0x33ea9d0
//
__m128i *__fastcall sub_33EA9D0(
        _QWORD *a1,
        __int32 a2,
        __int64 a3,
        unsigned __int64 a4,
        __int32 a5,
        const __m128i *a6,
        unsigned __int64 *a7,
        __int64 a8,
        __int64 a9,
        __int64 a10)
{
  __int32 v10; // r10d
  __int32 v15; // esi
  __int64 v16; // r9
  int v17; // eax
  __int64 v18; // rdx
  unsigned __int64 v19; // r8
  int v20; // eax
  __int64 v21; // r9
  __int64 v22; // rdx
  unsigned __int64 v23; // r8
  __int64 v24; // r8
  __int64 v25; // rax
  __int64 v26; // r9
  __int64 v27; // r8
  unsigned __int64 v28; // rdx
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rax
  unsigned __int64 v33; // r15
  __int64 v34; // r8
  unsigned __int64 v35; // r15
  __int64 v36; // rax
  __m128i *v37; // rax
  __m128i *v38; // r15
  __m128i *result; // rax
  __m128i *v40; // r15
  __int32 v41; // r11d
  __int32 v42; // r10d
  __int64 v43; // rsi
  unsigned __int64 v44; // rdx
  unsigned __int64 v45; // rcx
  char v46; // di
  int v47; // esi
  unsigned __int64 v48; // r8
  unsigned __int64 v49; // rdx
  unsigned __int64 v50; // r8
  unsigned __int64 v51; // r8
  unsigned __int64 v52; // r8
  __int64 v53; // rsi
  unsigned __int64 v54; // rdx
  __int64 v55; // rax
  __int64 v56; // rax
  __int128 v57; // [rsp-20h] [rbp-1D0h]
  __int128 v58; // [rsp-20h] [rbp-1D0h]
  __int128 v59; // [rsp-20h] [rbp-1D0h]
  unsigned int v60; // [rsp+8h] [rbp-1A8h]
  __int16 v61; // [rsp+18h] [rbp-198h]
  __int16 v62; // [rsp+18h] [rbp-198h]
  unsigned int v63; // [rsp+18h] [rbp-198h]
  int v64; // [rsp+18h] [rbp-198h]
  int v65; // [rsp+18h] [rbp-198h]
  int v66; // [rsp+18h] [rbp-198h]
  int v67; // [rsp+18h] [rbp-198h]
  __int32 v70; // [rsp+38h] [rbp-178h]
  __int32 v71; // [rsp+38h] [rbp-178h]
  unsigned __int8 *v73; // [rsp+68h] [rbp-148h] BYREF
  __m128i v74[2]; // [rsp+70h] [rbp-140h] BYREF
  char v75; // [rsp+90h] [rbp-120h]
  char v76; // [rsp+91h] [rbp-11Fh]
  __int64 v77[6]; // [rsp+C0h] [rbp-F0h] BYREF
  _BYTE *v78; // [rsp+F0h] [rbp-C0h] BYREF
  __int64 v79; // [rsp+F8h] [rbp-B8h]
  _BYTE v80[176]; // [rsp+100h] [rbp-B0h] BYREF

  v10 = a5;
  if ( *(_WORD *)(a4 + 16LL * (unsigned int)(a5 - 1)) == 262 )
  {
    v40 = (__m128i *)a1[52];
    v41 = *(_DWORD *)(a3 + 8);
    if ( v40 )
    {
      a1[52] = v40->m128i_i64[0];
    }
    else
    {
      v43 = a1[53];
      a1[63] += 120LL;
      v44 = (v43 + 7) & 0xFFFFFFFFFFFFFFF8LL;
      if ( a1[54] >= v44 + 120 && v43 )
      {
        a1[53] = v44 + 120;
        if ( !v44 )
          goto LABEL_30;
        v40 = (__m128i *)((v43 + 7) & 0xFFFFFFFFFFFFFFF8LL);
      }
      else
      {
        v70 = v41;
        v55 = sub_9D1E70((__int64)(a1 + 53), 120, 120, 3);
        v10 = a5;
        v41 = v70;
        v40 = (__m128i *)v55;
      }
    }
    *((_QWORD *)&v58 + 1) = a10;
    *(_QWORD *)&v58 = (unsigned __int16)a9;
    sub_33CF750(v40, a2, v41, (unsigned __int8 **)a3, a4, v10, v58, (__int64)a6);
    v40[2].m128i_i8[0] |= 2u;
LABEL_30:
    sub_33E4EC0((__int64)a1, (__int64)v40, (__int64)a7, a8);
LABEL_31:
    sub_33CC420((__int64)a1, (__int64)v40);
    return v40;
  }
  v78 = v80;
  v79 = 0x2000000000LL;
  sub_33C9670((__int64)&v78, a2, a4, a7, a8, (__int64)a6);
  v15 = *(_DWORD *)(a3 + 8);
  v73 = 0;
  *((_QWORD *)&v57 + 1) = a10;
  *(_QWORD *)&v57 = (unsigned __int16)a9;
  sub_33CF750(v74, a2, v15, &v73, a4, a5, v57, (__int64)a6);
  v75 |= 2u;
  BYTE1(v17) = v76;
  LOBYTE(v17) = v75 & 0xFA;
  if ( v77[0] )
  {
    v61 = v17;
    sub_B91220((__int64)v77, v77[0]);
    LOWORD(v17) = v61;
  }
  if ( v73 )
  {
    v62 = v17;
    sub_B91220((__int64)&v73, (__int64)v73);
    LOWORD(v17) = v62;
  }
  v18 = (unsigned int)v79;
  v17 = (unsigned __int16)v17;
  v19 = (unsigned int)v79 + 1LL;
  if ( v19 > HIDWORD(v79) )
  {
    v65 = (unsigned __int16)v17;
    sub_C8D5F0((__int64)&v78, v80, (unsigned int)v79 + 1LL, 4u, v19, v16);
    v18 = (unsigned int)v79;
    v17 = v65;
  }
  *(_DWORD *)&v78[4 * v18] = v17;
  LODWORD(v79) = v79 + 1;
  v20 = sub_2EAC1E0((__int64)a6);
  v22 = (unsigned int)v79;
  v23 = (unsigned int)v79 + 1LL;
  if ( v23 > HIDWORD(v79) )
  {
    v67 = v20;
    sub_C8D5F0((__int64)&v78, v80, (unsigned int)v79 + 1LL, 4u, v23, v21);
    v22 = (unsigned int)v79;
    v20 = v67;
  }
  *(_DWORD *)&v78[4 * v22] = v20;
  v24 = a6[2].m128i_u16[0];
  LODWORD(v79) = v79 + 1;
  v25 = (unsigned int)v79;
  if ( (unsigned __int64)(unsigned int)v79 + 1 > HIDWORD(v79) )
  {
    v66 = v24;
    sub_C8D5F0((__int64)&v78, v80, (unsigned int)v79 + 1LL, 4u, v24, v21);
    v25 = (unsigned int)v79;
    LODWORD(v24) = v66;
  }
  v26 = 0xFFFFFFFFLL;
  *(_DWORD *)&v78[4 * v25] = v24;
  v27 = 0xFFFFFFFFLL;
  v28 = a6[1].m128i_u64[1];
  v29 = (unsigned int)(v79 + 1);
  LODWORD(v79) = v79 + 1;
  if ( (v28 & 0xFFFFFFFFFFFFFFF9LL) != 0 )
  {
    v45 = v28 >> 3;
    v46 = a6[1].m128i_i8[8] & 2;
    if ( (a6[1].m128i_i8[8] & 6) == 2 || (a6[1].m128i_i8[8] & 1) != 0 )
    {
      v52 = HIWORD(v28);
      if ( !v46 )
        v52 = HIDWORD(v28);
      v51 = (v52 + 7) >> 3;
    }
    else
    {
      v47 = (unsigned __int16)((unsigned int)v28 >> 8);
      v48 = v28;
      v49 = HIDWORD(v28);
      v50 = HIWORD(v48);
      if ( !v46 )
        LODWORD(v50) = v49;
      v51 = ((unsigned __int64)(unsigned int)(v47 * v50) + 7) >> 3;
      if ( (v45 & 1) != 0 )
        v51 |= 0x4000000000000000uLL;
    }
    v26 = (unsigned int)v51;
    v27 = HIDWORD(v51);
  }
  if ( v29 + 1 > (unsigned __int64)HIDWORD(v79) )
  {
    v60 = v26;
    v63 = v27;
    sub_C8D5F0((__int64)&v78, v80, v29 + 1, 4u, v27, v26);
    v29 = (unsigned int)v79;
    v26 = v60;
    v27 = v63;
  }
  *(_DWORD *)&v78[4 * v29] = v26;
  LODWORD(v79) = v79 + 1;
  v30 = (unsigned int)v79;
  if ( (unsigned __int64)(unsigned int)v79 + 1 > HIDWORD(v79) )
  {
    v64 = v27;
    sub_C8D5F0((__int64)&v78, v80, (unsigned int)v79 + 1LL, 4u, v27, v26);
    v30 = (unsigned int)v79;
    LODWORD(v27) = v64;
  }
  *(_DWORD *)&v78[4 * v30] = v27;
  v31 = (unsigned __int16)a9;
  v32 = (unsigned int)(v79 + 1);
  if ( !(_WORD)a9 )
    v31 = a10;
  LODWORD(v79) = v79 + 1;
  v33 = v31;
  v34 = (unsigned int)v31;
  if ( v32 + 1 > (unsigned __int64)HIDWORD(v79) )
  {
    sub_C8D5F0((__int64)&v78, v80, v32 + 1, 4u, (unsigned int)v31, v26);
    v32 = (unsigned int)v79;
    v34 = (unsigned int)v33;
  }
  v35 = HIDWORD(v33);
  *(_DWORD *)&v78[4 * v32] = v34;
  LODWORD(v79) = v79 + 1;
  v36 = (unsigned int)v79;
  if ( (unsigned __int64)(unsigned int)v79 + 1 > HIDWORD(v79) )
  {
    sub_C8D5F0((__int64)&v78, v80, (unsigned int)v79 + 1LL, 4u, v34, v26);
    v36 = (unsigned int)v79;
  }
  *(_DWORD *)&v78[4 * v36] = v35;
  LODWORD(v79) = v79 + 1;
  v74[0].m128i_i64[0] = 0;
  v37 = (__m128i *)sub_33CCCF0((__int64)a1, (__int64)&v78, a3, v74[0].m128i_i64);
  v38 = v37;
  if ( !v37 )
  {
    v40 = (__m128i *)a1[52];
    v42 = *(_DWORD *)(a3 + 8);
    if ( v40 )
    {
      a1[52] = v40->m128i_i64[0];
    }
    else
    {
      v53 = a1[53];
      a1[63] += 120LL;
      v54 = (v53 + 7) & 0xFFFFFFFFFFFFFFF8LL;
      if ( a1[54] >= v54 + 120 && v53 )
      {
        a1[53] = v54 + 120;
        if ( !v54 )
        {
LABEL_35:
          sub_33E4EC0((__int64)a1, (__int64)v40, (__int64)a7, a8);
          sub_C657C0(a1 + 65, v40->m128i_i64, (__int64 *)v74[0].m128i_i64[0], (__int64)off_4A367D0);
          if ( v78 != v80 )
            _libc_free((unsigned __int64)v78);
          goto LABEL_31;
        }
        v40 = (__m128i *)((v53 + 7) & 0xFFFFFFFFFFFFFFF8LL);
      }
      else
      {
        v71 = v42;
        v56 = sub_9D1E70((__int64)(a1 + 53), 120, 120, 3);
        v42 = v71;
        v40 = (__m128i *)v56;
      }
    }
    *((_QWORD *)&v59 + 1) = a10;
    *(_QWORD *)&v59 = (unsigned __int16)a9;
    sub_33CF750(v40, a2, v42, (unsigned __int8 **)a3, a4, a5, v59, (__int64)a6);
    v40[2].m128i_i8[0] |= 2u;
    goto LABEL_35;
  }
  sub_2EAC4C0((__m128i *)v37[7].m128i_i64[0], a6);
  result = v38;
  if ( v78 != v80 )
  {
    _libc_free((unsigned __int64)v78);
    return v38;
  }
  return result;
}
