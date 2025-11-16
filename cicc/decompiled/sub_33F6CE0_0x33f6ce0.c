// Function: sub_33F6CE0
// Address: 0x33f6ce0
//
__m128i *__fastcall sub_33F6CE0(
        _QWORD *a1,
        unsigned __int64 a2,
        unsigned __int64 a3,
        __int64 a4,
        unsigned __int64 a5,
        unsigned __int64 a6,
        __int64 a7,
        unsigned __int64 a8,
        __int64 a9)
{
  __m128i *v11; // rbx
  __int32 v12; // edx
  __int64 v13; // r9
  __int64 v14; // rdx
  __int32 v15; // eax
  unsigned __int64 v16; // r15
  unsigned __int64 v17; // r8
  __int64 v18; // r9
  unsigned __int64 v19; // r15
  __int64 v20; // rdx
  __int32 v21; // edx
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
  __int64 v36; // rcx
  unsigned __int64 v37; // rax
  __int128 v38; // [rsp-20h] [rbp-1D0h]
  __int128 v39; // [rsp-20h] [rbp-1D0h]
  __int32 v40; // [rsp+18h] [rbp-198h]
  __int32 v41; // [rsp+20h] [rbp-190h]
  __int16 v42; // [rsp+20h] [rbp-190h]
  __int16 v43; // [rsp+20h] [rbp-190h]
  __int32 v44; // [rsp+20h] [rbp-190h]
  int v45; // [rsp+20h] [rbp-190h]
  int v46; // [rsp+20h] [rbp-190h]
  int v47; // [rsp+20h] [rbp-190h]
  unsigned __int8 *v49; // [rsp+48h] [rbp-168h] BYREF
  unsigned __int64 v50[4]; // [rsp+50h] [rbp-160h] BYREF
  __m128i v51[2]; // [rsp+70h] [rbp-140h] BYREF
  char v52; // [rsp+90h] [rbp-120h]
  char v53; // [rsp+91h] [rbp-11Fh]
  __int64 v54[6]; // [rsp+C0h] [rbp-F0h] BYREF
  _BYTE *v55; // [rsp+F0h] [rbp-C0h] BYREF
  __int64 v56; // [rsp+F8h] [rbp-B8h]
  _BYTE v57[176]; // [rsp+100h] [rbp-B0h] BYREF

  v50[2] = a5;
  v11 = sub_33ED250((__int64)a1, 1, 0);
  v55 = v57;
  v41 = v12;
  v56 = 0x2000000000LL;
  v50[3] = a6;
  v50[0] = a2;
  v50[1] = a3;
  sub_33C9670((__int64)&v55, 293, (unsigned __int64)v11, v50, 2, v13);
  v14 = (unsigned int)v56;
  v15 = v41;
  v16 = (unsigned __int16)a7;
  v17 = (unsigned int)v56 + 1LL;
  if ( !(_WORD)a7 )
    v16 = a8;
  v18 = (unsigned int)v16;
  if ( v17 > HIDWORD(v56) )
  {
    sub_C8D5F0((__int64)&v55, v57, (unsigned int)v56 + 1LL, 4u, v17, (unsigned int)v16);
    v14 = (unsigned int)v56;
    v15 = v41;
    v18 = (unsigned int)v16;
  }
  v19 = HIDWORD(v16);
  *(_DWORD *)&v55[4 * v14] = v18;
  LODWORD(v56) = v56 + 1;
  v20 = (unsigned int)v56;
  if ( (unsigned __int64)(unsigned int)v56 + 1 > HIDWORD(v56) )
  {
    v44 = v15;
    sub_C8D5F0((__int64)&v55, v57, (unsigned int)v56 + 1LL, 4u, (unsigned int)v56 + 1LL, v18);
    v20 = (unsigned int)v56;
    v15 = v44;
  }
  v40 = v15;
  *(_DWORD *)&v55[4 * v20] = v19;
  v21 = *(_DWORD *)(a4 + 8);
  *((_QWORD *)&v38 + 1) = a8;
  *(_QWORD *)&v38 = (unsigned __int16)a7;
  LODWORD(v56) = v56 + 1;
  v49 = 0;
  sub_33CF750(v51, 293, v21, &v49, (__int64)v11, v15, v38, a9);
  BYTE1(v23) = v53;
  LOBYTE(v23) = v52 & 0xFA;
  if ( v54[0] )
  {
    v42 = v23;
    sub_B91220((__int64)v54, v54[0]);
    LOWORD(v23) = v42;
  }
  if ( v49 )
  {
    v43 = v23;
    sub_B91220((__int64)&v49, (__int64)v49);
    LOWORD(v23) = v43;
  }
  v24 = (unsigned int)v56;
  v23 = (unsigned __int16)v23;
  v25 = (unsigned int)v56 + 1LL;
  if ( v25 > HIDWORD(v56) )
  {
    v45 = (unsigned __int16)v23;
    sub_C8D5F0((__int64)&v55, v57, (unsigned int)v56 + 1LL, 4u, v25, v22);
    v24 = (unsigned int)v56;
    v23 = v45;
  }
  *(_DWORD *)&v55[4 * v24] = v23;
  LODWORD(v56) = v56 + 1;
  v26 = sub_2EAC1E0(a9);
  v28 = (unsigned int)v56;
  v29 = (unsigned int)v56 + 1LL;
  if ( v29 > HIDWORD(v56) )
  {
    v46 = v26;
    sub_C8D5F0((__int64)&v55, v57, (unsigned int)v56 + 1LL, 4u, v29, v27);
    v28 = (unsigned int)v56;
    v26 = v46;
  }
  *(_DWORD *)&v55[4 * v28] = v26;
  v30 = *(unsigned __int16 *)(a9 + 32);
  LODWORD(v56) = v56 + 1;
  v31 = (unsigned int)v56;
  if ( (unsigned __int64)(unsigned int)v56 + 1 > HIDWORD(v56) )
  {
    v47 = v30;
    sub_C8D5F0((__int64)&v55, v57, (unsigned int)v56 + 1LL, 4u, v30, v27);
    v31 = (unsigned int)v56;
    LODWORD(v30) = v47;
  }
  *(_DWORD *)&v55[4 * v31] = v30;
  LODWORD(v56) = v56 + 1;
  v51[0].m128i_i64[0] = 0;
  v32 = (__m128i *)sub_33CCCF0((__int64)a1, (__int64)&v55, a4, v51[0].m128i_i64);
  if ( v32 )
  {
    v33 = v32;
    goto LABEL_19;
  }
  v33 = (__m128i *)a1[52];
  v35 = *(_DWORD *)(a4 + 8);
  if ( v33 )
  {
    a1[52] = v33->m128i_i64[0];
  }
  else
  {
    v36 = a1[53];
    a1[63] += 120LL;
    v37 = (v36 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    if ( a1[54] >= v37 + 120 && v36 )
    {
      a1[53] = v37 + 120;
      if ( !v37 )
        goto LABEL_25;
    }
    else
    {
      v37 = sub_9D1E70((__int64)(a1 + 53), 120, 120, 3);
    }
    v33 = (__m128i *)v37;
  }
  *((_QWORD *)&v39 + 1) = a8;
  *(_QWORD *)&v39 = (unsigned __int16)a7;
  sub_33CF750(v33, 293, v35, (unsigned __int8 **)a4, (__int64)v11, v40, v39, a9);
LABEL_25:
  sub_33E4EC0((__int64)a1, (__int64)v33, (__int64)v50, 2);
  sub_C657C0(a1 + 65, v33->m128i_i64, (__int64 *)v51[0].m128i_i64[0], (__int64)off_4A367D0);
  sub_33CC420((__int64)a1, (__int64)v33);
LABEL_19:
  if ( v55 != v57 )
    _libc_free((unsigned __int64)v55);
  return v33;
}
