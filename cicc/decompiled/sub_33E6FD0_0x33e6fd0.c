// Function: sub_33E6FD0
// Address: 0x33e6fd0
//
__m128i *__fastcall sub_33E6FD0(
        _QWORD *a1,
        unsigned __int64 a2,
        __int32 a3,
        unsigned __int16 a4,
        unsigned __int64 a5,
        __int64 a6,
        unsigned __int64 *a7,
        __int64 a8,
        const __m128i *a9,
        __int16 a10)
{
  __int64 v12; // r9
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // r15
  __int64 v15; // r8
  __int64 v16; // rax
  unsigned __int64 v17; // rdx
  unsigned __int64 v18; // r15
  __int64 v19; // rax
  __int32 v20; // r11d
  __int64 v21; // r9
  int v22; // eax
  __int64 v23; // rdx
  unsigned __int64 v24; // r8
  int v25; // eax
  __int64 v26; // r9
  __int64 v27; // rdx
  unsigned __int64 v28; // r8
  __int64 v29; // r8
  __int64 v30; // rax
  __m128i *v31; // rax
  __m128i *v32; // r14
  __int32 v34; // r15d
  __int64 v35; // rcx
  unsigned __int64 v36; // rax
  __int128 v37; // [rsp-20h] [rbp-1B0h]
  __int128 v38; // [rsp-20h] [rbp-1B0h]
  __int32 v39; // [rsp+8h] [rbp-188h]
  __int16 v41; // [rsp+20h] [rbp-170h]
  __int16 v42; // [rsp+20h] [rbp-170h]
  int v43; // [rsp+20h] [rbp-170h]
  int v44; // [rsp+20h] [rbp-170h]
  int v45; // [rsp+20h] [rbp-170h]
  unsigned __int8 *v48; // [rsp+48h] [rbp-148h] BYREF
  __m128i v49[2]; // [rsp+50h] [rbp-140h] BYREF
  __int16 v50; // [rsp+70h] [rbp-120h]
  __int64 v51[6]; // [rsp+A0h] [rbp-F0h] BYREF
  _BYTE *v52; // [rsp+D0h] [rbp-C0h] BYREF
  __int64 v53; // [rsp+D8h] [rbp-B8h]
  _BYTE v54[176]; // [rsp+E0h] [rbp-B0h] BYREF

  v52 = v54;
  v53 = 0x2000000000LL;
  sub_33C9670((__int64)&v52, 467, a2, a7, a8, a6);
  v13 = a4;
  if ( !a4 )
    v13 = a5;
  v14 = v13;
  v15 = (unsigned int)v13;
  v16 = (unsigned int)v53;
  v17 = (unsigned int)v53 + 1LL;
  if ( v17 > HIDWORD(v53) )
  {
    sub_C8D5F0((__int64)&v52, v54, v17, 4u, v15, v12);
    v16 = (unsigned int)v53;
    v15 = (unsigned int)v14;
  }
  v18 = HIDWORD(v14);
  *(_DWORD *)&v52[4 * v16] = v15;
  LODWORD(v53) = v53 + 1;
  v19 = (unsigned int)v53;
  if ( (unsigned __int64)(unsigned int)v53 + 1 > HIDWORD(v53) )
  {
    sub_C8D5F0((__int64)&v52, v54, (unsigned int)v53 + 1LL, 4u, v15, v12);
    v19 = (unsigned int)v53;
  }
  *(_DWORD *)&v52[4 * v19] = v18;
  v20 = *(_DWORD *)(a6 + 8);
  v39 = a3;
  *((_QWORD *)&v37 + 1) = a5;
  LODWORD(v53) = v53 + 1;
  *(_QWORD *)&v37 = a4;
  v48 = 0;
  sub_33CF750(v49, 467, v20, &v48, a2, a3, v37, (__int64)a9);
  LOWORD(v22) = ((a10 & 7) << 7) | v50 & 0xFC7F;
  v50 = v22;
  LOBYTE(v22) = v22 & 0xFA;
  if ( v51[0] )
  {
    v41 = v22;
    sub_B91220((__int64)v51, v51[0]);
    LOWORD(v22) = v41;
  }
  if ( v48 )
  {
    v42 = v22;
    sub_B91220((__int64)&v48, (__int64)v48);
    LOWORD(v22) = v42;
  }
  v23 = (unsigned int)v53;
  v22 = (unsigned __int16)v22;
  v24 = (unsigned int)v53 + 1LL;
  if ( v24 > HIDWORD(v53) )
  {
    v44 = (unsigned __int16)v22;
    sub_C8D5F0((__int64)&v52, v54, (unsigned int)v53 + 1LL, 4u, v24, v21);
    v23 = (unsigned int)v53;
    v22 = v44;
  }
  *(_DWORD *)&v52[4 * v23] = v22;
  LODWORD(v53) = v53 + 1;
  v25 = sub_2EAC1E0((__int64)a9);
  v27 = (unsigned int)v53;
  v28 = (unsigned int)v53 + 1LL;
  if ( v28 > HIDWORD(v53) )
  {
    v45 = v25;
    sub_C8D5F0((__int64)&v52, v54, (unsigned int)v53 + 1LL, 4u, v28, v26);
    v27 = (unsigned int)v53;
    v25 = v45;
  }
  *(_DWORD *)&v52[4 * v27] = v25;
  v29 = a9[2].m128i_u16[0];
  LODWORD(v53) = v53 + 1;
  v30 = (unsigned int)v53;
  if ( (unsigned __int64)(unsigned int)v53 + 1 > HIDWORD(v53) )
  {
    v43 = v29;
    sub_C8D5F0((__int64)&v52, v54, (unsigned int)v53 + 1LL, 4u, v29, v26);
    v30 = (unsigned int)v53;
    LODWORD(v29) = v43;
  }
  *(_DWORD *)&v52[4 * v30] = v29;
  LODWORD(v53) = v53 + 1;
  v49[0].m128i_i64[0] = 0;
  v31 = (__m128i *)sub_33CCCF0((__int64)a1, (__int64)&v52, a6, v49[0].m128i_i64);
  v32 = v31;
  if ( v31 )
  {
    sub_2EAC4C0((__m128i *)v31[7].m128i_i64[0], a9);
    goto LABEL_19;
  }
  v32 = (__m128i *)a1[52];
  v34 = *(_DWORD *)(a6 + 8);
  if ( v32 )
  {
    a1[52] = v32->m128i_i64[0];
  }
  else
  {
    v35 = a1[53];
    a1[63] += 120LL;
    v36 = (v35 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    if ( a1[54] >= v36 + 120 && v35 )
    {
      a1[53] = v36 + 120;
      if ( !v36 )
        goto LABEL_25;
    }
    else
    {
      v36 = sub_9D1E70((__int64)(a1 + 53), 120, 120, 3);
    }
    v32 = (__m128i *)v36;
  }
  *((_QWORD *)&v38 + 1) = a5;
  *(_QWORD *)&v38 = a4;
  sub_33CF750(v32, 467, v34, (unsigned __int8 **)a6, a2, v39, v38, (__int64)a9);
  v32[2].m128i_i16[0] = v32[2].m128i_i16[0] & 0xFC7F | ((a10 & 7) << 7);
LABEL_25:
  sub_33E4EC0((__int64)a1, (__int64)v32, (__int64)a7, a8);
  sub_C657C0(a1 + 65, v32->m128i_i64, (__int64 *)v49[0].m128i_i64[0], (__int64)off_4A367D0);
  sub_33CC420((__int64)a1, (__int64)v32);
LABEL_19:
  if ( v52 != v54 )
    _libc_free((unsigned __int64)v52);
  return v32;
}
