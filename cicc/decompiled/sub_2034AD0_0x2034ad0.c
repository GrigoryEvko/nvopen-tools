// Function: sub_2034AD0
// Address: 0x2034ad0
//
__int64 __fastcall sub_2034AD0(__int64 **a1, __int64 a2, double a3, double a4, __m128i a5)
{
  __int64 v6; // rax
  __m128 v7; // xmm0
  __m128i v8; // xmm1
  __int64 v9; // rax
  __int8 v10; // dl
  char *v11; // rax
  __int8 v12; // dl
  __int64 v13; // rax
  unsigned int v14; // eax
  __int64 v15; // rsi
  const void **v16; // rdx
  __int64 v17; // rsi
  unsigned int v18; // eax
  __int64 *v19; // r15
  const void **v20; // rdx
  __int64 v21; // rax
  unsigned int v22; // edx
  unsigned __int8 v23; // al
  __int128 v24; // rax
  __int64 *v25; // rax
  __int64 *v26; // r15
  unsigned int v27; // edx
  __int64 v28; // rax
  unsigned int v29; // edx
  unsigned __int8 v30; // al
  __int128 v31; // rax
  unsigned int v32; // edx
  __int128 v33; // rax
  __m128i v34; // xmm2
  __int64 *v35; // r15
  __int128 v36; // kr00_16
  bool v37; // r12
  int v38; // eax
  __int64 v39; // r14
  __int64 v41; // rax
  unsigned int v42; // edx
  unsigned int v43; // edx
  __int64 *v44; // [rsp+0h] [rbp-F0h]
  __int64 (__fastcall *v45)(__int64, __int64); // [rsp+8h] [rbp-E8h]
  unsigned int v46; // [rsp+10h] [rbp-E0h]
  const void **v47; // [rsp+18h] [rbp-D8h]
  unsigned int v48; // [rsp+20h] [rbp-D0h]
  const void **v49; // [rsp+28h] [rbp-C8h]
  __int64 *v50; // [rsp+30h] [rbp-C0h]
  unsigned __int64 v51; // [rsp+30h] [rbp-C0h]
  __int64 (__fastcall *v52)(__int64, __int64); // [rsp+38h] [rbp-B8h]
  unsigned int v53; // [rsp+38h] [rbp-B8h]
  __int128 v54; // [rsp+40h] [rbp-B0h]
  __int128 v55; // [rsp+50h] [rbp-A0h]
  __m128i v56; // [rsp+80h] [rbp-70h] BYREF
  __int64 v57; // [rsp+90h] [rbp-60h] BYREF
  int v58; // [rsp+98h] [rbp-58h]
  __m128i v59; // [rsp+A0h] [rbp-50h] BYREF

  v6 = *(_QWORD *)(a2 + 32);
  v7 = (__m128)_mm_loadu_si128((const __m128i *)v6);
  v8 = _mm_loadu_si128((const __m128i *)(v6 + 40));
  v9 = *(_QWORD *)(*(_QWORD *)v6 + 40LL) + 16LL * *(unsigned int *)(v6 + 8);
  v10 = *(_BYTE *)v9;
  v56.m128i_i64[1] = *(_QWORD *)(v9 + 8);
  v11 = *(char **)(a2 + 40);
  v56.m128i_i8[0] = v10;
  v12 = *v11;
  v13 = *((_QWORD *)v11 + 1);
  v59.m128i_i8[0] = v12;
  v59.m128i_i64[1] = v13;
  LOBYTE(v14) = sub_1F7E0F0((__int64)&v59);
  v15 = *(_QWORD *)(a2 + 72);
  v49 = v16;
  v48 = v14;
  v57 = v15;
  if ( v15 )
    sub_1623A60((__int64)&v57, v15, 2);
  v17 = (__int64)*a1;
  v58 = *(_DWORD *)(a2 + 64);
  sub_1F40D10((__int64)&v59, v17, a1[1][6], v56.m128i_u8[0], v56.m128i_i64[1]);
  if ( v59.m128i_i8[0] == 5 )
  {
    v41 = sub_2032580((__int64)a1, v7.m128_u64[0], v7.m128_i64[1]);
    v53 = v42;
    v51 = v41;
    *(_QWORD *)&v54 = sub_2032580((__int64)a1, v8.m128i_u64[0], v8.m128i_i64[1]);
    *((_QWORD *)&v54 + 1) = v43 | v8.m128i_i64[1] & 0xFFFFFFFF00000000LL;
  }
  else
  {
    LOBYTE(v18) = sub_1F7E0F0((__int64)&v56);
    v19 = a1[1];
    v46 = v18;
    v47 = v20;
    v50 = *a1;
    v52 = *(__int64 (__fastcall **)(__int64, __int64))(**a1 + 48);
    v21 = sub_1E0A0C0(v19[4]);
    if ( v52 == sub_1D13A20 )
    {
      v22 = 8 * sub_15A9520(v21, 0);
      if ( v22 == 32 )
      {
        v23 = 5;
      }
      else if ( v22 > 0x20 )
      {
        v23 = 6;
        if ( v22 != 64 )
        {
          v23 = 0;
          if ( v22 == 128 )
            v23 = 7;
        }
      }
      else
      {
        v23 = 3;
        if ( v22 != 8 )
          v23 = 4 * (v22 == 16);
      }
    }
    else
    {
      v23 = v52((__int64)v50, v21);
    }
    *(_QWORD *)&v24 = sub_1D38BB0(
                        (__int64)v19,
                        0,
                        (__int64)&v57,
                        v23,
                        0,
                        0,
                        (__m128i)v7,
                        *(double *)v8.m128i_i64,
                        a5,
                        0);
    v25 = sub_1D332F0(
            v19,
            106,
            (__int64)&v57,
            v46,
            v47,
            0,
            *(double *)v7.m128_u64,
            *(double *)v8.m128i_i64,
            a5,
            v7.m128_i64[0],
            v7.m128_u64[1],
            v24);
    v26 = a1[1];
    v51 = (unsigned __int64)v25;
    v53 = v27;
    v44 = *a1;
    v45 = *(__int64 (__fastcall **)(__int64, __int64))(**a1 + 48);
    v28 = sub_1E0A0C0(v26[4]);
    if ( v45 == sub_1D13A20 )
    {
      v29 = 8 * sub_15A9520(v28, 0);
      if ( v29 == 32 )
      {
        v30 = 5;
      }
      else if ( v29 > 0x20 )
      {
        v30 = 6;
        if ( v29 != 64 )
        {
          v30 = 0;
          if ( v29 == 128 )
            v30 = 7;
        }
      }
      else
      {
        v30 = 3;
        if ( v29 != 8 )
          v30 = 4 * (v29 == 16);
      }
    }
    else
    {
      v30 = v45((__int64)v44, v28);
    }
    *(_QWORD *)&v31 = sub_1D38BB0(
                        (__int64)v26,
                        0,
                        (__int64)&v57,
                        v30,
                        0,
                        0,
                        (__m128i)v7,
                        *(double *)v8.m128i_i64,
                        a5,
                        0);
    *(_QWORD *)&v54 = sub_1D332F0(
                        v26,
                        106,
                        (__int64)&v57,
                        v46,
                        v47,
                        0,
                        *(double *)v7.m128_u64,
                        *(double *)v8.m128i_i64,
                        a5,
                        v8.m128i_i64[0],
                        v8.m128i_u64[1],
                        v31);
    *((_QWORD *)&v54 + 1) = v32 | v8.m128i_i64[1] & 0xFFFFFFFF00000000LL;
  }
  *(_QWORD *)&v33 = sub_1D3A900(
                      a1[1],
                      0x89u,
                      (__int64)&v57,
                      2u,
                      0,
                      0,
                      v7,
                      *(double *)v8.m128i_i64,
                      a5,
                      v51,
                      (__int16 *)(v53 | v7.m128_u64[1] & 0xFFFFFFFF00000000LL),
                      v54,
                      *(_QWORD *)(*(_QWORD *)(a2 + 32) + 80LL),
                      *(_QWORD *)(*(_QWORD *)(a2 + 32) + 88LL));
  v34 = _mm_loadu_si128(&v56);
  v35 = *a1;
  v36 = v33;
  v59 = v34;
  if ( v56.m128i_i8[0] )
  {
    if ( (unsigned __int8)(v56.m128i_i8[0] - 14) > 0x5Fu )
    {
      if ( (unsigned __int8)(v56.m128i_i8[0] - 86) <= 0x17u || (unsigned __int8)(v56.m128i_i8[0] - 8) <= 5u )
        goto LABEL_18;
LABEL_32:
      v38 = *((_DWORD *)v35 + 15);
      goto LABEL_19;
    }
  }
  else
  {
    v55 = v33;
    v37 = sub_1F58CD0((__int64)&v59);
    v36 = v55;
    if ( !sub_1F58D20((__int64)&v59) )
    {
      if ( v37 )
      {
LABEL_18:
        v38 = *((_DWORD *)v35 + 16);
        goto LABEL_19;
      }
      goto LABEL_32;
    }
  }
  v38 = *((_DWORD *)v35 + 17);
LABEL_19:
  v39 = sub_1D309E0(
          a1[1],
          (unsigned int)(144 - v38),
          (__int64)&v57,
          v48,
          v49,
          0,
          *(double *)v7.m128_u64,
          *(double *)v8.m128i_i64,
          *(double *)v34.m128i_i64,
          v36);
  if ( v57 )
    sub_161E7C0((__int64)&v57, v57);
  return v39;
}
