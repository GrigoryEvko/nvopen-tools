// Function: sub_200DAC0
// Address: 0x200dac0
//
__int64 *__fastcall sub_200DAC0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __m128i a6,
        double a7,
        __m128i a8)
{
  __int64 v9; // r13
  __int64 v10; // rsi
  __int64 v12; // rbx
  __int64 v13; // rsi
  __int64 v14; // r13
  __int64 v15; // rbx
  char v16; // r14
  __int64 v17; // rax
  char v18; // r13
  __int64 v19; // rax
  int v20; // ebx
  int v21; // eax
  unsigned int v22; // esi
  __int64 v23; // r15
  unsigned int v24; // eax
  const void **v25; // rbx
  unsigned int v26; // ecx
  int v27; // r14d
  int v28; // eax
  __int64 v29; // r13
  __int64 v30; // rax
  unsigned int v31; // eax
  const void **v32; // rdx
  const void **v33; // r13
  __int64 v34; // rax
  unsigned int v35; // edx
  __int64 v36; // rax
  __int64 *v37; // r14
  unsigned int v38; // edx
  __int64 v39; // r15
  unsigned int v40; // eax
  __int128 v41; // rax
  unsigned int v42; // edx
  __int64 *v43; // r14
  const void **v45; // rdx
  __int128 v46; // [rsp-10h] [rbp-D0h]
  unsigned int v47; // [rsp+0h] [rbp-C0h]
  unsigned int v48; // [rsp+8h] [rbp-B8h]
  __int64 v49; // [rsp+10h] [rbp-B0h]
  unsigned int v50; // [rsp+18h] [rbp-A8h]
  unsigned int v51; // [rsp+28h] [rbp-98h]
  unsigned int v52; // [rsp+28h] [rbp-98h]
  __int128 v53; // [rsp+30h] [rbp-90h]
  __int128 v54; // [rsp+40h] [rbp-80h]
  __int64 *v55; // [rsp+40h] [rbp-80h]
  unsigned __int64 v56; // [rsp+48h] [rbp-78h]
  __int64 v57; // [rsp+50h] [rbp-70h] BYREF
  int v58; // [rsp+58h] [rbp-68h]
  __int64 v59; // [rsp+60h] [rbp-60h] BYREF
  int v60; // [rsp+68h] [rbp-58h]
  char v61[8]; // [rsp+70h] [rbp-50h] BYREF
  __int64 v62; // [rsp+78h] [rbp-48h]
  char v63[8]; // [rsp+80h] [rbp-40h] BYREF
  __int64 v64; // [rsp+88h] [rbp-38h]

  *(_QWORD *)&v53 = a2;
  *((_QWORD *)&v53 + 1) = a3;
  *(_QWORD *)&v54 = a4;
  *((_QWORD *)&v54 + 1) = a5;
  v9 = (unsigned int)a3;
  v10 = *(_QWORD *)(a4 + 72);
  v12 = (unsigned int)a5;
  v57 = v10;
  if ( v10 )
    sub_1623A60((__int64)&v57, v10, 2);
  v13 = *(_QWORD *)(a2 + 72);
  v58 = *(_DWORD *)(a4 + 64);
  v59 = v13;
  if ( v13 )
    sub_1623A60((__int64)&v59, v13, 2);
  v14 = *(_QWORD *)(a2 + 40) + 16 * v9;
  v15 = *(_QWORD *)(a4 + 40) + 16 * v12;
  v60 = *(_DWORD *)(a2 + 64);
  v16 = *(_BYTE *)v14;
  v17 = *(_QWORD *)(v14 + 8);
  v61[0] = v16;
  v62 = v17;
  v18 = *(_BYTE *)v15;
  v19 = *(_QWORD *)(v15 + 8);
  v63[0] = *(_BYTE *)v15;
  v64 = v19;
  if ( v16 )
    v20 = sub_200D0E0(v16);
  else
    v20 = sub_1F58D40((__int64)v61);
  if ( v18 )
    v21 = sub_200D0E0(v18);
  else
    v21 = sub_1F58D40((__int64)v63);
  v22 = v21 + v20;
  v23 = *(_QWORD *)(a1 + 8);
  if ( v21 + v20 == 32 )
  {
    LOBYTE(v24) = 5;
    goto LABEL_13;
  }
  if ( v22 <= 0x20 )
  {
    if ( v22 == 8 )
    {
      LOBYTE(v24) = 3;
    }
    else
    {
      LOBYTE(v24) = 4;
      if ( v22 != 16 )
      {
        LOBYTE(v24) = 2;
        if ( v22 != 1 )
          goto LABEL_28;
      }
    }
LABEL_13:
    v25 = 0;
    goto LABEL_14;
  }
  if ( v22 == 64 )
  {
    LOBYTE(v24) = 6;
    goto LABEL_13;
  }
  if ( v22 == 128 )
  {
    LOBYTE(v24) = 7;
    goto LABEL_13;
  }
LABEL_28:
  v24 = sub_1F58CC0(*(_QWORD **)(v23 + 48), v22);
  v18 = v63[0];
  v16 = v61[0];
  v51 = v24;
  v25 = v45;
  v23 = *(_QWORD *)(a1 + 8);
LABEL_14:
  v26 = v51;
  LOBYTE(v26) = v24;
  v52 = v26;
  if ( v16 )
  {
    v27 = sub_200D0E0(v16);
    if ( v18 )
      goto LABEL_16;
LABEL_35:
    v28 = sub_1F58D40((__int64)v63);
    goto LABEL_17;
  }
  v27 = sub_1F58D40((__int64)v61);
  if ( !v18 )
    goto LABEL_35;
LABEL_16:
  v28 = sub_200D0E0(v18);
LABEL_17:
  if ( v28 == v27 && ((v28 - 16) & 0xFFFFFFEF) == 0 )
  {
    v43 = sub_1D332F0(
            (__int64 *)v23,
            50,
            (__int64)&v57,
            v52,
            v25,
            0,
            *(double *)a6.m128i_i64,
            a7,
            a8,
            v53,
            *((unsigned __int64 *)&v53 + 1),
            v54);
  }
  else
  {
    v29 = *(_QWORD *)a1;
    v30 = sub_1E0A0C0(*(_QWORD *)(v23 + 32));
    v31 = sub_1F40B60(v29, v52, (__int64)v25, v30, 0);
    v33 = v32;
    v47 = v31;
    v34 = sub_1D309E0(
            *(__int64 **)(a1 + 8),
            143,
            (__int64)&v59,
            v52,
            v25,
            0,
            *(double *)a6.m128i_i64,
            a7,
            *(double *)a8.m128i_i64,
            v53);
    v48 = v35;
    v49 = v34;
    v36 = sub_1D309E0(
            *(__int64 **)(a1 + 8),
            144,
            (__int64)&v57,
            v52,
            v25,
            0,
            *(double *)a6.m128i_i64,
            a7,
            *(double *)a8.m128i_i64,
            v54);
    v37 = *(__int64 **)(a1 + 8);
    v50 = v38;
    v39 = v36;
    if ( v61[0] )
      v40 = sub_200D0E0(v61[0]);
    else
      v40 = sub_1F58D40((__int64)v61);
    *(_QWORD *)&v41 = sub_1D38BB0((__int64)v37, v40, (__int64)&v57, v47, v33, 0, a6, a7, a8, 0);
    v56 = v50 | *((_QWORD *)&v54 + 1) & 0xFFFFFFFF00000000LL;
    v55 = sub_1D332F0(v37, 122, (__int64)&v57, v52, v25, 0, *(double *)a6.m128i_i64, a7, a8, v39, v56, v41);
    *((_QWORD *)&v46 + 1) = v42 | v56 & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v46 = v55;
    v43 = sub_1D332F0(
            *(__int64 **)(a1 + 8),
            119,
            (__int64)&v57,
            v52,
            v25,
            0,
            *(double *)a6.m128i_i64,
            a7,
            a8,
            v49,
            v48 | *((_QWORD *)&v53 + 1) & 0xFFFFFFFF00000000LL,
            v46);
  }
  if ( v59 )
    sub_161E7C0((__int64)&v59, v59);
  if ( v57 )
    sub_161E7C0((__int64)&v57, v57);
  return v43;
}
