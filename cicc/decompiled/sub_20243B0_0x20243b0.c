// Function: sub_20243B0
// Address: 0x20243b0
//
unsigned __int64 __fastcall sub_20243B0(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        double a5,
        double a6,
        __m128i a7)
{
  __int64 v7; // r8
  const __m128i *v10; // rax
  __int64 v11; // rsi
  __m128i v12; // xmm0
  __int64 v13; // r14
  __int64 v14; // r15
  int v15; // eax
  __int64 v16; // rsi
  char *v17; // rax
  char v18; // dl
  __int64 v19; // rax
  __int64 *v20; // rdi
  __m128i v21; // xmm1
  int v22; // edx
  __int64 v23; // rax
  bool v24; // cc
  _QWORD *v25; // rax
  __int64 *v26; // r13
  __int64 v27; // r14
  __int64 (__fastcall *v28)(__int64, __int64); // r15
  __int64 v29; // rax
  unsigned int v30; // edx
  unsigned __int8 v31; // al
  unsigned int v32; // r15d
  __int8 v33; // al
  unsigned int v34; // eax
  __int128 v35; // rax
  __int64 *v36; // rax
  __int64 v37; // rsi
  unsigned int v38; // edx
  unsigned __int64 result; // rax
  unsigned __int8 v40; // al
  __int128 v41; // [rsp-10h] [rbp-E0h]
  const void **v42; // [rsp+0h] [rbp-D0h]
  __int64 v43; // [rsp+8h] [rbp-C8h]
  __int64 v44; // [rsp+8h] [rbp-C8h]
  __int64 v46; // [rsp+18h] [rbp-B8h]
  _QWORD *v47; // [rsp+18h] [rbp-B8h]
  __int64 v48; // [rsp+50h] [rbp-80h] BYREF
  int v49; // [rsp+58h] [rbp-78h]
  __m128i v50; // [rsp+60h] [rbp-70h] BYREF
  _QWORD v51[2]; // [rsp+70h] [rbp-60h] BYREF
  __m128i v52; // [rsp+80h] [rbp-50h] BYREF
  __int64 v53; // [rsp+90h] [rbp-40h]
  const void **v54; // [rsp+98h] [rbp-38h]

  v7 = a2;
  v10 = *(const __m128i **)(a2 + 32);
  v11 = *(_QWORD *)(a2 + 72);
  v12 = _mm_loadu_si128(v10);
  v13 = v10[2].m128i_i64[1];
  v48 = v11;
  v14 = v10[3].m128i_i64[0];
  v46 = v10[2].m128i_i64[1];
  if ( v11 )
  {
    v43 = v7;
    sub_1623A60((__int64)&v48, v11, 2);
    v7 = v43;
  }
  v15 = *(_DWORD *)(v7 + 64);
  v16 = a1[1];
  v50.m128i_i8[0] = 0;
  v50.m128i_i64[1] = 0;
  v49 = v15;
  v17 = *(char **)(v7 + 40);
  v18 = *v17;
  v19 = *((_QWORD *)v17 + 1);
  LOBYTE(v51[0]) = v18;
  v51[1] = v19;
  sub_1D19A30((__int64)&v52, v16, v51);
  v20 = (__int64 *)a1[1];
  *((_QWORD *)&v41 + 1) = v14;
  v21 = _mm_loadu_si128(&v52);
  *(_QWORD *)&v41 = v13;
  v44 = v53;
  v50 = v21;
  v42 = v54;
  *(_QWORD *)a3 = sub_1D332F0(
                    v20,
                    109,
                    (__int64)&v48,
                    v52.m128i_u32[0],
                    (const void **)v21.m128i_i64[1],
                    0,
                    *(double *)v12.m128i_i64,
                    *(double *)v21.m128i_i64,
                    a7,
                    v12.m128i_i64[0],
                    v12.m128i_u64[1],
                    v41);
  *(_DWORD *)(a3 + 8) = v22;
  v23 = *(_QWORD *)(v46 + 88);
  v24 = *(_DWORD *)(v23 + 32) <= 0x40u;
  v25 = *(_QWORD **)(v23 + 24);
  if ( !v24 )
    v25 = (_QWORD *)*v25;
  v26 = (__int64 *)a1[1];
  v27 = *a1;
  v47 = v25;
  v28 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)*a1 + 48LL);
  v29 = sub_1E0A0C0(v26[4]);
  if ( v28 == sub_1D13A20 )
  {
    v30 = 8 * sub_15A9520(v29, 0);
    if ( v30 == 32 )
    {
      v31 = 5;
    }
    else if ( v30 > 0x20 )
    {
      v31 = 6;
      if ( v30 != 64 )
      {
        v40 = 0;
        if ( v30 == 128 )
          v40 = 7;
        v32 = v40;
        v33 = v50.m128i_i8[0];
        if ( !v50.m128i_i8[0] )
          goto LABEL_11;
        goto LABEL_19;
      }
    }
    else
    {
      v31 = 3;
      if ( v30 != 8 )
        v31 = 4 * (v30 == 16);
    }
  }
  else
  {
    v31 = v28(v27, v29);
  }
  v32 = v31;
  v33 = v50.m128i_i8[0];
  if ( !v50.m128i_i8[0] )
  {
LABEL_11:
    v34 = sub_1F58D30((__int64)&v50);
    goto LABEL_12;
  }
LABEL_19:
  v34 = word_4305480[(unsigned __int8)(v33 - 14)];
LABEL_12:
  *(_QWORD *)&v35 = sub_1D38BB0(
                      (__int64)v26,
                      (__int64)v47 + v34,
                      (__int64)&v48,
                      v32,
                      0,
                      0,
                      v12,
                      *(double *)v21.m128i_i64,
                      a7,
                      0);
  v36 = sub_1D332F0(
          v26,
          109,
          (__int64)&v48,
          v44,
          v42,
          0,
          *(double *)v12.m128i_i64,
          *(double *)v21.m128i_i64,
          a7,
          v12.m128i_i64[0],
          v12.m128i_u64[1],
          v35);
  v37 = v48;
  *(_QWORD *)a4 = v36;
  result = v38;
  *(_DWORD *)(a4 + 8) = v38;
  if ( v37 )
    return sub_161E7C0((__int64)&v48, v37);
  return result;
}
