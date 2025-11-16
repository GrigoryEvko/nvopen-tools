// Function: sub_1FA7370
// Address: 0x1fa7370
//
__int64 __fastcall sub_1FA7370(__int64 a1, __int64 a2, double a3, double a4, __m128i a5)
{
  __int64 v7; // rax
  unsigned int v8; // r13d
  __int64 v9; // rbx
  __m128i v10; // xmm0
  __m128i v11; // xmm1
  __int64 v12; // r14
  __int64 v13; // rax
  char v14; // dl
  const void **v15; // rax
  __int64 v16; // rax
  __int64 v17; // rsi
  unsigned __int8 v18; // cl
  const void **v19; // rax
  __int64 v20; // r9
  int v21; // eax
  __int16 v22; // dx
  __int64 v23; // rax
  int v24; // edx
  __int64 v25; // r12
  __int64 *v27; // rdi
  _QWORD *v28; // r13
  int v29; // edx
  int v30; // ebx
  __int64 *v31; // rax
  unsigned int v32; // edx
  __int64 v33; // rax
  __int64 *v34; // rax
  __int64 v35; // rax
  int v36; // edx
  int v37; // r14d
  __int64 v38; // rbx
  __int64 *v39; // rax
  unsigned int v40; // edx
  __int64 *v41; // r14
  __int128 *v42; // rbx
  __int128 v43; // rax
  __int64 v44; // r9
  __int64 *v45; // r14
  int v46; // edx
  const void **v47; // [rsp+0h] [rbp-A0h]
  unsigned __int8 v48; // [rsp+Fh] [rbp-91h]
  unsigned int v49; // [rsp+30h] [rbp-70h] BYREF
  const void **v50; // [rsp+38h] [rbp-68h]
  __int64 v51; // [rsp+40h] [rbp-60h] BYREF
  int v52; // [rsp+48h] [rbp-58h]
  __int64 v53; // [rsp+50h] [rbp-50h] BYREF
  unsigned int v54; // [rsp+58h] [rbp-48h]
  _QWORD *v55; // [rsp+60h] [rbp-40h]
  int v56; // [rsp+68h] [rbp-38h]

  v7 = *(_QWORD *)(a2 + 32);
  v8 = *(_DWORD *)(v7 + 8);
  v9 = *(_QWORD *)v7;
  v10 = _mm_loadu_si128((const __m128i *)v7);
  v11 = _mm_loadu_si128((const __m128i *)(v7 + 40));
  v12 = *(_QWORD *)(v7 + 40);
  v13 = *(_QWORD *)(*(_QWORD *)v7 + 40LL) + 16LL * v8;
  v14 = *(_BYTE *)v13;
  v15 = *(const void ***)(v13 + 8);
  LOBYTE(v49) = v14;
  v50 = v15;
  if ( v14 )
  {
    if ( (unsigned __int8)(v14 - 14) > 0x5Fu )
      goto LABEL_3;
    return 0;
  }
  if ( sub_1F58D20((__int64)&v49) )
    return 0;
LABEL_3:
  v16 = *(_QWORD *)(a2 + 40);
  v17 = *(_QWORD *)(a2 + 72);
  v18 = *(_BYTE *)(v16 + 16);
  v19 = *(const void ***)(v16 + 24);
  v51 = v17;
  v48 = v18;
  v47 = v19;
  if ( v17 )
    sub_1623A60((__int64)&v51, v17, 2);
  v52 = *(_DWORD *)(a2 + 64);
  if ( !(unsigned __int8)sub_1D18C40(a2, 1) )
  {
    v27 = *(__int64 **)a1;
    v53 = 0;
    v54 = 0;
    v28 = sub_1D2B300(v27, 0x30u, (__int64)&v53, v48, (__int64)v47, v20);
    v30 = v29;
    if ( v53 )
      sub_161E7C0((__int64)&v53, v53);
    v31 = sub_1D332F0(
            *(__int64 **)a1,
            52,
            (__int64)&v51,
            v49,
            v50,
            0,
            *(double *)v10.m128i_i64,
            *(double *)v11.m128i_i64,
            a5,
            v10.m128i_i64[0],
            v10.m128i_u64[1],
            *(_OWORD *)&v11);
    v55 = v28;
    v53 = (__int64)v31;
    v56 = v30;
    v54 = v32;
    goto LABEL_22;
  }
  v21 = *(unsigned __int16 *)(v9 + 24);
  v22 = *(_WORD *)(v12 + 24);
  if ( (v21 == 32 || v21 == 10) && v22 != 32 && v22 != 10 )
  {
    v25 = (__int64)sub_1D37440(
                     *(__int64 **)a1,
                     71,
                     (__int64)&v51,
                     *(const void ****)(a2 + 40),
                     *(_DWORD *)(a2 + 60),
                     v20,
                     *(double *)v10.m128i_i64,
                     *(double *)v11.m128i_i64,
                     a5,
                     *(_OWORD *)&v11,
                     *(_OWORD *)&v10);
    goto LABEL_11;
  }
  if ( sub_1D185B0(v11.m128i_i64[0]) )
  {
    v23 = sub_1D38BB0(*(_QWORD *)a1, 0, (__int64)&v51, v48, v47, 0, v10, *(double *)v11.m128i_i64, a5, 0);
    v53 = v9;
    v54 = v8;
LABEL_10:
    v56 = v24;
    v55 = (_QWORD *)v23;
    v25 = sub_1F994A0(a1, a2, &v53, 2, 1);
    goto LABEL_11;
  }
  if ( !(unsigned int)sub_1D1FED0(*(_QWORD *)a1, v10.m128i_i64[0], v10.m128i_i64[1], v11.m128i_i64[0], v11.m128i_i64[1]) )
  {
    v35 = sub_1D38BB0(*(_QWORD *)a1, 0, (__int64)&v51, v48, v47, 0, v10, *(double *)v11.m128i_i64, a5, 0);
    v37 = v36;
    v38 = v35;
    v39 = sub_1D332F0(
            *(__int64 **)a1,
            52,
            (__int64)&v51,
            v49,
            v50,
            0,
            *(double *)v10.m128i_i64,
            *(double *)v11.m128i_i64,
            a5,
            v10.m128i_i64[0],
            v10.m128i_u64[1],
            *(_OWORD *)&v11);
    v55 = (_QWORD *)v38;
    v54 = v40;
    v53 = (__int64)v39;
    v56 = v37;
LABEL_22:
    v33 = sub_1F994A0(a1, a2, &v53, 2, 1);
    goto LABEL_23;
  }
  if ( sub_1D18970(v10.m128i_i64[0]) && (unsigned __int8)sub_1F706D0(v11.m128i_i64[0], v11.m128i_u32[2]) )
  {
    v41 = *(__int64 **)a1;
    v42 = *(__int128 **)(v9 + 32);
    *(_QWORD *)&v43 = sub_1D38BB0(*(_QWORD *)a1, 0, (__int64)&v51, v49, v50, 0, v10, *(double *)v11.m128i_i64, a5, 0);
    v45 = sub_1D37440(
            v41,
            73,
            (__int64)&v51,
            *(const void ****)(a2 + 40),
            *(_DWORD *)(a2 + 60),
            v44,
            *(double *)v10.m128i_i64,
            *(double *)v11.m128i_i64,
            a5,
            v43,
            *v42);
    LODWORD(v42) = v46;
    v23 = (__int64)sub_1F6DC60(
                     (__int64)v45,
                     1u,
                     (__int64)&v51,
                     v48,
                     v47,
                     *(__int64 **)a1,
                     v10,
                     *(double *)v11.m128i_i64,
                     a5,
                     *(_DWORD **)(a1 + 8));
    v53 = (__int64)v45;
    v54 = (unsigned int)v42;
    goto LABEL_10;
  }
  v33 = (__int64)sub_1F732B0(
                   (__int64 *)a1,
                   v10.m128i_i64[0],
                   v10.m128i_i64[1],
                   v11.m128i_i64[0],
                   v11.m128i_u64[1],
                   a2,
                   v10,
                   *(double *)v11.m128i_i64,
                   a5);
  if ( !v33 )
  {
    v25 = 0;
    v34 = sub_1F732B0(
            (__int64 *)a1,
            v11.m128i_i64[0],
            v11.m128i_i64[1],
            v10.m128i_i64[0],
            v10.m128i_u64[1],
            a2,
            v10,
            *(double *)v11.m128i_i64,
            a5);
    if ( v34 )
      v25 = (__int64)v34;
    goto LABEL_11;
  }
LABEL_23:
  v25 = v33;
LABEL_11:
  if ( v51 )
    sub_161E7C0((__int64)&v51, v51);
  return v25;
}
