// Function: sub_1FA2A00
// Address: 0x1fa2a00
//
__int64 __fastcall sub_1FA2A00(__int64 *a1, __int64 a2, double a3, double a4, __m128i a5)
{
  __int64 v7; // rax
  __m128i v8; // xmm0
  __int64 v9; // rbx
  unsigned int v10; // r13d
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
  __int64 v21; // r14
  __int64 v23; // rax
  int v24; // edx
  _QWORD *v25; // rdi
  _QWORD *v26; // r13
  int v27; // edx
  int v28; // ebx
  __int64 *v29; // rax
  unsigned int v30; // edx
  __int64 v31; // rax
  int v32; // edx
  int v33; // ebx
  __int64 v34; // r14
  __int64 v35; // rax
  unsigned int v36; // edx
  __int64 v37; // rax
  int v38; // edx
  int v39; // r14d
  __int64 v40; // rbx
  __int64 *v41; // rax
  unsigned int v42; // edx
  const void **v43; // [rsp+10h] [rbp-A0h]
  unsigned __int8 v44; // [rsp+1Bh] [rbp-95h]
  int v45; // [rsp+1Ch] [rbp-94h]
  unsigned int v46; // [rsp+40h] [rbp-70h] BYREF
  const void **v47; // [rsp+48h] [rbp-68h]
  __int64 v48; // [rsp+50h] [rbp-60h] BYREF
  int v49; // [rsp+58h] [rbp-58h]
  __int64 v50; // [rsp+60h] [rbp-50h] BYREF
  unsigned int v51; // [rsp+68h] [rbp-48h]
  _QWORD *v52; // [rsp+70h] [rbp-40h]
  int v53; // [rsp+78h] [rbp-38h]

  v7 = *(_QWORD *)(a2 + 32);
  v8 = _mm_loadu_si128((const __m128i *)v7);
  v9 = *(_QWORD *)v7;
  v10 = *(_DWORD *)(v7 + 8);
  v11 = _mm_loadu_si128((const __m128i *)(v7 + 40));
  v12 = *(_QWORD *)(v7 + 40);
  v45 = *(_DWORD *)(v7 + 48);
  v13 = *(_QWORD *)(*(_QWORD *)v7 + 40LL) + 16LL * v10;
  v14 = *(_BYTE *)v13;
  v15 = *(const void ***)(v13 + 8);
  LOBYTE(v46) = v14;
  v47 = v15;
  if ( v14 )
  {
    if ( (unsigned __int8)(v14 - 14) > 0x5Fu )
      goto LABEL_3;
    return 0;
  }
  if ( sub_1F58D20((__int64)&v46) )
    return 0;
LABEL_3:
  v16 = *(_QWORD *)(a2 + 40);
  v17 = *(_QWORD *)(a2 + 72);
  v18 = *(_BYTE *)(v16 + 16);
  v19 = *(const void ***)(v16 + 24);
  v48 = v17;
  v44 = v18;
  v43 = v19;
  if ( v17 )
    sub_1623A60((__int64)&v48, v17, 2);
  v49 = *(_DWORD *)(a2 + 64);
  if ( (unsigned __int8)sub_1D18C40(a2, 1) )
  {
    if ( v12 == v9 && v10 == v45 )
    {
      v31 = sub_1D38BB0(*a1, 0, (__int64)&v48, v44, v43, 0, v8, *(double *)v11.m128i_i64, a5, 0);
      v33 = v32;
      v34 = v31;
      v35 = sub_1D38BB0(*a1, 0, (__int64)&v48, v46, v47, 0, v8, *(double *)v11.m128i_i64, a5, 0);
      v52 = (_QWORD *)v34;
      v50 = v35;
      v51 = v36;
      v53 = v33;
    }
    else
    {
      if ( !sub_1D185B0(v11.m128i_i64[0]) )
      {
        v21 = 0;
        if ( !sub_1D188A0(v8.m128i_i64[0]) )
          goto LABEL_10;
        v37 = sub_1D38BB0(*a1, 0, (__int64)&v48, v44, v43, 0, v8, *(double *)v11.m128i_i64, a5, 0);
        v39 = v38;
        v40 = v37;
        v41 = sub_1D332F0(
                (__int64 *)*a1,
                120,
                (__int64)&v48,
                v46,
                v47,
                0,
                *(double *)v8.m128i_i64,
                *(double *)v11.m128i_i64,
                a5,
                v11.m128i_i64[0],
                v11.m128i_u64[1],
                *(_OWORD *)&v8);
        v52 = (_QWORD *)v40;
        v51 = v42;
        v50 = (__int64)v41;
        v53 = v39;
        goto LABEL_20;
      }
      v23 = sub_1D38BB0(*a1, 0, (__int64)&v48, v44, v43, 0, v8, *(double *)v11.m128i_i64, a5, 0);
      v50 = v9;
      v51 = v10;
      v52 = (_QWORD *)v23;
      v53 = v24;
    }
    v21 = sub_1F994A0((__int64)a1, a2, &v50, 2, 1);
    goto LABEL_10;
  }
  v25 = (_QWORD *)*a1;
  v50 = 0;
  v51 = 0;
  v26 = sub_1D2B300(v25, 0x30u, (__int64)&v50, v44, (__int64)v43, v20);
  v28 = v27;
  if ( v50 )
    sub_161E7C0((__int64)&v50, v50);
  v29 = sub_1D332F0(
          (__int64 *)*a1,
          53,
          (__int64)&v48,
          v46,
          v47,
          0,
          *(double *)v8.m128i_i64,
          *(double *)v11.m128i_i64,
          a5,
          v8.m128i_i64[0],
          v8.m128i_u64[1],
          *(_OWORD *)&v11);
  v52 = v26;
  v50 = (__int64)v29;
  v53 = v28;
  v51 = v30;
LABEL_20:
  v21 = sub_1F994A0((__int64)a1, a2, &v50, 2, 1);
LABEL_10:
  if ( v48 )
    sub_161E7C0((__int64)&v48, v48);
  return v21;
}
