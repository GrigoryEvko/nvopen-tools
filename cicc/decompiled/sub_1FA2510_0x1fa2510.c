// Function: sub_1FA2510
// Address: 0x1fa2510
//
__int64 __fastcall sub_1FA2510(__int64 **a1, __int64 a2, double a3, double a4, __m128i a5)
{
  __int64 v6; // rax
  __int64 v7; // rsi
  __m128i v8; // xmm0
  __int64 v9; // rbx
  unsigned int v10; // r13d
  __m128i v11; // xmm1
  unsigned __int8 *v12; // rax
  const void **v13; // rcx
  unsigned int v14; // r14d
  __int64 v15; // r9
  __int64 v16; // r9
  __int64 v17; // rbx
  __int64 v18; // r9
  _QWORD *v20; // rax
  int v21; // edx
  _QWORD *v22; // rax
  int v23; // edx
  int v24; // ebx
  _QWORD *v25; // r13
  __int64 *v26; // rax
  unsigned int v27; // edx
  _QWORD *v28; // rax
  int v29; // edx
  int v30; // ebx
  _QWORD *v31; // r13
  __int64 v32; // rax
  unsigned int v33; // edx
  _QWORD *v34; // rax
  int v35; // edx
  int v36; // [rsp+Ch] [rbp-94h]
  const void **v37; // [rsp+10h] [rbp-90h]
  __int64 v38; // [rsp+18h] [rbp-88h]
  __int64 v39; // [rsp+40h] [rbp-60h] BYREF
  int v40; // [rsp+48h] [rbp-58h]
  __int64 v41; // [rsp+50h] [rbp-50h] BYREF
  unsigned int v42; // [rsp+58h] [rbp-48h]
  _QWORD *v43; // [rsp+60h] [rbp-40h]
  int v44; // [rsp+68h] [rbp-38h]

  v6 = *(_QWORD *)(a2 + 32);
  v7 = *(_QWORD *)(a2 + 72);
  v8 = _mm_loadu_si128((const __m128i *)v6);
  v9 = *(_QWORD *)v6;
  v10 = *(_DWORD *)(v6 + 8);
  v11 = _mm_loadu_si128((const __m128i *)(v6 + 40));
  v38 = *(_QWORD *)(v6 + 40);
  v36 = *(_DWORD *)(v6 + 48);
  v12 = (unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)v6 + 40LL) + 16LL * v10);
  v13 = (const void **)*((_QWORD *)v12 + 1);
  v14 = *v12;
  v39 = v7;
  v37 = v13;
  if ( v7 )
    sub_1623A60((__int64)&v39, v7, 2);
  v40 = *(_DWORD *)(a2 + 64);
  if ( !(unsigned __int8)sub_1D18C40(a2, 1) )
  {
    v22 = sub_1D2B300(*a1, 0x3Fu, (__int64)&v39, 0x6Fu, 0, v15);
    v24 = v23;
    v25 = v22;
    v26 = sub_1D332F0(
            *a1,
            53,
            (__int64)&v39,
            v14,
            v37,
            0,
            *(double *)v8.m128i_i64,
            *(double *)v11.m128i_i64,
            a5,
            v8.m128i_i64[0],
            v8.m128i_u64[1],
            *(_OWORD *)&v11);
    goto LABEL_14;
  }
  if ( v38 != v9 || v36 != v10 )
  {
    if ( sub_1D185B0(v11.m128i_i64[0]) )
    {
      v20 = sub_1D2B300(*a1, 0x3Fu, (__int64)&v39, 0x6Fu, 0, v16);
      v41 = v9;
      v42 = v10;
      v43 = v20;
      v44 = v21;
LABEL_12:
      v17 = sub_1F994A0((__int64)a1, a2, &v41, 2, 1);
      goto LABEL_8;
    }
    v17 = 0;
    if ( !sub_1D188A0(v8.m128i_i64[0]) )
      goto LABEL_8;
    v34 = sub_1D2B300(*a1, 0x3Fu, (__int64)&v39, 0x6Fu, 0, v18);
    v24 = v35;
    v25 = v34;
    v26 = sub_1D332F0(
            *a1,
            120,
            (__int64)&v39,
            v14,
            v37,
            0,
            *(double *)v8.m128i_i64,
            *(double *)v11.m128i_i64,
            a5,
            v11.m128i_i64[0],
            v11.m128i_u64[1],
            *(_OWORD *)&v8);
LABEL_14:
    v43 = v25;
    v41 = (__int64)v26;
    v42 = v27;
    v44 = v24;
    goto LABEL_12;
  }
  v28 = sub_1D2B300(*a1, 0x3Fu, (__int64)&v39, 0x6Fu, 0, v15);
  v30 = v29;
  v31 = v28;
  v32 = sub_1D38BB0((__int64)*a1, 0, (__int64)&v39, v14, v37, 0, v8, *(double *)v11.m128i_i64, a5, 0);
  v42 = v33;
  v43 = v31;
  v44 = v30;
  v41 = v32;
  v17 = sub_1F994A0((__int64)a1, a2, &v41, 2, 1);
LABEL_8:
  if ( v39 )
    sub_161E7C0((__int64)&v39, v39);
  return v17;
}
