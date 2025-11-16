// Function: sub_2898E80
// Address: 0x2898e80
//
__int64 __fastcall sub_2898E80(
        __int64 a1,
        _BYTE *a2,
        __int64 a3,
        _BYTE *a4,
        char a5,
        unsigned int **a6,
        char a7,
        _DWORD *a8)
{
  __int64 v12; // rdi
  _BYTE *v13; // rdx
  __int64 v14; // rax
  double v15; // xmm0_8
  __int64 v16; // rax
  __m128d v17; // xmm0
  __int64 v18; // rdx
  double v19; // xmm1_8
  __m128d v20; // xmm1
  __int64 v21; // rdi
  __int64 v23; // rax
  __int64 v24; // rbx
  _BYTE *v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rdx
  double v29; // xmm1_8
  __m128d v30; // xmm0
  __m128d v31; // xmm1
  _BYTE *v32; // rax
  __int64 v33; // rax
  __int64 v34; // rbx
  _BYTE *v35; // rdx
  __int64 v36; // rax
  double v37; // xmm0_8
  __int64 v38; // rax
  __m128d v39; // xmm0
  __int64 v40; // rdx
  double v41; // xmm1_8
  __m128d v42; // xmm1
  _BYTE *v43; // rax
  __int64 v44; // [rsp+8h] [rbp-A8h]
  unsigned int v46; // [rsp+18h] [rbp-98h]
  double v47; // [rsp+18h] [rbp-98h]
  unsigned int v48; // [rsp+18h] [rbp-98h]
  double v49; // [rsp+18h] [rbp-98h]
  unsigned int v50; // [rsp+18h] [rbp-98h]
  double v51; // [rsp+18h] [rbp-98h]
  __int64 v52; // [rsp+20h] [rbp-90h] BYREF
  unsigned int v53; // [rsp+28h] [rbp-88h]
  int v54; // [rsp+2Ch] [rbp-84h]
  __int64 v55; // [rsp+30h] [rbp-80h] BYREF
  _BYTE *v56; // [rsp+38h] [rbp-78h]
  _BYTE *v57; // [rsp+40h] [rbp-70h]
  __int64 v58; // [rsp+50h] [rbp-60h] BYREF
  __int64 v59; // [rsp+58h] [rbp-58h]
  __int16 v60; // [rsp+70h] [rbp-40h]

  v12 = *(_QWORD *)(a3 + 8);
  v46 = *(_DWORD *)(v12 + 32);
  if ( (unsigned int)*(unsigned __int8 *)(v12 + 8) - 17 <= 1 )
    v12 = **(_QWORD **)(v12 + 16);
  v44 = *(_QWORD *)(a1 + 16);
  v55 = sub_BCAE30(v12);
  v56 = v13;
  v14 = v55 * v46;
  if ( v14 < 0 )
    v15 = (double)(int)(((_BYTE)v55 * (_BYTE)v46) & 1 | ((v55 * (unsigned __int64)v46) >> 1))
        + (double)(int)(((_BYTE)v55 * (_BYTE)v46) & 1 | ((v55 * (unsigned __int64)v46) >> 1));
  else
    v15 = (double)(int)v14;
  v47 = v15;
  v16 = sub_DFB1B0(v44);
  v17 = (__m128d)*(unsigned __int64 *)&v15;
  v58 = v16;
  v59 = v18;
  if ( v16 < 0 )
    v19 = (double)(int)(v16 & 1 | ((unsigned __int64)v16 >> 1)) + (double)(int)(v16 & 1 | ((unsigned __int64)v16 >> 1));
  else
    v19 = (double)(int)v16;
  v17.m128d_f64[0] = v15 / v19;
  if ( fabs(v47 / v19) < 4.503599627370496e15 )
  {
    v20.m128d_f64[0] = (double)(int)v17.m128d_f64[0];
    *(_QWORD *)&v17.m128d_f64[0] = COERCE_UNSIGNED_INT64(
                                     v20.m128d_f64[0]
                                   + COERCE_DOUBLE(*(_OWORD *)&_mm_cmpgt_sd(v17, v20) & 0x3FF0000000000000LL))
                                 | *(_QWORD *)&v17.m128d_f64[0] & 0x8000000000000000LL;
  }
  *a8 += (int)v17.m128d_f64[0];
  if ( a2 )
  {
    v21 = *(_QWORD *)(a3 + 8);
    if ( a5 )
    {
      if ( a7 )
      {
        v54 = 0;
        v60 = 257;
        v52 = v21;
        v55 = a3;
        v56 = a4;
        v57 = a2;
        return sub_B33D10((__int64)a6, 0xAEu, (__int64)&v52, 1, (int)&v55, 3, v53, (__int64)&v58);
      }
      else
      {
        v50 = *(_DWORD *)(v21 + 32);
        if ( (unsigned int)*(unsigned __int8 *)(v21 + 8) - 17 <= 1 )
          v21 = **(_QWORD **)(v21 + 16);
        v33 = sub_BCAE30(v21);
        v34 = *(_QWORD *)(a1 + 16);
        v55 = v33;
        v56 = v35;
        v36 = v33 * v50;
        if ( v36 < 0 )
          v37 = (double)(int)(v36 & 1 | ((unsigned __int64)v36 >> 1))
              + (double)(int)(v36 & 1 | ((unsigned __int64)v36 >> 1));
        else
          v37 = (double)(int)v36;
        v51 = v37;
        v38 = sub_DFB1B0(v34);
        v39 = (__m128d)*(unsigned __int64 *)&v37;
        v58 = v38;
        v59 = v40;
        if ( v38 < 0 )
          v41 = (double)(int)(v38 & 1 | ((unsigned __int64)v38 >> 1))
              + (double)(int)(v38 & 1 | ((unsigned __int64)v38 >> 1));
        else
          v41 = (double)(int)v38;
        v39.m128d_f64[0] = v37 / v41;
        if ( fabs(v51 / v41) < 4.503599627370496e15 )
        {
          v42.m128d_f64[0] = (double)(int)v39.m128d_f64[0];
          *(_QWORD *)&v39.m128d_f64[0] = COERCE_UNSIGNED_INT64(
                                           v42.m128d_f64[0]
                                         + COERCE_DOUBLE(*(_OWORD *)&_mm_cmpgt_sd(v39, v42) & 0x3FF0000000000000LL))
                                       | *(_QWORD *)&v39.m128d_f64[0] & 0x8000000000000000LL;
        }
        HIDWORD(v55) = 0;
        *a8 += (int)v39.m128d_f64[0];
        v60 = 257;
        v43 = (_BYTE *)sub_A826E0(a6, (_BYTE *)a3, a4, v55, (__int64)&v58, 0);
        HIDWORD(v55) = 0;
        v60 = 257;
        return sub_92A220(a6, a2, v43, (unsigned int)v55, (__int64)&v58, 0);
      }
    }
    else
    {
      v48 = *(_DWORD *)(v21 + 32);
      if ( (unsigned int)*(unsigned __int8 *)(v21 + 8) - 17 <= 1 )
        v21 = **(_QWORD **)(v21 + 16);
      v23 = sub_BCAE30(v21);
      v24 = *(_QWORD *)(a1 + 16);
      v55 = v23;
      v56 = v25;
      v26 = v23 * v48;
      if ( v26 < 0 )
        v49 = (double)(int)(v26 & 1 | ((unsigned __int64)v26 >> 1))
            + (double)(int)(v26 & 1 | ((unsigned __int64)v26 >> 1));
      else
        v49 = (double)(int)v26;
      v27 = sub_DFB1B0(v24);
      v58 = v27;
      v59 = v28;
      if ( v27 < 0 )
        v29 = (double)(int)(v27 & 1 | ((unsigned __int64)v27 >> 1))
            + (double)(int)(v27 & 1 | ((unsigned __int64)v27 >> 1));
      else
        v29 = (double)(int)v27;
      v30 = (__m128d)*(unsigned __int64 *)&v49;
      v30.m128d_f64[0] = v49 / v29;
      if ( fabs(v49 / v29) < 4.503599627370496e15 )
      {
        v31.m128d_f64[0] = (double)(int)v30.m128d_f64[0];
        *(_QWORD *)&v30.m128d_f64[0] = COERCE_UNSIGNED_INT64(
                                         v31.m128d_f64[0]
                                       + COERCE_DOUBLE(*(_OWORD *)&_mm_cmpgt_sd(v30, v31) & 0x3FF0000000000000LL))
                                     | *(_QWORD *)&v30.m128d_f64[0] & 0x8000000000000000LL;
      }
      *a8 += (int)v30.m128d_f64[0];
      v60 = 257;
      v32 = (_BYTE *)sub_A81850(a6, (_BYTE *)a3, a4, (__int64)&v58, 0, 0);
      v60 = 257;
      return sub_929C50(a6, a2, v32, (__int64)&v58, 0, 0);
    }
  }
  else
  {
    v60 = 257;
    if ( a5 )
    {
      HIDWORD(v55) = 0;
      return sub_A826E0(a6, (_BYTE *)a3, a4, (unsigned int)v55, (__int64)&v58, 0);
    }
    else
    {
      return sub_A81850(a6, (_BYTE *)a3, a4, (__int64)&v58, 0, 0);
    }
  }
}
