// Function: sub_32EF9B0
// Address: 0x32ef9b0
//
__int64 __fastcall sub_32EF9B0(_QWORD *a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  __int64 v7; // r15
  const __m128i *v10; // rdx
  __int64 v11; // rsi
  __int32 v12; // ecx
  __m128i v13; // xmm1
  _OWORD *v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // r14
  int v18; // edx
  __int64 v19; // rax
  __int64 v20; // rcx
  __int64 v21; // rax
  int v22; // eax
  int v23; // eax
  _OWORD **v24; // rdi
  __int64 v25; // rax
  __m128i v26; // xmm0
  __int16 v27; // ax
  unsigned __int16 v28; // ax
  __int64 v29; // rdx
  __int64 v30; // rax
  __int64 v31; // rdx
  __m128i v32; // xmm0
  __int64 v33; // rax
  __int64 v34; // [rsp-190h] [rbp-190h]
  __m128i v35; // [rsp-188h] [rbp-188h] BYREF
  __m128i v36; // [rsp-178h] [rbp-178h] BYREF
  __m128i v37; // [rsp-168h] [rbp-168h] BYREF
  __int64 v38; // [rsp-158h] [rbp-158h]
  __int64 v39; // [rsp-150h] [rbp-150h]
  __int64 v40; // [rsp-148h] [rbp-148h] BYREF
  __int32 v41; // [rsp-140h] [rbp-140h]
  __int64 v42; // [rsp-138h] [rbp-138h]
  int v43; // [rsp-130h] [rbp-130h]
  __int64 v44; // [rsp-128h] [rbp-128h]
  char v45; // [rsp-120h] [rbp-120h]
  char v46; // [rsp-118h] [rbp-118h]
  unsigned __int16 v47; // [rsp-108h] [rbp-108h] BYREF
  __int64 v48; // [rsp-100h] [rbp-100h]
  _OWORD *v49; // [rsp-C8h] [rbp-C8h] BYREF
  __int64 v50; // [rsp-C0h] [rbp-C0h]
  _OWORD v51[11]; // [rsp-B8h] [rbp-B8h] BYREF

  if ( (__int64)a2[13] < 0 )
    return 0;
  LODWORD(v7) = 1;
  v10 = (const __m128i *)a2[5];
  v44 = a2[13];
  v42 = 0;
  v11 = v10[2].m128i_i64[1];
  v12 = v10[3].m128i_i32[0];
  v43 = 0;
  v45 = 1;
  v40 = v11;
  v41 = v12;
  v46 = 0;
  v13 = _mm_loadu_si128(v10);
  v14 = v51;
  v50 = 0x800000001LL;
  v49 = v51;
  v51[0] = v13;
  while ( 1 )
  {
    v15 = (unsigned int)v7;
    v7 = (unsigned int)(v7 - 1);
    v16 = (__int64)&v14[v15 - 1];
    v17 = *(_QWORD *)v16;
    v18 = *(_DWORD *)(v16 + 8);
    LODWORD(v50) = v7;
    v19 = *(_QWORD *)(v17 + 56);
    if ( !v19 )
      goto LABEL_21;
    v20 = 1;
    do
    {
      while ( *(_DWORD *)(v19 + 8) != v18 )
      {
        v19 = *(_QWORD *)(v19 + 32);
        if ( !v19 )
          goto LABEL_12;
      }
      if ( !(_DWORD)v20 )
        goto LABEL_21;
      v21 = *(_QWORD *)(v19 + 32);
      if ( !v21 )
        goto LABEL_13;
      if ( v18 == *(_DWORD *)(v21 + 8) )
        goto LABEL_21;
      v19 = *(_QWORD *)(v21 + 32);
      v20 = 0;
    }
    while ( v19 );
LABEL_12:
    if ( (_DWORD)v20 == 1 )
      goto LABEL_21;
LABEL_13:
    v22 = *(_DWORD *)(v17 + 24);
    if ( v22 == 299 )
      break;
    if ( v22 <= 299 )
    {
      if ( v22 == 2 )
      {
        v23 = *(_DWORD *)(v17 + 64);
        if ( v23 )
        {
          v24 = &v49;
          v25 = 40LL * (unsigned int)(v23 - 1);
          do
          {
            v26 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(v17 + 40) + v25));
            if ( v7 + 1 > (unsigned __int64)HIDWORD(v50) )
            {
              v34 = v25;
              v36.m128i_i64[0] = (__int64)v24;
              v35 = v26;
              sub_C8D5F0((__int64)v24, v51, v7 + 1, 0x10u, a5, a6);
              v7 = (unsigned int)v50;
              v25 = v34;
              v26 = _mm_load_si128(&v35);
              v24 = (_OWORD **)v36.m128i_i64[0];
            }
            v25 -= 40;
            v49[v7] = v26;
            v7 = (unsigned int)(v50 + 1);
            LODWORD(v50) = v50 + 1;
          }
          while ( v25 != -40 );
        }
      }
      goto LABEL_21;
    }
    if ( (unsigned int)(v22 - 366) > 1 )
      goto LABEL_21;
    if ( !sub_3267610(a1, v17, (__int64)a2) )
    {
      v32 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(v17 + 40));
      v33 = (unsigned int)v50;
      if ( (unsigned __int64)(unsigned int)v50 + 1 > HIDWORD(v50) )
      {
        v36 = v32;
        sub_C8D5F0((__int64)&v49, v51, (unsigned int)v50 + 1LL, 0x10u, a5, a6);
        v33 = (unsigned int)v50;
        v32 = _mm_load_si128(&v36);
      }
      v49[v33] = v32;
      LODWORD(v7) = v50 + 1;
      LODWORD(v50) = v50 + 1;
LABEL_21:
      if ( !(_DWORD)v7 )
        goto LABEL_26;
      goto LABEL_22;
    }
LABEL_25:
    LODWORD(v7) = v50;
    if ( !(_DWORD)v50 )
    {
LABEL_26:
      result = 0;
      goto LABEL_27;
    }
LABEL_22:
    v14 = v49;
  }
  if ( (*(_BYTE *)(*(_QWORD *)(v17 + 112) + 37LL) & 0xF) != 0 )
    goto LABEL_21;
  v27 = *(_WORD *)(v17 + 32);
  if ( (v27 & 8) != 0 || (v27 & 0x380) != 0 )
    goto LABEL_21;
  v28 = *(_WORD *)(v17 + 96);
  v29 = *(_QWORD *)(v17 + 104);
  v47 = v28;
  v48 = v29;
  if ( v28 )
  {
    if ( v28 == 1 || (unsigned __int16)(v28 - 504) <= 7u )
      BUG();
    v31 = 16LL * (v28 - 1) + 71615648;
    v30 = *(_QWORD *)&byte_444C4A0[16 * v28 - 16];
    LOBYTE(v31) = *(_BYTE *)(v31 + 8);
  }
  else
  {
    v30 = sub_3007260((__int64)&v47);
    v38 = v30;
    v39 = v31;
  }
  v36.m128i_i64[0] = v30;
  if ( (_BYTE)v31 )
    goto LABEL_21;
  sub_33644B0(&v47, v17, *a1, v20);
  if ( !(unsigned __int8)sub_3364440(&v40, *a1, 8LL * a2[12], &v47, (v36.m128i_i64[0] + 7) & 0xFFFFFFFFFFFFFFF8LL, &v37) )
    goto LABEL_25;
  v37 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(v17 + 40));
  sub_32EB790((__int64)a1, v17, v37.m128i_i64, 1, 1);
  result = (__int64)a2;
LABEL_27:
  if ( v49 != v51 )
  {
    v35.m128i_i64[0] = 0;
    v36.m128i_i64[0] = result;
    _libc_free((unsigned __int64)v49);
    return v36.m128i_i64[0];
  }
  return result;
}
