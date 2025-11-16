// Function: sub_1F76BF0
// Address: 0x1f76bf0
//
__int64 *__fastcall sub_1F76BF0(__int64 a1, _QWORD *a2, double a3, __m128i a4, __m128i a5)
{
  __int64 *v6; // rax
  char *v7; // rdx
  __int64 v8; // rdi
  __int64 v9; // r15
  __int64 v10; // r12
  __int64 v11; // r13
  unsigned __int8 v12; // r10
  __int64 v13; // r14
  const void **v14; // rdx
  __int64 v15; // rsi
  __int64 v16; // rax
  __int64 v17; // r10
  __int64 v18; // r8
  __int64 v19; // rsi
  __int64 *v20; // r15
  __int64 v21; // rax
  __int64 *v22; // r15
  __int64 v24; // rdx
  __int64 v25; // rax
  __int64 v26; // rax
  int v27; // eax
  __int64 v28; // rax
  __int64 v29; // rax
  char v30; // al
  __int64 v31; // rsi
  __int64 *v32; // r15
  bool v33; // al
  __int64 v34; // rsi
  __int64 v35; // rdi
  __int64 v36; // rax
  _QWORD *v37; // rax
  __int64 v38; // rdi
  __int64 v39; // rdx
  __int64 v40; // rdx
  __int64 v41; // r12
  __int64 v42; // rax
  __int64 v43; // rsi
  __int64 v44; // rdi
  _QWORD *v45; // rax
  __int64 v46; // rdi
  __int64 v47; // rdx
  __int64 v48; // rdx
  __int64 v49; // rax
  bool v50; // al
  __int128 v51; // [rsp-10h] [rbp-D0h]
  __int128 v52; // [rsp-10h] [rbp-D0h]
  __int128 v53; // [rsp-10h] [rbp-D0h]
  unsigned __int8 v54; // [rsp+Fh] [rbp-B1h]
  unsigned __int8 v56; // [rsp+10h] [rbp-B0h]
  __int64 v57; // [rsp+10h] [rbp-B0h]
  __int64 v58; // [rsp+10h] [rbp-B0h]
  unsigned __int8 v59; // [rsp+18h] [rbp-A8h]
  _QWORD *v60; // [rsp+18h] [rbp-A8h]
  __int64 v61; // [rsp+18h] [rbp-A8h]
  __int64 v62; // [rsp+18h] [rbp-A8h]
  __int64 v63; // [rsp+18h] [rbp-A8h]
  __int64 v64; // [rsp+18h] [rbp-A8h]
  unsigned int v65; // [rsp+20h] [rbp-A0h] BYREF
  const void **v66; // [rsp+28h] [rbp-98h]
  __int64 v67; // [rsp+30h] [rbp-90h] BYREF
  int v68; // [rsp+38h] [rbp-88h]
  __m128i v69; // [rsp+40h] [rbp-80h] BYREF
  __m128i v70; // [rsp+50h] [rbp-70h]
  _QWORD *v71; // [rsp+60h] [rbp-60h]
  __int64 v72; // [rsp+68h] [rbp-58h]
  _QWORD *v73; // [rsp+70h] [rbp-50h]
  __int64 v74; // [rsp+78h] [rbp-48h]
  __m128i v75; // [rsp+80h] [rbp-40h]

  v6 = (__int64 *)a2[4];
  v7 = (char *)a2[5];
  v8 = *(_QWORD *)a1;
  v9 = *v6;
  v10 = *v6;
  v11 = v6[1];
  v12 = *v7;
  v13 = 16LL * *((unsigned int *)v6 + 2);
  v14 = (const void **)*((_QWORD *)v7 + 1);
  v15 = *v6;
  LOBYTE(v6) = *(_BYTE *)(*(_QWORD *)(*v6 + 40) + v13);
  LOBYTE(v65) = v12;
  v66 = v14;
  v59 = v12;
  v54 = (unsigned __int8)v6;
  v16 = sub_1D23600(v8, v15);
  v17 = v59;
  v18 = (__int64)a2;
  if ( v16 )
  {
    if ( !*(_BYTE *)(a1 + 24)
      || ((v24 = *(_QWORD *)(a1 + 8), v25 = 1, v59 == 1) || v59 && (v25 = v59, *(_QWORD *)(v24 + 8LL * v59 + 120)))
      && (*(_BYTE *)(v24 + 259 * v25 + 2433) & 0xFB) == 0 )
    {
      v19 = a2[9];
      v20 = *(__int64 **)a1;
      v69.m128i_i64[0] = v19;
      if ( v19 )
      {
        sub_1623A60((__int64)&v69, v19, 2);
        v18 = (__int64)a2;
      }
      *((_QWORD *)&v51 + 1) = v11;
      *(_QWORD *)&v51 = v10;
      v69.m128i_i32[2] = *(_DWORD *)(v18 + 64);
      v21 = sub_1D309E0(v20, 146, (__int64)&v69, v65, v66, 0, a3, *(double *)a4.m128i_i64, *(double *)a5.m128i_i64, v51);
LABEL_6:
      v22 = (__int64 *)v21;
      if ( v69.m128i_i64[0] )
        sub_161E7C0((__int64)&v69, v69.m128i_i64[0]);
      return v22;
    }
  }
  else
  {
    v24 = *(_QWORD *)(a1 + 8);
  }
  if ( v54 == 1 )
  {
    v29 = 1;
    if ( (*(_BYTE *)(v24 + 2827) & 0xFB) == 0 )
      goto LABEL_17;
  }
  else
  {
    if ( !v54 )
      goto LABEL_17;
    v29 = v54;
    if ( !*(_QWORD *)(v24 + 8LL * v54 + 120)
      || (*(_BYTE *)(v24 + 259LL * v54 + 2568) & 0xFB) == 0
      || !*(_QWORD *)(v24 + 8 * (v54 + 14LL) + 8) )
    {
      goto LABEL_17;
    }
  }
  if ( (*(_BYTE *)(v24 + 259 * v29 + 2569) & 0xFB) == 0 )
  {
    v60 = a2;
    v56 = v17;
    v30 = sub_1D1F9F0(*(_QWORD *)a1, v10, v11, 0);
    v18 = (__int64)v60;
    if ( v30 )
    {
      v31 = v60[9];
      v32 = *(__int64 **)a1;
      v69.m128i_i64[0] = v31;
      if ( v31 )
      {
        sub_1623A60((__int64)&v69, v31, 2);
        v18 = (__int64)v60;
      }
      *((_QWORD *)&v52 + 1) = v11;
      *(_QWORD *)&v52 = v10;
      v69.m128i_i32[2] = *(_DWORD *)(v18 + 64);
      v21 = sub_1D309E0(v32, 147, (__int64)&v69, v65, v66, 0, a3, *(double *)a4.m128i_i64, *(double *)a5.m128i_i64, v52);
      goto LABEL_6;
    }
    v24 = *(_QWORD *)(a1 + 8);
    v17 = v56;
  }
LABEL_17:
  v26 = 1;
  if ( (_BYTE)v17 != 1
    && (!(_BYTE)v17 || (v26 = (unsigned __int8)v17, !*(_QWORD *)(v24 + 8LL * (unsigned __int8)v17 + 120)))
    || (*(_BYTE *)(v24 + 259 * v26 + 2558) & 0xFB) != 0 )
  {
    if ( *(_BYTE *)(a1 + 24) )
      goto LABEL_23;
    v27 = *(unsigned __int16 *)(v9 + 24);
    if ( v27 == 137 )
      goto LABEL_34;
LABEL_22:
    if ( v27 != 143 )
      goto LABEL_23;
    v41 = *(_QWORD *)(v9 + 32);
    if ( *(_WORD *)(*(_QWORD *)v41 + 24LL) != 137 )
      goto LABEL_23;
    if ( (_BYTE)v17 )
    {
      if ( (unsigned __int8)(v17 - 14) <= 0x5Fu )
        goto LABEL_23;
      if ( *(_BYTE *)(a1 + 24) )
      {
        v42 = 1;
        if ( (_BYTE)v17 != 1 )
        {
          v42 = (unsigned __int8)v17;
          if ( !*(_QWORD *)(v24 + 8 * v17 + 120) )
            goto LABEL_23;
        }
        if ( (*(_BYTE *)(v24 + 259 * v42 + 2433) & 0xFB) != 0 )
          goto LABEL_23;
      }
    }
    else
    {
      v58 = v18;
      v64 = v24;
      v50 = sub_1F58D20((__int64)&v65);
      v24 = v64;
      v18 = v58;
      if ( v50 || *(_BYTE *)(a1 + 24) )
        goto LABEL_23;
    }
    v43 = *(_QWORD *)(v18 + 72);
    v67 = v43;
    if ( v43 )
    {
      v63 = v18;
      sub_1623A60((__int64)&v67, v43, 2);
      v41 = *(_QWORD *)(v9 + 32);
      v18 = v63;
    }
    v44 = *(_QWORD *)a1;
    v68 = *(_DWORD *)(v18 + 64);
    v69 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(*(_QWORD *)v41 + 32LL));
    v70 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(*(_QWORD *)v41 + 32LL) + 40LL));
    v45 = sub_1D364E0(v44, (__int64)&v67, v65, v66, 0, 1.0, *(double *)a4.m128i_i64, a5);
    v46 = *(_QWORD *)a1;
    v72 = v47;
    v71 = v45;
    v73 = sub_1D364E0(v46, (__int64)&v67, v65, v66, 0, 0.0, *(double *)a4.m128i_i64, a5);
    v74 = v48;
    v75 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(**(_QWORD **)(v9 + 32) + 32LL) + 80LL));
    goto LABEL_54;
  }
  v27 = *(unsigned __int16 *)(v9 + 24);
  if ( v27 != 137 )
    goto LABEL_22;
LABEL_34:
  if ( *(_BYTE *)(*(_QWORD *)(v9 + 40) + v13) == 2 )
  {
    if ( (_BYTE)v17 )
    {
      if ( (unsigned __int8)(v17 - 14) > 0x5Fu )
      {
        if ( !*(_BYTE *)(a1 + 24) )
          goto LABEL_38;
        v49 = 1;
        if ( (_BYTE)v17 == 1 || (v49 = (unsigned __int8)v17, *(_QWORD *)(v24 + 8 * v17 + 120)) )
        {
          if ( (*(_BYTE *)(v24 + 259 * v49 + 2433) & 0xFB) == 0 )
            goto LABEL_38;
        }
      }
    }
    else
    {
      v57 = v18;
      v61 = v24;
      v33 = sub_1F58D20((__int64)&v65);
      v24 = v61;
      v18 = v57;
      if ( !v33 && !*(_BYTE *)(a1 + 24) )
      {
LABEL_38:
        v34 = *(_QWORD *)(v18 + 72);
        v67 = v34;
        if ( v34 )
        {
          v62 = v18;
          sub_1623A60((__int64)&v67, v34, 2);
          v18 = v62;
        }
        v35 = *(_QWORD *)a1;
        v68 = *(_DWORD *)(v18 + 64);
        v36 = *(_QWORD *)(v9 + 32);
        a4 = _mm_loadu_si128((const __m128i *)v36);
        v69 = a4;
        a5 = _mm_loadu_si128((const __m128i *)(v36 + 40));
        v70 = a5;
        v37 = sub_1D364E0(v35, (__int64)&v67, v65, v66, 0, -1.0, *(double *)a4.m128i_i64, a5);
        v38 = *(_QWORD *)a1;
        v72 = v39;
        v71 = v37;
        v73 = sub_1D364E0(v38, (__int64)&v67, v65, v66, 0, 0.0, *(double *)a4.m128i_i64, a5);
        v74 = v40;
        v75 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(v9 + 32) + 80LL));
LABEL_54:
        *((_QWORD *)&v53 + 1) = 5;
        *(_QWORD *)&v53 = &v69;
        v22 = sub_1D359D0(*(__int64 **)a1, 136, (__int64)&v67, v65, v66, 0, 0.0, *(double *)a4.m128i_i64, a5, v53);
        if ( v67 )
          sub_161E7C0((__int64)&v67, v67);
        return v22;
      }
    }
  }
LABEL_23:
  v22 = 0;
  v28 = sub_1F73BC0(v18, *(__int64 **)a1, v24, a3, *(double *)a4.m128i_i64, *(double *)a5.m128i_i64);
  if ( v28 )
    return (__int64 *)v28;
  return v22;
}
