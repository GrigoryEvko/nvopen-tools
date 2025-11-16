// Function: sub_2ED41E0
// Address: 0x2ed41e0
//
__int64 __fastcall sub_2ED41E0(__int64 a1, __int64 a2, int a3)
{
  __int64 v6; // r12
  __int64 v7; // rsi
  __int64 v8; // rdi
  __int64 (*v9)(); // rax
  __int64 (__fastcall *v10)(__int64); // rax
  _DWORD *v12; // r15
  _DWORD *v13; // r14
  int v14; // edx
  bool v15; // zf
  _BYTE *v16; // rsi
  _BYTE *v17; // rdx
  __m128i *v18; // r12
  unsigned __int64 v19; // rsi
  __m128i *v20; // rdi
  __int64 v21; // rdx
  _BYTE *v22; // rsi
  _BYTE *v23; // rdx
  __int16 v24; // ax
  _QWORD v25[2]; // [rsp+30h] [rbp-1D0h] BYREF
  char v26; // [rsp+40h] [rbp-1C0h]
  __m128i v27; // [rsp+50h] [rbp-1B0h]
  _BYTE v28[16]; // [rsp+60h] [rbp-1A0h] BYREF
  void (__fastcall *v29)(_BYTE *, _BYTE *, __int64); // [rsp+70h] [rbp-190h]
  unsigned __int8 (__fastcall *v30)(_BYTE *); // [rsp+78h] [rbp-188h]
  _DWORD *v31; // [rsp+80h] [rbp-180h]
  __int64 v32; // [rsp+88h] [rbp-178h]
  _BYTE v33[16]; // [rsp+90h] [rbp-170h] BYREF
  void (__fastcall *v34)(_BYTE *, _BYTE *, __int64); // [rsp+A0h] [rbp-160h]
  __int64 v35; // [rsp+A8h] [rbp-158h]
  __m128i v36; // [rsp+B0h] [rbp-150h]
  _BYTE v37[16]; // [rsp+C0h] [rbp-140h] BYREF
  void (__fastcall *v38)(_BYTE *, _BYTE *, __int64); // [rsp+D0h] [rbp-130h]
  unsigned __int8 (__fastcall *v39)(_BYTE *, __m128i *); // [rsp+D8h] [rbp-128h]
  _QWORD v40[2]; // [rsp+E0h] [rbp-120h] BYREF
  __int64 (__fastcall *v41)(_QWORD *, _DWORD *, int); // [rsp+F0h] [rbp-110h] BYREF
  bool (__fastcall *v42)(_DWORD *, __int64); // [rsp+F8h] [rbp-108h]
  void (__fastcall *v43)(_QWORD, _QWORD, _QWORD); // [rsp+100h] [rbp-100h]
  __int64 v44; // [rsp+108h] [rbp-F8h]
  __m128i v45; // [rsp+110h] [rbp-F0h] BYREF
  _QWORD v46[2]; // [rsp+120h] [rbp-E0h] BYREF
  void (__fastcall *v47)(_BYTE *, _BYTE *, __int64); // [rsp+130h] [rbp-D0h]
  unsigned __int8 (__fastcall *v48)(_BYTE *); // [rsp+138h] [rbp-C8h]
  _DWORD *v49; // [rsp+140h] [rbp-C0h]
  __int64 v50; // [rsp+148h] [rbp-B8h]
  _BYTE v51[16]; // [rsp+150h] [rbp-B0h] BYREF
  void (__fastcall *v52)(_BYTE *, _BYTE *, __int64); // [rsp+160h] [rbp-A0h]
  __int64 v53; // [rsp+168h] [rbp-98h]
  __m128i v54; // [rsp+170h] [rbp-90h] BYREF
  _QWORD v55[2]; // [rsp+180h] [rbp-80h] BYREF
  void (__fastcall *v56)(_BYTE *, _BYTE *, __int64); // [rsp+190h] [rbp-70h]
  unsigned __int8 (__fastcall *v57)(_BYTE *, __m128i *); // [rsp+198h] [rbp-68h]
  __int64 v58; // [rsp+1A0h] [rbp-60h]
  __int64 v59; // [rsp+1A8h] [rbp-58h]
  _BYTE v60[16]; // [rsp+1B0h] [rbp-50h] BYREF
  void (__fastcall *v61)(__int64 (__fastcall **)(_QWORD *, _DWORD *, int), _BYTE *, __int64); // [rsp+1C0h] [rbp-40h]
  __int64 v62; // [rsp+1C8h] [rbp-38h]

  v6 = *(_QWORD *)(sub_2E88D60(a1) + 32);
  v7 = 0;
  v8 = *(_QWORD *)(sub_2E88D60(a1) + 16);
  v9 = *(__int64 (**)())(*(_QWORD *)v8 + 128LL);
  if ( v9 != sub_2DAC790 )
    v7 = ((__int64 (__fastcall *)(__int64, _QWORD))v9)(v8, 0);
  if ( *(_WORD *)(a1 + 68) == 20 )
  {
    v12 = *(_DWORD **)(a1 + 32);
    v13 = v12 + 10;
  }
  else
  {
    v10 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v7 + 520LL);
    if ( v10 == sub_2DCA430 )
      return 0;
    ((void (__fastcall *)(_QWORD *, __int64, __int64))v10)(v25, v7, a1);
    v12 = (_DWORD *)v25[0];
    v13 = (_DWORD *)v25[1];
    if ( !v26 )
      return 0;
  }
  if ( a3 < 0 != (int)v13[2] < 0 )
    return 0;
  v14 = *(_DWORD *)(v6 + 64);
  if ( (v14 == 0) != a3 >= 0 )
    return 0;
  if ( v14 )
  {
    LODWORD(v40[0]) = a3;
    v42 = sub_2E85490;
    v55[0] = 0;
    v41 = sub_2E854D0;
    sub_2E854D0(&v54, v40, 2);
    v15 = *(_WORD *)(a2 + 68) == 14;
    v22 = *(_BYTE **)(a2 + 32);
    v55[1] = v42;
    v55[0] = v41;
    if ( v15 )
    {
      v23 = v22 + 40;
    }
    else
    {
      v23 = &v22[40 * (*(_DWORD *)(a2 + 40) & 0xFFFFFF)];
      v22 += 80;
    }
    v20 = &v45;
    sub_2ED3660(&v45, v22, v23, (__int64)&v54);
    if ( v55[0] )
    {
      v20 = &v54;
      ((void (__fastcall *)(__m128i *, __m128i *, __int64))v55[0])(&v54, &v54, 3);
    }
    if ( v41 )
    {
      v20 = (__m128i *)v40;
      v41(v40, v40, 3);
    }
    v29 = 0;
    v27 = v45;
    if ( v47 )
    {
      v20 = (__m128i *)v28;
      v47(v28, v46, 2);
      v30 = v48;
      v29 = v47;
    }
    v34 = 0;
    v31 = v49;
    v32 = v50;
    if ( v52 )
    {
      v20 = (__m128i *)v33;
      v52(v33, v51, 2);
      v35 = v53;
      v34 = v52;
    }
    while ( 1 )
    {
      v19 = v27.m128i_i64[0];
      if ( (_DWORD *)v27.m128i_i64[0] == v31 )
        break;
      while ( 1 )
      {
        v24 = (*v13 >> 8) & 0xFFF;
        if ( v24 != ((*(_DWORD *)v19 >> 8) & 0xFFF) || (v21 = *v12 >> 8, LOWORD(v21) = v21 & 0xFFF, v24 != (_WORD)v21) )
        {
          if ( v34 )
            v34(v33, v33, 3);
          if ( v29 )
            v29(v28, v28, 3);
          if ( v52 )
            v52(v51, v51, 3);
          if ( v47 )
            v47(v46, v46, 3);
          return 0;
        }
        v19 += 40LL;
        v27.m128i_i64[0] = v19;
        if ( v19 != v27.m128i_i64[1] )
          break;
LABEL_54:
        if ( v31 == (_DWORD *)v19 )
          goto LABEL_55;
      }
      while ( 1 )
      {
        if ( !v29 )
LABEL_73:
          sub_4263D6(v20, v19, v21);
        v20 = (__m128i *)v28;
        if ( v30(v28) )
          break;
        v19 = v27.m128i_i64[0] + 40;
        v27.m128i_i64[0] = v19;
        if ( v27.m128i_i64[1] == v19 )
          goto LABEL_54;
      }
    }
LABEL_55:
    if ( v34 )
      v34(v33, v33, 3);
    if ( v29 )
      v29(v28, v28, 3);
    if ( v52 )
      v52(v51, v51, 3);
    if ( v47 )
      v47(v46, v46, 3);
    goto LABEL_11;
  }
  if ( v12[2] != a3 )
    return 0;
LABEL_11:
  LODWORD(v40[0]) = a3;
  v46[0] = 0;
  v42 = sub_2E85490;
  v41 = sub_2E854D0;
  sub_2E854D0(&v45, v40, 2);
  v15 = *(_WORD *)(a2 + 68) == 14;
  v16 = *(_BYTE **)(a2 + 32);
  v46[1] = v42;
  v46[0] = v41;
  if ( v15 )
  {
    v17 = v16 + 40;
  }
  else
  {
    v17 = &v16[40 * (*(_DWORD *)(a2 + 40) & 0xFFFFFF)];
    v16 += 80;
  }
  sub_2ED3660(&v54, v16, v17, (__int64)&v45);
  if ( v46[0] )
    ((void (__fastcall *)(__m128i *, __m128i *, __int64))v46[0])(&v45, &v45, 3);
  if ( v41 )
    v41(v40, v40, 3);
  v38 = 0;
  v36 = v54;
  if ( v56 )
  {
    v56(v37, v55, 2);
    v39 = v57;
    v38 = v56;
  }
  v43 = 0;
  v40[0] = v58;
  v40[1] = v59;
  if ( v61 )
  {
    v61(&v41, v60, 2);
    v44 = v62;
    v43 = (void (__fastcall *)(_QWORD, _QWORD, _QWORD))v61;
  }
LABEL_21:
  v18 = (__m128i *)v36.m128i_i64[0];
  if ( v36.m128i_i64[0] != v40[0] )
  {
    do
    {
      v19 = (unsigned int)v13[2];
      v20 = v18;
      sub_2EAB0C0((__int64)v18, v19);
      v21 = *v13 & 0xFFF00;
      v18->m128i_i32[0] = v21 | v18->m128i_i32[0] & 0xFFF000FF;
      while ( 1 )
      {
        v18 = (__m128i *)(v36.m128i_i64[0] + 40);
        v36.m128i_i64[0] = (__int64)v18;
        if ( (__m128i *)v36.m128i_i64[1] == v18 )
          break;
        if ( !v38 )
          goto LABEL_73;
        v19 = (unsigned __int64)v18;
        v20 = (__m128i *)v37;
        if ( v39(v37, v18) )
          goto LABEL_21;
      }
    }
    while ( (__m128i *)v40[0] != v18 );
  }
  if ( v43 )
    v43(&v41, &v41, 3);
  if ( v38 )
    v38(v37, v37, 3);
  if ( v61 )
    v61((__int64 (__fastcall **)(_QWORD *, _DWORD *, int))v60, v60, 3);
  if ( v56 )
    v56(v55, v55, 3);
  return 1;
}
