// Function: sub_281E640
// Address: 0x281e640
//
__int64 __fastcall sub_281E640(__int64 *a1, int a2, __int64 a3, char a4)
{
  __int64 *v8; // rax
  __m128i *v9; // rdi
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  __m128i v13; // xmm4
  __m128i v14; // xmm5
  __m128i v15; // xmm6
  __m128i v16; // xmm7
  void (__fastcall *v17)(_BYTE *, _BYTE *, __int64); // rax
  __int64 v18; // rsi
  __int64 v19; // r12
  __int64 v20; // rdx
  unsigned int v21; // r13d
  __int64 v22; // rax
  int v23; // edx
  __int64 v25; // [rsp+0h] [rbp-220h] BYREF
  __int64 v26; // [rsp+8h] [rbp-218h]
  __m128i v27; // [rsp+10h] [rbp-210h] BYREF
  __m128i v28; // [rsp+20h] [rbp-200h] BYREF
  _BYTE v29[16]; // [rsp+30h] [rbp-1F0h] BYREF
  void (__fastcall *v30)(_BYTE *, _BYTE *, __int64); // [rsp+40h] [rbp-1E0h]
  unsigned __int8 (__fastcall *v31)(_BYTE *, __int64); // [rsp+48h] [rbp-1D8h]
  __m128i v32; // [rsp+50h] [rbp-1D0h] BYREF
  __m128i v33; // [rsp+60h] [rbp-1C0h] BYREF
  _BYTE v34[16]; // [rsp+70h] [rbp-1B0h] BYREF
  void (__fastcall *v35)(char *, char *, __int64); // [rsp+80h] [rbp-1A0h]
  __int64 v36; // [rsp+88h] [rbp-198h]
  __m128i v37; // [rsp+90h] [rbp-190h]
  __m128i v38; // [rsp+A0h] [rbp-180h]
  _BYTE v39[16]; // [rsp+B0h] [rbp-170h] BYREF
  void (__fastcall *v40)(_BYTE *, _BYTE *, __int64); // [rsp+C0h] [rbp-160h]
  unsigned __int8 (__fastcall *v41)(_BYTE *, __int64); // [rsp+C8h] [rbp-158h]
  __m128i v42; // [rsp+D0h] [rbp-150h] BYREF
  __m128i v43; // [rsp+E0h] [rbp-140h] BYREF
  _BYTE v44[16]; // [rsp+F0h] [rbp-130h] BYREF
  void (__fastcall *v45)(_BYTE *, _BYTE *, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64); // [rsp+100h] [rbp-120h]
  unsigned __int8 (__fastcall *v46)(_BYTE *, __int64); // [rsp+108h] [rbp-118h]
  __m128i v47; // [rsp+110h] [rbp-110h] BYREF
  __m128i v48; // [rsp+120h] [rbp-100h] BYREF
  _BYTE v49[16]; // [rsp+130h] [rbp-F0h] BYREF
  void (__fastcall *v50)(char *, char *, __int64); // [rsp+140h] [rbp-E0h]
  __int64 v51; // [rsp+148h] [rbp-D8h]
  __m128i v52; // [rsp+150h] [rbp-D0h] BYREF
  __m128i v53; // [rsp+160h] [rbp-C0h]
  char v54[8]; // [rsp+170h] [rbp-B0h] BYREF
  char v55; // [rsp+178h] [rbp-A8h] BYREF
  void (__fastcall *v56)(char *, char *, __int64); // [rsp+180h] [rbp-A0h]
  __int64 v57; // [rsp+188h] [rbp-98h]
  char *v58; // [rsp+198h] [rbp-88h]
  char v59; // [rsp+1A8h] [rbp-78h] BYREF

  v25 = a3;
  v8 = (__int64 *)sub_BD5C60(a3);
  v9 = &v42;
  v26 = sub_ACD760(v8, a4);
  sub_AA72C0(&v42, **(_QWORD **)(*a1 + 32), 1);
  v35 = 0;
  v32 = _mm_loadu_si128(&v47);
  v33 = _mm_loadu_si128(&v48);
  if ( v50 )
  {
    v9 = (__m128i *)v34;
    v50(v34, v49, 2);
    v36 = v51;
    v35 = v50;
  }
  v30 = 0;
  v27 = _mm_loadu_si128(&v42);
  v28 = _mm_loadu_si128(&v43);
  if ( v45 )
  {
    v9 = (__m128i *)v29;
    v45(v29, v44, 2, v10, v11, v12, v25, v26, v27.m128i_i64[0], v27.m128i_i64[1], v28.m128i_i64[0], v28.m128i_i64[1]);
    v31 = v46;
    v30 = (void (__fastcall *)(_BYTE *, _BYTE *, __int64))v45;
  }
  v13 = _mm_loadu_si128(&v32);
  v14 = _mm_loadu_si128(&v33);
  v56 = 0;
  v52 = v13;
  v53 = v14;
  if ( v35 )
  {
    v9 = (__m128i *)v54;
    v35(v54, v34, 2);
    v57 = v36;
    v56 = v35;
  }
  v15 = _mm_loadu_si128(&v27);
  v16 = _mm_loadu_si128(&v28);
  v40 = 0;
  v17 = v30;
  v37 = v15;
  v38 = v16;
  if ( !v30 )
  {
    v18 = v37.m128i_i64[0];
    v19 = 0;
    if ( v37.m128i_i64[0] == v52.m128i_i64[0] )
      goto LABEL_22;
    goto LABEL_9;
  }
  v9 = (__m128i *)v39;
  v30(v39, v29, 2);
  v18 = v37.m128i_i64[0];
  v41 = v31;
  v17 = v30;
  v40 = v30;
  if ( v37.m128i_i64[0] != v52.m128i_i64[0] )
  {
LABEL_9:
    LODWORD(v19) = 0;
    do
    {
      v18 = *(_QWORD *)(v18 + 8);
      v37.m128i_i16[4] = 0;
      v37.m128i_i64[0] = v18;
      if ( v38.m128i_i64[0] != v18 )
      {
        while ( 1 )
        {
          v20 = v18 - 24;
          if ( v18 )
            v18 -= 24;
          if ( !v17 )
            sub_4263D6(v9, v18, v20);
          v9 = (__m128i *)v39;
          if ( v41(v39, v18) )
            break;
          v18 = *(_QWORD *)(v37.m128i_i64[0] + 8);
          v37.m128i_i16[4] = 0;
          v17 = v40;
          v37.m128i_i64[0] = v18;
          if ( v38.m128i_i64[0] == v18 )
            goto LABEL_18;
        }
        v18 = v37.m128i_i64[0];
        v17 = v40;
      }
LABEL_18:
      LODWORD(v19) = v19 + 1;
    }
    while ( v52.m128i_i64[0] != v18 );
    v19 = (unsigned int)v19;
    goto LABEL_20;
  }
  v19 = 0;
LABEL_20:
  if ( v17 )
    v17(v39, v39, 3);
LABEL_22:
  if ( v56 )
    v56(v54, v54, 3);
  if ( v30 )
    v30(v29, v29, 3);
  if ( v35 )
    v35(v34, v34, 3);
  sub_DF8D10((__int64)&v52, a2, *(_QWORD *)(a3 + 8), (char *)&v25, 2);
  v21 = 1;
  v22 = sub_DFD690(a1[6], (__int64)&v52);
  if ( v19 != 6 )
  {
    if ( v23 )
      LOBYTE(v21) = v23 <= 0;
    else
      LOBYTE(v21) = v22 <= 1;
  }
  if ( v58 != &v59 )
    _libc_free((unsigned __int64)v58);
  if ( (char *)v53.m128i_i64[1] != &v55 )
    _libc_free(v53.m128i_u64[1]);
  if ( v50 )
    v50(v49, v49, 3);
  if ( v45 )
    ((void (__fastcall *)(_BYTE *, _BYTE *, __int64))v45)(v44, v44, 3);
  return v21;
}
