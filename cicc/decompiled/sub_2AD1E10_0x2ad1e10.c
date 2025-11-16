// Function: sub_2AD1E10
// Address: 0x2ad1e10
//
unsigned __int64 __fastcall sub_2AD1E10(__int64 a1, __int64 a2)
{
  unsigned int v3; // r13d
  char v4; // bl
  char *v5; // rsi
  __int64 v6; // rax
  __int64 *v7; // rdi
  __int64 v8; // rax
  __m128i *v9; // rdi
  __m128i v10; // xmm0
  __m128i v11; // xmm1
  __m128i v12; // xmm2
  __m128i v13; // xmm3
  void (__fastcall *v14)(_BYTE *, _BYTE *, __int64); // rcx
  unsigned __int8 *v15; // rax
  unsigned __int8 *v16; // rsi
  __int64 v17; // r12
  __int64 v18; // rdx
  __int64 v19; // rcx
  int v20; // eax
  unsigned __int64 v21; // rax
  unsigned __int8 **v23; // rax
  unsigned __int8 **v24; // rax
  unsigned __int64 v25; // rax
  int v26; // edx
  int v27; // ecx
  int v28; // eax
  bool v29; // of
  int v30; // eax
  __int64 v31; // [rsp+8h] [rbp-1A8h]
  __int64 *v32; // [rsp+10h] [rbp-1A0h]
  int v33; // [rsp+24h] [rbp-18Ch]
  unsigned __int64 v34; // [rsp+28h] [rbp-188h]
  __int64 *v35; // [rsp+30h] [rbp-180h]
  int v36; // [rsp+38h] [rbp-178h]
  char v37; // [rsp+3Fh] [rbp-171h]
  unsigned __int64 v38; // [rsp+40h] [rbp-170h]
  __int64 v39; // [rsp+48h] [rbp-168h]
  __int64 v40; // [rsp+50h] [rbp-160h] BYREF
  unsigned __int8 **v41; // [rsp+58h] [rbp-158h]
  __int64 v42; // [rsp+60h] [rbp-150h]
  int v43; // [rsp+68h] [rbp-148h]
  char v44; // [rsp+6Ch] [rbp-144h]
  char v45; // [rsp+70h] [rbp-140h] BYREF
  __m128i v46; // [rsp+80h] [rbp-130h]
  __m128i v47; // [rsp+90h] [rbp-120h]
  _BYTE v48[16]; // [rsp+A0h] [rbp-110h] BYREF
  void (__fastcall *v49)(_BYTE *, _BYTE *, __int64); // [rsp+B0h] [rbp-100h]
  unsigned __int8 (__fastcall *v50)(_BYTE *, unsigned __int8 *); // [rsp+B8h] [rbp-F8h]
  __m128i v51; // [rsp+C0h] [rbp-F0h]
  __m128i v52; // [rsp+D0h] [rbp-E0h]
  _BYTE v53[16]; // [rsp+E0h] [rbp-D0h] BYREF
  void (__fastcall *v54)(_BYTE *, _BYTE *, __int64); // [rsp+F0h] [rbp-C0h]
  __int64 v55; // [rsp+F8h] [rbp-B8h]
  __m128i v56; // [rsp+100h] [rbp-B0h] BYREF
  __m128i v57; // [rsp+110h] [rbp-A0h] BYREF
  _BYTE v58[16]; // [rsp+120h] [rbp-90h] BYREF
  void (__fastcall *v59)(_BYTE *, _BYTE *, __int64); // [rsp+130h] [rbp-80h]
  unsigned __int8 (__fastcall *v60)(_BYTE *, unsigned __int8 *); // [rsp+138h] [rbp-78h]
  __m128i v61; // [rsp+140h] [rbp-70h] BYREF
  __m128i v62; // [rsp+150h] [rbp-60h] BYREF
  _BYTE v63[16]; // [rsp+160h] [rbp-50h] BYREF
  void (__fastcall *v64)(_BYTE *, _BYTE *, __int64); // [rsp+170h] [rbp-40h]
  __int64 v65; // [rsp+178h] [rbp-38h]

  v3 = a2;
  v4 = BYTE4(a2) ^ 1;
  v37 = BYTE4(a2);
  HIDWORD(v39) = HIDWORD(a2);
  v5 = *(char **)(a1 + 416);
  v41 = (unsigned __int8 **)&v45;
  v6 = *(_QWORD *)(a1 + 424);
  v40 = 0;
  v7 = *(__int64 **)(v6 + 112);
  v44 = 1;
  v42 = 2;
  v43 = 0;
  if ( (unsigned int)sub_DCF980(v7, v5) == v3 && v4 && (!*(_BYTE *)(a1 + 108) || !*(_DWORD *)(a1 + 100)) )
    sub_2AB9470(*(_QWORD *)(a1 + 416), *(_QWORD *)(a1 + 440) + 128LL, (__int64)&v40);
  v8 = *(_QWORD *)(a1 + 416);
  v32 = *(__int64 **)(v8 + 40);
  if ( v32 == *(__int64 **)(v8 + 32) )
  {
    v34 = 0;
    goto LABEL_41;
  }
  v35 = *(__int64 **)(v8 + 32);
  v34 = 0;
  v33 = 0;
  do
  {
    v9 = &v56;
    v31 = *v35;
    sub_AA72C0(&v56, *v35, 1);
    v10 = _mm_loadu_si128(&v56);
    v49 = 0;
    v11 = _mm_loadu_si128(&v57);
    v46 = v10;
    v47 = v11;
    if ( v59 )
    {
      v9 = (__m128i *)v48;
      v59(v48, v58, 2);
      v50 = v60;
      v49 = v59;
    }
    v12 = _mm_loadu_si128(&v61);
    v13 = _mm_loadu_si128(&v62);
    v54 = 0;
    v51 = v12;
    v52 = v13;
    if ( v64 )
    {
      v9 = (__m128i *)v53;
      v64(v53, v63, 2);
      v14 = v64;
      v55 = v65;
      v15 = (unsigned __int8 *)v46.m128i_i64[0];
      v54 = v64;
      v16 = (unsigned __int8 *)v46.m128i_i64[0];
      if ( v46.m128i_i64[0] == v51.m128i_i64[0] )
      {
        v36 = 0;
        v17 = 0;
        goto LABEL_27;
      }
    }
    else
    {
      v15 = (unsigned __int8 *)v46.m128i_i64[0];
      v16 = (unsigned __int8 *)v46.m128i_i64[0];
      if ( v46.m128i_i64[0] == v51.m128i_i64[0] )
      {
        v36 = 0;
        v17 = 0;
        goto LABEL_29;
      }
    }
    v36 = 0;
    v17 = 0;
    do
    {
      if ( v16 )
        v16 -= 24;
      if ( *(_BYTE *)(a1 + 540) )
      {
        v18 = *(_QWORD *)(a1 + 520);
        v19 = v18 + 8LL * *(unsigned int *)(a1 + 532);
        if ( v18 != v19 )
        {
          while ( v16 != *(unsigned __int8 **)v18 )
          {
            v18 += 8;
            if ( v19 == v18 )
              goto LABEL_45;
          }
          goto LABEL_17;
        }
      }
      else
      {
        v9 = (__m128i *)(a1 + 512);
        if ( sub_C8CA60(a1 + 512, (__int64)v16) )
          goto LABEL_50;
      }
LABEL_45:
      if ( v44 )
      {
        v23 = v41;
        v18 = (__int64)&v41[HIDWORD(v42)];
        if ( v41 != (unsigned __int8 **)v18 )
        {
          while ( v16 != *v23 )
          {
            if ( (unsigned __int8 **)v18 == ++v23 )
              goto LABEL_52;
          }
          goto LABEL_50;
        }
      }
      else
      {
        v9 = (__m128i *)&v40;
        if ( sub_C8CA60((__int64)&v40, (__int64)v16) )
          goto LABEL_50;
      }
LABEL_52:
      if ( v37 )
      {
        if ( !v3 )
          goto LABEL_61;
      }
      else if ( v3 <= 1 )
      {
        goto LABEL_61;
      }
      if ( !*(_BYTE *)(a1 + 700) )
      {
        v9 = (__m128i *)(a1 + 672);
        if ( sub_C8CA60(a1 + 672, (__int64)v16) )
          goto LABEL_50;
        goto LABEL_61;
      }
      v24 = *(unsigned __int8 ***)(a1 + 680);
      v18 = (__int64)&v24[*(unsigned int *)(a1 + 692)];
      if ( v24 != (unsigned __int8 **)v18 )
      {
        while ( v16 != *v24 )
        {
          if ( (unsigned __int8 **)v18 == ++v24 )
            goto LABEL_61;
        }
LABEL_50:
        v15 = (unsigned __int8 *)v46.m128i_i64[0];
        goto LABEL_17;
      }
LABEL_61:
      LODWORD(v39) = v3;
      v9 = (__m128i *)a1;
      BYTE4(v39) = v37;
      v25 = sub_2AD0150(a1, v16, v39);
      v27 = v26;
      v18 = v25;
      if ( v27 )
      {
        v28 = 1;
        if ( v27 != 1 )
          v28 = v36;
        v36 = v28;
      }
      else
      {
        v38 = v25;
        v9 = (__m128i *)&qword_500DDC0[1];
        v30 = sub_23DF0D0((int *)&qword_500DDC0[1]);
        v18 = v38;
        if ( v30 > 0 )
          v18 = LODWORD(qword_500DDC0[17]);
      }
      v29 = __OFADD__(v18, v17);
      v17 += v18;
      if ( !v29 )
        goto LABEL_50;
      v17 = 0x7FFFFFFFFFFFFFFFLL;
      if ( v18 > 0 )
        goto LABEL_50;
      v15 = (unsigned __int8 *)v46.m128i_i64[0];
      v17 = 0x8000000000000000LL;
LABEL_17:
      v15 = (unsigned __int8 *)*((_QWORD *)v15 + 1);
      v46.m128i_i16[4] = 0;
      v46.m128i_i64[0] = (__int64)v15;
      v16 = v15;
      if ( v15 != (unsigned __int8 *)v47.m128i_i64[0] )
      {
        while ( 1 )
        {
          if ( v16 )
            v16 -= 24;
          if ( !v49 )
            sub_4263D6(v9, v16, v18);
          v9 = (__m128i *)v48;
          if ( v50(v48, v16) )
            break;
          v18 = 0;
          v16 = *(unsigned __int8 **)(v46.m128i_i64[0] + 8);
          v46.m128i_i16[4] = 0;
          v46.m128i_i64[0] = (__int64)v16;
          v15 = v16;
          if ( (unsigned __int8 *)v47.m128i_i64[0] == v16 )
            goto LABEL_25;
        }
        v16 = (unsigned __int8 *)v46.m128i_i64[0];
        v15 = (unsigned __int8 *)v46.m128i_i64[0];
      }
LABEL_25:
      ;
    }
    while ( v16 != (unsigned __int8 *)v51.m128i_i64[0] );
    v14 = v54;
LABEL_27:
    if ( v14 )
      v14(v53, v53, 3);
LABEL_29:
    if ( v49 )
      v49(v48, v48, 3);
    if ( v64 )
      v64(v63, v63, 3);
    if ( v59 )
      v59(v58, v58, 3);
    if ( v3 == 1 && v4 && (unsigned __int8)sub_31A6C30(*(_QWORD *)(a1 + 440), v31) && *(_DWORD *)(a1 + 992) != 2 )
      v17 /= 2;
    v20 = 1;
    if ( v36 != 1 )
      v20 = v33;
    v33 = v20;
    v21 = v17 + v34;
    if ( __OFADD__(v17, v34) )
    {
      v21 = 0x8000000000000000LL;
      if ( v17 > 0 )
        v21 = 0x7FFFFFFFFFFFFFFFLL;
    }
    ++v35;
    v34 = v21;
  }
  while ( v32 != v35 );
LABEL_41:
  if ( !v44 )
    _libc_free((unsigned __int64)v41);
  return v34;
}
