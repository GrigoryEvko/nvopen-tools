// Function: sub_C13B90
// Address: 0xc13b90
//
__int64 __fastcall sub_C13B90(__int64 a1, _QWORD *a2)
{
  __m128i *v3; // r15
  __int64 *v4; // r13
  __m128i *v5; // rdi
  __int64 (__fastcall *v6)(__int64); // rax
  __int64 v7; // rax
  __int64 v8; // rdi
  unsigned __int64 v9; // rax
  char *v10; // rsi
  _QWORD *v11; // r15
  _BYTE *v12; // r13
  __m128i *v13; // rdi
  __int64 (__fastcall *v14)(__int64); // rax
  __int64 v15; // rdi
  unsigned __int64 v17; // [rsp+0h] [rbp-1C0h]
  unsigned __int64 v18; // [rsp+8h] [rbp-1B8h]
  unsigned __int64 v19; // [rsp+10h] [rbp-1B0h]
  unsigned __int64 v20; // [rsp+18h] [rbp-1A8h]
  unsigned __int64 v21; // [rsp+20h] [rbp-1A0h]
  unsigned __int64 v22; // [rsp+28h] [rbp-198h]
  unsigned __int64 v24; // [rsp+38h] [rbp-188h]
  unsigned __int64 v25; // [rsp+40h] [rbp-180h]
  __m128i v26; // [rsp+50h] [rbp-170h] BYREF
  __m128i v27; // [rsp+60h] [rbp-160h]
  __m128i v28; // [rsp+70h] [rbp-150h]
  __m128i v29; // [rsp+80h] [rbp-140h]
  _BYTE v30[16]; // [rsp+90h] [rbp-130h] BYREF
  __int64 (__fastcall *v31)(__int64); // [rsp+A0h] [rbp-120h]
  __int64 v32; // [rsp+A8h] [rbp-118h]
  __int64 (__fastcall *v33)(__int64); // [rsp+B0h] [rbp-110h]
  __int64 v34; // [rsp+B8h] [rbp-108h]
  __int64 (__fastcall *v35)(_QWORD *); // [rsp+C0h] [rbp-100h]
  __int64 v36; // [rsp+C8h] [rbp-F8h]
  _QWORD v37[8]; // [rsp+D0h] [rbp-F0h] BYREF
  __m128i v38; // [rsp+110h] [rbp-B0h] BYREF
  __m128i v39; // [rsp+120h] [rbp-A0h] BYREF
  __m128i v40; // [rsp+130h] [rbp-90h] BYREF
  __m128i v41; // [rsp+140h] [rbp-80h] BYREF
  unsigned __int64 v42; // [rsp+150h] [rbp-70h]
  unsigned __int64 v43; // [rsp+158h] [rbp-68h]
  unsigned __int64 v44; // [rsp+160h] [rbp-60h]
  unsigned __int64 v45; // [rsp+168h] [rbp-58h]
  unsigned __int64 v46; // [rsp+170h] [rbp-50h]
  unsigned __int64 v47; // [rsp+178h] [rbp-48h]
  unsigned __int64 v48; // [rsp+180h] [rbp-40h]
  unsigned __int64 v49; // [rsp+188h] [rbp-38h]

  if ( !*(_QWORD *)a1 )
    *(_QWORD *)a1 = a2;
  sub_BA9680(&v38, a2);
  v18 = v42;
  v26 = _mm_loadu_si128(&v38);
  v20 = v43;
  v27 = _mm_loadu_si128(&v39);
  v22 = v44;
  v28 = _mm_loadu_si128(&v40);
  v25 = v45;
  v29 = _mm_loadu_si128(&v41);
  v17 = v46;
  v19 = v47;
  v21 = v48;
  v24 = v49;
  while ( *(_OWORD *)&v27 != __PAIR128__(v25, v22)
       || *(_OWORD *)&v26 != __PAIR128__(v20, v18)
       || *(_OWORD *)&v29 != __PAIR128__(v24, v21)
       || __PAIR128__(v19, v17) != *(_OWORD *)&v28 )
  {
    v3 = (__m128i *)v37;
    v37[3] = 0;
    v4 = v37;
    v5 = &v26;
    v37[2] = sub_C11C50;
    v37[5] = 0;
    v37[4] = sub_C11C70;
    v37[7] = 0;
    v37[6] = sub_C11C90;
    v6 = sub_C11C30;
    if ( ((unsigned __int8)sub_C11C30 & 1) == 0 )
      goto LABEL_7;
    while ( 1 )
    {
      v6 = *(__int64 (__fastcall **)(__int64))((char *)v6 + v5->m128i_i64[0] - 1);
LABEL_7:
      v7 = v6((__int64)v5);
      if ( v7 )
        break;
      while ( 1 )
      {
        if ( &v38 == ++v3 )
LABEL_26:
          BUG();
        v8 = v4[3];
        v6 = (__int64 (__fastcall *)(__int64))v4[2];
        v4 = (__int64 *)v3;
        v5 = (__m128i *)((char *)&v26 + v8);
        if ( ((unsigned __int8)v6 & 1) != 0 )
          break;
        v7 = v6((__int64)v5);
        if ( v7 )
          goto LABEL_11;
      }
    }
LABEL_11:
    v9 = v7 & 0xFFFFFFFFFFFFFFFBLL;
    v10 = *(char **)(a1 + 112);
    v37[0] = v9;
    if ( v10 == *(char **)(a1 + 120) )
    {
      sub_C13810((char **)(a1 + 104), v10, v37);
    }
    else
    {
      if ( v10 )
      {
        *(_QWORD *)v10 = v9;
        v10 = *(char **)(a1 + 112);
      }
      *(_QWORD *)(a1 + 112) = v10 + 8;
    }
    v11 = v30;
    v32 = 0;
    v34 = 0;
    v12 = v30;
    v13 = &v26;
    v31 = sub_C11BA0;
    v36 = 0;
    v33 = sub_C11BD0;
    v35 = sub_C11C00;
    v14 = sub_C11B70;
    if ( ((unsigned __int8)sub_C11B70 & 1) != 0 )
LABEL_16:
      v14 = *(__int64 (__fastcall **)(__int64))((char *)v14 + v13->m128i_i64[0] - 1);
    while ( !(unsigned __int8)v14((__int64)v13) )
    {
      v11 += 2;
      if ( v37 == v11 )
        goto LABEL_26;
      v15 = *((_QWORD *)v12 + 3);
      v14 = (__int64 (__fastcall *)(__int64))*((_QWORD *)v12 + 2);
      v12 = v11;
      v13 = (__m128i *)((char *)&v26 + v15);
      if ( ((unsigned __int8)v14 & 1) != 0 )
        goto LABEL_16;
    }
  }
  v38.m128i_i64[0] = a1;
  return sub_C136E0(
           (__int64)a2,
           (__int64 (__fastcall *)(__int64, const char *, __int64, __int64))sub_C13990,
           (__int64)&v38);
}
