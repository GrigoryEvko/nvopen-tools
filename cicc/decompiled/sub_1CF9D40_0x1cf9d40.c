// Function: sub_1CF9D40
// Address: 0x1cf9d40
//
__int64 __fastcall sub_1CF9D40(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __m128 a4,
        __m128i a5,
        __m128i a6,
        __m128i a7,
        double a8,
        double a9,
        __m128i a10,
        __m128 a11)
{
  int *v11; // rbx
  __int64 v12; // r14
  __int64 v13; // rax
  __int64 v14; // r15
  __int64 v15; // rax
  __int64 v16; // rdx
  int v17; // r8d
  int v18; // r9d
  __m128i v19; // xmm4
  __m128i v20; // xmm5
  double v21; // xmm4_8
  double v22; // xmm5_8
  const __m128i *v23; // rdi
  const __m128i *v24; // rbx
  const __m128i *v25; // r13
  __m128i *v26; // rsi
  const __m128i *v27; // rcx
  __m128i *v28; // rdx
  const __m128i *v29; // rax
  const __m128i *v30; // rsi
  const __m128i *v31; // rsi
  const __m128i *v32; // rdi
  const __m128i *v33; // rbx
  const __m128i *v34; // r13
  __m128i *v35; // rsi
  __m128i *v36; // rdx
  const __m128i *v37; // rax
  const __m128i *v38; // rcx
  const __m128i *v39; // rsi
  _QWORD **v40; // rdi
  _QWORD **v41; // r12
  _QWORD **v43; // rbx
  int *v44; // [rsp+10h] [rbp-7A0h]
  unsigned __int8 v45; // [rsp+1Fh] [rbp-791h]
  const __m128i *v46; // [rsp+28h] [rbp-788h]
  int *v47; // [rsp+28h] [rbp-788h]
  __int64 v51; // [rsp+58h] [rbp-758h]
  const __m128i *v52; // [rsp+60h] [rbp-750h] BYREF
  __m128i *v53; // [rsp+68h] [rbp-748h]
  const __m128i *v54; // [rsp+70h] [rbp-740h]
  const __m128i *v55; // [rsp+80h] [rbp-730h] BYREF
  __m128i *v56; // [rsp+88h] [rbp-728h]
  const __m128i *v57; // [rsp+90h] [rbp-720h]
  __int64 v58; // [rsp+A0h] [rbp-710h] BYREF
  _QWORD **v59; // [rsp+A8h] [rbp-708h]
  __int64 v60; // [rsp+B0h] [rbp-700h]
  __int64 v61; // [rsp+B8h] [rbp-6F8h]
  int v62[4]; // [rsp+C0h] [rbp-6F0h] BYREF
  __int64 v63; // [rsp+D0h] [rbp-6E0h]
  __int64 v64; // [rsp+F0h] [rbp-6C0h] BYREF
  __int64 v65; // [rsp+F8h] [rbp-6B8h]
  int *v66; // [rsp+100h] [rbp-6B0h]
  __int64 v67; // [rsp+108h] [rbp-6A8h]
  __int64 v68; // [rsp+110h] [rbp-6A0h]
  __int64 v69; // [rsp+118h] [rbp-698h]
  __int64 v70; // [rsp+120h] [rbp-690h]
  __int64 v71[6]; // [rsp+128h] [rbp-688h] BYREF
  const __m128i *v72; // [rsp+158h] [rbp-658h]
  __m128i *v73; // [rsp+160h] [rbp-650h]
  const __m128i *v74; // [rsp+168h] [rbp-648h]
  const __m128i *v75; // [rsp+170h] [rbp-640h]
  __m128i *v76; // [rsp+178h] [rbp-638h]
  const __m128i *v77; // [rsp+180h] [rbp-630h]
  _BYTE *v78; // [rsp+188h] [rbp-628h]
  __int64 v79; // [rsp+190h] [rbp-620h]
  _BYTE v80[128]; // [rsp+198h] [rbp-618h] BYREF
  __int64 v81; // [rsp+218h] [rbp-598h]
  __int64 v82; // [rsp+220h] [rbp-590h]
  __int64 v83; // [rsp+228h] [rbp-588h]
  __int64 v84; // [rsp+230h] [rbp-580h]
  __int64 v85; // [rsp+238h] [rbp-578h]
  __int64 v86; // [rsp+240h] [rbp-570h]
  __int64 v87[2]; // [rsp+250h] [rbp-560h] BYREF
  __int64 v88; // [rsp+260h] [rbp-550h]
  _BYTE *v89; // [rsp+268h] [rbp-548h]
  __int64 v90; // [rsp+270h] [rbp-540h]
  _BYTE v91[1280]; // [rsp+278h] [rbp-538h] BYREF
  __int64 *v92; // [rsp+778h] [rbp-38h]

  v11 = v62;
  v12 = *(_QWORD *)(**(_QWORD **)(a1 + 56) + 56LL);
  sub_1CF7290(v62, a1);
  v13 = *(_QWORD *)(v12 + 80);
  v58 = 0;
  v59 = 0;
  v60 = 0;
  v61 = 0;
  if ( !v13 )
    BUG();
  v14 = *(_QWORD *)(v13 + 24);
  v51 = v13 + 16;
  if ( v14 == v13 + 16 )
  {
    v45 = 0;
    v40 = 0;
    goto LABEL_45;
  }
  v45 = 0;
  do
  {
    if ( !v14 )
      BUG();
    if ( *(_BYTE *)(v14 - 8) == 53 )
    {
      v87[0] = v14 - 24;
      v87[1] = a1;
      v88 = 0;
      v89 = v91;
      v90 = 0x2000000000LL;
      v92 = &v58;
      if ( !sub_1CF8EB0((__int64)v87, (__int64)v11) )
      {
        sub_1B33900(v14 - 24, 1);
        sub_1CF8600(v87);
        sub_1CF7D60(
          (__int64)v87,
          a4,
          *(double *)a5.m128i_i64,
          *(double *)a6.m128i_i64,
          *(double *)a7.m128i_i64,
          v21,
          v22,
          *(double *)a10.m128i_i64,
          a11);
        v45 = 1;
LABEL_11:
        if ( v89 != v91 )
          _libc_free((unsigned __int64)v89);
        if ( v88 )
          j_j___libc_free_0(v88, 32);
        goto LABEL_15;
      }
      v15 = sub_1632FA0(*(_QWORD *)(v12 + 40));
      v66 = v11;
      v67 = v15;
      v64 = a1;
      v68 = a3;
      v78 = v80;
      v65 = a2;
      v69 = v14 - 24;
      v70 = 0;
      v71[0] = 0;
      v71[1] = -1;
      memset(&v71[2], 0, 32);
      v72 = 0;
      v73 = 0;
      v74 = 0;
      v75 = 0;
      v76 = 0;
      v77 = 0;
      v79 = 0x1000000000LL;
      v81 = 0;
      v82 = 0;
      v83 = 0;
      v84 = 0;
      v85 = 0;
      v86 = 0;
      v46 = (const __m128i *)sub_1CF99F0((__int64)&v64, (__int64)v11, v16, a2, v17, v18);
      if ( !v46 && v70 )
      {
        v52 = 0;
        v53 = 0;
        v54 = 0;
        v23 = v73;
        if ( v72 != v73 )
        {
          v44 = v11;
          v24 = v72;
          v25 = v73;
          while ( 1 )
          {
            while ( !(unsigned __int8)sub_1CF4D90(&v24->m128i_i64[1], v71, v65) )
            {
LABEL_23:
              v24 += 4;
              if ( v25 == v24 )
                goto LABEL_29;
            }
            v26 = v53;
            if ( v53 == v54 )
            {
              sub_1CF5F80(&v52, v53, v24);
              goto LABEL_23;
            }
            if ( v53 )
            {
              a4 = (__m128)_mm_loadu_si128(v24);
              *v53 = (__m128i)a4;
              a5 = _mm_loadu_si128(v24 + 1);
              v26[1] = a5;
              a6 = _mm_loadu_si128(v24 + 2);
              v26[2] = a6;
              a7 = _mm_loadu_si128(v24 + 3);
              v26[3] = a7;
              v26 = v53;
            }
            v24 += 4;
            v53 = v26 + 4;
            if ( v25 == v24 )
            {
LABEL_29:
              v11 = v44;
              v27 = v52;
              v28 = v53;
              v23 = v72;
              v29 = v54;
              v30 = v73;
              goto LABEL_30;
            }
          }
        }
        v30 = v73;
        v28 = 0;
        v29 = 0;
        v27 = 0;
LABEL_30:
        v53 = (__m128i *)v30;
        v31 = v74;
        v72 = v27;
        v74 = v29;
        v52 = v23;
        v54 = v31;
        v32 = v76;
        v73 = v28;
        v55 = 0;
        v56 = 0;
        v57 = 0;
        if ( v75 != v76 )
        {
          v47 = v11;
          v33 = v75;
          v34 = v76;
          while ( 1 )
          {
            while ( (unsigned __int8)sub_1CF4D90(&v33->m128i_i64[1], v71, v65) != 3 )
            {
LABEL_32:
              v33 += 4;
              if ( v34 == v33 )
                goto LABEL_38;
            }
            v35 = v56;
            if ( v56 == v57 )
            {
              sub_1CF5F80(&v55, v56, v33);
              goto LABEL_32;
            }
            if ( v56 )
            {
              v19 = _mm_loadu_si128(v33);
              *v56 = v19;
              v20 = _mm_loadu_si128(v33 + 1);
              v35[1] = v20;
              a10 = _mm_loadu_si128(v33 + 2);
              v35[2] = a10;
              a11 = (__m128)_mm_loadu_si128(v33 + 3);
              v35[3] = (__m128i)a11;
              v35 = v56;
            }
            v33 += 4;
            v56 = v35 + 4;
            if ( v34 == v33 )
            {
LABEL_38:
              v11 = v47;
              v36 = v56;
              v46 = v55;
              v32 = v75;
              v37 = v57;
              v38 = v76;
              goto LABEL_39;
            }
          }
        }
        v38 = v76;
        v36 = 0;
        v37 = 0;
LABEL_39:
        v39 = v77;
        v56 = (__m128i *)v38;
        v55 = v32;
        v57 = v77;
        v75 = v46;
        v76 = v36;
        v77 = v37;
        if ( v32 )
          j_j___libc_free_0(v32, (char *)v39 - (char *)v32);
        if ( v52 )
          j_j___libc_free_0(v52, (char *)v54 - (char *)v52);
        sub_1CF6830(
          &v64,
          a4,
          a5,
          a6,
          a7,
          *(double *)v19.m128i_i64,
          *(double *)v20.m128i_i64,
          *(double *)a10.m128i_i64,
          a11);
      }
      j___libc_free_0(v82);
      if ( v78 != v80 )
        _libc_free((unsigned __int64)v78);
      if ( v75 )
        j_j___libc_free_0(v75, (char *)v77 - (char *)v75);
      if ( v72 )
        j_j___libc_free_0(v72, (char *)v74 - (char *)v72);
      goto LABEL_11;
    }
LABEL_15:
    v14 = *(_QWORD *)(v14 + 8);
  }
  while ( v51 != v14 );
  v40 = v59;
  v41 = &v59[(unsigned int)v61];
  if ( (_DWORD)v60 && v59 != v41 )
  {
    v43 = v59;
    while ( *v43 == (_QWORD *)-8LL || *v43 == (_QWORD *)-16LL )
    {
      if ( ++v43 == v41 )
        goto LABEL_45;
    }
    if ( v43 != v41 )
    {
LABEL_52:
      sub_15F20C0(*v43);
      while ( ++v43 != v41 )
      {
        if ( *v43 != (_QWORD *)-16LL && *v43 != (_QWORD *)-8LL )
        {
          if ( v43 != v41 )
            goto LABEL_52;
          break;
        }
      }
      v40 = v59;
    }
  }
LABEL_45:
  j___libc_free_0(v40);
  j___libc_free_0(v63);
  return v45;
}
