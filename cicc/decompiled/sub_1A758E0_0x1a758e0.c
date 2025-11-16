// Function: sub_1A758E0
// Address: 0x1a758e0
//
__int64 __fastcall sub_1A758E0(__int64 a1, __int64 a2, __int64 a3)
{
  _BYTE *v4; // rsi
  _QWORD *v7; // rdi
  const __m128i *v8; // rcx
  const __m128i *v9; // rdx
  unsigned __int64 v10; // r15
  __m128i *v11; // rax
  __m128i *v12; // rcx
  __m128i *v13; // rax
  __m128i *v14; // rax
  __int8 *v15; // rax
  const __m128i *v16; // rcx
  unsigned __int64 v17; // r15
  __m128i *v18; // rax
  __m128i *v19; // rcx
  __m128i *v20; // rax
  __m128i *v21; // rax
  __int8 *v22; // rax
  const __m128i *v23; // rcx
  unsigned __int64 v24; // rbx
  __m128i *v25; // rax
  __m128i *v26; // rcx
  const __m128i *v27; // rcx
  unsigned __int64 v28; // rbx
  __m128i *v29; // rax
  __m128i *v30; // rcx
  const __m128i *v31; // rcx
  unsigned __int64 v32; // rbx
  __m128i *v33; // rax
  __m128i *v34; // rcx
  const __m128i *v35; // rcx
  unsigned __int64 v36; // r13
  __int64 v37; // rax
  __m128i *v38; // rdi
  __m128i *v39; // rax
  __m128i *v40; // rdx
  __m128i *v41; // rcx
  _BYTE *v42; // rsi
  __m128i *v43; // rax
  _QWORD v45[2]; // [rsp+0h] [rbp-430h] BYREF
  unsigned __int64 v46; // [rsp+10h] [rbp-420h]
  _BYTE v47[64]; // [rsp+28h] [rbp-408h] BYREF
  __m128i *v48; // [rsp+68h] [rbp-3C8h]
  __m128i *v49; // [rsp+70h] [rbp-3C0h]
  __int8 *v50; // [rsp+78h] [rbp-3B8h]
  _QWORD v51[2]; // [rsp+80h] [rbp-3B0h] BYREF
  unsigned __int64 v52; // [rsp+90h] [rbp-3A0h]
  char v53[64]; // [rsp+A8h] [rbp-388h] BYREF
  const __m128i *v54; // [rsp+E8h] [rbp-348h]
  const __m128i *v55; // [rsp+F0h] [rbp-340h]
  __int8 *v56; // [rsp+F8h] [rbp-338h]
  _QWORD v57[2]; // [rsp+100h] [rbp-330h] BYREF
  unsigned __int64 v58; // [rsp+110h] [rbp-320h]
  _BYTE v59[64]; // [rsp+128h] [rbp-308h] BYREF
  __m128i *v60; // [rsp+168h] [rbp-2C8h]
  __m128i *v61; // [rsp+170h] [rbp-2C0h]
  __int8 *v62; // [rsp+178h] [rbp-2B8h]
  _QWORD v63[2]; // [rsp+180h] [rbp-2B0h] BYREF
  unsigned __int64 v64; // [rsp+190h] [rbp-2A0h]
  char v65[64]; // [rsp+1A8h] [rbp-288h] BYREF
  const __m128i *v66; // [rsp+1E8h] [rbp-248h]
  const __m128i *v67; // [rsp+1F0h] [rbp-240h]
  __int8 *v68; // [rsp+1F8h] [rbp-238h]
  _QWORD v69[2]; // [rsp+200h] [rbp-230h] BYREF
  unsigned __int64 v70; // [rsp+210h] [rbp-220h]
  _BYTE v71[64]; // [rsp+228h] [rbp-208h] BYREF
  __m128i *v72; // [rsp+268h] [rbp-1C8h]
  __m128i *v73; // [rsp+270h] [rbp-1C0h]
  __int8 *v74; // [rsp+278h] [rbp-1B8h]
  _QWORD v75[2]; // [rsp+280h] [rbp-1B0h] BYREF
  unsigned __int64 v76; // [rsp+290h] [rbp-1A0h]
  _BYTE v77[64]; // [rsp+2A8h] [rbp-188h] BYREF
  __m128i *v78; // [rsp+2E8h] [rbp-148h]
  __m128i *v79; // [rsp+2F0h] [rbp-140h]
  __int8 *v80; // [rsp+2F8h] [rbp-138h]
  _QWORD v81[2]; // [rsp+300h] [rbp-130h] BYREF
  unsigned __int64 v82; // [rsp+310h] [rbp-120h]
  _BYTE v83[64]; // [rsp+328h] [rbp-108h] BYREF
  __m128i *v84; // [rsp+368h] [rbp-C8h]
  __m128i *v85; // [rsp+370h] [rbp-C0h]
  __int8 *v86; // [rsp+378h] [rbp-B8h]
  _QWORD v87[2]; // [rsp+380h] [rbp-B0h] BYREF
  unsigned __int64 v88; // [rsp+390h] [rbp-A0h]
  _BYTE v89[64]; // [rsp+3A8h] [rbp-88h] BYREF
  __m128i *v90; // [rsp+3E8h] [rbp-48h]
  __m128i *v91; // [rsp+3F0h] [rbp-40h]
  __int8 *v92; // [rsp+3F8h] [rbp-38h]

  v4 = v59;
  v7 = v57;
  sub_16CCCB0(v57, (__int64)v59, a2);
  v8 = *(const __m128i **)(a2 + 112);
  v9 = *(const __m128i **)(a2 + 104);
  v60 = 0;
  v61 = 0;
  v62 = 0;
  v10 = (char *)v8 - (char *)v9;
  if ( v8 == v9 )
  {
    v11 = 0;
  }
  else
  {
    if ( v10 > 0x7FFFFFFFFFFFFFE0LL )
      goto LABEL_112;
    v11 = (__m128i *)sub_22077B0((char *)v8 - (char *)v9);
    v8 = *(const __m128i **)(a2 + 112);
    v9 = *(const __m128i **)(a2 + 104);
  }
  v60 = v11;
  v61 = v11;
  v62 = &v11->m128i_i8[v10];
  if ( v8 == v9 )
  {
    v12 = v11;
  }
  else
  {
    v12 = (__m128i *)((char *)v11 + (char *)v8 - (char *)v9);
    do
    {
      if ( v11 )
      {
        *v11 = _mm_loadu_si128(v9);
        v11[1] = _mm_loadu_si128(v9 + 1);
      }
      v11 += 2;
      v9 += 2;
    }
    while ( v11 != v12 );
  }
  v61 = v12;
  sub_16CCEE0(v63, (__int64)v65, 8, (__int64)v57);
  v13 = v60;
  v7 = v45;
  v4 = v47;
  v60 = 0;
  v66 = v13;
  v14 = v61;
  v61 = 0;
  v67 = v14;
  v15 = v62;
  v62 = 0;
  v68 = v15;
  sub_16CCCB0(v45, (__int64)v47, a1);
  v16 = *(const __m128i **)(a1 + 112);
  v9 = *(const __m128i **)(a1 + 104);
  v48 = 0;
  v49 = 0;
  v50 = 0;
  v17 = (char *)v16 - (char *)v9;
  if ( v16 == v9 )
  {
    v18 = 0;
  }
  else
  {
    if ( v17 > 0x7FFFFFFFFFFFFFE0LL )
      goto LABEL_112;
    v18 = (__m128i *)sub_22077B0((char *)v16 - (char *)v9);
    v16 = *(const __m128i **)(a1 + 112);
    v9 = *(const __m128i **)(a1 + 104);
  }
  v48 = v18;
  v49 = v18;
  v50 = &v18->m128i_i8[v17];
  if ( v9 == v16 )
  {
    v19 = v18;
  }
  else
  {
    v19 = (__m128i *)((char *)v18 + (char *)v16 - (char *)v9);
    do
    {
      if ( v18 )
      {
        *v18 = _mm_loadu_si128(v9);
        v18[1] = _mm_loadu_si128(v9 + 1);
      }
      v18 += 2;
      v9 += 2;
    }
    while ( v19 != v18 );
  }
  v49 = v19;
  sub_16CCEE0(v51, (__int64)v53, 8, (__int64)v45);
  v20 = v48;
  v7 = v75;
  v4 = v77;
  v48 = 0;
  v54 = v20;
  v21 = v49;
  v49 = 0;
  v55 = v21;
  v22 = v50;
  v50 = 0;
  v56 = v22;
  sub_16CCCB0(v75, (__int64)v77, (__int64)v63);
  v23 = v67;
  v9 = v66;
  v78 = 0;
  v79 = 0;
  v80 = 0;
  v24 = (char *)v67 - (char *)v66;
  if ( v67 == v66 )
  {
    v25 = 0;
  }
  else
  {
    if ( v24 > 0x7FFFFFFFFFFFFFE0LL )
      goto LABEL_112;
    v25 = (__m128i *)sub_22077B0((char *)v67 - (char *)v66);
    v23 = v67;
    v9 = v66;
  }
  v78 = v25;
  v79 = v25;
  v80 = &v25->m128i_i8[v24];
  if ( v9 == v23 )
  {
    v26 = v25;
  }
  else
  {
    v26 = (__m128i *)((char *)v25 + (char *)v23 - (char *)v9);
    do
    {
      if ( v25 )
      {
        *v25 = _mm_loadu_si128(v9);
        v25[1] = _mm_loadu_si128(v9 + 1);
      }
      v25 += 2;
      v9 += 2;
    }
    while ( v25 != v26 );
  }
  v4 = v71;
  v79 = v26;
  v7 = v69;
  sub_16CCCB0(v69, (__int64)v71, (__int64)v51);
  v27 = v55;
  v9 = v54;
  v72 = 0;
  v73 = 0;
  v74 = 0;
  v28 = (char *)v55 - (char *)v54;
  if ( v55 == v54 )
  {
    v28 = 0;
    v29 = 0;
  }
  else
  {
    if ( v28 > 0x7FFFFFFFFFFFFFE0LL )
      goto LABEL_112;
    v29 = (__m128i *)sub_22077B0((char *)v55 - (char *)v54);
    v27 = v55;
    v9 = v54;
  }
  v72 = v29;
  v73 = v29;
  v74 = &v29->m128i_i8[v28];
  if ( v27 == v9 )
  {
    v30 = v29;
  }
  else
  {
    v30 = (__m128i *)((char *)v29 + (char *)v27 - (char *)v9);
    do
    {
      if ( v29 )
      {
        *v29 = _mm_loadu_si128(v9);
        v29[1] = _mm_loadu_si128(v9 + 1);
      }
      v29 += 2;
      v9 += 2;
    }
    while ( v30 != v29 );
  }
  v7 = v87;
  v4 = v89;
  v73 = v30;
  sub_16CCCB0(v87, (__int64)v89, (__int64)v75);
  v31 = v79;
  v9 = v78;
  v90 = 0;
  v91 = 0;
  v92 = 0;
  v32 = (char *)v79 - (char *)v78;
  if ( v79 == v78 )
  {
    v33 = 0;
  }
  else
  {
    if ( v32 > 0x7FFFFFFFFFFFFFE0LL )
      goto LABEL_112;
    v33 = (__m128i *)sub_22077B0((char *)v79 - (char *)v78);
    v31 = v79;
    v9 = v78;
  }
  v90 = v33;
  v91 = v33;
  v92 = &v33->m128i_i8[v32];
  if ( v31 == v9 )
  {
    v34 = v33;
  }
  else
  {
    v34 = (__m128i *)((char *)v33 + (char *)v31 - (char *)v9);
    do
    {
      if ( v33 )
      {
        *v33 = _mm_loadu_si128(v9);
        v33[1] = _mm_loadu_si128(v9 + 1);
      }
      v33 += 2;
      v9 += 2;
    }
    while ( v34 != v33 );
  }
  v4 = v83;
  v91 = v34;
  v7 = v81;
  sub_16CCCB0(v81, (__int64)v83, (__int64)v69);
  v9 = v73;
  v35 = v72;
  v84 = 0;
  v85 = 0;
  v86 = 0;
  v36 = (char *)v73 - (char *)v72;
  if ( v73 == v72 )
  {
    v38 = 0;
    goto LABEL_44;
  }
  if ( v36 > 0x7FFFFFFFFFFFFFE0LL )
LABEL_112:
    sub_4261EA(v7, v4, v9);
  v37 = sub_22077B0((char *)v73 - (char *)v72);
  v9 = v73;
  v35 = v72;
  v38 = (__m128i *)v37;
LABEL_44:
  v84 = v38;
  v85 = v38;
  v86 = &v38->m128i_i8[v36];
  if ( v9 == v35 )
  {
    v40 = v38;
  }
  else
  {
    v39 = v38;
    v40 = (__m128i *)((char *)v38 + (char *)v9 - (char *)v35);
    do
    {
      if ( v39 )
      {
        *v39 = _mm_loadu_si128(v35);
        v39[1] = _mm_loadu_si128(v35 + 1);
      }
      v39 += 2;
      v35 += 2;
    }
    while ( v40 != v39 );
  }
  v85 = v40;
  while ( 1 )
  {
    v41 = v90;
    if ( (char *)v40 - (char *)v38 != (char *)v91 - (char *)v90 )
      goto LABEL_51;
    if ( v38 == v40 )
      break;
    v43 = v38;
    while ( v43->m128i_i64[0] == v41->m128i_i64[0] )
    {
      if ( ((v43->m128i_i64[1] >> 1) & 3) != 0 )
      {
        if ( ((v43->m128i_i64[1] >> 1) & 3) != ((v41->m128i_i64[1] >> 1) & 3) )
          break;
        v43 += 2;
        v41 += 2;
        if ( v40 == v43 )
          goto LABEL_63;
      }
      else
      {
        if ( v43[1].m128i_i32[2] != v41[1].m128i_i32[2] )
          break;
        v43 += 2;
        v41 += 2;
        if ( v40 == v43 )
          goto LABEL_63;
      }
    }
LABEL_51:
    v42 = *(_BYTE **)(a3 + 8);
    if ( v42 == *(_BYTE **)(a3 + 16) )
    {
      sub_1A75750(a3, v42, (__m128i *)v40[-2].m128i_i64);
      v40 = v85;
    }
    else
    {
      if ( v42 )
      {
        *(_QWORD *)v42 = v40[-2].m128i_i64[0];
        v42 = *(_BYTE **)(a3 + 8);
        v40 = v85;
      }
      *(_QWORD *)(a3 + 8) = v42 + 8;
    }
    v38 = v84;
    v40 -= 2;
    v85 = v40;
    if ( v40 != v84 )
    {
      sub_1A75460((__int64)v81);
      v38 = v84;
      v40 = v85;
    }
  }
LABEL_63:
  if ( v38 )
    j_j___libc_free_0(v38, v86 - (__int8 *)v38);
  if ( v82 != v81[1] )
    _libc_free(v82);
  if ( v90 )
    j_j___libc_free_0(v90, v92 - (__int8 *)v90);
  if ( v88 != v87[1] )
    _libc_free(v88);
  if ( v72 )
    j_j___libc_free_0(v72, v74 - (__int8 *)v72);
  if ( v70 != v69[1] )
    _libc_free(v70);
  if ( v78 )
    j_j___libc_free_0(v78, v80 - (__int8 *)v78);
  if ( v76 != v75[1] )
    _libc_free(v76);
  if ( v54 )
    j_j___libc_free_0(v54, v56 - (__int8 *)v54);
  if ( v52 != v51[1] )
    _libc_free(v52);
  if ( v48 )
    j_j___libc_free_0(v48, v50 - (__int8 *)v48);
  if ( v46 != v45[1] )
    _libc_free(v46);
  if ( v66 )
    j_j___libc_free_0(v66, v68 - (__int8 *)v66);
  if ( v64 != v63[1] )
    _libc_free(v64);
  if ( v60 )
    j_j___libc_free_0(v60, v62 - (__int8 *)v60);
  if ( v58 != v57[1] )
    _libc_free(v58);
  return a3;
}
