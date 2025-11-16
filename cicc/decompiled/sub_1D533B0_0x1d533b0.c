// Function: sub_1D533B0
// Address: 0x1d533b0
//
__int64 __fastcall sub_1D533B0(__int64 a1, __int64 a2, __int64 a3)
{
  _BYTE *v4; // rsi
  __int64 *v7; // rdi
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
  unsigned __int64 v24; // r13
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
  const __m128i *v35; // rax
  unsigned __int64 v36; // r13
  __int64 v37; // rax
  const __m128i *v38; // rdi
  __m128i *v39; // rsi
  __m128i *v40; // rax
  __int64 *v41; // rdx
  _QWORD *v42; // r8
  __int64 *v43; // rax
  __int64 v44; // r14
  __int64 *v45; // rax
  char v46; // dl
  __int64 v47; // rax
  __int64 *v48; // rcx
  __int64 *v49; // rsi
  __int64 *v50; // rdx
  __m128i *v51; // rax
  __m128i v53; // [rsp+0h] [rbp-440h] BYREF
  _QWORD v54[2]; // [rsp+10h] [rbp-430h] BYREF
  unsigned __int64 v55; // [rsp+20h] [rbp-420h]
  _BYTE v56[64]; // [rsp+38h] [rbp-408h] BYREF
  __m128i *v57; // [rsp+78h] [rbp-3C8h]
  __m128i *v58; // [rsp+80h] [rbp-3C0h]
  __int8 *v59; // [rsp+88h] [rbp-3B8h]
  _QWORD v60[2]; // [rsp+90h] [rbp-3B0h] BYREF
  unsigned __int64 v61; // [rsp+A0h] [rbp-3A0h]
  char v62[64]; // [rsp+B8h] [rbp-388h] BYREF
  const __m128i *v63; // [rsp+F8h] [rbp-348h]
  const __m128i *v64; // [rsp+100h] [rbp-340h]
  __int8 *v65; // [rsp+108h] [rbp-338h]
  _QWORD v66[2]; // [rsp+110h] [rbp-330h] BYREF
  unsigned __int64 v67; // [rsp+120h] [rbp-320h]
  _BYTE v68[64]; // [rsp+138h] [rbp-308h] BYREF
  __m128i *v69; // [rsp+178h] [rbp-2C8h]
  __m128i *v70; // [rsp+180h] [rbp-2C0h]
  __int8 *v71; // [rsp+188h] [rbp-2B8h]
  _QWORD v72[2]; // [rsp+190h] [rbp-2B0h] BYREF
  unsigned __int64 v73; // [rsp+1A0h] [rbp-2A0h]
  char v74[64]; // [rsp+1B8h] [rbp-288h] BYREF
  const __m128i *v75; // [rsp+1F8h] [rbp-248h]
  const __m128i *v76; // [rsp+200h] [rbp-240h]
  __int8 *v77; // [rsp+208h] [rbp-238h]
  _QWORD v78[2]; // [rsp+210h] [rbp-230h] BYREF
  unsigned __int64 v79; // [rsp+220h] [rbp-220h]
  _BYTE v80[64]; // [rsp+238h] [rbp-208h] BYREF
  __m128i *v81; // [rsp+278h] [rbp-1C8h]
  __m128i *v82; // [rsp+280h] [rbp-1C0h]
  __int8 *v83; // [rsp+288h] [rbp-1B8h]
  _QWORD v84[2]; // [rsp+290h] [rbp-1B0h] BYREF
  unsigned __int64 v85; // [rsp+2A0h] [rbp-1A0h]
  _BYTE v86[64]; // [rsp+2B8h] [rbp-188h] BYREF
  __m128i *v87; // [rsp+2F8h] [rbp-148h]
  __m128i *v88; // [rsp+300h] [rbp-140h]
  __int8 *v89; // [rsp+308h] [rbp-138h]
  __int64 v90; // [rsp+310h] [rbp-130h] BYREF
  __int64 *v91; // [rsp+318h] [rbp-128h]
  __int64 *v92; // [rsp+320h] [rbp-120h]
  unsigned int v93; // [rsp+328h] [rbp-118h]
  unsigned int v94; // [rsp+32Ch] [rbp-114h]
  int v95; // [rsp+330h] [rbp-110h]
  _BYTE v96[64]; // [rsp+338h] [rbp-108h] BYREF
  const __m128i *v97; // [rsp+378h] [rbp-C8h] BYREF
  __m128i *v98; // [rsp+380h] [rbp-C0h]
  __m128i *v99; // [rsp+388h] [rbp-B8h]
  _QWORD v100[2]; // [rsp+390h] [rbp-B0h] BYREF
  unsigned __int64 v101; // [rsp+3A0h] [rbp-A0h]
  _BYTE v102[64]; // [rsp+3B8h] [rbp-88h] BYREF
  __m128i *v103; // [rsp+3F8h] [rbp-48h]
  __m128i *v104; // [rsp+400h] [rbp-40h]
  __int8 *v105; // [rsp+408h] [rbp-38h]

  v4 = v68;
  v7 = v66;
  sub_16CCCB0(v66, (__int64)v68, a2);
  v8 = *(const __m128i **)(a2 + 112);
  v9 = *(const __m128i **)(a2 + 104);
  v69 = 0;
  v70 = 0;
  v71 = 0;
  v10 = (char *)v8 - (char *)v9;
  if ( v8 == v9 )
  {
    v11 = 0;
  }
  else
  {
    if ( v10 > 0x7FFFFFFFFFFFFFF0LL )
      goto LABEL_130;
    v11 = (__m128i *)sub_22077B0((char *)v8 - (char *)v9);
    v8 = *(const __m128i **)(a2 + 112);
    v9 = *(const __m128i **)(a2 + 104);
  }
  v69 = v11;
  v70 = v11;
  v71 = &v11->m128i_i8[v10];
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
        *v11 = _mm_loadu_si128(v9);
      ++v11;
      ++v9;
    }
    while ( v11 != v12 );
  }
  v70 = v12;
  sub_16CCEE0(v72, (__int64)v74, 8, (__int64)v66);
  v13 = v69;
  v7 = v54;
  v4 = v56;
  v69 = 0;
  v75 = v13;
  v14 = v70;
  v70 = 0;
  v76 = v14;
  v15 = v71;
  v71 = 0;
  v77 = v15;
  sub_16CCCB0(v54, (__int64)v56, a1);
  v16 = *(const __m128i **)(a1 + 112);
  v9 = *(const __m128i **)(a1 + 104);
  v57 = 0;
  v58 = 0;
  v59 = 0;
  v17 = (char *)v16 - (char *)v9;
  if ( v16 == v9 )
  {
    v18 = 0;
  }
  else
  {
    if ( v17 > 0x7FFFFFFFFFFFFFF0LL )
      goto LABEL_130;
    v18 = (__m128i *)sub_22077B0((char *)v16 - (char *)v9);
    v16 = *(const __m128i **)(a1 + 112);
    v9 = *(const __m128i **)(a1 + 104);
  }
  v57 = v18;
  v58 = v18;
  v59 = &v18->m128i_i8[v17];
  if ( v16 == v9 )
  {
    v19 = v18;
  }
  else
  {
    v19 = (__m128i *)((char *)v18 + (char *)v16 - (char *)v9);
    do
    {
      if ( v18 )
        *v18 = _mm_loadu_si128(v9);
      ++v18;
      ++v9;
    }
    while ( v18 != v19 );
  }
  v58 = v19;
  sub_16CCEE0(v60, (__int64)v62, 8, (__int64)v54);
  v20 = v57;
  v7 = v84;
  v4 = v86;
  v57 = 0;
  v63 = v20;
  v21 = v58;
  v58 = 0;
  v64 = v21;
  v22 = v59;
  v59 = 0;
  v65 = v22;
  sub_16CCCB0(v84, (__int64)v86, (__int64)v72);
  v23 = v76;
  v9 = v75;
  v87 = 0;
  v88 = 0;
  v89 = 0;
  v24 = (char *)v76 - (char *)v75;
  if ( v76 == v75 )
  {
    v25 = 0;
  }
  else
  {
    if ( v24 > 0x7FFFFFFFFFFFFFF0LL )
      goto LABEL_130;
    v25 = (__m128i *)sub_22077B0((char *)v76 - (char *)v75);
    v23 = v76;
    v9 = v75;
  }
  v87 = v25;
  v88 = v25;
  v89 = &v25->m128i_i8[v24];
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
        *v25 = _mm_loadu_si128(v9);
      ++v25;
      ++v9;
    }
    while ( v25 != v26 );
  }
  v4 = v80;
  v88 = v26;
  v7 = v78;
  sub_16CCCB0(v78, (__int64)v80, (__int64)v60);
  v27 = v64;
  v9 = v63;
  v81 = 0;
  v82 = 0;
  v83 = 0;
  v28 = (char *)v64 - (char *)v63;
  if ( v64 == v63 )
  {
    v28 = 0;
    v29 = 0;
  }
  else
  {
    if ( v28 > 0x7FFFFFFFFFFFFFF0LL )
      goto LABEL_130;
    v29 = (__m128i *)sub_22077B0((char *)v64 - (char *)v63);
    v27 = v64;
    v9 = v63;
  }
  v81 = v29;
  v82 = v29;
  v83 = &v29->m128i_i8[v28];
  if ( v9 == v27 )
  {
    v30 = v29;
  }
  else
  {
    v30 = (__m128i *)((char *)v29 + (char *)v27 - (char *)v9);
    do
    {
      if ( v29 )
        *v29 = _mm_loadu_si128(v9);
      ++v29;
      ++v9;
    }
    while ( v29 != v30 );
  }
  v7 = v100;
  v4 = v102;
  v82 = v30;
  sub_16CCCB0(v100, (__int64)v102, (__int64)v84);
  v31 = v88;
  v9 = v87;
  v103 = 0;
  v104 = 0;
  v105 = 0;
  v32 = (char *)v88 - (char *)v87;
  if ( v88 == v87 )
  {
    v33 = 0;
  }
  else
  {
    if ( v32 > 0x7FFFFFFFFFFFFFF0LL )
      goto LABEL_130;
    v33 = (__m128i *)sub_22077B0((char *)v88 - (char *)v87);
    v31 = v88;
    v9 = v87;
  }
  v103 = v33;
  v104 = v33;
  v105 = &v33->m128i_i8[v32];
  if ( v9 == v31 )
  {
    v34 = v33;
  }
  else
  {
    v34 = (__m128i *)((char *)v33 + (char *)v31 - (char *)v9);
    do
    {
      if ( v33 )
        *v33 = _mm_loadu_si128(v9);
      ++v33;
      ++v9;
    }
    while ( v33 != v34 );
  }
  v4 = v96;
  v104 = v34;
  v7 = &v90;
  sub_16CCCB0(&v90, (__int64)v96, (__int64)v78);
  v35 = v82;
  v9 = v81;
  v97 = 0;
  v98 = 0;
  v99 = 0;
  v36 = (char *)v82 - (char *)v81;
  if ( v82 == v81 )
  {
    v38 = 0;
    goto LABEL_44;
  }
  if ( v36 > 0x7FFFFFFFFFFFFFF0LL )
LABEL_130:
    sub_4261EA(v7, v4, v9);
  v37 = sub_22077B0((char *)v82 - (char *)v81);
  v9 = v81;
  v38 = (const __m128i *)v37;
  v35 = v82;
LABEL_44:
  v97 = v38;
  v98 = (__m128i *)v38;
  v99 = (__m128i *)((char *)v38 + v36);
  if ( v9 == v35 )
  {
    v39 = (__m128i *)v38;
  }
  else
  {
    v39 = (__m128i *)((char *)v38 + (char *)v35 - (char *)v9);
    v40 = (__m128i *)v38;
    do
    {
      if ( v40 )
        *v40 = _mm_loadu_si128(v9);
      ++v40;
      ++v9;
    }
    while ( v40 != v39 );
  }
  v98 = v39;
LABEL_50:
  while ( 1 )
  {
    v41 = (__int64 *)v103;
    if ( (char *)v39 - (char *)v38 == (char *)v104 - (char *)v103 )
      break;
LABEL_51:
    v42 = *(_QWORD **)(a3 + 8);
    if ( v42 == *(_QWORD **)(a3 + 16) )
    {
      sub_1D4AF10(a3, *(_BYTE **)(a3 + 8), (__m128i *)v39[-1].m128i_i64);
      v39 = v98;
    }
    else
    {
      if ( v42 )
      {
        *v42 = v39[-1].m128i_i64[0];
        v42 = *(_QWORD **)(a3 + 8);
        v39 = v98;
      }
      *(_QWORD *)(a3 + 8) = v42 + 1;
    }
    v38 = v97;
    v98 = --v39;
    if ( v39 != v97 )
    {
LABEL_56:
      v43 = (__int64 *)v39[-1].m128i_i64[1];
      if ( *(__int64 **)(v39[-1].m128i_i64[0] + 96) == v43 )
        goto LABEL_63;
      while ( 1 )
      {
        v39[-1].m128i_i64[1] = (__int64)(v43 + 1);
        v44 = *v43;
        v45 = v91;
        if ( v92 != v91 )
        {
LABEL_58:
          sub_16CCBA0((__int64)&v90, v44);
          v39 = v98;
          if ( v46 )
            goto LABEL_59;
          goto LABEL_56;
        }
        v48 = &v91[v94];
        if ( v91 == v48 )
          goto LABEL_76;
        v49 = 0;
        while ( 1 )
        {
          if ( v44 == *v45 )
          {
            v39 = v98;
            goto LABEL_56;
          }
          v50 = v45 + 1;
          if ( *v45 != -2 )
            break;
          if ( v48 == v50 )
          {
            v49 = v45;
LABEL_73:
            *v49 = v44;
            v39 = v98;
            --v95;
            ++v90;
            goto LABEL_59;
          }
LABEL_68:
          v49 = v45;
          v45 = v50;
        }
        if ( v48 != v50 )
          break;
        if ( v49 )
          goto LABEL_73;
LABEL_76:
        if ( v94 >= v93 )
          goto LABEL_58;
        ++v94;
        *v48 = v44;
        v39 = v98;
        ++v90;
LABEL_59:
        v47 = *(_QWORD *)(v44 + 88);
        v53.m128i_i64[0] = v44;
        v53.m128i_i64[1] = v47;
        if ( v99 == v39 )
        {
          sub_1D530F0(&v97, v39, &v53);
          v39 = v98;
          goto LABEL_56;
        }
        if ( v39 )
        {
          *v39 = _mm_loadu_si128(&v53);
          v39 = v98;
        }
        v98 = ++v39;
        v43 = (__int64 *)v39[-1].m128i_i64[1];
        if ( *(__int64 **)(v39[-1].m128i_i64[0] + 96) == v43 )
        {
LABEL_63:
          v38 = v97;
          goto LABEL_50;
        }
      }
      v45 = v49;
      goto LABEL_68;
    }
  }
  if ( v38 != v39 )
  {
    v51 = (__m128i *)v38;
    while ( v51->m128i_i64[0] == *v41 && v51->m128i_i64[1] == v41[1] )
    {
      ++v51;
      v41 += 2;
      if ( v39 == v51 )
        goto LABEL_83;
    }
    goto LABEL_51;
  }
LABEL_83:
  if ( v38 )
    j_j___libc_free_0(v38, (char *)v99 - (char *)v38);
  if ( v92 != v91 )
    _libc_free((unsigned __int64)v92);
  if ( v103 )
    j_j___libc_free_0(v103, v105 - (__int8 *)v103);
  if ( v101 != v100[1] )
    _libc_free(v101);
  if ( v81 )
    j_j___libc_free_0(v81, v83 - (__int8 *)v81);
  if ( v79 != v78[1] )
    _libc_free(v79);
  if ( v87 )
    j_j___libc_free_0(v87, v89 - (__int8 *)v87);
  if ( v85 != v84[1] )
    _libc_free(v85);
  if ( v63 )
    j_j___libc_free_0(v63, v65 - (__int8 *)v63);
  if ( v61 != v60[1] )
    _libc_free(v61);
  if ( v57 )
    j_j___libc_free_0(v57, v59 - (__int8 *)v57);
  if ( v55 != v54[1] )
    _libc_free(v55);
  if ( v75 )
    j_j___libc_free_0(v75, v77 - (__int8 *)v75);
  if ( v73 != v72[1] )
    _libc_free(v73);
  if ( v69 )
    j_j___libc_free_0(v69, v71 - (__int8 *)v69);
  if ( v67 != v66[1] )
    _libc_free(v67);
  return a3;
}
